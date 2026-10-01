"""Evaluate frozen checkpoints through the framework-neutral rollout path.

Compares learned policies against fixed seed splits and optional baselines using
:class:`~aresim.training.runner.RolloutRunner`. Does not reimplement simulator rules.

When trajectory recording is off, validation episodes and baselines run in a
CPU process pool (not training EnvRunners). Recording stays sequential.

**Last updated:** September 22, 2026

**Contains:** :func:`evaluate_checkpoint`.

**Inputs:** RLlib checkpoint sidecar JSON, seed manifest YAML, experiment provenance.

**Outputs:** ``summary.json``, optional trajectory shards under an output directory.

**See also:** :mod:`aresim.algorithms.ppo.checkpoint` (loader),
:mod:`aresim.training.seeds` (manifest).
"""

from __future__ import annotations

import csv
import json
import multiprocessing
import os
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean
from typing import Any

from ..defaults import DEFAULT_ENVIRONMENT_CONFIG
from ..factory import make_agent, make_env
from .experiments import ExperimentSpec, experiment_payload_hash, parse_experiment
from .runner import EpisodeSpec, RolloutConfig, RolloutResult, RolloutRunner
from .seeds import SeedSplitManifest, load_seed_manifest
from .trajectories import TrajectoryWriter, validate_trajectory_dataset


_CHECKPOINT_AGENT: Any = None


@dataclass(frozen=True)
class _EvalJob:
    """Picklable eval work item. Do not send EnvironmentConfig — its mappings are mappingproxies."""

    scenario_id: str
    observation: str
    action: str
    reward: str
    task: str
    max_episode_steps: int
    episode: EpisodeSpec
    agent_name: str
    checkpoint: str | None


@dataclass(frozen=True)
class _EvalRow:
    """Compact per-episode eval record; avoids pickling full trajectories."""

    policy_id: str
    episode_id: str
    environment_seed: int
    agent_seed: int
    length: int
    shaped_return: float
    sparse_return: float
    engine_return: float
    terminated: bool
    truncated: bool
    ending_reason: str | None


def _expected_task_id(task_name: str) -> str:
    from ..components.tasks import RESOURCE_MISSION_ID, TASK_ID

    return {"open_exploration": TASK_ID, "resource_mission": RESOURCE_MISSION_ID}.get(task_name, task_name)


def _validate_provenance(spec: ExperimentSpec, seeds: SeedSplitManifest, sidecar: dict[str, object]) -> None:
    """Seed lists are world-generation seeds. Task may differ (open exploration vs mission)."""
    expected = (
        (seeds.scenario_id, spec.environment.scenario_id, "scenario"),
        (seeds.observation_schema, "aresim.obs.local.v1", "observation schema"),
        (seeds.action_schema, "aresim.action.rover.v1", "action schema"),
        (seeds.reward_profile, str(sidecar.get("reward_profile", "")), "reward profile"),
    )
    for actual, configured, label in expected:
        if actual != configured:
            raise ValueError(f"seed manifest {label} is incompatible with the experiment")
    sidecar_task = str(sidecar.get("task_id", ""))
    if sidecar_task != _expected_task_id(spec.environment.task):
        raise ValueError("checkpoint task is incompatible with the experiment")


def _sparse_return(episode) -> float:
    selected = {"mission_success", "terminal_failure", "invalid_action"}
    total = 0.0
    for breakdown in episode.reward_breakdowns:
        terms = breakdown.get("terms", {})
        for name in selected:
            term = terms.get(name, {})
            value = term.get("value", 0.0) if isinstance(term, dict) else 0.0
            total += float(value)
    return total


def _rows_from_result(result: RolloutResult) -> list[_EvalRow]:
    return [
        _EvalRow(
            policy_id=summary.policy_id,
            episode_id=summary.episode_id,
            environment_seed=summary.environment_seed,
            agent_seed=summary.agent_seed,
            length=summary.length,
            shaped_return=summary.episode_return,
            sparse_return=_sparse_return(episode),
            engine_return=summary.engine_return,
            terminated=summary.terminated,
            truncated=summary.truncated,
            ending_reason=summary.ending_reason,
        )
        for summary, episode in zip(result.summaries, result.episodes, strict=True)
    ]


def _baseline_payload(rows: list[_EvalRow]) -> dict[str, object]:
    return {
        "policy_id": rows[0].policy_id,
        "episode_count": len(rows),
        "mean_shaped_return": fmean(item.shaped_return for item in rows),
        "mean_sparse_evaluation_return": fmean(item.sparse_return for item in rows),
        "mean_engine_return": fmean(item.engine_return for item in rows),
        "mean_episode_length": fmean(item.length for item in rows),
    }


def _summary_payload(rows: list[_EvalRow], split: str, checkpoint: Path, baselines: list[dict[str, object]]) -> dict[str, object]:
    endings: dict[str, int] = {}
    for row in rows:
        key = row.ending_reason or "unknown"
        endings[key] = endings.get(key, 0) + 1
    return {
        "schema_version": "aresim.evaluation.v1",
        "split": split,
        "checkpoint": str(checkpoint.resolve()),
        "episode_count": len(rows),
        "transition_count": sum(row.length for row in rows),
        "mean_shaped_return": fmean(item.shaped_return for item in rows),
        "mean_sparse_evaluation_return": fmean(item.sparse_return for item in rows),
        "mean_engine_return": fmean(item.engine_return for item in rows),
        "mean_episode_length": fmean(item.length for item in rows),
        "terminal_reasons": endings,
        "success_metric": None,
        "promoted_checkpoint": None,
        "baseline_comparisons": baselines,
    }


def _load_evaluation(checkpoint: Path, split: str) -> tuple[ExperimentSpec, tuple, dict[str, object]]:
    sidecar = json.loads(checkpoint.read_text(encoding="utf-8"))
    experiment = sidecar.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("checkpoint sidecar is missing its resolved experiment")
    spec = parse_experiment(experiment)
    if sidecar.get("config_hash") != experiment_payload_hash(experiment):
        raise ValueError("checkpoint configuration hash does not match its experiment")
    seeds = load_seed_manifest(spec.evaluation.seed_manifest)
    _validate_provenance(spec, seeds, sidecar)
    return spec, seeds.episodes(split), sidecar


def _environment_config(spec: ExperimentSpec):
    return replace(
        DEFAULT_ENVIRONMENT_CONFIG,
        scenario_id=spec.environment.scenario_id,
        observation=spec.environment.observation,
        action=spec.environment.action,
        reward=spec.environment.reward,
        task=spec.environment.task,
    )


def _validate_environment(config, seed: int, sidecar: dict[str, object], registry) -> None:
    reset = make_env(config, registry=registry).reset(seed=seed)
    for key in ("observation_schema", "action_schema", "task_id", "reward_profile"):
        if str(reset.info[key]) != str(sidecar.get(key, "")):
            raise ValueError(f"checkpoint {key} is incompatible with the evaluation environment")


def _eval_worker_init(checkpoint: str | None) -> None:
    """Pin each worker to CPU before it loads torch/Ray."""
    global _CHECKPOINT_AGENT
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["RAY_NUM_CPUS"] = "1"
    _CHECKPOINT_AGENT = None
    if not checkpoint:
        return
    from ..algorithms.ppo.checkpoint import make_checkpoint_agent

    _CHECKPOINT_AGENT = make_checkpoint_agent(checkpoint)


def _job_environment_config(job: _EvalJob):
    return replace(
        DEFAULT_ENVIRONMENT_CONFIG,
        scenario_id=job.scenario_id,
        observation=job.observation,
        action=job.action,
        reward=job.reward,
        task=job.task,
    )


def _eval_episode_job(job: _EvalJob) -> _EvalRow:
    if job.agent_name == "checkpoint":
        agent = _CHECKPOINT_AGENT
        if agent is None:
            from ..algorithms.ppo.checkpoint import make_checkpoint_agent

            agent = make_checkpoint_agent(job.checkpoint)
    else:
        agent = job.agent_name
    return _summary_episode(
        agent, job.episode, _job_environment_config(job), job.max_episode_steps
    )


def _summary_episode(agent, episode: EpisodeSpec, config, max_steps: int, registry=None) -> _EvalRow:
    """Evaluate one episode without constructing replay snapshots or trajectories."""
    policy = make_agent(agent, config, registry) if isinstance(agent, str) else agent
    environment = make_env(config, registry=registry, max_episode_steps=max_steps, audit=False)
    reset = environment.reset(seed=episode.environment_seed)
    if policy.action_schema != reset.info["action_schema"]:
        raise ValueError("evaluation agent action schema is incompatible")
    if policy.observation_schema is not None and policy.observation_schema != reset.info["observation_schema"]:
        raise ValueError("evaluation agent observation schema is incompatible")
    policy.reset(episode.agent_seed)
    observation, mask = reset.observation, reset.action_mask
    length = 0
    shaped_rewards: list[float] = []
    engine_rewards: list[float] = []
    sparse_return = 0.0
    while True:
        action = policy.act(deepcopy(observation), mask.copy())
        if not environment.action_space.contains(action):
            raise ValueError("agent returned an action outside the environment action space")
        result = environment.step(action)
        length += 1
        shaped_rewards.append(float(result.reward))
        engine_rewards.append(float(result.transition.reward))
        for name in {"mission_success", "terminal_failure", "invalid_action"}:
            sparse_return += result.reward_breakdown.terms[name].value
        if result.terminated or result.truncated:
            return _EvalRow(
                policy_id=policy.policy_id,
                episode_id=episode.episode_id,
                environment_seed=episode.environment_seed,
                agent_seed=episode.agent_seed,
                length=length,
                shaped_return=float(sum(shaped_rewards)),
                sparse_return=sparse_return,
                engine_return=float(sum(engine_rewards)),
                terminated=result.terminated,
                truncated=result.truncated,
                ending_reason=result.info.get("terminal_reason") or result.info.get("truncation_reason"),
            )
        observation, mask = result.observation, result.action_mask


def _summary_episodes(agent, episodes: tuple[EpisodeSpec, ...], config, max_steps: int, registry=None) -> list[_EvalRow]:
    policy = make_agent(agent, config, registry) if isinstance(agent, str) else agent
    return [_summary_episode(policy, episode, config, max_steps, registry) for episode in episodes]


def _eval_worker_count(job_count: int) -> int:
    cpus = os.process_cpu_count() if hasattr(os, "process_cpu_count") else os.cpu_count()
    return max(1, min(job_count, cpus or 1))


def _report_eval_progress(done: int, total: int) -> None:
    if done in {1, total} or done % max(1, total // 4) == 0:
        print(f"Frozen evaluation {done}/{total} rollouts", flush=True)


def _map_eval_jobs(
    jobs: list[_EvalJob],
    *,
    checkpoint: str | None,
    workers: int,
) -> list[_EvalRow]:
    if workers <= 1:
        return [_eval_episode_job(job) for job in jobs]
    rows: list[_EvalRow | None] = [None] * len(jobs)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_eval_worker_init,
        initargs=(checkpoint,),
    ) as pool:
        pending = {pool.submit(_eval_episode_job, job): index for index, job in enumerate(jobs)}
        for done, future in enumerate(as_completed(pending), start=1):
            rows[pending[future]] = future.result()
            _report_eval_progress(done, len(jobs))
    filled = []
    for row in rows:
        if row is None:
            raise RuntimeError("evaluation worker returned no result")
        filled.append(row)
    return filled


def _evaluate_sequential(
    checkpoint_path: Path,
    spec: ExperimentSpec,
    episodes: tuple[EpisodeSpec, ...],
    environment_config,
    component_registry,
    registry,
    destination: Path,
    split: str,
    record: bool,
) -> tuple[list[_EvalRow], list[dict[str, object]]]:
    from ..algorithms.ppo.checkpoint import make_checkpoint_agent

    agent = make_checkpoint_agent(checkpoint_path, registry=registry)
    writer = (
        TrajectoryWriter(destination / "trajectories", f"{spec.experiment_id}-{spec.trial_id}-{split}", compression="gzip")
        if record
        else None
    )
    if writer is None:
        rows = _summary_episodes(agent, episodes, environment_config, spec.environment.max_episode_steps, component_registry)
    else:
        result = RolloutRunner(
            RolloutConfig(episodes=episodes, max_episode_steps=spec.environment.max_episode_steps),
            agent,
            environment_config=environment_config,
            registry=component_registry,
        ).run(writer)
        validate_trajectory_dataset(destination / "trajectories")
        rows = _rows_from_result(result)
    baselines = []
    for policy in ("random_valid", "scripted"):
        baseline_rows = _summary_episodes(policy, episodes, environment_config, spec.environment.max_episode_steps, component_registry)
        baselines.append(_baseline_payload(baseline_rows))
    return rows, baselines


def _evaluate_parallel(
    checkpoint_path: Path,
    spec: ExperimentSpec,
    episodes: tuple[EpisodeSpec, ...],
    environment_config,
    split: str,
) -> tuple[list[_EvalRow], list[dict[str, object]]]:
    max_steps = spec.environment.max_episode_steps
    checkpoint = str(checkpoint_path)
    ids = (
        environment_config.scenario_id,
        environment_config.observation,
        environment_config.action,
        environment_config.reward,
        environment_config.task,
    )
    jobs = [
        _EvalJob(*ids, max_steps, episode, "checkpoint", checkpoint)
        for episode in episodes
    ]
    jobs.extend(_EvalJob(*ids, max_steps, episode, "random_valid", None) for episode in episodes)
    jobs.extend(_EvalJob(*ids, max_steps, episode, "scripted", None) for episode in episodes)
    workers = _eval_worker_count(len(jobs))
    print(
        f"Frozen evaluation: {len(episodes)} {split} episodes plus random_valid and scripted "
        f"({len(jobs)} rollouts, {workers} workers)",
        flush=True,
    )
    rows = _map_eval_jobs(jobs, checkpoint=checkpoint, workers=workers)
    count = len(episodes)
    return rows[:count], [_baseline_payload(rows[count : 2 * count]), _baseline_payload(rows[2 * count :])]


def _write_evaluation(destination: Path, rows: list[_EvalRow], summary: dict[str, object]) -> None:
    partial = destination / "summary.json.partial"
    partial.write_text(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    partial.replace(destination / "summary.json")
    fieldnames = [
        "episode_id",
        "environment_seed",
        "agent_seed",
        "length",
        "shaped_return",
        "sparse_evaluation_return",
        "engine_return",
        "terminated",
        "truncated",
        "ending_reason",
    ]
    with (destination / "seed_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "episode_id": row.episode_id,
                "environment_seed": row.environment_seed,
                "agent_seed": row.agent_seed,
                "length": row.length,
                "shaped_return": row.shaped_return,
                "engine_return": row.engine_return,
                "sparse_evaluation_return": row.sparse_return,
                "terminated": row.terminated,
                "truncated": row.truncated,
                "ending_reason": row.ending_reason,
            })


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    split: str = "validation",
    output_directory: str | Path | None = None,
    registry=None,
    record_trajectories: bool | None = None,
    component_registry=None,
) -> Path:
    """Evaluate a native checkpoint adapter on a fixed validation or test split."""
    checkpoint_path = Path(checkpoint).resolve()
    spec, episodes, sidecar = _load_evaluation(checkpoint_path, split)
    environment_config = _environment_config(spec)
    _validate_environment(environment_config, episodes[0].environment_seed, sidecar, component_registry)
    destination = Path(output_directory) if output_directory is not None else checkpoint_path.parents[2] / "evaluation" / split
    if destination.exists():
        raise FileExistsError(f"evaluation output already exists: {destination}")
    destination.mkdir(parents=True)
    should_record = spec.evaluation.record_trajectories if record_trajectories is None else record_trajectories
    job_count = len(episodes) * 3
    if should_record or _eval_worker_count(job_count) <= 1:
        print("Frozen evaluation running sequentially", flush=True)
        rows, baselines = _evaluate_sequential(
            checkpoint_path, spec, episodes, environment_config, component_registry, registry, destination, split, should_record,
        )
    else:
        rows, baselines = _evaluate_parallel(checkpoint_path, spec, episodes, environment_config, split)
    _write_evaluation(destination, rows, _summary_payload(rows, split, checkpoint_path, baselines))
    return destination


__all__ = ["evaluate_checkpoint"]
