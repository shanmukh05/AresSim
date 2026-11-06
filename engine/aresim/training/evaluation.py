"""Evaluate frozen checkpoints through the framework-neutral rollout path.

Evaluates learned policies against fixed seed splits using
:class:`~aresim.training.runner.RolloutRunner`. Does not reimplement simulator rules.

Evaluation runs only the checkpoint policy. With recording off and multiple evaluation workers,
rollouts run in a CPU process pool (not training EnvRunners).

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
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import fmean
from typing import Any

from ..defaults import DEFAULT_ENVIRONMENT_CONFIG
from ..factory import make_agent, make_env
from .provenance import digest, environment_config, environment_from_payload, experiment_seeds, serialize, verify_provenance
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
    checkpoint: str
    resolved_config: dict[str, object] | None = None
    deterministic: bool = True


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
    success: bool = False
    ice_delivered: float = 0.0
    samples_delivered: float = 0.0
    action_counts: dict[str, int] = field(default_factory=dict)
    extract_opportunities: int = 0
    action_mode: str = "deterministic"


def _expected_task_id(task_name: str) -> str:
    from ..components.tasks import RESOURCE_MISSION_ID, TASK_ID

    return {"open_exploration": TASK_ID, "resource_mission": RESOURCE_MISSION_ID}.get(task_name, task_name)


def _validate_provenance(spec: ExperimentSpec, seeds: SeedSplitManifest, sidecar: dict[str, object]) -> None:
    """Seed lists are world-generation seeds. Task may differ (open exploration vs mission)."""
    expected = (
        (seeds.scenario_id, spec.environment.scenario_id, "scenario"),
        (seeds.observation_schema, str(sidecar.get("observation_schema", "")), "observation schema"),
        (seeds.action_schema, "aresim.action.rover.v1", "action schema"),
        (seeds.reward_profile, str(sidecar.get("reward_profile", "")), "reward profile"),
    )
    for actual, configured, label in expected:
        if actual != configured:
            raise ValueError(f"seed manifest {label} is incompatible with the experiment")
    sidecar_task = str(sidecar.get("task_id", ""))
    if spec.environment.task in {"resource_mission", "open_exploration"} and sidecar_task != _expected_task_id(spec.environment.task):
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



def _recorded_delivery(episode, cargo_index):
    capacity = episode.environment_config.engine.payload.capacity_kg
    return sum(max(0., float(episode.observations[index]["self"][cargo_index]) -
                       float(episode.observations[index + 1]["self"][cargo_index])) * capacity
               for index, action in enumerate(episode.effective_actions) if action == "unload")

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
            success=summary.ending_reason == "mission_complete",
            ice_delivered=_recorded_delivery(episode, 4),
            samples_delivered=_recorded_delivery(episode, 6),
            action_counts={name: episode.effective_actions.count(name) for name in set(episode.effective_actions)},
            extract_opportunities=sum(int(bool(mask[6])) for mask in episode.action_masks[:-1]),
        )
        for summary, episode in zip(result.summaries, result.episodes, strict=True)
    ]


def _summary_payload(rows: list[_EvalRow], split: str, checkpoint: Path) -> dict[str, object]:
    endings: dict[str, int] = {}
    actions: dict[str, int] = {}
    for row in rows:
        key = row.ending_reason or "unknown"
        endings[key] = endings.get(key, 0) + 1
        for name, count in row.action_counts.items():
            actions[name] = actions.get(name, 0) + count
    return {
        "schema_version": "aresim.evaluation.v2",
        "split": split,
        "checkpoint": str(checkpoint.resolve()),
        "episode_count": len(rows),
        "transition_count": sum(row.length for row in rows),
        "mean_shaped_return": fmean(item.shaped_return for item in rows),
        "mean_sparse_evaluation_return": fmean(item.sparse_return for item in rows),
        "mean_engine_return": fmean(item.engine_return for item in rows),
        "mean_episode_length": fmean(item.length for item in rows),
        "success_count": sum(item.success for item in rows),
        "success_rate": fmean(item.success for item in rows),
        "mean_ice_delivered": fmean(item.ice_delivered for item in rows),
        "mean_samples_delivered": fmean(item.samples_delivered for item in rows),
        "terminal_reasons": endings,
        "action_counts": actions,
        "extract_opportunities": sum(row.extract_opportunities for row in rows),
        "episodes_with_extraction": sum(row.action_counts.get("extract", 0) > 0 for row in rows),
        "episodes_with_ice_delivery": sum(row.ice_delivered > 0 for row in rows),
        "episodes_with_sample_delivery": sum(row.samples_delivered > 0 for row in rows),
        "success_metric": fmean(item.success for item in rows),
        "promoted_checkpoint": None,
    }


def _load_evaluation(checkpoint: Path, split: str, *, allow_environment_mismatch=False) -> tuple[ExperimentSpec, tuple, dict[str, object]]:
    sidecar = json.loads(checkpoint.read_text(encoding="utf-8"))
    experiment = sidecar.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("checkpoint sidecar is missing its resolved experiment")
    spec = parse_experiment(experiment)
    if sidecar.get("config_hash") != experiment_payload_hash(experiment):
        raise ValueError("checkpoint configuration hash does not match its experiment")
    verify_provenance(spec, allow_environment_mismatch=allow_environment_mismatch)
    seeds = experiment_seeds(spec)
    if not allow_environment_mismatch:
        _validate_provenance(spec, seeds, sidecar)
    return spec, seeds.episodes(split), sidecar


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
    from . import make_checkpoint_agent

    _CHECKPOINT_AGENT = make_checkpoint_agent(checkpoint)


def _job_environment_config(job: _EvalJob):
    return environment_from_payload(job.resolved_config)


def _eval_episode_job(job: _EvalJob) -> _EvalRow:
    agent = _CHECKPOINT_AGENT
    if agent is None:
        from . import make_checkpoint_agent

        agent = make_checkpoint_agent(job.checkpoint, deterministic=job.deterministic)
    agent.deterministic = job.deterministic
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
    action_counts = {}
    extract_opportunities = 0
    while True:
        action = policy.act(deepcopy(observation), mask.copy())
        if not environment.action_space.contains(action):
            raise ValueError("agent returned an action outside the environment action space")
        extract_opportunities += int(bool(mask[6]))
        result = environment.step(action)
        name = result.info["effective_action"]
        action_counts[name] = action_counts.get(name, 0) + 1
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
                success=bool(result.info["success"]),
                ice_delivered=result.info["objective_stats"]["ice_delivered"],
                samples_delivered=result.info["objective_stats"]["samples_delivered"],
                action_counts=action_counts, extract_opportunities=extract_opportunities,
                action_mode="stochastic" if not getattr(policy, "deterministic", True) else "deterministic",
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
    deterministic: bool = True,
) -> list[_EvalRow]:
    from . import make_checkpoint_agent

    agent = make_checkpoint_agent(checkpoint_path, registry=registry, deterministic=deterministic)
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
    return rows


def _evaluate_parallel(
    checkpoint_path: Path,
    spec: ExperimentSpec,
    episodes: tuple[EpisodeSpec, ...],
    environment_config,
    split: str,
    deterministic: bool = True,
) -> list[_EvalRow]:
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
        _EvalJob(*ids, max_steps, episode, checkpoint, serialize(environment_config), deterministic)
        for episode in episodes
    ]
    workers = min(spec.evaluation.workers, _eval_worker_count(len(jobs)))
    print(
        f"Frozen evaluation: {len(episodes)} {split} episodes "
        f"({len(jobs)} rollouts, {workers} workers)",
        flush=True,
    )
    return _map_eval_jobs(jobs, checkpoint=checkpoint, workers=workers)


def _write_evaluation(destination: Path, rows: list[_EvalRow], summary: dict[str, object]) -> None:
    partial = destination / "summary.json.partial"
    partial.write_text(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    partial.replace(destination / "summary.json")
    fieldnames = [
        "episode_id", "action_mode", "success", "ice_delivered", "samples_delivered", "extract_opportunities", "action_counts",
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
                "episode_id": row.episode_id, "action_mode": row.action_mode, "success": row.success,
                "ice_delivered": row.ice_delivered, "samples_delivered": row.samples_delivered,
                "extract_opportunities": row.extract_opportunities, "action_counts": json.dumps(row.action_counts, sort_keys=True),
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
    action_mode: str | None = None,
    allow_environment_mismatch: bool = False,
) -> Path:
    """Evaluate only the checkpoint policy on a fixed validation or test split."""
    checkpoint_path = Path(checkpoint).resolve()
    spec, episodes, sidecar = _load_evaluation(checkpoint_path, split, allow_environment_mismatch=allow_environment_mismatch)
    config = environment_config(spec)
    if not allow_environment_mismatch:
        _validate_environment(config, episodes[0].environment_seed, sidecar, component_registry)
    destination = Path(output_directory) if output_directory is not None else checkpoint_path.parents[2] / "evaluation" / split
    if destination.exists():
        raise FileExistsError(f"evaluation output already exists: {destination}")
    mode = action_mode or spec.evaluation.action_mode
    if mode not in {"deterministic", "stochastic", "both"}:
        raise ValueError("invalid evaluation action mode")
    modes = ("deterministic", "stochastic") if mode == "both" else (mode,)
    destination.mkdir(parents=True)
    should_record = spec.evaluation.record_trajectories if record_trajectories is None else record_trajectories
    summaries, all_rows = {}, []
    for selected_mode in modes:
        deterministic = selected_mode == "deterministic"
        selected_episodes = episodes if deterministic else tuple(
            EpisodeSpec(f"{ep.episode_id}_repeat_{repeat}", ep.environment_seed,
                        ep.agent_seed + repeat * 1000003)
            for ep in episodes for repeat in range(spec.evaluation.stochastic_repeats))
        mode_destination = destination / selected_mode
        mode_destination.mkdir()
        if should_record or spec.evaluation.workers == 1 or registry is not None or component_registry is not None:
            rows = _evaluate_sequential(checkpoint_path, spec, selected_episodes, config,
                component_registry, registry, mode_destination, split, should_record, deterministic)
        else:
            rows = _evaluate_parallel(checkpoint_path, spec, selected_episodes, config, split, deterministic)
        rows = [replace(row, action_mode=selected_mode) for row in rows]
        summary = _summary_payload(rows, split, checkpoint_path)
        targets = config.task_config
        requirements = [("ice_delivered", targets.ice_target_kg), ("samples_delivered", targets.sample_target_kg)]
        requirements = [(name, target) for name, target in requirements if target > 0]
        summary["mean_capped_delivery_progress"] = fmean(
            sum(min(1., getattr(row, name) / target) for name, target in requirements) / len(requirements)
            for row in rows)
        summary["success_metric"] = summary["success_rate"] if spec.environment.task == "resource_mission" else None
        summary.update(action_mode=selected_mode, environment_comparable=verify_provenance(
            spec, allow_environment_mismatch=allow_environment_mismatch))
        summary.update(config_hash=spec.config_hash, checkpoint_sha256=digest(sidecar))
        if spec.algorithm in {"dreamerv3", "plan2explore"}:
            summary["policy_inference"] = {"latent_mode": "modal" if deterministic else "seeded_sample",
                                           "phase": sidecar.get("phase"), "memory_reset_each_episode": True}
        _write_evaluation(mode_destination, rows, summary)
        summaries[selected_mode] = summary
        all_rows.extend(rows)
    # Keep top-level fields for report consumers, using deterministic when available.
    primary = dict(summaries.get("deterministic", summaries[modes[0]]))
    primary.update(schema_version="aresim.evaluation.v2", action_modes=summaries,
                   checkpoint_selection_metric="validation mission success, then capped deliveries, then sparse return")
    _write_evaluation(destination, all_rows, primary)
    return destination


__all__ = ["evaluate_checkpoint"]
