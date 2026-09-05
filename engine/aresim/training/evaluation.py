"""Evaluate frozen checkpoints through the framework-neutral rollout path.

Compares learned policies against fixed seed splits and optional baselines using
:class:`~aresim.training.runner.RolloutRunner`. Does not reimplement simulator rules.

**Last updated:** September 1, 2026

**Contains:** :func:`evaluate_checkpoint`.

**Inputs:** RLlib checkpoint sidecar JSON, seed manifest YAML, experiment provenance.

**Outputs:** ``summary.json``, optional trajectory shards under an output directory.

**See also:** :mod:`aresim.algorithms.ppo.checkpoint` (loader),
:mod:`aresim.training.seeds` (manifest).
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from statistics import fmean

from ..defaults import DEFAULT_ENVIRONMENT_CONFIG
from ..factory import make_env
from .runner import RolloutConfig, RolloutRunner
from .trajectories import TrajectoryWriter, validate_trajectory_dataset
from ..algorithms.ppo.checkpoint import make_checkpoint_agent
from .experiments import ExperimentSpec, parse_experiment
from .seeds import SeedSplitManifest, load_seed_manifest


def _validate_provenance(spec: ExperimentSpec, seeds: SeedSplitManifest, sidecar: dict[str, object]) -> None:
    expected = (
        (seeds.scenario_id, spec.environment.scenario_id, "scenario"),
        (seeds.observation_schema, "aresim.obs.local.v1", "observation schema"),
        (seeds.action_schema, "aresim.action.rover.v1", "action schema"),
        (seeds.task_id, str(sidecar.get("task_id", "")), "task"),
        (seeds.reward_profile, str(sidecar.get("reward_profile", "")), "reward profile"),
    )
    for actual, configured, label in expected:
        if actual != configured:
            raise ValueError(f"seed manifest {label} is incompatible with the experiment")


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


def _summary_payload(result, split: str, checkpoint: Path, baselines: list[dict[str, object]]) -> dict[str, object]:
    summaries = result.summaries
    endings: dict[str, int] = {}
    for summary in summaries:
        key = summary.ending_reason or "unknown"
        endings[key] = endings.get(key, 0) + 1
    return {
        "schema_version": "aresim.evaluation.v1",
        "split": split,
        "checkpoint": str(checkpoint.resolve()),
        "episode_count": len(summaries),
        "transition_count": result.transition_count,
        "mean_shaped_return": fmean(item.episode_return for item in summaries),
        "mean_sparse_evaluation_return": fmean(_sparse_return(item) for item in result.episodes),
        "mean_engine_return": fmean(item.engine_return for item in summaries),
        "mean_episode_length": fmean(item.length for item in summaries),
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
    if sidecar.get("config_hash") != spec.config_hash:
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


def _baseline_comparisons(episodes, spec: ExperimentSpec, environment_config, registry) -> list[dict[str, object]]:
    comparisons = []
    for policy in ("random_valid", "scripted"):
        baseline = RolloutRunner(
            RolloutConfig(episodes=episodes, max_episode_steps=spec.environment.max_episode_steps),
            policy,
            environment_config=environment_config,
            registry=registry,
        ).run()
        comparisons.append({
            "policy_id": baseline.summaries[0].policy_id,
            "episode_count": len(baseline.summaries),
            "mean_shaped_return": fmean(item.episode_return for item in baseline.summaries),
            "mean_sparse_evaluation_return": fmean(_sparse_return(item) for item in baseline.episodes),
            "mean_engine_return": fmean(item.engine_return for item in baseline.summaries),
            "mean_episode_length": fmean(item.length for item in baseline.summaries),
        })
    return comparisons


def _write_evaluation(destination: Path, result, summary: dict[str, object]) -> None:
    partial = destination / "summary.json.partial"
    partial.write_text(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    partial.replace(destination / "summary.json")
    fieldnames = ["episode_id", "environment_seed", "agent_seed", "length", "shaped_return", "sparse_evaluation_return", "engine_return", "terminated", "truncated", "ending_reason"]
    with (destination / "seed_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for item, episode in zip(result.summaries, result.episodes, strict=True):
            writer.writerow({
                "episode_id": item.episode_id, "environment_seed": item.environment_seed,
                "agent_seed": item.agent_seed, "length": item.length,
                "shaped_return": item.episode_return, "engine_return": item.engine_return,
                "sparse_evaluation_return": _sparse_return(episode),
                "terminated": item.terminated, "truncated": item.truncated,
                "ending_reason": item.ending_reason,
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
    agent = make_checkpoint_agent(checkpoint_path, registry=registry)
    runner = RolloutRunner(
        RolloutConfig(episodes=episodes, max_episode_steps=spec.environment.max_episode_steps),
        agent,
        environment_config=environment_config,
        registry=component_registry,
    )
    destination = Path(output_directory) if output_directory is not None else checkpoint_path.parents[2] / "evaluation" / split
    if destination.exists():
        raise FileExistsError(f"evaluation output already exists: {destination}")
    destination.mkdir(parents=True)
    should_record = spec.evaluation.record_trajectories if record_trajectories is None else record_trajectories
    writer = TrajectoryWriter(destination / "trajectories", f"{spec.experiment_id}-{spec.trial_id}-{split}", compression="gzip") if should_record else None
    result = runner.run(writer)
    if writer is not None:
        validate_trajectory_dataset(destination / "trajectories")
    baselines = _baseline_comparisons(episodes, spec, environment_config, component_registry)
    _write_evaluation(destination, result, _summary_payload(result, split, checkpoint_path, baselines))
    return destination


__all__ = ["evaluate_checkpoint"]
