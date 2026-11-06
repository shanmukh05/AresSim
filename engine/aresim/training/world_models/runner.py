"""Train Dreamer and Plan2Explore with bounded Ray collection and full-state exports.

The runner owns budgets, provenance, transfer/resume, tracking, and cleanup. It never evaluates,
selects checkpoints on held-out seeds, or runs baseline policies during training.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import time
import numpy as np
import torch
from ...algorithms.registry import TrainingContext, create_training_registry
from ..resume import require_resume_compatible
from ...algorithms.common.tracking import new_wandb_run_id
from ...algorithms.dreamerv3.config import DreamerConfig, DreamerModelConfig, WorldModelResources
from ...algorithms.dreamerv3.checkpoint import validate_sidecar, write_sidecar
from ...algorithms.plan2explore.config import Plan2ExploreConfig
from ...algorithms.plan2explore.checkpoint import validate_transfer
from ...factory import make_env
from ..artifacts import atomic_json, numbered_checkpoints, latest_checkpoint
from ..experiments import ExperimentSpec
from ..provenance import environment_config, experiment_seeds, freeze_experiment, verify_provenance, serialize
from .learner import DreamerLearner
from .replay import SequenceReplay
from .collection import RayCollectionPool


def cpu_state(value):
    """Copy tensors before publishing immutable CPU inference snapshots."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_state(item) for item in value)
    return value


def _preflight(spec, registry, component_registry, *, resuming: bool):
    if not isinstance(spec.algorithm_config, DreamerConfig) or not isinstance(spec.model_config, DreamerModelConfig) or not isinstance(spec.resources, WorldModelResources):
        raise TypeError("world-model experiment has incompatible config types")
    _validate_resources(spec)
    context = TrainingContext(spec, component_registry=component_registry)
    model = registry.build_model(spec.model, context)
    algorithm = registry.build_algorithm(spec.algorithm, TrainingContext(spec, model, component_registry))
    algorithm.build(TrainingContext(spec, model, component_registry))
    environment = make_env(environment_config(spec), registry=component_registry, max_episode_steps=spec.environment.max_episode_steps, audit=False)
    reset = environment.reset(seed=spec.learner_seed)
    declared = (algorithm.observation_schema, algorithm.action_schema)
    if (model.observation_schema, model.action_schema) != declared or (reset.info["observation_schema"], reset.info["action_schema"]) != declared or environment.action_space.n != 10:
        raise ValueError("world-model environment and model schemas differ")
    _validate_initialization(spec, resuming)
    experiment_seeds(spec)
    schemas = {key: str(reset.info[key]) for key in ("observation_schema", "action_schema", "task_id", "reward_profile")}
    return environment.observation_space, schemas



def _validate_resources(spec: ExperimentSpec) -> None:
    if spec.resources.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("world-model experiment requests unavailable CUDA")
    active_budget = spec.environment.max_episode_steps * spec.resources.num_collectors * spec.resources.num_envs_per_collector
    if spec.algorithm_config.replay_capacity < active_budget:
        raise ValueError("replay capacity must contain the maximum active histories")


def _validate_initialization(spec: ExperimentSpec, resuming: bool) -> None:
    config = spec.algorithm_config
    if not isinstance(config, Plan2ExploreConfig) or config.phase != "adapt" or resuming:
        return
    if config.initialization_checkpoint is None:
        raise ValueError("fresh Plan2Explore adaptation requires initialization_checkpoint")

def _transfer(spec, learner: DreamerLearner):
    requested = spec.algorithm_config.initialization_checkpoint
    sidecar = latest_checkpoint(requested)
    payload, parent, native = validate_transfer(sidecar, spec)
    learner.load_model_state(torch.load(native / "model.pt", map_location=learner.device, weights_only=True), transfer=True)
    trainer = torch.load(native / "trainer.pt", map_location="cpu", weights_only=False)
    replay = trainer["replay"].for_adaptation()
    replay.resize(spec.algorithm_config.replay_capacity)
    lineage = {"policy_id": payload["policy_id"], "checkpoint_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
               "config_hash": payload["config_hash"], "environment_steps": payload["environment_steps"], "experiment": payload["experiment"]}
    return replay, lineage


def _manifest(directory: Path, spec, status: str, counters: dict, run_id: str, parent, elapsed: float) -> None:
    atomic_json(directory / "manifest.json", {"schema_version": "aresim.run.v1", "framework_id": "torch", "status": status,
        "config_hash": spec.config_hash, "experiment": spec.as_dict(), "environment_steps": counters["steps"],
        "elapsed_seconds": elapsed, "wandb_run_id": run_id, "phase": getattr(spec.algorithm_config, "phase", "task"),
        "parent": parent, "open_exploration_has_success": spec.environment.task != "open_exploration", "promoted_checkpoint": None,
        "installed_versions": {"python": platform.python_version(), "torch": torch.__version__},
        "artifacts": [{"path": str(p.relative_to(directory)), "size": p.stat().st_size} for p in sorted(directory.glob("*.yaml"))]})


def _checkpoint(directory: Path, label: str, spec, schemas, learner: DreamerLearner, replay: SequenceReplay,
                rng, counters, pool: RayCollectionPool, parent) -> Path:
    exports = directory / "checkpoints"
    exports.mkdir(exist_ok=True)
    staging = exports / ("." + label + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    native = staging / "native"
    native.mkdir(parents=True)
    torch.save(cpu_state(learner.model_state()), native / "model.pt")
    torch.save({"optimizers": learner.optimizer_state(), "replay": replay, "rng": rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if spec.resources.device == "cuda" else None,
        "counters": counters, "collectors": pool.get_states(), "collector_snapshot": pool.snapshot, "parent": parent}, native / "trainer.pt")
    write_sidecar(staging, spec, schemas, counters["steps"], parent)
    payload = json.loads((staging / "checkpoint.json").read_text())
    payload["policy_id"] = f"{spec.experiment_id}:{spec.trial_id}:{label}"
    atomic_json(staging / "checkpoint.json", payload)
    destination = exports / label
    if destination.exists():
        if label != "final":
            raise FileExistsError(f"checkpoint already exists: {destination}")
        backup = exports / ".final.previous"
        if backup.exists():
            shutil.rmtree(backup)
        destination.replace(backup)
        try:
            staging.replace(destination)
        except OSError:
            backup.replace(destination)
            raise
        shutil.rmtree(backup)
    else:
        staging.replace(destination)
    for expired in numbered_checkpoints(directory)[:-spec.checkpoint.keep]:
        shutil.rmtree(expired)
    return destination / "checkpoint.json"



@dataclass
class TrainingSession:
    """Mutable state of one concrete world-model run, shared by its lifecycle helpers."""

    spec: ExperimentSpec
    directory: Path
    learner: DreamerLearner
    replay: SequenceReplay
    rng: np.random.Generator
    schemas: dict[str, str]
    counters: dict[str, int | float]
    run_id: str
    parent: dict | None = None
    collector_states: list[dict] | None = None
    snapshot: dict | None = None
    previous_elapsed: float = 0.0


def _restore(session: TrainingSession, target: str | Path) -> None:
    sidecar = latest_checkpoint(session.directory if str(target) == "." else target)
    _, stored, native = validate_sidecar(sidecar, session.spec.algorithm)
    require_resume_compatible(stored, session.spec)
    if sidecar.parents[2] != session.directory:
        raise ValueError("resume belongs to a different experiment/trial")
    session.learner.load_model_state(torch.load(native / "model.pt", map_location=session.learner.device, weights_only=True))
    saved = torch.load(native / "trainer.pt", map_location="cpu", weights_only=False)
    session.learner.load_optimizer_state(saved["optimizers"])
    session.replay = saved["replay"]
    session.counters = saved["counters"]
    session.collector_states = saved["collectors"]
    session.snapshot = saved["collector_snapshot"]
    session.parent = saved["parent"]
    session.rng.bit_generator.state = saved["rng"]
    torch.set_rng_state(saved["torch_rng"])
    random.setstate(saved["python_rng"])
    if session.spec.resources.device == "cuda" and saved["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    old_manifest = json.loads((session.directory / "manifest.json").read_text())
    session.run_id = old_manifest["wandb_run_id"]
    session.previous_elapsed = old_manifest["elapsed_seconds"]


def _prepare_session(spec, registry, component_registry, resume_from) -> TrainingSession:
    from .. import load_experiment
    resolved = load_experiment(spec, registry=registry) if isinstance(spec, (str, Path)) else spec
    resolved = freeze_experiment(resolved, component_registry=component_registry)
    resolved.validate()
    target = resume_from or resolved.artifacts.resume_from
    observation_space, schemas = _preflight(resolved, registry, component_registry, resuming=bool(target))
    directory = Path(resolved.artifacts.root).resolve() / resolved.experiment_id / resolved.trial_id
    if directory.exists() and not target and resolved.artifacts.reject_existing:
        raise FileExistsError(f"run already exists: {directory}")
    torch.set_num_threads(resolved.resources.torch_threads)
    torch.manual_seed(resolved.learner_seed)
    random.seed(resolved.learner_seed)
    session = TrainingSession(resolved, directory, DreamerLearner(resolved, observation_space),
        SequenceReplay(resolved.algorithm_config.replay_capacity), np.random.default_rng(resolved.learner_seed), schemas,
        {"steps": 0, "updates": 0, "learned_timesteps": 0, "update_credit": 0.0, "episodes": 0}, new_wandb_run_id())
    if target:
        _restore(session, target)
    elif isinstance(resolved.algorithm_config, Plan2ExploreConfig) and resolved.algorithm_config.phase == "adapt":
        session.replay, session.parent = _transfer(resolved, session.learner)
    return session


def _write_resolved(session: TrainingSession) -> None:
    import yaml
    session.directory.mkdir(parents=True, exist_ok=True)
    (session.directory / "resolved_config.yaml").write_text(yaml.safe_dump(session.spec.as_dict(), sort_keys=True), encoding="utf-8")
    if session.parent is not None:
        atomic_json(session.directory / "transfer.json", session.parent)


def _tracking(session: TrainingSession):
    tracking = session.spec.tracking
    if tracking.mode == "disabled":
        return None
    import wandb
    if tracking.mode == "online" and not wandb.login(timeout=30):
        raise RuntimeError("W&B authentication failed")
    return wandb.init(project=tracking.project, entity=tracking.entity, group=tracking.group,
        tags=list(tracking.tags), name=f"{session.spec.experiment_id}-{session.spec.trial_id}", id=session.run_id,
        resume="allow", mode=tracking.mode, dir=str(session.directory), config=session.spec.as_dict())


def _collect(session: TrainingSession, pool: RayCollectionPool):
    learner = session.learner
    current = learner.model_state()
    behavior = current["exploration"] if learner.pretraining else current["task"]
    snapshot = cpu_state({"model": current["model"], "behavior": behavior})
    config = session.spec.algorithm_config
    results = pool.collect(snapshot, session.counters["updates"], config.total_environment_steps - session.counters["steps"],
                           warmup=session.counters["steps"] < config.learning_starts)
    added = sum(result.environment_steps for result in results)
    if added <= 0:
        raise RuntimeError("collectors made no progress")
    for result in results:
        for chunk in result.chunks:
            session.replay.add(chunk)
    session.counters["steps"] += added
    session.counters["episodes"] += sum(len(result.episode_metrics) for result in results)
    session.counters["update_credit"] += added * config.replay_ratio
    return results


def _learn(session: TrainingSession) -> dict[str, float]:
    config = session.spec.algorithm_config
    if session.counters["steps"] < config.learning_starts:
        return {}
    metrics = {}
    while session.counters["update_credit"] >= config.batch_size * config.sequence_length:
        batch = session.replay.sample(config.batch_size, config.sequence_length, session.rng, session.learner.device)
        metrics = session.learner.update(batch)
        learned = int(batch.valid.sum())
        session.counters["learned_timesteps"] += learned
        session.counters["update_credit"] -= learned
        session.counters["updates"] += 1
    return metrics


def _episode_statistics(results) -> dict[str, float]:
    episodes = [episode for result in results for episode in result.episode_metrics]
    keys = {key for episode in episodes for key in episode}
    return {key: float(np.mean([episode[key] for episode in episodes if key in episode])) for key in keys}


def _metrics(session: TrainingSession, results, started: float, initial_steps: int, collection_seconds: float,
             learner_seconds: float) -> dict:
    counters = session.counters
    pretraining_steps = 0 if session.parent is None else session.parent["environment_steps"]
    return {"train/environment_steps": counters["steps"], "train/episodes": counters["episodes"],
        "train/learner_updates": counters["updates"], "train/replay_steps": session.replay.steps,
        "train/actual_replay_ratio": counters["learned_timesteps"] / max(1, counters["steps"]),
        "train/update_credit": counters["update_credit"], "train/invalid_actions": 0,
        "train/phase": getattr(session.spec.algorithm_config, "phase", "task"),
        "train/pretraining_environment_steps": pretraining_steps,
        "train/cumulative_environment_steps": counters["steps"] + pretraining_steps,
        "system/collection_seconds": collection_seconds, "system/learner_seconds": learner_seconds,
        "system/collector_reconstruction_seconds": sum(item.reconstruction_seconds for item in results),
        "system/elapsed_seconds": session.previous_elapsed + time.monotonic() - started,
        "system/environment_steps_per_second": (counters["steps"] - initial_steps) / max(1e-6, time.monotonic() - started)}


def _save(session: TrainingSession, label: str, pool: RayCollectionPool) -> Path:
    return _checkpoint(session.directory, label, session.spec, session.schemas, session.learner,
        session.replay, session.rng, session.counters, pool, session.parent)


def _publish_status(session: TrainingSession, status: str, started: float, error: str | None = None) -> None:
    _manifest(session.directory, session.spec, status, session.counters, session.run_id, session.parent,
              session.previous_elapsed + time.monotonic() - started)
    atomic_json(session.directory / "status.json", {"status": status, "error": error})


def _train(session: TrainingSession, pool: RayCollectionPool, tracking, started: float) -> None:
    spec = session.spec
    initial_steps = session.counters["steps"]
    interval = spec.checkpoint.interval_environment_steps
    next_checkpoint = (initial_steps // interval + 1) * interval
    metrics = {}
    while session.counters["steps"] < spec.algorithm_config.total_environment_steps:
        phase_started = time.monotonic()
        results = _collect(session, pool)
        collection_seconds = time.monotonic() - phase_started
        phase_started = time.monotonic()
        metrics.update(_learn(session))
        learner_seconds = time.monotonic() - phase_started
        metrics.update(_episode_statistics(results))
        metrics.update(_metrics(session, results, started, initial_steps, collection_seconds, learner_seconds))
        atomic_json(session.directory / "last_metrics.json", metrics)
        if tracking is not None:
            tracking.log(metrics, step=session.counters["steps"])
        _publish_status(session, "running", started)
        if session.counters["steps"] >= next_checkpoint:
            _save(session, f"step_{session.counters['steps']:06d}", pool)
            next_checkpoint = (session.counters["steps"] // interval + 1) * interval
    if session.counters["updates"] == 0:
        raise RuntimeError("training budget ended without a learner update")
    _save(session, "final", pool)


def run_experiment(spec, registry=None, *, report: bool = False, component_registry=None, resume_from=None) -> Path:
    """Train one phase, restoring its full state on resume rather than reinitializing transfer."""
    selected = create_training_registry() if registry is None else registry
    session = _prepare_session(spec, selected, component_registry, resume_from)
    _write_resolved(session)
    started = time.monotonic()
    tracking = pool = None
    _publish_status(session, "running", started)
    try:
        pool = RayCollectionPool(session.spec, component_registry, session.collector_states, session.snapshot)
        tracking = _tracking(session)
        _train(session, pool, tracking, started)
        if report and session.spec.tracking.mode == "online":
            from ..reports import generate_report
            generate_report(session.directory)
        _publish_status(session, "completed", started)
    except Exception as error:
        _publish_status(session, "failed", started, f"{type(error).__name__}: {error}")
        raise
    finally:
        if pool is not None:
            pool.close()
        if tracking is not None:
            tracking.finish()
    return session.directory
