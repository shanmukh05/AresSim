"""Run registered RLlib algorithms through Tune and publish portable sidecars.

Owns trial lifecycle, tracking and checkpoint schedules shared by PPO and both DQNs.
Evaluation remains an explicit separate command.
"""
from __future__ import annotations
import json
import platform
import shutil
from collections.abc import Mapping
from pathlib import Path
from ray import tune
from ...factory import make_env
from ...algorithms.registry import TrainingContext, TrainingRegistry, create_training_registry
from .checkpoint import write_checkpoint_sidecar
from ..artifacts import atomic_json, numbered_checkpoints
from ...algorithms.common.resume import ResolvedResume, resolve_resume_target
from ...algorithms.common.tracking import load_wandb_run_id, new_wandb_run_id, resolve_wandb_run_id
from ..experiments import ExperimentSpec
from ..provenance import environment_config, freeze_experiment
from .metrics import canonicalize_rllib_metrics

RUN_SCHEMA = "aresim.run.v1"

def _installed_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    values = {"python": platform.python_version()}
    for package in ("aresim", "ray", "torch", "gymnasium", "numpy", "wandb"):
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            continue
    return values

def _artifact_inventory(run_directory: Path) -> list[dict[str, object]]:
    import hashlib

    inventory = []
    for path in sorted(item for item in run_directory.rglob("*") if item.is_file() and not item.name.endswith(".partial")):
        if path.name in {"manifest.json", "status.json"} or "ray" in path.parts or "native" in path.parts:
            continue
        inventory.append({"path": path.relative_to(run_directory).as_posix(), "size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return inventory

def _write_status(run_directory: Path, status: str, error: str | None = None) -> None:
    atomic_json(run_directory / "status.json", {"status": status, "error": error})

def _write_run_manifest(
    run_directory: Path,
    spec: ExperimentSpec,
    status: str,
    result_path: str | None = None,
    wandb_run_id: str | None = None,
    environment_steps: int | None = None,
) -> None:
    resolved_wandb_run_id = resolve_wandb_run_id(run_directory, wandb_run_id)
    if environment_steps is None:
        previous = run_directory / "manifest.json"
        environment_steps = json.loads(previous.read_text()).get("environment_steps", 0) if previous.is_file() else 0
    payload = {"schema_version": RUN_SCHEMA, "framework_id": "rllib", "status": status, "config_hash": spec.config_hash,
               "experiment": spec.as_dict(), "result_path": result_path, "environment_steps": environment_steps,
               "open_exploration_has_success": spec.environment.task != "open_exploration",
               "promoted_checkpoint": "checkpoints/best/checkpoint.json" if (run_directory / "checkpoint_selection.json").is_file() else None, "wandb_run_id": resolved_wandb_run_id, "installed_versions": _installed_versions(),
               "artifacts": _artifact_inventory(run_directory) if status == "completed" else []}
    atomic_json(run_directory / "manifest.json", payload)

def _checkpoint_label(environment_steps: int) -> str:
    """Return a stable exported checkpoint directory name for one env-step count."""
    return f"step_{environment_steps:06d}"

def _environment_steps_from_result(result: Mapping[str, object], rollout_batch_size: int) -> int:
    """Read lifetime environment steps from one Ray Tune train result."""
    metrics = canonicalize_rllib_metrics(result)
    if "train/environment_steps" in metrics:
        return int(metrics["train/environment_steps"])
    lifetime = result.get("num_env_steps_sampled_lifetime")
    if isinstance(lifetime, (int, float)) and not isinstance(lifetime, bool):
        return int(lifetime)
    iteration = result.get("training_iteration")
    if isinstance(iteration, int) and not isinstance(iteration, bool):
        return iteration * rollout_batch_size
    raise ValueError("checkpoint export requires environment step metrics in the trial result")

def _export_checkpoint(
    run_directory: Path,
    checkpoint,
    label: str,
    spec: ExperimentSpec,
    algorithm,
    schemas: Mapping[str, str],
    *, environment_steps: int | None = None,
) -> Path:
    """Copy one Ray checkpoint and publish a UI-loadable ``checkpoint.json`` sidecar."""
    checkpoint_directory = run_directory / "checkpoints" / label
    if checkpoint_directory.exists():
        shutil.rmtree(checkpoint_directory)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    native_copy = checkpoint_directory / "native"
    with checkpoint.as_directory() as source_directory:
        shutil.copytree(source_directory, native_copy)
    return write_checkpoint_sidecar(
        checkpoint_directory / "checkpoint.json",
        native_path=str(native_copy),
        experiment=spec.as_dict(),
        config_hash=spec.config_hash,
        loader_id=algorithm.checkpoint_loader_id,
        observation_schema=schemas["observation_schema"],
        action_schema=schemas["action_schema"],
        task_id=schemas["task_id"],
        reward_profile=schemas["reward_profile"],
        environment_steps=environment_steps if environment_steps is not None else (int(label.removeprefix("step_")) if label.startswith("step_") else None),
    )

def _prune_exported_checkpoints(run_directory: Path, keep: int) -> None:
    """Drop oldest ``step_*`` exports; ``final`` is managed separately."""
    for expired in numbered_checkpoints(run_directory)[:-keep]:
        shutil.rmtree(expired)

class AresCheckpointExportCallback(tune.Callback):
    """Publish UI-loadable checkpoint sidecars whenever Ray Tune checkpoints."""

    def __init__(
        self,
        run_directory: Path,
        spec: ExperimentSpec,
        algorithm,
        schemas: Mapping[str, str],
    ) -> None:
        self._run_directory = Path(run_directory)
        self._spec = spec
        self._algorithm = algorithm
        self._schemas = schemas
        self._rollout_batch_size = spec.algorithm_config.rollout_batch_size

    def on_checkpoint(self, *, trial, checkpoint, **kwargs) -> None:
        """Export ``checkpoints/step_<env_steps>/checkpoint.json`` for the UI."""
        result = trial.last_result or {}
        label = _checkpoint_label(_environment_steps_from_result(result, self._rollout_batch_size))
        _export_checkpoint(self._run_directory, checkpoint, label, self._spec, self._algorithm, self._schemas)
        steps = _environment_steps_from_result(result, self._rollout_batch_size)
        _write_run_manifest(self._run_directory, self._spec, "running", environment_steps=steps)
        _prune_exported_checkpoints(self._run_directory, self._spec.checkpoint.keep)

def _training_callbacks(
    spec: ExperimentSpec,
    run_directory: Path,
    algorithm,
    schemas: Mapping[str, str],
) -> list[tune.Callback]:
    return [
        * _wandb_callbacks(spec, run_directory),
        AresCheckpointExportCallback(run_directory, spec, algorithm, schemas),
    ]

def _wandb_callbacks(spec: ExperimentSpec, run_directory: Path) -> list[tune.Callback]:
    if spec.tracking.mode == "disabled":
        return []
    from ray.air.integrations.wandb import WandbLoggerCallback

    wandb_run_id = load_wandb_run_id(run_directory)
    if wandb_run_id is None:
        raise RuntimeError("manifest is missing wandb_run_id before training")
    return [WandbLoggerCallback(
        project=spec.tracking.project,
        entity=spec.tracking.entity,
        group=spec.tracking.group,
        tags=list(spec.tracking.tags),
        name=f"{spec.experiment_id}-{spec.trial_id}",
        id=wandb_run_id,
        mode=spec.tracking.mode,
        log_config=True,
        upload_checkpoints=False,
    )]

def _preflight(spec: ExperimentSpec, component_registry=None) -> dict[str, str]:
    if spec.resources.gpus_per_learner > 0:
        import torch as torch_module

        if not torch_module.cuda.is_available():
            raise RuntimeError("the experiment requests CUDA learners but CUDA is unavailable")
    config = environment_config(spec)
    reset = make_env(config, registry=component_registry).reset(seed=spec.learner_seed)
    return {
        key: str(reset.info[key])
        for key in ("observation_schema", "action_schema", "task_id", "reward_profile", "scenario_id")
    }

def _authenticate_tracking(spec: ExperimentSpec) -> None:
    """Authenticate online tracking after the failed-run artifact exists."""
    if spec.tracking.mode == "online":
        import wandb

        if not wandb.login(timeout=30):
            raise RuntimeError("W&B authentication failed before Ray startup")

def _resolve_training(spec: ExperimentSpec, registry: TrainingRegistry | None, component_registry):
    schemas = _preflight(spec, component_registry)
    selected = create_training_registry() if registry is None else registry
    context = TrainingContext(spec, component_registry=component_registry)
    model = selected.build_model(spec.model, context)
    algorithm = selected.build_algorithm(spec.algorithm, TrainingContext(spec, model, component_registry))
    declared = (algorithm.observation_schema, algorithm.action_schema)
    if (model.observation_schema, model.action_schema) != declared:
        raise ValueError("selected algorithm and model declare incompatible schemas")
    if (schemas["observation_schema"], schemas["action_schema"]) != declared:
        raise ValueError("selected environment schemas are incompatible with the algorithm")
    return schemas, selected, algorithm, algorithm.build(TrainingContext(spec, model, component_registry))

def _run_config(
    spec: ExperimentSpec,
    run_directory: Path,
    algorithm,
    schemas: Mapping[str, str],
) -> tune.RunConfig:
    batch = spec.algorithm_config.rollout_batch_size
    return tune.RunConfig(
        name="ray",
        storage_path=str(run_directory),
        stop={"num_env_steps_sampled_lifetime": spec.algorithm_config.total_environment_steps},
        callbacks=_training_callbacks(spec, run_directory, algorithm, schemas),
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=max(1, spec.checkpoint.interval_environment_steps // batch),
            checkpoint_at_end=True,
            num_to_keep=spec.checkpoint.keep,
        ),
    )

class _TuneTrialResult:
    """Minimal adapter so checkpoint export works after ``tune.run`` resume."""

    def __init__(self, trial) -> None:
        self.checkpoint = trial.checkpoint
        self.path = trial.path
        self.metrics = trial.last_result or {}

def _fit_from_native_checkpoint(
    spec: ExperimentSpec,
    algorithm,
    config,
    run_directory: Path,
    schemas: Mapping[str, str],
    native_checkpoint: Path,
) -> _TuneTrialResult:
    analysis = tune.run(
        algorithm.trainable,
        config=config.to_dict(),
        storage_path=str(run_directory),
        name="ray",
        stop={"num_env_steps_sampled_lifetime": spec.algorithm_config.total_environment_steps},
        restore=str(native_checkpoint.resolve()),
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=max(1, spec.checkpoint.interval_environment_steps // spec.algorithm_config.rollout_batch_size),
            checkpoint_at_end=True,
            num_to_keep=spec.checkpoint.keep,
        ),
        callbacks=_training_callbacks(spec, run_directory, algorithm, schemas),
        resume=False,
    )
    if not analysis.trials:
        raise RuntimeError("Ray Tune resume trial did not start")
    trial = analysis.trials[0]
    if trial.status != "TERMINATED" or trial.error_file:
        raise RuntimeError(f"Ray Tune resume trial failed: {trial.error_file}")
    return _TuneTrialResult(trial)

def _prepare_run(spec: ExperimentSpec, resume: ResolvedResume | None = None) -> Path:
    run_directory = Path(spec.artifacts.root).resolve() / spec.experiment_id / spec.trial_id
    if resume is not None:
        if resume.run_directory != run_directory:
            raise ValueError("resume target run directory does not match experiment_id/trial_id")
        if not run_directory.is_dir():
            raise FileNotFoundError(f"resume run directory does not exist: {run_directory}")
    elif run_directory.exists() and spec.artifacts.reject_existing:
        raise FileExistsError(f"run already exists: {run_directory}")
    else:
        run_directory.mkdir(parents=True, exist_ok=True)
    import yaml

    (run_directory / "resolved_config.yaml").write_text(yaml.safe_dump(spec.as_dict(), sort_keys=True), encoding="utf-8")
    wandb_run_id = None if resume is not None else new_wandb_run_id()
    _write_run_manifest(run_directory, spec, "running", wandb_run_id=wandb_run_id)
    _write_status(run_directory, "running")
    return run_directory

def _fit(
    spec: ExperimentSpec,
    algorithm,
    config,
    run_directory: Path,
    schemas: Mapping[str, str],
    resume: ResolvedResume | None = None,
):
    ray_directory = run_directory / "ray"
    if resume is not None and resume.explicit_checkpoint and resume.native_checkpoint is not None:
        return _fit_from_native_checkpoint(spec, algorithm, config, run_directory, schemas, resume.native_checkpoint)
    if resume is not None and tune.Tuner.can_restore(str(ray_directory)):
        from ray.tune.tune_config import ResumeConfig

        tuner = tune.Tuner.restore(
            str(ray_directory),
            trainable=algorithm.trainable,
            param_space=config.to_dict(),
            resume_unfinished=True,
            _resume_config=ResumeConfig(
                finished=ResumeConfig.ResumeType.RESUME,
                unfinished=ResumeConfig.ResumeType.RESUME,
            ),
        )
        internal = tuner._local_tuner
        internal._run_config = _run_config(spec, run_directory, algorithm, schemas)
        grid = tuner.fit()
    elif resume is not None:
        raise ValueError("cannot resume: no Ray experiment state or checkpoint found for the requested run")
    else:
        tuner = tune.Tuner(
            algorithm.trainable,
            param_space=config.to_dict(),
            run_config=_run_config(spec, run_directory, algorithm, schemas),
        )
        grid = tuner.fit()
    if grid.errors:
        raise RuntimeError(f"Ray Tune trial failed: {grid.errors[0]}")
    return grid[0]

def _save_checkpoint(run_directory: Path, result, spec: ExperimentSpec, algorithm, schemas: Mapping[str, str]) -> Path | None:
    if result.checkpoint is None:
        return None
    return _export_checkpoint(run_directory, result.checkpoint, "final", spec, algorithm, schemas,
                              environment_steps=_environment_steps_from_result(result.metrics, spec.algorithm_config.rollout_batch_size))

def _shutdown_ray() -> None:
    import ray

    if ray.is_initialized():
        ray.shutdown()

def _finish_run(spec: ExperimentSpec, run_directory: Path, *, report: bool) -> None:
    if report and spec.tracking.mode == "online":
        from ..reports import generate_report

        generate_report(run_directory)

def run_experiment(
    spec: ExperimentSpec | str | Path,
    registry: TrainingRegistry | None = None,
    *,
    report: bool = False,
    component_registry=None,
    resume_from: str | Path | None = None,
) -> Path:
    """Run one validated Ray Tune trial and publish an immutable run manifest.

    Accepts a resolved :class:`~aresim.training.experiments.ExperimentSpec` or a
    path to experiment YAML. Writes ``resolved_config.yaml``, ``manifest.json``,
    and ``status.json`` under the artifact root. Saves periodic and final
    checkpoints without evaluation rollouts. Evaluation is a separate command.
    W&B-backed reporting is optional and disabled by default. Shuts down Ray
    after Tune and again in ``finally``.

    Pass ``resume_from`` (or ``artifacts.resume_from`` in YAML) to continue a
    prior run from its Ray experiment state or an exported ``checkpoints/*`` sidecar.

    Returns the run directory path on success; leaves a failed manifest on error.
    """
    selected = create_training_registry() if registry is None else registry
    if isinstance(spec, (str, Path)):
        from .. import load_experiment

        resolved = load_experiment(spec, registry=selected)
    else:
        resolved = spec
    resolved = freeze_experiment(resolved, component_registry=component_registry)
    resolved.validate()
    resume_target = (
        resolve_resume_target(resume_from or resolved.artifacts.resume_from, resolved)
        if (resume_from or resolved.artifacts.resume_from)
        else None
    )
    schemas, selected, algorithm, config = _resolve_training(resolved, selected, component_registry)
    run_directory = _prepare_run(resolved, resume_target)
    import ray
    owns_runtime = not ray.is_initialized()
    try:
        _authenticate_tracking(resolved)
        result = _fit(resolved, algorithm, config, run_directory, schemas, resume_target)
        _save_checkpoint(run_directory, result, resolved, algorithm, schemas)
        if owns_runtime:
            _shutdown_ray()
        _finish_run(resolved, run_directory, report=report)
        _write_run_manifest(run_directory, resolved, "completed", str(getattr(result, "path", run_directory)),
                            environment_steps=_environment_steps_from_result(result.metrics, resolved.algorithm_config.rollout_batch_size))
        _write_status(run_directory, "completed")
    except Exception as error:
        _write_run_manifest(run_directory, resolved, "failed")
        _write_status(run_directory, "failed", f"{type(error).__name__}: {error}")
        raise
    finally:
        if owns_runtime:
            _shutdown_ray()
    return run_directory
