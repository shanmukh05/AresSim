"""Validated experiment configuration for the optional learned-policy stack.

Immutable dataclasses decode repository YAML before Ray or W&B starts. Algorithm-
specific hyperparameters and model/resources decode through the explicit training
registry; checked-in YAML lives in repository
``configs/<algorithm>/``, not the installable package. Training
run artifacts default to ``results/``.

**Last updated:** October 16, 2026

**Contains:** ``ExperimentSpec``, resource/tracking/artifact configs,
:func:`parse_experiment`, :func:`load_experiment`, :func:`apply_overrides`.

**Schema:** ``aresim.experiment.v1``.

**See also:** :mod:`aresim.algorithms.registry` (algorithm config decoders).
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Generic, Literal, Mapping, TypeVar

from ..algorithms.common.config_decode import decode_dataclass, finite_number, mapping, positive_integer
from ..algorithms.ppo.config import MaskedPPOConfig
from ..algorithms.common.model_config import ModelConfig
from .seeds import resolve_checkout_file


EXPERIMENT_SCHEMA = "aresim.experiment.v1"
AlgorithmConfigT = TypeVar("AlgorithmConfigT")


@dataclass(frozen=True)
class EnvironmentTrainingConfig:
    """Select registered environment components for training."""

    scenario_id: str = "phase1_default_v1"
    observation: str = "local"
    action: str = "discrete"
    reward: str = "shaped_train"
    task: str = "open_exploration"
    max_episode_steps: int = 1200
    resolved_config: dict[str, object] | None = None
    task_config: dict[str, object] = field(default_factory=dict)
    engine_overrides: dict[str, object] = field(default_factory=dict)
    observation_overrides: dict[str, object] = field(default_factory=dict)
    reward_overrides: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        for name in (self.scenario_id, self.observation, self.action, self.reward, self.task):
            if not isinstance(name, str) or not name.strip():
                raise ValueError("environment identifiers must be non-empty strings")
        positive_integer(self.max_episode_steps, "max_episode_steps")
        from ..config import TaskConfig
        TaskConfig(**self.task_config).validate()
        for value in (self.engine_overrides, self.observation_overrides, self.reward_overrides):
            if not isinstance(value, dict):
                raise ValueError("environment overrides must be mappings")

@dataclass(frozen=True)
class RLlibResourceConfig:
    """Ray EnvRunner and learner resources."""

    num_env_runners: int = 0
    num_envs_per_env_runner: int = 1
    num_learners: int = 0
    cpus_per_env_runner: int = 1
    gpus_per_learner: float = 0.0

    def validate(self) -> None:
        positive_integer(self.num_env_runners, "num_env_runners", allow_zero=True)
        positive_integer(self.num_envs_per_env_runner, "num_envs_per_env_runner")
        positive_integer(self.num_learners, "num_learners", allow_zero=True)
        positive_integer(self.cpus_per_env_runner, "cpus_per_env_runner")
        if finite_number(self.gpus_per_learner, "gpus_per_learner") < 0:
            raise ValueError("gpus_per_learner cannot be negative")


@dataclass(frozen=True)
class EvaluationConfig:
    """Manual frozen-policy evaluation protocol; training never evaluates."""

    seed_manifest: str = "configs/seeds/phase1_open_exploration_split_v1.yaml"
    # Retain old snapshots/configs for inference; this field no longer schedules work.
    interval_environment_steps: int | None = None
    split: Literal["validation", "test"] = "validation"
    record_trajectories: bool = True
    action_mode: Literal["deterministic", "stochastic", "both"] = "both"
    stochastic_repeats: int = 3
    workers: int = 1
    minimum_success_rate: float | None = None
    selection_action_mode: Literal["deterministic", "stochastic"] = "deterministic"

    def validate(self) -> None:
        if not isinstance(self.seed_manifest, str) or not self.seed_manifest.strip():
            raise ValueError("evaluation seed manifest is required")
        if self.interval_environment_steps is not None:
            positive_integer(self.interval_environment_steps, "legacy evaluation interval_environment_steps")
        positive_integer(self.stochastic_repeats, "stochastic_repeats")
        positive_integer(self.workers, "evaluation workers")
        if self.action_mode not in {"deterministic", "stochastic", "both"}:
            raise ValueError("invalid evaluation action_mode")
        if self.selection_action_mode not in {"deterministic", "stochastic"} or (self.action_mode != "both" and self.selection_action_mode != self.action_mode):
            raise ValueError("selection_action_mode must be an evaluated action mode")
        if self.minimum_success_rate is not None and not 0 <= finite_number(self.minimum_success_rate, "minimum_success_rate") <= 1:
            raise ValueError("minimum_success_rate must be in [0, 1]")
        if not isinstance(self.record_trajectories, bool):
            raise ValueError("record_trajectories must be a boolean")
        if self.split not in {"validation", "test"}:
            raise ValueError("evaluation split must be validation or test")


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint schedule for one trial."""

    interval_environment_steps: int = 20480
    keep: int = 5

    def validate(self) -> None:
        positive_integer(self.interval_environment_steps, "checkpoint interval_environment_steps")
        positive_integer(self.keep, "checkpoint keep")


@dataclass(frozen=True)
class TrackingConfig:
    """W&B tracking policy."""

    mode: Literal["online", "offline", "disabled"] = "online"
    project: str = "aresim"
    entity: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.mode not in {"online", "offline", "disabled"} or not isinstance(self.project, str) or not self.project.strip():
            raise ValueError("tracking mode or project is invalid")
        if not isinstance(self.tags, tuple) or any(not isinstance(tag, str) or not tag.strip() for tag in self.tags):
            raise ValueError("tracking tags must be non-empty strings")


@dataclass(frozen=True)
class ArtifactConfig:
    """Run output identity and root directory."""

    root: str = "results"
    reject_existing: bool = True
    resume_from: str | None = None

    def validate(self) -> None:
        if not isinstance(self.root, str) or not self.root.strip() or not isinstance(self.reject_existing, bool):
            raise ValueError("artifact root cannot be empty")
        if self.resume_from is not None and (not isinstance(self.resume_from, str) or not self.resume_from.strip()):
            raise ValueError("resume_from must be a non-empty path when set")


@dataclass(frozen=True)
class ExperimentSpec(Generic[AlgorithmConfigT]):
    """Fully resolved registered experiment selected by one YAML document."""

    schema_version: str
    experiment_id: str
    trial_id: str
    algorithm: str
    model: str
    learner_seed: int
    environment: EnvironmentTrainingConfig
    algorithm_config: AlgorithmConfigT
    model_config: object
    resources: object
    evaluation: EvaluationConfig
    checkpoint: CheckpointConfig
    tracking: TrackingConfig
    artifacts: ArtifactConfig
    provenance: dict[str, object] | None = None

    def validate(self) -> None:
        if self.schema_version != EXPERIMENT_SCHEMA:
            raise ValueError(f"unsupported experiment schema: {self.schema_version}")
        _require_identifiers(self.experiment_id, self.trial_id, self.algorithm, self.model)
        positive_integer(self.learner_seed, "learner_seed", allow_zero=True)
        _validate_components(self.environment, self.algorithm_config, self.model_config, self.resources, self.evaluation, self.checkpoint, self.tracking, self.artifacts)
        _validate_step_schedule(self.algorithm_config, self.checkpoint)
        if self.evaluation.minimum_success_rate is not None and self.environment.task != "resource_mission":
            raise ValueError("success gates require a resource_mission task")

    def as_dict(self) -> dict[str, object]:
        """Return the stable JSON/YAML representation."""
        return asdict(self)

    @property
    def config_hash(self) -> str:
        """Return a deterministic SHA-256 over resolved configuration."""
        return experiment_payload_hash(self.as_dict())


def experiment_payload_hash(payload: Mapping[str, object]) -> str:
    """Hash the serialized experiment, preserving older resolved sidecar schemas."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _require_identifiers(*values: object) -> None:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("experiment identifiers must be non-empty")


def _validate_components(*components: object) -> None:
    for component in components:
        validate = getattr(component, "validate", None)
        if not callable(validate):
            raise TypeError("experiment component configuration must provide validate()")
        validate()


def _validate_step_schedule(algorithm_config: object, checkpoint: CheckpointConfig) -> None:
    if getattr(algorithm_config, "threshold_schedule", False):
        return
    batch = getattr(algorithm_config, "rollout_batch_size", None)
    total = getattr(algorithm_config, "total_environment_steps", None)
    if not isinstance(batch, int) or isinstance(batch, bool) or not isinstance(total, int) or isinstance(total, bool):
        return
    schedules = (
        (total, "total_environment_steps"),
        (checkpoint.interval_environment_steps, "checkpoint interval"),
    )
    for value, label in schedules:
        if value % batch:
            raise ValueError(f"{label} must be divisible by rollout_batch_size")


_TOP_LEVEL = {
    "schema_version", "experiment_id", "trial_id", "algorithm", "model", "learner_seed",
    "environment", "algorithm_config", "model_config", "resources", "evaluation", "checkpoint", "tracking", "artifacts", "provenance",
}




def _validate_model_selection(entry, values: dict[str, object]) -> None:
    if entry.model_id is not None and values["model"] != entry.model_id:
        raise ValueError(f"{values['algorithm']} requires model {entry.model_id}")

def parse_experiment(
    payload: object,
    *,
    algorithm_config_decoder: Callable[[object], object] | None = None,
    registry=None,
) -> ExperimentSpec[object]:
    """Decode configs strictly, allowing retired MuZero metadata in frozen snapshots."""
    from ..algorithms.registry import create_training_registry
    selected = create_training_registry() if registry is None else registry
    values = deepcopy(mapping(payload, "experiment"))
    unknown = sorted(set(values) - _TOP_LEVEL)
    if unknown:
        raise ValueError(f"unknown experiment fields: {', '.join(unknown)}")
    required = _TOP_LEVEL - {"environment", "algorithm_config", "model_config", "resources", "evaluation", "checkpoint", "tracking", "artifacts", "provenance"}
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"missing experiment fields: {', '.join(missing)}")
    # Old hashed checkpoints record a backend selector. It has no effect on
    # frozen inference; new training YAML must not expose the retired option.
    resources = values.get("resources", {})
    if (values["algorithm"] == "muzero" and values.get("provenance") is not None
            and isinstance(resources, dict) and resources.get("collection_backend") in ("local", "ray")):
        resources.pop("collection_backend")
    _validate_model_selection(selected.registration(str(values["algorithm"])), values)
    spec = ExperimentSpec(
        schema_version=str(values["schema_version"]), experiment_id=str(values["experiment_id"]),
        trial_id=str(values["trial_id"]), algorithm=str(values["algorithm"]), model=str(values["model"]),
        learner_seed=values["learner_seed"],
        environment=decode_dataclass(EnvironmentTrainingConfig, values.get("environment", {}), "environment"),
        algorithm_config=(algorithm_config_decoder or (lambda raw: selected.decode_algorithm_config(str(values["algorithm"]), raw)))(values.get("algorithm_config", {})),
        model_config=selected.decode_model_config(str(values["algorithm"]), values.get("model_config", {})),
        resources=selected.decode_resources(str(values["algorithm"]), values.get("resources", {})),
        evaluation=decode_dataclass(EvaluationConfig, values.get("evaluation", {}), "evaluation"),
        checkpoint=decode_dataclass(CheckpointConfig, values.get("checkpoint", {}), "checkpoint"),
        tracking=decode_dataclass(TrackingConfig, values.get("tracking", {}), "tracking"),
        artifacts=decode_dataclass(ArtifactConfig, values.get("artifacts", {}), "artifacts"),
        provenance=values.get("provenance"),
    )
    spec.validate()
    return spec


def load_experiment(
    path: str | Path,
    registry=None,
    *,
    algorithm_config_decoder: Callable[[object], object] | None = None,
) -> ExperimentSpec[object]:
    """Safely load and validate an experiment YAML file."""
    import yaml

    payload = yaml.safe_load(resolve_checkout_file(path, kind="experiment config").read_text(encoding="utf-8"))
    return parse_experiment(payload, algorithm_config_decoder=algorithm_config_decoder, registry=registry)


def _override_target(values: dict[str, object], dotted: str) -> tuple[dict[str, object], str]:
    path = dotted.split(".")
    if not dotted or any(not part for part in path):
        raise ValueError(f"invalid override path: {dotted}")
    cursor = values
    for part in path[:-1]:
        nested = cursor.get(part)
        if not isinstance(nested, Mapping):
            raise ValueError(f"unknown override path: {dotted}")
        nested_copy = dict(nested)
        cursor[part] = nested_copy
        cursor = nested_copy
    if path[-1] not in cursor:
        raise ValueError(f"unknown override path: {dotted}")
    return cursor, path[-1]


def _same_override_type(current: object, replacement: object) -> bool:
    if isinstance(current, bool):
        return isinstance(replacement, bool)
    if isinstance(current, int):
        return isinstance(replacement, int) and not isinstance(replacement, bool)
    if isinstance(current, float):
        return isinstance(replacement, (int, float)) and not isinstance(replacement, bool)
    if current is None:
        return replacement is None or isinstance(replacement, str)
    return isinstance(replacement, type(current))


def apply_overrides(payload: object, overrides: tuple[str, ...]) -> ExperimentSpec:
    """Apply strict dotted ``path=value`` overrides to an experiment mapping."""
    import yaml

    values = mapping(payload, "experiment")
    for expression in overrides:
        if "=" not in expression:
            raise ValueError(f"override must use path=value syntax: {expression}")
        dotted, raw_value = expression.split("=", 1)
        cursor, leaf = _override_target(values, dotted)
        replacement = yaml.safe_load(raw_value)
        if not _same_override_type(cursor[leaf], replacement):
            raise TypeError(f"override type mismatch for {dotted}")
        cursor[leaf] = replacement
    return parse_experiment(values)


def load_experiment_with_overrides(path: str | Path, overrides: tuple[str, ...]) -> ExperimentSpec:
    """Load safe YAML and apply validated CLI overrides before any side effects."""
    import yaml

    payload = yaml.safe_load(resolve_checkout_file(path, kind="experiment config").read_text(encoding="utf-8"))
    return apply_overrides(payload, overrides)


__all__ = [
    "ArtifactConfig", "CheckpointConfig", "EnvironmentTrainingConfig", "EvaluationConfig", "EXPERIMENT_SCHEMA",
    "ExperimentSpec", "MaskedPPOConfig", "ModelConfig", "RLlibResourceConfig", "TrackingConfig", "apply_overrides", "load_experiment", "load_experiment_with_overrides", "parse_experiment",
]
