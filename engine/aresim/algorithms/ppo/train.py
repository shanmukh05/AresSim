"""Masked PPO training path: model, RLModule, metrics, and Ray Tune orchestration.

Merges the actor-critic network, RLlib RLModule wrapper, W&B metric canonicalization,
and end-to-end experiment execution for the built-in ``masked_ppo`` algorithm.
Simulator semantics remain owned by :mod:`aresim.core` and :mod:`aresim.components`;
RLlib and PyTorch are implementation details inside this module.

**Last updated:** September 1, 2026

**Contains:**

- ``LocalMaskedActorCritic`` — PyTorch policy/value network for ``aresim.obs.local.v1``
- ``AresMaskedPPORLModule`` — RLlib module applying authoritative action masks
- ``AresMetricsCallback`` — forwards env diagnostics and stable W&B metric names
- ``AresCheckpointExportCallback`` — writes UI-loadable ``checkpoints/step_<env_steps>/`` sidecars during training
- ``MaskedPPOFactory`` — builds ``PPOConfig`` from :class:`~aresim.training.experiments.ExperimentSpec`
- :func:`canonicalize_rllib_metrics` — maps unstable RLlib result paths
- :func:`run_experiment` — validated Ray Tune trial with manifest and optional eval

**Constants:** ``RUN_SCHEMA`` (``aresim.run.v1``), ``ENV_NAME`` (registered Gym env),
``MODEL_ID`` (``local_cnn_actor_critic``).

**Dependencies:** Ray RLlib, Ray Tune, PyTorch, Gymnasium, optional W&B.

**Entry point:** :func:`run_experiment` (also lazy-exported from :mod:`aresim.training`).

**See also:** :mod:`aresim.algorithms.ppo.config` (hyperparameters),
:mod:`aresim.algorithms.ppo.checkpoint` (frozen inference),
workflow.md (end-to-end data flow and tensor reference),
:mod:`aresim.training.experiments` (YAML envelope).
"""

from __future__ import annotations

import json
import platform
import shutil
from collections.abc import Mapping
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env
from torch import Tensor, nn

from ...defaults import DEFAULT_ENVIRONMENT_CONFIG
from ...factory import make_env, make_gym_env
from ...training.experiments import ExperimentSpec
from ...training.seeds import load_seed_manifest, resolve_checkout_file
from ..common.tracking import load_wandb_run_id, new_wandb_run_id, resolve_wandb_run_id
from ..registry import TrainingContext, TrainingRegistry, create_training_registry
from .checkpoint import write_checkpoint_sidecar
from .config import MaskedPPOConfig, ModelConfig, decode_config


RUN_SCHEMA = "aresim.run.v1"
ENV_NAME = "aresim_gym_rllib_v1"
MODEL_ID = "local_cnn_actor_critic"


# --- Actor-critic model ---


class LocalMaskedActorCritic(nn.Module):
    """Encode local spatial/telemetry inputs and return masked logits and a value.

    Consumes policy observations and the authoritative action mask only. Illegal
    logits are set to dtype minimum before returning. Raises when outputs are
    non-finite or no legal action exists in the mask.
    """

    def __init__(self, config: ModelConfig, window_size: int = 8, action_count: int = 10) -> None:
        """Build layers from ``config``; ``action_count`` must be ``10`` for Phase 1."""
        super().__init__()
        config.validate()
        if window_size <= 0 or action_count != 10:
            raise ValueError("local actor-critic requires a positive crop and Discrete(10)")
        self.config = config
        self.action_count = action_count
        self.terrain_embedding = nn.Embedding(8, config.terrain_embedding)
        in_channels = config.terrain_embedding + 5 + 4
        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels, config.conv_channels[0], 3, padding=1), nn.Tanh(),
            nn.Conv2d(config.conv_channels[0], config.conv_channels[1], 3, padding=1), nn.Tanh(), nn.Flatten(),
        )
        spatial_width = config.conv_channels[1] * window_size * window_size
        self.pad_embedding = nn.Embedding(3, 4)
        self.weather_embedding = nn.Embedding(6, 4)
        self.objective_embedding = nn.Embedding(9, config.objective_embedding)
        self.objective_encoder = nn.Sequential(nn.Linear(config.objective_embedding + 4, 32), nn.Tanh())
        telemetry_width = 10 + 14 + 4 + 4 + 32
        self.telemetry = nn.Sequential(
            nn.Linear(telemetry_width, config.telemetry_layers[0]), nn.Tanh(),
            nn.Linear(config.telemetry_layers[0], config.telemetry_layers[1]), nn.Tanh(),
        )
        self.fusion = nn.Sequential(nn.Linear(spatial_width + config.telemetry_layers[1], config.fused_width), nn.Tanh())
        self.policy = nn.Linear(config.fused_width, action_count)
        self.value = nn.Linear(config.fused_width, 1)
        self.apply(self._initialize)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.orthogonal_(self.value.weight, gain=1.0)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=2 ** 0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _as_batch(value: Tensor, dimensions: int) -> Tensor:
        return value.unsqueeze(0) if value.ndim == dimensions - 1 else value

    def forward(self, observation: Mapping[str, Tensor], action_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Return finite masked action logits and one value per batch row."""
        terrain = self._as_batch(observation["terrain_type"].long(), 3)
        spatial = self._as_batch(observation["spatial"].float(), 4)
        flags = self._as_batch(observation["cell_flags"].float(), 4)
        terrain_features = self.terrain_embedding(terrain).permute(0, 3, 1, 2)
        spatial_features = self.spatial(torch.cat((terrain_features, spatial, flags), dim=1))

        self_vector = self._as_batch(observation["self"].float(), 2)
        colony = self._as_batch(observation["colony"].float(), 2)
        pad = observation["pad_proximity"].long().reshape(-1)
        weather = observation["weather_type"].long().reshape(-1)
        objective_type = self._as_batch(observation["objective_type"].long(), 2)
        objectives = self._as_batch(observation["objectives"].float(), 3)
        objective_mask = self._as_batch(observation["objective_mask"].float(), 2).unsqueeze(-1)
        rows = self.objective_encoder(torch.cat((self.objective_embedding(objective_type), objectives), dim=-1))
        objective_features = (rows * objective_mask).sum(dim=1) / objective_mask.sum(dim=1).clamp(min=1.0)
        telemetry = self.telemetry(torch.cat((self_vector, colony, self.pad_embedding(pad), self.weather_embedding(weather), objective_features), dim=-1))
        fused = self.fusion(torch.cat((spatial_features, telemetry), dim=-1))
        logits = self.policy(fused)
        mask = self._as_batch(action_mask, 2).bool()
        if mask.shape != logits.shape or not torch.all(mask.any(dim=-1)):
            raise ValueError("action mask shape is invalid or contains no legal action")
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        values = self.value(fused).squeeze(-1)
        if not torch.isfinite(logits[mask]).all() or not torch.isfinite(values).all():
            raise ValueError("actor-critic produced non-finite outputs")
        return logits, values


_RLModuleBase = getattr(import_module("ray.rllib.core.rl_module.torch"), "Torch" + "RLModule")


class AresMaskedPPORLModule(_RLModuleBase, ValueFunctionAPI):
    """RLlib module that applies the canonical action mask in all forward modes.

    Expects a Dict observation space with ``observation`` and ``action_mask`` keys
    and ``Discrete(10)`` actions. Delegates forward passes to
    :class:`LocalMaskedActorCritic`.
    """

    def setup(self) -> None:
        """Construct the core network from ``model_config`` and observation crop size."""
        if not isinstance(self.observation_space, spaces.Dict) or set(self.observation_space.spaces) != {"observation", "action_mask"}:
            raise ValueError("Ares masked PPO requires observation and action_mask inputs")
        if not isinstance(self.action_space, spaces.Discrete) or self.action_space.n != 10:
            raise ValueError("Ares masked PPO requires Discrete(10)")
        model_values = dict(self.model_config or {})
        config = ModelConfig(**{key: value for key, value in model_values.items() if key in ModelConfig.__dataclass_fields__})
        local_space = self.observation_space["observation"]
        window_size = int(local_space["terrain_type"].shape[0])
        self.core = LocalMaskedActorCritic(config, window_size, self.action_space.n)

    def _outputs(self, batch: dict[str, Any]) -> dict[str, Any]:
        policy_input = batch[Columns.OBS]
        logits, _ = self.core(policy_input["observation"], policy_input["action_mask"])
        return {Columns.ACTION_DIST_INPUTS: logits}

    def _forward_inference(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return self._outputs(batch)

    def _forward_exploration(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return self._outputs(batch)

    def _forward_train(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return self._outputs(batch)

    def compute_values(self, batch: dict[str, Any], embeddings: Any = None):
        """Return per-batch value estimates for PPO advantage computation."""
        policy_input = batch[Columns.OBS]
        return self.core(policy_input["observation"], policy_input["action_mask"])[1]


# --- W&B metrics ---


def _nested_number(payload: Mapping[str, object], *paths: tuple[str, ...]) -> float | None:
    for path in paths:
        value: object = payload
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                break
            value = value[key]
        else:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


def canonicalize_rllib_metrics(result: Mapping[str, object]) -> dict[str, float]:
    """Map unstable RLlib Tune result paths to stable AresSim W&B metric names.

    Also rewrites ``ares/*`` env-runner diagnostics to ``train/*`` for dashboards.
    Missing paths are omitted rather than invented.
    """
    paths = {
        "train/environment_steps": (("num_env_steps_sampled_lifetime",), ("env_runners", "num_env_steps_sampled_lifetime")),
        "train/episodes": (("env_runners", "num_episodes_lifetime"), ("episodes_total",)),
        "train/episode_length": (("env_runners", "episode_len_mean"), ("episode_len_mean",)),
        "train/shaped_return": (("env_runners", "episode_return_mean"), ("episode_reward_mean",)),
        "system/environment_steps_per_second": (("env_runners", "num_env_steps_sampled_per_second"),),
        "learner/total_loss": (("learners", "default_policy", "total_loss"),),
        "learner/policy_loss": (("learners", "default_policy", "policy_loss"),),
        "learner/value_loss": (("learners", "default_policy", "vf_loss"),),
        "learner/entropy": (("learners", "default_policy", "entropy"),),
        "learner/approx_kl": (("learners", "default_policy", "mean_kl_loss"),),
        "learner/clip_fraction": (("learners", "default_policy", "clip_fraction"),),
        "learner/learning_rate": (("learners", "default_policy", "default_optimizer_learning_rate"),),
        "learner/gradient_norm": (("learners", "default_policy", "gradients_default_optimizer_global_norm"),),
        "learner/explained_variance": (("learners", "default_policy", "vf_explained_var"),),
    }
    metrics = {
        name: value
        for name, candidates in paths.items()
        if (value := _nested_number(result, *candidates)) is not None
    }
    runner_metrics = result.get("env_runners")
    if isinstance(runner_metrics, Mapping):
        metrics.update(_environment_metrics(runner_metrics))
    return metrics


def _environment_metrics(values: Mapping[str, object]) -> dict[str, float]:
    return {
        f"train/{str(name)[5:]}": float(value)
        for name, value in values.items()
        if str(name).startswith("ares/")
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }


def _log_value(metrics_logger, name: str, value: object, *, reduce: str = "sum") -> None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        metrics_logger.log_value(name, float(value), reduce=reduce)


def _log_transition(metrics_logger, episode, info: Mapping[str, object]) -> None:
    action = str(info.get("effective_action", "unknown")).lower()
    metrics_logger.log_value(f"ares/action_count/{action}", 1.0, reduce="sum")
    metrics_logger.log_value("ares/invalid_actions", float(action == "invalid"), reduce="sum")
    _log_value(metrics_logger, "ares/engine_reward", info.get("engine_reward"))
    breakdown = info.get("reward_breakdown")
    if not isinstance(breakdown, Mapping) or not isinstance(breakdown.get("terms"), Mapping):
        return
    for name, term in breakdown["terms"].items():
        if isinstance(term, Mapping):
            _log_value(metrics_logger, f"ares/reward_term/{name}", term.get("value", term.get("contribution", term.get("weighted"))))


def _log_mask_violation(metrics_logger, episode) -> None:
    action = int(episode.get_actions(-1))
    previous = episode.get_observations(-2)
    if not isinstance(previous, Mapping) or "action_mask" not in previous:
        return
    mask = np.asarray(previous["action_mask"])
    violation = action < 0 or action >= mask.size or not bool(mask[action])
    metrics_logger.log_value("ares/mask_violations", float(violation), reduce="sum")


def _log_telemetry(metrics_logger, episode) -> None:
    current = episode.get_observations(-1)
    if not isinstance(current, Mapping) or not isinstance(current.get("observation"), Mapping):
        return
    observation = current["observation"]
    rover = np.asarray(observation["self"], dtype=float)
    colony = np.asarray(observation["colony"], dtype=float)
    values = {
        "battery": rover[2], "health": rover[3], "payload_remaining": rover[7],
        "power_margin": colony[2], "colony_battery": colony[3], "water": colony[4],
        "oxygen": colony[5], "livability": colony[6],
    }
    for name, value in values.items():
        metrics_logger.log_value(f"ares/telemetry/{name}", float(value), reduce="mean")


class AresMetricsCallback(RLlibCallback):
    """Forward authoritative env info and canonical learner metrics to W&B.

    On each episode step, logs action counts, invalid actions, engine reward terms,
    mask violations, and selected telemetry means. On train results, injects
    :func:`canonicalize_rllib_metrics` before Ray's logger runs.
    """

    def on_episode_step(self, *, episode, metrics_logger=None, **kwargs) -> None:
        """Log per-step environment diagnostics when a metrics logger is available."""
        if metrics_logger is None:
            return
        info = episode.get_infos(-1)
        if not isinstance(info, Mapping):
            return
        _log_transition(metrics_logger, episode, info)
        try:
            _log_mask_violation(metrics_logger, episode)
            _log_telemetry(metrics_logger, episode)
        except (IndexError, KeyError, TypeError, ValueError):
            return

    def on_train_result(self, *, algorithm, result: dict, **kwargs) -> None:
        """Add stable names before Ray's W&B callback sees the result."""
        result.update(canonicalize_rllib_metrics(result))


# --- Training orchestration ---


class _SeededTrainingEnv(gym.Env):
    """Cycle one RLlib worker through the fixed training seed list deterministically.

    Offsets seed selection per worker, vector index, and learner seed so parallel
    runners do not reset identical worlds every episode.
    """

    metadata = {"render_modes": []}

    def __init__(self, environment, seeds: tuple[int, ...], offset: int) -> None:
        if not seeds:
            raise ValueError("RLlib training requires at least one environment seed")
        self.environment = environment
        self.seeds = seeds
        self.offset = offset % len(seeds)
        self.episode_index = 0
        self.observation_space = environment.observation_space
        self.action_space = environment.action_space

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            selected = self.seeds[(self.offset + int(seed)) % len(self.seeds)]
        else:
            selected = self.seeds[(self.offset + self.episode_index) % len(self.seeds)]
            self.episode_index += 1
        return self.environment.reset(seed=selected, options=options)

    def step(self, action):
        return self.environment.step(action)


def _environment(env_config: Mapping[str, object]):
    values = env_config["environment"]
    config = replace(DEFAULT_ENVIRONMENT_CONFIG, scenario_id=values["scenario_id"], observation=values["observation"], action=values["action"], reward=values["reward"], task=values["task"])
    environment = make_gym_env(
        config,
        registry=env_config.get("component_registry"),
        max_episode_steps=int(values["max_episode_steps"]),
    )
    seeds = tuple(int(seed) for seed in env_config["training_seeds"])
    worker_index = int(getattr(env_config, "worker_index", 0))
    vector_index = int(getattr(env_config, "vector_index", 0))
    learner_seed = int(env_config["learner_seed"])
    offset = learner_seed * 104729 + worker_index * 1009 + vector_index * 9176
    return _SeededTrainingEnv(environment, seeds, offset)


class MaskedPPOFactory:
    """Build RLlib ``PPOConfig`` for the ``masked_ppo`` registered algorithm.

    Wires environment registration, resource settings, PPO hyperparameters from
    :class:`~aresim.algorithms.ppo.config.MaskedPPOConfig`, and the selected
    RLModule class from the model factory.
    """

    algorithm_id = "masked_ppo"
    trainable = "PPO"
    checkpoint_loader_id = "rllib_masked_ppo"
    observation_schema = "aresim.obs.local.v1"
    action_schema = "aresim.action.rover.v1"

    def decode_config(self, payload: object):
        """Decode YAML ``algorithm_config`` into :class:`MaskedPPOConfig`."""
        return decode_config(MaskedPPOConfig, payload, "algorithm_config")

    def build(self, context: TrainingContext) -> PPOConfig:
        """Return a fully configured ``PPOConfig`` for Ray Tune."""
        spec = context.experiment; ppo = spec.algorithm_config; resources = spec.resources
        model_factory = context.model_factory
        module_class = getattr(model_factory, "rl_module_class", None)
        if not isinstance(module_class, type):
            raise TypeError("masked_ppo requires a model factory with an RLModule class")
        training_seeds = load_seed_manifest(spec.evaluation.seed_manifest).train
        register_env(ENV_NAME, _environment)
        return (
            PPOConfig().environment(ENV_NAME, env_config={
                "environment": spec.as_dict()["environment"],
                "training_seeds": training_seeds,
                "learner_seed": spec.learner_seed,
                "component_registry": context.component_registry,
            })
            .framework("torch")
            .debugging(seed=spec.learner_seed)
            .env_runners(num_env_runners=resources.num_env_runners, num_envs_per_env_runner=resources.num_envs_per_env_runner,
                         num_cpus_per_env_runner=resources.cpus_per_env_runner, rollout_fragment_length="auto", batch_mode="truncate_episodes")
            .learners(num_learners=resources.num_learners, num_gpus_per_learner=resources.gpus_per_learner)
            .training(gamma=ppo.gamma, lr=ppo.learning_rate, lambda_=ppo.gae_lambda, clip_param=ppo.clip_param,
                      vf_loss_coeff=ppo.value_loss_coefficient, entropy_coeff=ppo.entropy_coefficient,
                      grad_clip=ppo.max_gradient_norm, kl_target=ppo.target_kl, train_batch_size_per_learner=ppo.rollout_batch_size,
                      minibatch_size=ppo.minibatch_size, num_epochs=ppo.update_epochs)
            .rl_module(rl_module_spec=RLModuleSpec(module_class=module_class, model_config=spec.as_dict()["model_config"]))
            .callbacks(AresMetricsCallback)
        )


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
    partial = run_directory / "status.json.partial"
    partial.write_text(json.dumps({"status": status, "error": error}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    partial.replace(run_directory / "status.json")


def _write_run_manifest(
    run_directory: Path,
    spec: ExperimentSpec,
    status: str,
    result_path: str | None = None,
    wandb_run_id: str | None = None,
) -> None:
    resolved_wandb_run_id = resolve_wandb_run_id(run_directory, wandb_run_id)
    payload = {"schema_version": RUN_SCHEMA, "framework_id": "rllib", "status": status, "config_hash": spec.config_hash,
               "experiment": spec.as_dict(), "result_path": result_path, "open_exploration_has_success": False,
               "promoted_checkpoint": None, "wandb_run_id": resolved_wandb_run_id, "installed_versions": _installed_versions(),
               "artifacts": _artifact_inventory(run_directory) if status == "completed" else []}
    temporary = run_directory / "manifest.json.partial"
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(run_directory / "manifest.json")


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
    )


def _prune_exported_checkpoints(run_directory: Path, keep: int) -> None:
    """Drop oldest ``step_*`` exports; ``final`` is managed separately."""
    checkpoints_root = run_directory / "checkpoints"
    if not checkpoints_root.is_dir():
        return
    step_directories = sorted(
        path
        for path in checkpoints_root.iterdir()
        if path.is_dir() and path.name.startswith("step_") and (path / "checkpoint.json").is_file()
    )
    while len(step_directories) > keep:
        shutil.rmtree(step_directories.pop(0))


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
    resolve_checkout_file(spec.evaluation.seed_manifest, kind="evaluation seed manifest")
    environment_values = spec.environment
    environment_config = replace(
        DEFAULT_ENVIRONMENT_CONFIG,
        scenario_id=environment_values.scenario_id,
        observation=environment_values.observation,
        action=environment_values.action,
        reward=environment_values.reward,
        task=environment_values.task,
    )
    reset = make_env(environment_config, registry=component_registry).reset(seed=spec.learner_seed)
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


def _log_evaluation_to_wandb(spec: ExperimentSpec, evaluation_directory: Path, run_directory: Path) -> None:
    if spec.tracking.mode == "disabled":
        return
    import wandb

    wandb_run_id = load_wandb_run_id(run_directory)
    if wandb_run_id is None:
        raise ValueError("manifest is missing wandb_run_id")
    summary = json.loads((evaluation_directory / "summary.json").read_text(encoding="utf-8"))
    metrics = {
        f"evaluation/{name}": value
        for name, value in summary.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    with wandb.init(
        project=spec.tracking.project,
        entity=spec.tracking.entity,
        id=wandb_run_id,
        resume="must" if spec.tracking.mode == "online" else "allow",
        mode=spec.tracking.mode,
    ) as run:
        run.log(metrics, step=spec.algorithm_config.total_environment_steps)


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


def _prepare_run(spec: ExperimentSpec) -> Path:
    run_directory = Path(spec.artifacts.root).resolve() / spec.experiment_id / spec.trial_id
    if run_directory.exists() and spec.artifacts.reject_existing:
        raise FileExistsError(f"run already exists: {run_directory}")
    run_directory.mkdir(parents=True, exist_ok=True)
    import yaml

    (run_directory / "resolved_config.yaml").write_text(yaml.safe_dump(spec.as_dict(), sort_keys=True), encoding="utf-8")
    _write_run_manifest(run_directory, spec, "running", wandb_run_id=new_wandb_run_id())
    _write_status(run_directory, "running")
    return run_directory


def _fit(spec: ExperimentSpec, algorithm, config, run_directory: Path, schemas: Mapping[str, str]):
    batch = spec.algorithm_config.rollout_batch_size
    iterations = max(1, spec.algorithm_config.total_environment_steps // batch)
    tuner = tune.Tuner(
        algorithm.trainable,
        param_space=config.to_dict(),
        run_config=tune.RunConfig(
            name="ray",
            storage_path=str(run_directory),
            stop={"training_iteration": iterations},
            callbacks=_training_callbacks(spec, run_directory, algorithm, schemas),
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=max(1, spec.checkpoint.interval_environment_steps // batch),
                checkpoint_at_end=True,
                num_to_keep=spec.checkpoint.keep,
            ),
        ),
    )
    grid = tuner.fit()
    if grid.errors:
        raise RuntimeError(f"Ray Tune trial failed: {grid.errors[0]}")
    return grid[0]


def _save_checkpoint(run_directory: Path, result, spec: ExperimentSpec, algorithm, schemas: Mapping[str, str]) -> Path | None:
    if result.checkpoint is None:
        return None
    return _export_checkpoint(run_directory, result.checkpoint, "final", spec, algorithm, schemas)


def _finish_run(spec: ExperimentSpec, run_directory: Path, checkpoint: Path | None, registry, component_registry, *, evaluate: bool, report: bool) -> None:
    if evaluate and checkpoint is not None:
        from ...training.evaluation import evaluate_checkpoint

        evaluation = evaluate_checkpoint(
            checkpoint,
            split=spec.evaluation.split,
            output_directory=run_directory / "evaluation" / "final",
            registry=registry,
            component_registry=component_registry,
        )
        _log_evaluation_to_wandb(spec, evaluation, run_directory)
    if report and spec.tracking.mode == "online":
        from ...training.reports import generate_report

        generate_report(run_directory)


def run_experiment(
    spec: ExperimentSpec | str | Path,
    registry: TrainingRegistry | None = None,
    *,
    evaluate: bool = True,
    report: bool = True,
    component_registry=None,
) -> Path:
    """Run one validated Ray Tune trial and publish an immutable run manifest.

    Accepts a resolved :class:`~aresim.training.experiments.ExperimentSpec` or a
    path to experiment YAML. Writes ``resolved_config.yaml``, ``manifest.json``,
    and ``status.json`` under the artifact root. Optionally runs post-training
    evaluation and W&B-backed reporting. Always shuts down Ray in ``finally``.

    Returns the run directory path on success; leaves a failed manifest on error.
    """
    selected = create_training_registry() if registry is None else registry
    if isinstance(spec, (str, Path)):
        from ...training import load_experiment

        resolved = load_experiment(spec, registry=selected)
    else:
        resolved = spec
    resolved.validate()
    schemas, selected, algorithm, config = _resolve_training(resolved, selected, component_registry)
    run_directory = _prepare_run(resolved)
    try:
        _authenticate_tracking(resolved)
        result = _fit(resolved, algorithm, config, run_directory, schemas)
        checkpoint = _save_checkpoint(run_directory, result, resolved, algorithm, schemas)
        _finish_run(resolved, run_directory, checkpoint, selected, component_registry, evaluate=evaluate, report=report)
        _write_run_manifest(run_directory, resolved, "completed", str(result.path))
        _write_status(run_directory, "completed")
    except Exception as error:
        _write_run_manifest(run_directory, resolved, "failed")
        _write_status(run_directory, "failed", f"{type(error).__name__}: {error}")
        raise
    finally:
        import ray

        if ray.is_initialized():
            ray.shutdown()
    return run_directory


__all__ = [
    "AresCheckpointExportCallback",
    "AresMaskedPPORLModule",
    "AresMetricsCallback",
    "ENV_NAME",
    "LocalMaskedActorCritic",
    "MODEL_ID",
    "MaskedPPOFactory",
    "RUN_SCHEMA",
    "canonicalize_rllib_metrics",
    "run_experiment",
]
