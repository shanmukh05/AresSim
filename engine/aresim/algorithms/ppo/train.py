"""Masked PPO networks and RLlib factory.

Shared environment, metrics, and Tune orchestration live in training.rllib.
Compatibility exports retain the existing public PPO entry points.
"""
from __future__ import annotations
import os
from collections.abc import Mapping
from importlib import import_module
from typing import Any
import torch
from torch import Tensor, nn
from gymnasium import spaces
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env
from ...defaults import DEFAULT_ENVIRONMENT_CONFIG
from ...training.provenance import experiment_seeds
from ...training.rllib.environment import ENV_NAME, _environment
from ...training.rllib.metrics import AresMetricsCallback, canonicalize_rllib_metrics
from ...training.rllib.runner import AresCheckpointExportCallback, RUN_SCHEMA, run_experiment
from ..common.encoder import LocalObservationEncoder, mask_illegal_actions
from ..registry import TrainingContext
from .config import MaskedPPOConfig, ModelConfig, decode_config

MODEL_ID = "local_cnn_actor_critic"
_RLModuleBase = getattr(import_module("ray.rllib.core.rl_module.torch"), "Torch" + "RLModule")

class LocalMaskedActorCritic(nn.Module):
    """Encode local spatial/telemetry inputs and return masked logits and a value.

    Consumes policy observations and the authoritative action mask only. Illegal
    logits are set to dtype minimum before returning. Standalone/checkpoint
    instances validate output finiteness and mask contents; RLlib training can
    skip those GPU-synchronizing diagnostics after CPU environment validation.
    """

    def __init__(
        self, config: ModelConfig, window_size: int | None = None, action_count: int = 10,
        *, validate_outputs: bool = True, memory: bool = False,
    ) -> None:
        """Build layers from ``config``; ``action_count`` must be ``10`` for Phase 1."""
        super().__init__()
        config.validate()
        size = DEFAULT_ENVIRONMENT_CONFIG.observation_config.window_size if window_size is None else window_size
        if size <= 0 or action_count != 10:
            raise ValueError("local actor-critic requires a positive crop and Discrete(10)")
        self.config = config
        self.action_count = action_count
        self.validate_outputs = validate_outputs
        self.encoder = LocalObservationEncoder(config, size, memory=memory)
        self.policy = nn.Linear(config.fused_width, action_count)
        self.value = nn.Linear(config.fused_width, 1)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.policy.bias)
        nn.init.zeros_(self.value.bias)

    def forward_with_embedding(
        self, observation: Mapping[str, Tensor], action_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return logits, value, and the shared embedding for PPO's learner."""
        fused = self.encoder(observation)
        logits = mask_illegal_actions(
            self.policy(fused), action_mask, validate_contents=self.validate_outputs
        )
        values = self.value(fused).squeeze(-1)
        if self.validate_outputs:
            legal = action_mask.unsqueeze(0).bool() if action_mask.ndim == 1 else action_mask.bool()
            if not torch.isfinite(logits[legal]).all() or not torch.isfinite(values).all():
                raise ValueError("actor-critic produced non-finite outputs")
        return logits, values, fused

    def forward(self, observation: Mapping[str, Tensor], action_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Return masked action logits and one value per batch row."""
        logits, values, _ = self.forward_with_embedding(observation, action_mask)
        return logits, values

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
        self.core = LocalMaskedActorCritic(
            config, window_size, self.action_space.n, memory="navigation" in local_space.spaces,
            validate_outputs=os.environ.get("ARESIM_VALIDATE_MODEL_OUTPUTS") == "1",
        )

    def _outputs(self, batch: dict[str, Any]) -> dict[str, Any]:
        policy_input = batch[Columns.OBS]
        logits, _ = self.core(policy_input["observation"], policy_input["action_mask"])
        return {Columns.ACTION_DIST_INPUTS: logits}

    def _forward_inference(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return self._outputs(batch)

    def _forward_exploration(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return self._outputs(batch)

    def _forward_train(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        policy_input = batch[Columns.OBS]
        logits, _, fused = self.core.forward_with_embedding(
            policy_input["observation"], policy_input["action_mask"]
        )
        return {Columns.ACTION_DIST_INPUTS: logits, Columns.EMBEDDINGS: fused}

    def compute_values(self, batch: dict[str, Any], embeddings: Any = None):
        """Return per-batch value estimates for PPO advantage computation."""
        if embeddings is not None:
            return self.core.value(embeddings).squeeze(-1)
        policy_input = batch[Columns.OBS]
        return self.core(policy_input["observation"], policy_input["action_mask"])[1]

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
        training_seeds = experiment_seeds(spec).train
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
