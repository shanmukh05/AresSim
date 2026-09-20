"""Mask-aware DQN RLModule, Q-network, and RLlib factory.

Builds Double/dueling DQN on the shared local encoder. Illegal actions are set
to ``-inf`` in ``compute_q_values`` so selection, epsilon-greedy, and
Double-DQN targets all exclude them. Simulator rules stay in ``aresim.core``.

**Last updated:** September 12, 2026

**Contains:** ``LocalMaskedQNetwork``, ``AresMaskedDQNRLModule``, ``MaskedDQNFactory``.

**See also:** :mod:`aresim.algorithms.dqn.config`, :mod:`aresim.algorithms.ppo.train`
(shared EnvRunners, metrics, and Tune driver).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from gymnasium import spaces
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.dqn.torch.default_dqn_torch_rl_module import DefaultDQNTorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.schedules.scheduler import Scheduler
from torch import Tensor, nn

from ...training.experiments import ExperimentSpec
from ...training.seeds import load_seed_manifest
from ..common.encoder import LocalObservationEncoder, combine_q_values
from ..ppo.config import ModelConfig
from ..ppo.train import ENV_NAME, AresMetricsCallback, _environment
from ..registry import TrainingContext
from .config import MaskedDQNConfig, decode_config

MODEL_ID = "local_cnn_q"


class LocalMaskedQNetwork(nn.Module):
    """Encode local observations and return masked Q-values for Discrete(10)."""

    def __init__(
        self,
        config: ModelConfig,
        window_size: int = 8,
        action_count: int = 10,
        *,
        dueling: bool = True,
    ) -> None:
        """Build the shared encoder plus Q / dueling heads."""
        super().__init__()
        if window_size <= 0 or action_count != 10:
            raise ValueError("local Q-network requires a positive crop and Discrete(10)")
        self.encoder = LocalObservationEncoder(config, window_size)
        self.advantage, self.value = _q_heads(config.fused_width, action_count, dueling)

    def forward(self, observation: Mapping[str, Tensor], action_mask: Tensor) -> Tensor:
        """Return finite masked Q-values, one row per batch item."""
        q_values = combine_q_values(self.encoder(observation), self.advantage, self.value, action_mask)
        legal = action_mask.unsqueeze(0).bool() if action_mask.ndim == 1 else action_mask.bool()
        if not torch.isfinite(q_values[legal]).all():
            raise ValueError("Q-network produced non-finite legal values")
        return q_values


class AresMaskedDQNRLModule(DefaultDQNTorchRLModule):
    """RLlib DQN module that masks Q-values in inference, exploration, and training.

    Expects Dict observations with ``observation`` and ``action_mask``. Target
    networks copy ``encoder``, ``af``, and optional ``vf`` so Double DQN uses the
    next state's mask stored on ``NEXT_OBS``.
    """

    def setup(self) -> None:
        """Build the Ares encoder and Q heads; skip RLlib's default catalog nets."""
        local_space, action_count = _require_local_spaces(self.observation_space, self.action_space)
        model = dict(self.model_config or {})
        self.uses_dueling, self.uses_double_q, self.num_atoms = _dqn_runtime_flags(model)
        self.epsilon_schedule = Scheduler(fixed_value_or_schedule=model["epsilon"], framework=self.framework)
        config = ModelConfig(**{key: value for key, value in model.items() if key in ModelConfig.__dataclass_fields__})
        self.encoder = LocalObservationEncoder(config, int(local_space["terrain_type"].shape[0]))
        self.af, value = _q_heads(config.fused_width, action_count, self.uses_dueling)
        if value is not None:
            self.vf = value

    def _qf_forward_helper(self, batch: dict[str, Tensor], encoder: nn.Module, head: nn.Module | dict[str, nn.Module]):
        """Compute masked Q-values from the current or target encoder/heads."""
        policy_input = batch[Columns.OBS]
        advantage, value = _heads(head)
        return {"qf_preds": combine_q_values(encoder(policy_input["observation"]), advantage, value, policy_input["action_mask"])}


def _require_local_spaces(observation_space: object, action_space: object) -> tuple[spaces.Dict, int]:
    if not isinstance(observation_space, spaces.Dict) or set(observation_space.spaces) != {"observation", "action_mask"}:
        raise ValueError("Ares masked DQN requires observation and action_mask inputs")
    if not isinstance(action_space, spaces.Discrete) or action_space.n != 10:
        raise ValueError("Ares masked DQN requires Discrete(10)")
    return observation_space["observation"], int(action_space.n)


def _dqn_runtime_flags(model: Mapping[str, Any]) -> tuple[bool, bool, int]:
    if "epsilon" not in model:
        raise ValueError("Ares masked DQN requires an epsilon schedule in model_config")
    atoms = int(model.get("num_atoms", 1))
    if atoms != 1:
        raise ValueError("Ares masked DQN does not implement distributional Q")
    return bool(model.get("dueling", True)), bool(model.get("double_q", True)), atoms


def _init_linear(layer: nn.Linear, gain: float = 1.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


def _q_heads(fused_width: int, action_count: int, dueling: bool) -> tuple[nn.Linear, nn.Linear | None]:
    advantage = _init_linear(nn.Linear(fused_width, action_count))
    value = _init_linear(nn.Linear(fused_width, 1)) if dueling else None
    return advantage, value


def _heads(head: nn.Module | dict[str, nn.Module]) -> tuple[nn.Module, nn.Module | None]:
    if isinstance(head, dict):
        return head["af"], head["vf"]
    return head, None


class MaskedDQNFactory:
    """Build RLlib ``DQNConfig`` for the ``masked_dqn`` registered algorithm."""

    algorithm_id = "masked_dqn"
    trainable = "DQN"
    checkpoint_loader_id = "rllib_masked_dqn"
    observation_schema = "aresim.obs.local.v1"
    action_schema = "aresim.action.rover.v1"

    def decode_config(self, payload: object):
        """Decode YAML ``algorithm_config`` into :class:`MaskedDQNConfig`."""
        return decode_config(MaskedDQNConfig, payload, "algorithm_config")

    def build(self, context: TrainingContext) -> DQNConfig:
        """Return a fully configured ``DQNConfig`` for Ray Tune."""
        spec = context.experiment
        dqn = spec.algorithm_config
        if not isinstance(dqn, MaskedDQNConfig):
            raise TypeError("masked_dqn requires MaskedDQNConfig")
        module_class = getattr(context.model_factory, "rl_module_class", None)
        if not isinstance(module_class, type):
            raise TypeError("masked_dqn requires a model factory with an RLModule class")
        from ray.tune.registry import register_env

        register_env(ENV_NAME, _environment)
        resources = spec.resources
        return (
            DQNConfig()
            .environment(ENV_NAME, env_config={
                "environment": spec.as_dict()["environment"],
                "training_seeds": load_seed_manifest(spec.evaluation.seed_manifest).train,
                "learner_seed": spec.learner_seed,
                "component_registry": context.component_registry,
            })
            .framework("torch")
            .debugging(seed=spec.learner_seed)
            .env_runners(
                num_env_runners=resources.num_env_runners,
                num_envs_per_env_runner=resources.num_envs_per_env_runner,
                num_cpus_per_env_runner=resources.cpus_per_env_runner,
                rollout_fragment_length="auto",
                batch_mode="truncate_episodes",
            )
            .learners(num_learners=resources.num_learners, num_gpus_per_learner=resources.gpus_per_learner)
            .reporting(min_sample_timesteps_per_iteration=dqn.rollout_batch_size)
            .training(**_training_kwargs(dqn))
            .rl_module(rl_module_spec=RLModuleSpec(module_class=module_class, model_config=_module_config(spec, dqn)))
            .callbacks(AresMetricsCallback)
        )


def _training_kwargs(dqn: MaskedDQNConfig) -> dict[str, Any]:
    return {
        "gamma": dqn.gamma,
        "lr": dqn.learning_rate,
        "grad_clip": dqn.max_gradient_norm,
        "train_batch_size_per_learner": dqn.train_batch_size,
        "double_q": dqn.double_q,
        "dueling": dqn.dueling,
        "n_step": dqn.n_step,
        "epsilon": dqn.epsilon_schedule(),
        "num_steps_sampled_before_learning_starts": dqn.learning_starts,
        "target_network_update_freq": dqn.target_network_update_freq,
        "tau": dqn.tau,
        "td_error_loss_fn": "huber",
        "store_buffer_in_checkpoints": False,
        "replay_buffer_config": {"type": "EpisodeReplayBuffer", "capacity": dqn.replay_capacity},
    }


def _module_config(spec: ExperimentSpec, dqn: MaskedDQNConfig) -> dict[str, Any]:
    values = dict(spec.as_dict()["model_config"])
    values.update(dueling=dqn.dueling, double_q=dqn.double_q, num_atoms=1, epsilon=dqn.epsilon_schedule())
    return values


__all__ = [
    "AresMaskedDQNRLModule",
    "LocalMaskedQNetwork",
    "MODEL_ID",
    "MaskedDQNFactory",
]
