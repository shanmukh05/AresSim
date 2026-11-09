"""Framework-independent configuration for symbolic DreamerV3 and its Ray runtime.

All counts and numerical settings validate before training allocates workers or artifacts.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Literal
from ..common.model_config import ModelConfig
from ..common.config_decode import finite_number, positive_integer


@dataclass(frozen=True)
class DreamerConfig:
    """Sequence-model and imagined-behavior learning settings."""

    threshold_schedule: ClassVar[bool] = True
    total_environment_steps: int = 102400
    batch_size: int = 16
    sequence_length: int = 32
    replay_capacity: int = 250000
    learning_starts: int = 2048
    replay_ratio: float = 16.0
    gamma: float = 0.995
    imagination_horizon: int = 15
    lambda_return: float = 0.95
    model_learning_rate: float = 0.0001
    actor_learning_rate: float = 0.00003
    critic_learning_rate: float = 0.00003
    max_gradient_norm: float = 100.0
    free_nats: float = 1.0
    dynamics_loss_weight: float = 1.0
    representation_loss_weight: float = 0.1
    entropy_coefficient: float = 0.0003
    slow_critic_rate: float = 0.02
    slow_regularization_weight: float = 1.0
    replay_value_loss_weight: float = 0.3
    legality_threshold: float = 0.5
    legality_positive_weights: tuple[float, ...] = (1.0,) * 10

    def validate(self) -> None:
        for name in ("total_environment_steps", "batch_size", "sequence_length", "replay_capacity", "learning_starts", "imagination_horizon"):
            positive_integer(getattr(self, name), name)
        self._validate_rates()
        if self.learning_starts > min(self.replay_capacity, self.total_environment_steps):
            raise ValueError("warmup must fit the replay and total step budget")
        if len(self.legality_positive_weights) != 10 or any(finite_number(w, "legality weight") < 1 for w in self.legality_positive_weights):
            raise ValueError("legality_positive_weights must contain ten weights >=1")

    def _validate_rates(self) -> None:
        for name in ("replay_ratio", "model_learning_rate", "actor_learning_rate", "critic_learning_rate", "max_gradient_norm"):
            if finite_number(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("free_nats", "dynamics_loss_weight", "representation_loss_weight", "entropy_coefficient", "slow_regularization_weight", "replay_value_loss_weight"):
            if finite_number(getattr(self, name), name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in ("gamma", "lambda_return", "slow_critic_rate", "legality_threshold"):
            if not 0 < finite_number(getattr(self, name), name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")


@dataclass(frozen=True)
class DreamerModelConfig(ModelConfig):
    """Small categorical RSSM plus structured decoder and actor/critic heads."""

    deterministic_size: int = 512
    stochastic_size: int = 32
    categories: int = 32
    hidden_size: int = 256
    unimix: float = 0.01
    value_bins: int = 255
    # Missing fields in historical checkpoints must retain independent decoder heads.
    decoder_layout: Literal["independent", "grouped"] = "independent"

    def validate(self) -> None:
        super().validate()
        for name in ("deterministic_size", "stochastic_size", "categories", "hidden_size", "value_bins"):
            positive_integer(getattr(self, name), name)
        if self.categories < 2 or self.value_bins < 3 or self.value_bins % 2 == 0:
            raise ValueError("RSSM needs >=2 categories and an odd value_bins >=3")
        if not 0 <= finite_number(self.unimix, "unimix") < 1:
            raise ValueError("unimix must be in [0, 1)")
        if self.decoder_layout not in ("independent", "grouped"):
            raise ValueError("decoder_layout must be independent or grouped")


@dataclass(frozen=True)
class WorldModelResources:
    """Bounded CPU collection waves and one CPU/CUDA learner; no RLlib dependency."""

    device: Literal["cpu", "cuda"] = "cpu"
    torch_threads: int = 1
    num_collectors: int = 1
    num_envs_per_collector: int = 1
    collection_steps_per_env: int = 32
    collector_torch_threads: int = 1

    def validate(self) -> None:
        if self.device not in ("cpu", "cuda"):
            raise ValueError("world-model device must be cpu or cuda")
        for name in ("torch_threads", "num_collectors", "num_envs_per_collector", "collection_steps_per_env", "collector_torch_threads"):
            positive_integer(getattr(self, name), name)
