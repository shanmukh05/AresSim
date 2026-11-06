"""Validated MuZero experiment settings independent of Ray and PyTorch.

The common experiment envelope selects these dataclasses for algorithm
``muzero``; other algorithms retain their existing configuration types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from ..common.config_decode import finite_number, positive_integer


@dataclass(frozen=True)
class MuZeroConfig:
    """Search, replay, auxiliary supervision and learner settings for MuZero."""

    threshold_schedule: ClassVar[bool] = True
    total_environment_steps: int = 4096
    rollout_batch_size: int = 256
    simulations: int = 12
    evaluation_simulations: int = 12
    unroll_steps: int = 5
    td_steps: int = 10
    batch_size: int = 32
    replay_capacity: int = 8192
    learning_starts: int = 512
    updates_per_environment_step: float = 0.1
    gamma: float = 0.995
    learning_rate: float = 0.0001
    max_gradient_norm: float = 5.0
    weight_decay: float = 0.0001
    priority_alpha: float = 1.0
    priority_beta: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25
    temperature: float = 1.0
    puct_c1: float = 1.25
    puct_c2: float = 19652.0
    reanalysis_interval_updates: int = 100
    reanalysis_positions: int = 16
    reanalysis_fraction: float = 0.25
    reanalysis_batch_size: int = 128
    reanalysis_bootstrap_search: bool = True
    legality_positive_weights: tuple[float, ...] = (1.0,) * 10
    consistency_loss_weight: float = 0.0

    def validate(self) -> None:
        for name in ("total_environment_steps", "rollout_batch_size", "simulations", "evaluation_simulations",
                     "unroll_steps", "td_steps", "batch_size", "replay_capacity", "learning_starts",
                     "reanalysis_interval_updates", "reanalysis_positions", "reanalysis_batch_size"):
            positive_integer(getattr(self, name), name)
        if self.replay_capacity < self.batch_size or self.learning_starts > self.replay_capacity:
            raise ValueError("MuZero replay capacity must contain the warmup and one batch")
        if self.total_environment_steps < self.learning_starts:
            raise ValueError("MuZero total steps must reach learning_starts")
        for name in ("updates_per_environment_step", "learning_rate", "max_gradient_norm", "puct_c1", "puct_c2", "dirichlet_alpha", "temperature"):
            if finite_number(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("priority_alpha", "priority_beta", "weight_decay", "dirichlet_fraction"):
            value = finite_number(getattr(self, name), name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0 < finite_number(self.gamma, "gamma") <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if not 0 <= finite_number(self.reanalysis_fraction, "reanalysis_fraction") <= 1:
            raise ValueError("reanalysis_fraction must be in [0, 1]")
        if not isinstance(self.reanalysis_bootstrap_search, bool):
            raise ValueError("reanalysis_bootstrap_search must be boolean")
        if not isinstance(self.legality_positive_weights, (tuple, list)) or len(self.legality_positive_weights) != 10:
            raise ValueError("legality_positive_weights must contain ten action weights")
        if any(finite_number(value, "legality_positive_weights") < 1 for value in self.legality_positive_weights):
            raise ValueError("legality_positive_weights must be at least one")
        if finite_number(self.consistency_loss_weight, "consistency_loss_weight") < 0:
            raise ValueError("consistency_loss_weight must be non-negative")


@dataclass(frozen=True)
class MuZeroModelConfig:
    """Compact symbolic-observation architecture and categorical support."""

    terrain_embedding: int = 4
    objective_embedding: int = 4
    conv_channels: tuple[int, int] = (32, 64)
    telemetry_layers: tuple[int, int] = (128, 128)
    fused_width: int = 256
    history_length: int = 4
    latent_width: int = 256
    dynamics_blocks: int = 2
    support_size: int = 300
    predict_constraints: bool = True
    consistency_projection_width: int = 0

    def validate(self) -> None:
        from ..common.model_config import ModelConfig

        ModelConfig(self.terrain_embedding, self.objective_embedding, self.conv_channels,
                    self.telemetry_layers, self.fused_width).validate()
        for name in ("history_length", "latent_width", "dynamics_blocks", "support_size"):
            positive_integer(getattr(self, name), name)
        if not isinstance(self.predict_constraints, bool):
            raise ValueError("predict_constraints must be boolean")
        positive_integer(self.consistency_projection_width, "consistency_projection_width", allow_zero=True)
        if self.consistency_projection_width == 1:
            raise ValueError("consistency_projection_width must be zero or at least two")


@dataclass(frozen=True)
class MuZeroResourceConfig:
    """Configure bounded Ray CPU search collectors and the learner device.

    Ray collectors own CPU search models; the driver owns the learner device.
    One pending collection per actor and wave boundaries bound memory and policy age.
    """

    device: Literal["cpu", "cuda"] = "cpu"
    torch_threads: int = 1
    num_collectors: int = 1
    num_envs_per_collector: int = 1
    collection_steps_per_env: int = 32
    collector_torch_threads: int = 1
    max_policy_lag_updates: int = 32

    def validate(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("MuZero device must be cpu or cuda")
        positive_integer(self.torch_threads, "torch_threads")
        for name in ("num_collectors", "num_envs_per_collector", "collection_steps_per_env",
                     "collector_torch_threads", "max_policy_lag_updates"):
            positive_integer(getattr(self, name), name)


__all__ = ["MuZeroConfig", "MuZeroModelConfig", "MuZeroResourceConfig"]
