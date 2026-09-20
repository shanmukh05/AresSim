"""Typed configuration for the built-in mask-aware DQN algorithm.

Immutable hyperparameters for RLlib DQN. Safe to import without Ray or PyTorch;
used by experiment YAML decoding and ``DQNConfig`` construction.

**Last updated:** September 12, 2026

**Contains:** ``MaskedDQNConfig``, :func:`decode_config`.

**YAML location:** ``algorithm_config`` in repository ``configs/masked_dqn/*.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..common.config_decode import decode_dataclass, finite_number, positive_integer

decode_config = decode_dataclass


@dataclass(frozen=True)
class MaskedDQNConfig:
    """Canonical action-masked DQN hyperparameters for RLlib training.

    ``rollout_batch_size`` is environment steps per Tune iteration (mapped to
    ``min_sample_timesteps_per_iteration``). ``train_batch_size`` is the replay
    minibatch. Parent evaluation/checkpoint intervals must divide ``rollout_batch_size``.
    """

    total_environment_steps: int = 4096
    rollout_batch_size: int = 256
    train_batch_size: int = 32
    replay_capacity: int = 8192
    learning_starts: int = 512
    n_step: int = 1
    double_q: bool = True
    dueling: bool = True
    gamma: float = 0.995
    learning_rate: float = 0.0001
    max_gradient_norm: float = 40.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_timesteps: int = 2000
    target_network_update_freq: int = 250
    tau: float = 1.0

    def epsilon_schedule(self) -> list[tuple[int, float]]:
        """Return the RLlib epsilon timestep schedule."""
        return [(0, self.epsilon_start), (self.epsilon_timesteps, self.epsilon_end)]

    def validate(self) -> None:
        """Raise ``ValueError`` when any hyperparameter is outside allowed ranges."""
        self._validate_counts()
        self._validate_rates()
        self._validate_exploration()

    def _validate_counts(self) -> None:
        for name, value in (
            ("total_environment_steps", self.total_environment_steps),
            ("rollout_batch_size", self.rollout_batch_size),
            ("train_batch_size", self.train_batch_size),
            ("replay_capacity", self.replay_capacity),
            ("learning_starts", self.learning_starts),
            ("n_step", self.n_step),
            ("epsilon_timesteps", self.epsilon_timesteps),
            ("target_network_update_freq", self.target_network_update_freq),
        ):
            positive_integer(value, name)
        if self.replay_capacity < self.train_batch_size:
            raise ValueError("replay_capacity must be at least train_batch_size")
        if self.total_environment_steps < self.learning_starts:
            raise ValueError("total_environment_steps must be at least learning_starts")
        if self.learning_starts > self.replay_capacity:
            raise ValueError("learning_starts cannot exceed replay_capacity")

    def _validate_rates(self) -> None:
        gamma = finite_number(self.gamma, "gamma")
        if not 0 < gamma <= 1:
            raise ValueError("gamma is outside its valid range")
        if min(finite_number(self.learning_rate, "learning_rate"), finite_number(self.max_gradient_norm, "max_gradient_norm")) <= 0:
            raise ValueError("learning rate and gradient norm must be positive")
        tau = finite_number(self.tau, "tau")
        if not 0 < tau <= 1:
            raise ValueError("tau must be in (0, 1]")

    def _validate_exploration(self) -> None:
        for name, value in (("epsilon_start", self.epsilon_start), ("epsilon_end", self.epsilon_end)):
            epsilon = finite_number(value, name)
            if not 0 <= epsilon <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not isinstance(self.double_q, bool) or not isinstance(self.dueling, bool):
            raise ValueError("double_q and dueling must be booleans")


__all__ = ["MaskedDQNConfig", "decode_config"]
