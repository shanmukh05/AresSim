"""Typed configuration for the built-in masked PPO algorithm.

Immutable dataclasses for PPO hyperparameters and the local CNN actor-critic
architecture. Safe to import without Ray or PyTorch; used by experiment YAML
decoding and RLlib module construction.

**Last updated:** September 5, 2026

**Contains:** ``MaskedPPOConfig``, ``ModelConfig``, :func:`decode_config`.

**YAML location:** ``algorithm_config`` and ``model_config`` sections in repository
``configs/<algorithm>/*.yaml``; envelope parsing lives in
:mod:`aresim.training.experiments`.

**Validation:** each dataclass exposes ``validate()``; unknown YAML keys are rejected
by :func:`decode_config`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..common.config_decode import decode_dataclass, finite_number, positive_integer

decode_config = decode_dataclass


@dataclass(frozen=True)
class MaskedPPOConfig:
    """Canonical action-masked PPO hyperparameters for RLlib training.

    ``rollout_batch_size`` must divide ``minibatch_size`` and all environment-step
    schedules in the parent :class:`~aresim.training.experiments.ExperimentSpec`.
    """

    total_environment_steps: int = 4096
    rollout_batch_size: int = 4096
    minibatch_size: int = 256
    update_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    learning_rate: float = 0.0003
    max_gradient_norm: float = 0.5
    target_kl: float = 0.01

    def validate(self) -> None:
        """Raise ``ValueError`` when any hyperparameter is outside allowed ranges."""
        integers = (self.total_environment_steps, self.rollout_batch_size, self.minibatch_size, self.update_epochs)
        for name, value in zip(("total_environment_steps", "rollout_batch_size", "minibatch_size", "update_epochs"), integers, strict=True):
            positive_integer(value, name)
        if self.rollout_batch_size % self.minibatch_size:
            raise ValueError("rollout_batch_size must be divisible by minibatch_size")
        gamma = finite_number(self.gamma, "gamma")
        gae_lambda = finite_number(self.gae_lambda, "gae_lambda")
        if not 0 < gamma <= 1 or not 0 <= gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda are outside their valid ranges")
        positive = tuple(finite_number(value, name) for name, value in (
            ("clip_param", self.clip_param), ("learning_rate", self.learning_rate),
            ("max_gradient_norm", self.max_gradient_norm),
        ))
        finite_number(self.value_loss_coefficient, "value_loss_coefficient")
        finite_number(self.entropy_coefficient, "entropy_coefficient")
        finite_number(self.target_kl, "target_kl")
        if min(positive) <= 0:
            raise ValueError("PPO clip, learning rate, and gradient norm must be positive")


@dataclass(frozen=True)
class ModelConfig:
    """Architecture dimensions for :class:`~aresim.algorithms.ppo.train.LocalMaskedActorCritic`.

    Expects exactly two convolution and two telemetry hidden layers. Embeddings
    size the terrain, objective-type, pad-proximity, and weather branches.
    """

    terrain_embedding: int = 4
    objective_embedding: int = 4
    conv_channels: tuple[int, int] = (32, 64)
    telemetry_layers: tuple[int, int] = (128, 128)
    fused_width: int = 256

    def validate(self) -> None:
        """Raise ``ValueError`` when layer counts or dimensions are invalid."""
        if len(self.conv_channels) != 2 or len(self.telemetry_layers) != 2:
            raise ValueError("the built-in model requires two convolution and telemetry layers")
        values = (self.terrain_embedding, self.objective_embedding, *self.conv_channels, *self.telemetry_layers, self.fused_width)
        for value in values:
            positive_integer(value, "model dimension")


__all__ = ["MaskedPPOConfig", "ModelConfig", "decode_config"]
