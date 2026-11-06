"""Learn a value-equivalent latent model for AresSim's symbolic rover input.

The representation consumes recent policy observations and actions. Dynamics
predict latent transitions and rewards without invoking simulator rules.
Optional training-only projection heads align predictions with recorded future
observations; search and frozen action selection do not execute these heads.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from ..common.encoder import LocalObservationEncoder, initialize_linear_conv
from ..common.model_config import ModelConfig
from .config import MuZeroModelConfig


def scalar_transform(value: Tensor) -> Tensor:
    """Map arbitrary-scale reward and value to the paper's bounded support scale."""
    return value.sign() * ((value.abs() + 1).sqrt() - 1 + 0.001 * value.abs())


def inverse_scalar_transform(value: Tensor) -> Tensor:
    """Invert :func:`scalar_transform` after categorical expectation."""
    magnitude = (((1 + 4 * 0.001 * (value.abs() + 1 + 0.001)).sqrt() - 1) / (2 * 0.001)) ** 2 - 1
    return value.sign() * magnitude.clamp(min=0)


def support_target(value: Tensor, support_size: int) -> Tensor:
    """Interpolate a transformed scalar between adjacent categorical bins."""
    shifted = scalar_transform(value).clamp(-support_size, support_size) + support_size
    low = shifted.floor().long()
    high = shifted.ceil().long()
    result = torch.zeros(*value.shape, support_size * 2 + 1, device=value.device, dtype=torch.float32)
    result.scatter_add_(-1, low.unsqueeze(-1), (high.float() - shifted).unsqueeze(-1))
    result.scatter_add_(-1, high.unsqueeze(-1), (shifted - low.float()).unsqueeze(-1))
    same = low == high
    result.scatter_add_(-1, low.unsqueeze(-1), same.float().unsqueeze(-1))
    return result


def support_value(logits: Tensor, support_size: int) -> Tensor:
    bins = torch.arange(-support_size, support_size + 1, device=logits.device, dtype=logits.dtype)
    return inverse_scalar_transform((logits.softmax(-1) * bins).sum(-1))


class _Residual(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))

    def forward(self, value: Tensor) -> Tensor:
        return torch.relu(value + self.layers(value))


class MuZeroNetwork(nn.Module):
    """Expose initial and recurrent inference for search and unrolled learning."""

    def __init__(self, config: MuZeroModelConfig, window_size: int = 8, *, memory: bool = False) -> None:
        super().__init__()
        config.validate()
        self.config = config
        encoder_config = ModelConfig(config.terrain_embedding, config.objective_embedding,
                                     config.conv_channels, config.telemetry_layers, config.fused_width)
        self.encoder = LocalObservationEncoder(encoder_config, window_size, memory=memory)
        self.representation = nn.Sequential(
            nn.Linear(config.history_length * (config.fused_width + 11), config.latent_width), nn.ReLU(),
            _Residual(config.latent_width), nn.Sigmoid(),
        )
        self.dynamics = nn.Sequential(nn.Linear(config.latent_width + 10, config.latent_width), nn.ReLU(),
                                      *(_Residual(config.latent_width) for _ in range(config.dynamics_blocks)))
        self.policy = nn.Linear(config.latent_width, 10)
        self.value = nn.Linear(config.latent_width, 2 * config.support_size + 1)
        self.reward = nn.Linear(config.latent_width, 2 * config.support_size + 1)
        if config.predict_constraints:
            self.legality = nn.Linear(config.latent_width, 10)
            self.continuation = nn.Linear(config.latent_width, 1)
        if config.consistency_projection_width:
            width = config.consistency_projection_width
            self.consistency_projector = nn.Sequential(nn.Linear(config.latent_width, width),
                nn.LayerNorm(width), nn.ReLU(), nn.Linear(width, width))
            self.consistency_predictor = nn.Sequential(nn.Linear(width, width),
                nn.LayerNorm(width), nn.ReLU(), nn.Linear(width, width))
        self.apply(initialize_linear_conv)
        if config.predict_constraints:
            nn.init.constant_(self.continuation.bias, 2.0)

    def constraints(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        """Predict hypothetical legality and nonterminal continuation."""
        if not self.config.predict_constraints:
            raise ValueError("legacy MuZero model does not predict constraints")
        return self.legality(latent), self.continuation(latent).squeeze(-1)

    def initial(self, observations: Mapping[str, Tensor], previous_actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode tensors shaped ``[batch, history, ...]`` and predict root policy/value."""
        if previous_actions.shape[1] != self.config.history_length:
            raise ValueError("MuZero observation history has the wrong length")
        latent = self.encode_sequence(observations, previous_actions)[:, 0]
        return latent, self.policy(latent), self.value(latent)

    def encode_sequence(self, observations: Mapping[str, Tensor], previous_actions: Tensor) -> Tensor:
        """Encode overlapping histories after one encoder pass per sequence frame.

        Returns ``[batch, sequence-history+1, latent]``. This also packs stopped-
        gradient future targets without re-encoding each overlapping crop.
        """
        batch, history = previous_actions.shape
        if history < self.config.history_length:
            raise ValueError("MuZero sequence is shorter than its history")
        flattened = {key: value.reshape(batch * history, *value.shape[2:]) for key, value in observations.items()}
        embedded = self.encoder(flattened).reshape(batch, history, -1)
        action_planes = torch.nn.functional.one_hot(previous_actions.long(), 11).float()
        features = torch.cat((embedded, action_planes), dim=-1)
        windows = features.unfold(1, self.config.history_length, 1).transpose(-1, -2).flatten(-2)
        return self.representation(windows)

    def project_consistency(self, latent: Tensor, *, prediction: bool = False) -> Tensor:
        """Project a latent for auxiliary learning, optionally through the predictor."""
        if not self.config.consistency_projection_width:
            raise ValueError("MuZero consistency projection is disabled")
        projected = self.consistency_projector(latent)
        return self.consistency_predictor(projected) if prediction else projected

    def recurrent(self, latent: Tensor, action: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Predict one hypothetical transition, reward, next policy, and value."""
        one_hot = torch.nn.functional.one_hot(action.long().reshape(-1), 10).float()
        next_latent = self.dynamics(torch.cat((latent, one_hot), dim=-1)).sigmoid()
        return next_latent, self.reward(next_latent), self.policy(next_latent), self.value(next_latent)


__all__ = ["MuZeroNetwork", "scalar_transform", "inverse_scalar_transform", "support_target", "support_value"]
