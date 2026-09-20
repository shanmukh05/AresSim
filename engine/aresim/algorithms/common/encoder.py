"""Shared local-observation encoder for learned rover policies.

Owns the CNN + telemetry trunk for ``aresim.obs.local.v1``. Masked PPO and
masked DQN attach different heads; neither this module nor those heads may
read ``WorldState`` or rebuild legality.

**Last updated:** September 12, 2026

**See also:** :mod:`aresim.algorithms.ppo.train`, :mod:`aresim.algorithms.dqn.train`.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from ..ppo.config import ModelConfig


class LocalObservationEncoder(nn.Module):
    """Encode an 8×8 local crop plus rover/colony telemetry into a fused vector.

    Illegal-action masking is applied by the policy or Q head, not here.
    """

    def __init__(self, config: ModelConfig, window_size: int = 8) -> None:
        """Build spatial and telemetry branches from ``config``."""
        super().__init__()
        config.validate()
        if window_size <= 0:
            raise ValueError("local encoder requires a positive crop")
        self.config = config
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
        self.apply(initialize_linear_conv)

    @staticmethod
    def _as_batch(value: Tensor, dimensions: int) -> Tensor:
        return value.unsqueeze(0) if value.ndim == dimensions - 1 else value

    def forward(self, observation: Mapping[str, Tensor]) -> Tensor:
        """Return one fused embedding per batch row."""
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
        telemetry = self.telemetry(torch.cat(
            (self_vector, colony, self.pad_embedding(pad), self.weather_embedding(weather), objective_features),
            dim=-1,
        ))
        return self.fusion(torch.cat((spatial_features, telemetry), dim=-1))


def initialize_linear_conv(module: nn.Module) -> None:
    """Orthogonal init for linear and convolution weights."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=2 ** 0.5)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def mask_illegal_actions(scores: Tensor, action_mask: Tensor, *, illegal_value: float | None = None) -> Tensor:
    """Replace illegal action scores. PPO keeps the default finite min; DQN passes ``-inf``."""
    mask = action_mask.unsqueeze(0) if action_mask.ndim == scores.ndim - 1 else action_mask
    mask = mask.bool()
    if mask.shape != scores.shape or not mask.any(dim=-1).all():
        raise ValueError("action mask shape is invalid or contains no legal action")
    fill = torch.finfo(scores.dtype).min if illegal_value is None else illegal_value
    return scores.masked_fill(~mask, fill)


def combine_q_values(fused: Tensor, advantage_head: nn.Module, value_head: nn.Module | None, action_mask: Tensor) -> Tensor:
    """Build masked Q-values, centering advantages over legal actions when dueling.

    Illegal Q is ``-inf`` so RLlib epsilon-greedy (``neginf`` → 0) and dueling
    ``nanmean`` skip those actions. PPO logits still use the finite default.
    """
    advantages = mask_illegal_actions(advantage_head(fused), action_mask, illegal_value=float("-inf"))
    if value_head is None:
        return advantages
    centered = advantages - torch.nan_to_num(advantages, neginf=torch.nan).nanmean(dim=-1, keepdim=True)
    return mask_illegal_actions(centered + value_head(fused), action_mask, illegal_value=float("-inf"))


__all__ = ["LocalObservationEncoder", "combine_q_values", "initialize_linear_conv", "mask_illegal_actions"]
