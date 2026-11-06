"""Imagined actor-critic parameters and robust return normalization.

Task and exploration behaviors own separate critics, slow targets, and normalization state.
"""
from __future__ import annotations
from copy import deepcopy
import torch
from torch import Tensor, nn
from .config import DreamerModelConfig
from .distributions import scalar_prediction
from .model import mlp


class ActorCritic(nn.Module):
    """Categorical actor and two-hot critic with a separately updated slow critic."""

    def __init__(self, features: int, config: DreamerModelConfig) -> None:
        super().__init__()
        self.actor = mlp(features, config.hidden_size, 10)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)
        self.critic = mlp(features, config.hidden_size, config.value_bins, zero_output=True)
        self.slow_critic = deepcopy(self.critic).requires_grad_(False)
        self.register_buffer("return_low", torch.zeros(()))
        self.register_buffer("return_high", torch.zeros(()))

    def value(self, features: Tensor) -> Tensor:
        return scalar_prediction(self.critic(features))

    @torch.no_grad()
    def return_scale(self, targets: Tensor) -> Tensor:
        quantiles = torch.quantile(targets.detach().float(), torch.tensor([0.05, 0.95], device=targets.device))
        self.return_low.lerp_(quantiles[0], 0.01)
        self.return_high.lerp_(quantiles[1], 0.01)
        return (self.return_high - self.return_low).clamp(min=1)

    @torch.no_grad()
    def update_slow(self, rate: float) -> None:
        for slow, current in zip(self.slow_critic.parameters(), self.critic.parameters(), strict=True):
            slow.lerp_(current, rate)
