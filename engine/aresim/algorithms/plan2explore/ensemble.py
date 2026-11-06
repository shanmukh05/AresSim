"""Bootstrapped next-embedding predictors and latent disagreement reward.

Targets are detached encoder features; ensemble fitting must not alter the shared world model.
"""
from __future__ import annotations
import torch
from torch import Tensor, nn
from ..dreamerv3.model import mlp


class DisagreementEnsemble(nn.Module):
    """Independently initialized predictors trained on independently resampled pairs."""

    def __init__(self, features: int, embedding: int, hidden: int, members: int = 5) -> None:
        super().__init__()
        self.members = nn.ModuleList(mlp(features + 10, hidden, embedding) for _ in range(members))

    def forward(self, features: Tensor, actions: Tensor) -> Tensor:
        inputs = torch.cat((features, actions), -1)
        return torch.stack([member(inputs) for member in self.members], 0)

    def reward(self, features: Tensor, actions: Tensor) -> Tensor:
        return self(features, actions).var(0, unbiased=True).mean(-1)

    def loss(self, features: Tensor, actions: Tensor, targets: Tensor) -> Tensor:
        inputs = torch.cat((features.detach(), actions.detach()), -1)
        targets = targets.detach()
        losses = []
        for member in self.members:
            indices = torch.randint(len(inputs), (len(inputs),), device=inputs.device)
            losses.append((member(inputs[indices]) - targets[indices]).square().mean())
        return torch.stack(losses).mean()
