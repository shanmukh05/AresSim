"""Categorical latent sampling, masked policies, and Dreamer scalar predictions.

MuZero uses a different scalar transform; these functions implement signed-log/exponential support.
"""
from __future__ import annotations
import torch
from torch import Tensor
import torch.nn.functional as F


def symlog(value: Tensor) -> Tensor:
    return value.sign() * torch.log1p(value.abs())


def symexp(value: Tensor) -> Tensor:
    return value.sign() * torch.expm1(value.abs())


def latent_probs(logits: Tensor, unimix: float) -> Tensor:
    return (1 - unimix) * logits.softmax(-1) + unimix / logits.shape[-1]


def sample_latent(logits: Tensor, unimix: float, *, generator=None, deterministic: bool = False) -> Tensor:
    """Sample categorical variables with straight-through probability gradients."""
    probabilities = latent_probs(logits, unimix)
    indices = probabilities.argmax(-1) if deterministic else torch.multinomial(
        probabilities.reshape(-1, probabilities.shape[-1]), 1, generator=generator).reshape(probabilities.shape[:-1])
    sample = F.one_hot(indices, probabilities.shape[-1]).to(probabilities.dtype)
    return sample + probabilities - probabilities.detach()


def policy_probs(logits: Tensor, mask: Tensor, unimix: float) -> Tensor:
    """Mix uniform mass only over legal actions, retaining exactly zero illegal mass."""
    legal = mask.bool()
    if legal.shape != logits.shape or not legal.any(-1).all():
        raise ValueError("actor requires aligned, nonempty legal masks")
    probabilities = logits.masked_fill(~legal, torch.finfo(logits.dtype).min).softmax(-1)
    uniform = legal.to(logits.dtype) / legal.sum(-1, keepdim=True)
    return (1 - unimix) * probabilities + unimix * uniform


def support(bins: int, device) -> Tensor:
    return symexp(torch.linspace(-20, 20, bins, device=device))


def twohot_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Interpolate scalar targets on signed exponential bins and score categorical logits."""
    values = support(logits.shape[-1], logits.device)
    target = targets.detach().clamp(values[0], values[-1]).contiguous()
    high = torch.searchsorted(values, target).clamp(1, len(values) - 1)
    low = high - 1
    fraction = (target - values[low]) / (values[high] - values[low])
    log_probs = logits.log_softmax(-1)
    return -(1 - fraction) * log_probs.gather(-1, low.unsqueeze(-1)).squeeze(-1) - fraction * log_probs.gather(-1, high.unsqueeze(-1)).squeeze(-1)


def scalar_prediction(logits: Tensor) -> Tensor:
    """Compute the scalar expectation with paired bins to avoid symmetric cancellation."""
    probabilities = logits.softmax(-1)
    middle = logits.shape[-1] // 2
    positive = support(logits.shape[-1], logits.device)[middle + 1:]
    return ((probabilities[..., middle + 1:] - probabilities[..., :middle].flip(-1)) * positive).sum(-1)


def lambda_returns(rewards: Tensor, discounts: Tensor, values: Tensor, lambda_: float, valid: Tensor | None = None) -> Tensor:
    """Return bootstrapped returns for T rewards and T+1 state values."""
    result = []
    tail = values[..., -1]
    for index in range(rewards.shape[-1] - 1, -1, -1):
        tail = rewards[..., index] + discounts[..., index] * ((1 - lambda_) * values[..., index + 1] + lambda_ * tail)
        if valid is not None:
            tail = torch.where(valid[..., index], tail, values[..., index])
        result.append(tail)
    return torch.stack(result[::-1], -1)
