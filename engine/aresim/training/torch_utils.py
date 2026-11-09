"""Tensor copies shared by native training snapshots and checkpoints.

Copies preserve module devices and parameter storage while detaching immutable CPU payloads.
Only the optional PyTorch training stack imports this module.
"""
from __future__ import annotations

import torch


def cpu_state(value):
    """Copy a nested state without moving live modules or cloning a CUDA transfer twice."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_state(item) for item in value)
    return value
