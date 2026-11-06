"""Dreamer registry factory; numerical training is shared with Plan2Explore.

The factory validates its configuration without making RLlib a world-model dependency.
"""
from __future__ import annotations
from ..registry import TrainingContext
from .config import DreamerConfig
from ..common.config_decode import decode_dataclass


class DreamerFactory:
    """Select the native PyTorch/Ray sequence-model runner."""

    algorithm_id = "dreamerv3"
    trainable = "dreamerv3"
    checkpoint_loader_id = "torch_dreamerv3"
    observation_schema = "aresim.obs.local.v1"
    action_schema = "aresim.action.rover.v1"

    def decode_config(self, payload: object) -> DreamerConfig:
        return decode_dataclass(DreamerConfig, payload, "algorithm_config")

    def build(self, context: TrainingContext) -> DreamerConfig:
        if not isinstance(context.experiment.algorithm_config, DreamerConfig):
            raise TypeError("Dreamer requires DreamerConfig")
        return context.experiment.algorithm_config
