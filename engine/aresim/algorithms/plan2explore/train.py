"""Plan2Explore registration and explicit reward-free/adaptation phase validation.

The shared world-model runner owns collection and checkpoint lifecycle.
"""
from __future__ import annotations
from ..registry import TrainingContext
from ..dreamerv3.train import DreamerFactory
from ..common.config_decode import decode_dataclass
from .config import Plan2ExploreConfig


class Plan2ExploreFactory(DreamerFactory):
    """Select disagreement exploration or downstream task behavior from phase config."""

    algorithm_id = "plan2explore"
    trainable = "plan2explore"
    checkpoint_loader_id = "torch_plan2explore"

    def decode_config(self, payload: object) -> Plan2ExploreConfig:
        return decode_dataclass(Plan2ExploreConfig, payload, "algorithm_config")

    def build(self, context: TrainingContext) -> Plan2ExploreConfig:
        config = context.experiment.algorithm_config
        if not isinstance(config, Plan2ExploreConfig):
            raise TypeError("Plan2Explore requires Plan2ExploreConfig")
        expected = "open_exploration" if config.phase == "pretrain" else "resource_mission"
        if context.experiment.environment.task != expected:
            raise ValueError(f"Plan2Explore {config.phase} requires task={expected}")
        return config
