"""Phase and disagreement settings for reward-free exploration and task adaptation.

Transfer is an explicit fresh-run operation; ordinary resume restores the current phase.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from ..dreamerv3.config import DreamerConfig
from ..common.config_decode import finite_number, positive_integer


@dataclass(frozen=True)
class Plan2ExploreConfig(DreamerConfig):
    """DreamerV3 plus a bootstrapped disagreement ensemble and phase identity."""

    phase: Literal["pretrain", "adapt"] = "pretrain"
    initialization_checkpoint: str | None = None
    ensemble_size: int = 5
    intrinsic_scale: float = 1.0
    demonstration_fraction: float = 0.0
    cycle_penalty: float = 0.0

    def validate(self) -> None:
        super().validate()
        if not 0 <= finite_number(self.demonstration_fraction, "demonstration_fraction") < 1:
            raise ValueError("demonstration_fraction must be in [0, 1)")
        if finite_number(self.cycle_penalty, "cycle_penalty") < 0:
            raise ValueError("cycle_penalty must be nonnegative")
        if self.phase != "adapt" and (self.demonstration_fraction or self.cycle_penalty):
            raise ValueError("demonstrations and cycle shaping require adaptation")
        positive_integer(self.ensemble_size, "ensemble_size")
        if self.ensemble_size < 2:
            raise ValueError("disagreement requires at least two ensemble members")
        if finite_number(self.intrinsic_scale, "intrinsic_scale") <= 0:
            raise ValueError("intrinsic_scale must be positive")
        if self.phase not in ("pretrain", "adapt"):
            raise ValueError("phase must be pretrain or adapt")
        if self.initialization_checkpoint is not None and (not isinstance(self.initialization_checkpoint, str) or not self.initialization_checkpoint.strip()):
            raise ValueError("initialization_checkpoint must be a nonempty path")
        if self.phase == "pretrain" and self.initialization_checkpoint is not None:
            raise ValueError("pretraining forbids initialization_checkpoint")
