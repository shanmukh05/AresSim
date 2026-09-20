"""Typed configuration for the inference-only Jev rover agent.

Safe to import without the TypeSafe SDK. Used by ``JevAgent``, rollout YAML, and
live-client construction.

**Last updated:** September 20, 2026

**Contains:** ``JevAgentConfig`` and :func:`decode_config`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..common.config_decode import decode_dataclass, finite_number, positive_integer

decode_config = decode_dataclass


@dataclass(frozen=True)
class JevAgentConfig:
    """Hyperparameters for one Jev Choice call per environment step.

    ``explore_second_prob`` is the chance to take the second-highest legal Choice
    mass instead of the top. ``log_path`` is a directory (default ``results/jev``);
    each ``reset`` opens ``jev-<UTC start>.log`` there. An empty string disables
    the file. A path with a suffix is used as a single file. ``neighborhood_size``
    must be odd so the rover sits at the crop center.
    """

    model: str = "jev-latest"
    timeout_seconds: float = 8.0
    confidence_floor: float = 0.4
    fallback: str = "wait"
    max_calls: int = 0
    memory_length: int = 10
    neighborhood_size: int = 5
    explore_second_prob: float = 0.3
    log_path: str = "results/jev"

    def validate(self) -> None:
        """Reject empty model ids, non-positive budgets, and invalid floors."""
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model cannot be empty")
        if finite_number(self.timeout_seconds, "timeout_seconds") <= 0:
            raise ValueError("timeout_seconds must be positive")
        floor = finite_number(self.confidence_floor, "confidence_floor")
        if not 0 <= floor <= 1:
            raise ValueError("confidence_floor must be in [0, 1]")
        if self.fallback != "wait":
            raise ValueError("fallback must be wait")
        positive_integer(self.max_calls, "max_calls", allow_zero=True)
        positive_integer(self.memory_length, "memory_length")
        positive_integer(self.neighborhood_size, "neighborhood_size")
        if self.neighborhood_size % 2 == 0:
            raise ValueError("neighborhood_size must be odd")
        explore = finite_number(self.explore_second_prob, "explore_second_prob")
        if not 0 <= explore <= 1:
            raise ValueError("explore_second_prob must be in [0, 1]")
        if not isinstance(self.log_path, str):
            raise ValueError("log_path must be a string")


__all__ = ["JevAgentConfig", "decode_config"]
