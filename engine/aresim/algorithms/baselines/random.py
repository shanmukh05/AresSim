"""Uniform-random baseline policy for mask and invalid-action diagnosis.

Samples all ten macro actions with equal probability regardless of the current
action mask. Useful for stress-testing invalid-action handling and measuring how
much legality alone improves returns.

**Last updated:** September 5, 2026

**Contains:** ``UniformRandomAgent``.

**Registry name:** ``random`` → ``policy_id`` ``aresim.agent.random.v1``.

**Dependencies:** NumPy only.
"""

from __future__ import annotations

import numpy as np

from ...components.actions import ACTION_COUNT, ACTION_SCHEMA
from ..common.masks import require_mask


class UniformRandomAgent:
    """Sample every rover action uniformly, including currently masked actions.

    Validates mask shape but intentionally ignores legality bits. Requires
    ``reset(seed)`` before the first ``act`` call each episode.
    """

    policy_id = "aresim.agent.random.v1"
    observation_schema = None
    action_schema = ACTION_SCHEMA

    def __init__(self) -> None:
        self._generator: np.random.Generator | None = None

    def reset(self, seed: int) -> None:
        """Start the policy RNG from an explicit episode seed for reproducibility."""
        self._generator = np.random.default_rng(seed)

    def act(self, observation: object, action_mask: np.ndarray) -> int:
        """Return a uniform action in ``[0, ACTION_COUNT)`` without consulting legality."""
        require_mask(action_mask)
        if self._generator is None:
            raise RuntimeError("agent has not been reset")
        return int(self._generator.integers(ACTION_COUNT))


__all__ = ["UniformRandomAgent"]
