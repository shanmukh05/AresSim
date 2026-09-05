"""Mask-respecting uniform-random baseline policy.

Samples uniformly among legal action indices only. This is the standard untrained
comparison for masked PPO because it respects legality but has no observation-dependent
strategy.

**Last updated:** September 5, 2026

**Contains:** ``RandomValidAgent``.

**Registry name:** ``random_valid`` → ``policy_id`` ``aresim.agent.random_valid.v1``.

**Dependencies:** NumPy only.
"""

from __future__ import annotations

import numpy as np

from ...components.actions import ACTION_SCHEMA
from ..common.masks import require_mask


class RandomValidAgent:
    """Sample uniformly from indices whose mask entry is one.

    Raises ``ValueError`` when no legal action exists. Observation contents are
    ignored (``observation_schema`` is ``None``).
    """

    policy_id = "aresim.agent.random_valid.v1"
    observation_schema = None
    action_schema = ACTION_SCHEMA

    def __init__(self) -> None:
        self._generator: np.random.Generator | None = None

    def reset(self, seed: int) -> None:
        """Start the policy RNG from an explicit episode seed for reproducibility."""
        self._generator = np.random.default_rng(seed)

    def act(self, observation: object, action_mask: np.ndarray) -> int:
        """Return one uniformly chosen legal action index."""
        mask = require_mask(action_mask)
        if self._generator is None:
            raise RuntimeError("agent has not been reset")
        legal_actions = np.flatnonzero(mask)
        if legal_actions.size == 0:
            raise ValueError("action mask contains no legal actions")
        return int(self._generator.choice(legal_actions))


__all__ = ["RandomValidAgent"]
