"""Passive Wait baseline policy for survival and decay measurement.

Always selects action ``0`` (Wait) after verifying the environment keeps Wait legal.
Useful for measuring passive resource dynamics without navigation decisions.

**Last updated:** September 5, 2026

**Contains:** ``WaitAgent``.

**Registry name:** ``wait`` → ``policy_id`` ``aresim.agent.wait.v1``.

**Dependencies:** NumPy only.
"""

from __future__ import annotations

import numpy as np

from ...components.actions import ACTION_SCHEMA
from ..common.masks import require_mask


class WaitAgent:
    """Choose the canonical Wait action to expose passive-survival behavior.

    Stateless aside from a reset flag; the seed argument is accepted for API
    compatibility but does not affect behavior.
    """

    policy_id = "aresim.agent.wait.v1"
    observation_schema = None
    action_schema = ACTION_SCHEMA

    def __init__(self) -> None:
        self._reset = False

    def reset(self, seed: int) -> None:
        """Mark the policy ready for a new episode."""
        self._reset = True

    def act(self, observation: object, action_mask: np.ndarray) -> int:
        """Return Wait (``0``) after asserting ``mask[0] == 1``."""
        mask = require_mask(action_mask)
        if not self._reset:
            raise RuntimeError("agent has not been reset")
        if mask[0] != 1:
            raise ValueError("Wait must remain legal")
        return 0


__all__ = ["WaitAgent"]
