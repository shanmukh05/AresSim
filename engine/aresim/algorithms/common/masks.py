"""Shared action-mask validation for baseline policies.

Centralizes the Phase 1 ``Discrete(10)`` mask shape and binary-value checks so each
baseline does not duplicate validation logic.

**Last updated:** September 1, 2026

**Contains:** :func:`require_mask`.

**Used by:** :mod:`aresim.algorithms.baselines.random`,
:mod:`aresim.algorithms.baselines.random_valid`, :mod:`aresim.algorithms.baselines.wait`,
:mod:`aresim.algorithms.baselines.scripted`.
"""

from __future__ import annotations

import numpy as np

from ...components.actions import ACTION_COUNT


def require_mask(action_mask: np.ndarray) -> np.ndarray:
    """Validate and return an authoritative ``(ACTION_COUNT,)`` binary mask.

    Raises ``ValueError`` when the array is missing, wrong-shaped, or contains
    values other than ``0`` and ``1``.
    """
    if not isinstance(action_mask, np.ndarray) or action_mask.shape != (ACTION_COUNT,):
        raise ValueError(f"action mask must have shape ({ACTION_COUNT},)")
    if not np.isin(action_mask, (0, 1)).all():
        raise ValueError("action mask must contain only zero and one")
    return action_mask


__all__ = ["require_mask"]
