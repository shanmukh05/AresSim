"""Deterministic simulation core: generation, rules, and engine coordination.

Owns all gameplay truth. Must stay free of FastAPI, React, NumPy, Gymnasium,
PettingZoo, Ray, and wall-clock randomness.

**Last updated:** September 1, 2026

**Contains:** :class:`~aresim.core.engine.AresEngine`, :func:`~aresim.core.engine.state_checksum`.

**Submodules:** :mod:`aresim.core.generation` (world build),
:mod:`aresim.core.rules` (legality, transitions, rewards).
"""

from .engine import AresEngine, state_checksum

__all__ = ["AresEngine", "state_checksum"]
