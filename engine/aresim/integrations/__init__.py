"""Adapters that project engine state for consumers outside the core.

UI snapshots live in :mod:`aresim.integrations.ui`. Future RL framework adapters
belong here and must not reimplement rules from :mod:`aresim.core`.

**Last updated:** September 1, 2026

**Re-exports:** :func:`aresim.integrations.ui.snapshot_from_state`.
"""

from .ui import snapshot_from_state

__all__ = ["snapshot_from_state"]
