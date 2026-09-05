"""Public Python import surface for the deterministic gameplay engine.

Re-exports the minimum API for in-process simulator use: ``AresEngine``,
``EngineConfig``, command types, and defaults. The React UI uses
:mod:`aresim.api` over HTTP instead of importing this package directly.

**Last updated:** September 1, 2026

**Contains:** ``AresEngine``, ``EngineConfig``, ``DEFAULT_ENGINE_CONFIG``,
``ActionCommand``, ``ActionType``, ``Actor``.

**See also:** :mod:`aresim.factory` (RL env construction),
:mod:`aresim.core` (deterministic simulation), :mod:`aresim.api` (REST).
"""

from .config import EngineConfig
from .core.engine import AresEngine
from .defaults import DEFAULT_ENGINE_CONFIG
from .types import ActionCommand, ActionType, Actor

__all__ = [
    "ActionCommand",
    "ActionType",
    "Actor",
    "AresEngine",
    "DEFAULT_ENGINE_CONFIG",
    "EngineConfig",
]
