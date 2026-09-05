"""Built-in baseline policies for rollouts, evaluation, and UI algorithm mode.

Each module implements one registered :class:`~aresim.algorithms.base.Agent`.
Registry names: ``random``, ``random_valid``, ``wait``, ``scripted``.

**Last updated:** September 5, 2026
"""

from .random import UniformRandomAgent
from .random_valid import RandomValidAgent
from .scripted import ScriptedAgent
from .wait import WaitAgent

__all__ = [
    "RandomValidAgent",
    "ScriptedAgent",
    "UniformRandomAgent",
    "WaitAgent",
]
