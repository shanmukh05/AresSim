"""Public entry point for baseline policies and the shared ``Agent`` contract.

Re-exports every built-in deterministic-seeded baseline from
:mod:`aresim.algorithms.baselines`. Learned masked PPO lives under
under :mod:`aresim.algorithms.ppo`; rollout collection and experiment YAML remain in
:mod:`aresim.training`.

**Last updated:** September 1, 2026

**Contains:** ``Agent``, ``UniformRandomAgent``, ``RandomValidAgent``, ``WaitAgent``,
``ScriptedAgent``.

**Registry names:** ``random``, ``random_valid``, ``wait``, ``scripted`` (via
:func:`aresim.registry.create_default_registry`).

**Import note:** Does not import Ray or PyTorch. Safe for lightweight tests and
baseline-only rollouts.
"""

from .base import Agent
from .baselines import RandomValidAgent, ScriptedAgent, UniformRandomAgent, WaitAgent

__all__ = [
    "Agent",
    "RandomValidAgent",
    "ScriptedAgent",
    "UniformRandomAgent",
    "WaitAgent",
]
