"""RL environment boundaries built over one composed :class:`~aresim.envs.environment.AresEnvironment`.

Re-exports framework-neutral, Gymnasium, and PettingZoo adapters. Gameplay rules
always remain in :mod:`aresim.core`.

**Last updated:** September 1, 2026

**Contains:** ``AresEnvironment``, ``AresGymEnv``, ``AresParallelEnv``,
``AresTimeLimit``, result dataclasses.
"""

from .environment import AresEnvironment, AresTimeLimit, Environment, EnvironmentReset, EnvironmentStep
from .gymnasium import AresGymEnv
from .pettingzoo import AresParallelEnv

__all__ = [
    "AresEnvironment",
    "AresGymEnv",
    "AresParallelEnv",
    "AresTimeLimit",
    "Environment",
    "EnvironmentReset",
    "EnvironmentStep",
]
