"""RL environment boundaries built over one composed :class:`~aresim.envs.environment.AresEnvironment`.

Re-exports framework-neutral and Gymnasium adapters. The PettingZoo adapter
(`AresParallelEnv`) loads on first use so MuZero and other Gym-only extras do
not need that dependency. Gameplay rules always remain in :mod:`aresim.core`.

**Last updated:** September 1, 2026

**Contains:** ``AresEnvironment``, ``AresGymEnv``, ``AresParallelEnv``,
``AresTimeLimit``, result dataclasses.
"""

from .environment import AresEnvironment, AresTimeLimit, Environment, EnvironmentReset, EnvironmentStep
from .gymnasium import AresGymEnv

__all__ = [
    "AresEnvironment",
    "AresGymEnv",
    "AresParallelEnv",
    "AresTimeLimit",
    "Environment",
    "EnvironmentReset",
    "EnvironmentStep",
]


def __getattr__(name: str):
    if name == "AresParallelEnv":
        from .pettingzoo import AresParallelEnv

        return AresParallelEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
