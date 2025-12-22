"""
Mars Survival RL Environment

A Gymnasium environment for training autonomous rover agents in extreme
Martian conditions.
"""

from gymnasium.envs.registration import register

__version__ = "0.1.0"


def register_envs():
    """Register the Mars Survival environment with Gymnasium."""
    register(
        id="gym_mars/MarsSurvival-v0",
        entry_point="gym_mars.envs:MarsSurvivalEnv",
        max_episode_steps=1000,
    )


# Auto-register on import
register_envs()

# Convenience imports
from gym_mars.envs import MarsSurvivalEnv

__all__ = ["MarsSurvivalEnv", "register_envs", "__version__"]
