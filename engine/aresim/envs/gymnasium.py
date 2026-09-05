"""Thin exactly-one-rover Gymnasium adapter over :class:`~aresim.envs.environment.AresEnvironment`.

Scalar shape conversion only; observations, masks, rewards, and checksums come
from the shared composed environment.

**Last updated:** September 1, 2026

**Contains:** ``AresGymEnv``.

**Observation space:** Dict with ``observation`` and ``action_mask`` keys.

**Used by:** RLlib training (:mod:`aresim.algorithms.ppo.train`).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces

from .environment import Environment, policy_input


class AresGymEnv(gym.Env):
    """Expose the `rover_0` composition through Gymnasium's scalar API."""

    metadata = {"render_modes": []}

    def __init__(self, environment: Environment[Any, Any]) -> None:
        super().__init__()
        if tuple(environment.possible_agents) != ("rover_0",):
            raise ValueError("AresGymEnv requires exactly one rover_0 agent")
        self.environment = environment
        self.action_space = environment.action_space
        self.observation_space = spaces.Dict({
            "observation": environment.observation_space,
            "action_mask": environment.action_mask_space,
        })

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset the shared composition and remove only the agent-ID dictionary."""
        super().reset(seed=seed)
        result = self.environment.reset(seed=seed, options=options)
        return policy_input(result.observation, result.action_mask), result.info

    def step(self, action: Any):
        """Return the composed result as Gymnasium scalar reward and end flags."""
        result = self.environment.step(action)
        return (
            policy_input(result.observation, result.action_mask),
            result.reward,
            result.terminated,
            result.truncated,
            result.info,
        )

    def render(self) -> None:
        """Rendering is intentionally absent from the headless RL environment."""
        return None
