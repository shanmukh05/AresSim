"""Canonical one-rover PettingZoo Parallel adapter over :class:`~aresim.envs.environment.AresEnvironment`.

Dictionary API with stable agent IDs for future multi-rover growth. Delegates all
semantics to the shared composition.

**Last updated:** September 1, 2026

**Contains:** ``AresParallelEnv``.

**Metadata:** ``aresim_parallel_v0``.

**See also:** :mod:`aresim.envs.gymnasium` (single-agent Gym path).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from gymnasium import spaces
from pettingzoo import ParallelEnv

from .environment import Environment, policy_input


class AresParallelEnv(ParallelEnv):
    """Expose one live `rover_0` through PettingZoo's Parallel API contract."""

    metadata = {"name": "aresim_parallel_v0", "render_modes": [], "is_parallelizable": True}

    def __init__(self, environment: Environment[Any, Any]) -> None:
        if tuple(environment.possible_agents) != ("rover_0",):
            raise ValueError("AresParallelEnv currently requires exactly rover_0")
        self.environment = environment
        self.possible_agents = ["rover_0"]
        self.agents: list[str] = []
        self._observation_space = spaces.Dict({
            "observation": environment.observation_space,
            "action_mask": environment.action_mask_space,
        })

    @lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        """Return the stable policy-input space for a known rover ID."""
        if agent not in self.possible_agents:
            raise ValueError(f"unknown agent: {agent}")
        return self._observation_space

    @lru_cache(maxsize=None)
    def action_space(self, agent: str):
        """Return the stable action space for a known rover ID."""
        if agent not in self.possible_agents:
            raise ValueError(f"unknown agent: {agent}")
        return self.environment.action_space

    def reset(self, seed: int | None = None, options: dict[str, object] | None = None):
        """Reset the composition and wrap its one result by stable agent ID."""
        # PettingZoo's contract probe passes an arbitrary options mapping. This
        # adapter has no scenario options yet, so it accepts and ignores it.
        result = self.environment.reset(seed=seed)
        self.agents = self.possible_agents.copy()
        return (
            {result.agent_id: policy_input(result.observation, result.action_mask)},
            {result.agent_id: result.info},
        )

    def step(self, actions: dict[str, Any]):
        """Require one action per live agent and return standard Parallel dictionaries."""
        if not self.agents:
            raise RuntimeError("episode has ended; reset before stepping again")
        if set(actions) != set(self.agents):
            raise ValueError("actions must contain exactly every live agent ID")
        agent_id = self.agents[0]
        result = self.environment.step(actions[agent_id])
        observations = {agent_id: policy_input(result.observation, result.action_mask)}
        rewards = {agent_id: result.reward}
        terminations = {agent_id: result.terminated}
        truncations = {agent_id: result.truncated}
        infos = {agent_id: result.info}
        if result.terminated or result.truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def render(self) -> None:
        """Rendering is intentionally absent from the headless RL environment."""
        return None
