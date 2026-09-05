"""Composes the deterministic engine with observation, action, reward, and task components.

Framework-neutral RL boundary: orchestrates reset/step and learning-facing results
while all state mutation stays inside :class:`~aresim.core.engine.AresEngine`.

**Last updated:** September 1, 2026

**Contains:** ``AresEnvironment``, ``AresTimeLimit``, ``Environment`` protocol,
``EnvironmentReset``, ``EnvironmentStep``, :func:`policy_input`.

**Agent IDs:** ``rover_0`` (Phase 1 single-rover).

**See also:** :mod:`aresim.envs.gymnasium`, :mod:`aresim.envs.pettingzoo`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Generic, Protocol, TypeVar

from gymnasium import Space

import numpy as np

from ..components.base import ActionAdapter, ObservationBuilder, RewardFunction, TaskEvaluator
from ..components.rewards import RewardBreakdown
from ..config import EngineConfig
from ..core.engine import AresEngine, state_checksum
from ..types import Actor, EngineTransition, WorldState


ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


@dataclass(frozen=True)
class EnvironmentReset(Generic[ObservationT, ActionT]):
    """Initial actor input and metadata returned by a composed reset."""

    agent_id: str
    observation: ObservationT
    action_mask: np.ndarray
    info: dict[str, object]


@dataclass(frozen=True)
class EnvironmentStep(Generic[ObservationT, ActionT]):
    """One learning-facing result plus the authoritative engine transition."""

    agent_id: str
    observation: ObservationT
    action_mask: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]
    transition: EngineTransition
    reward_breakdown: RewardBreakdown


class Environment(Protocol[ObservationT, ActionT]):
    """Minimal framework-neutral environment surface consumed by adapters."""

    possible_agents: tuple[str, ...]
    observation_space: Space[ObservationT]
    action_space: Space[ActionT]
    action_mask_space: Space[np.ndarray]

    @property
    def world_state(self) -> WorldState:
        """Return a copy of the authoritative state for audit artifact recording."""
        ...

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> EnvironmentReset[ObservationT, ActionT]:
        """Start one episode."""
        ...

    def step(self, action_id: ActionT) -> EnvironmentStep[ObservationT, ActionT]:
        """Advance one environment transition."""
        ...


def policy_input(observation: ObservationT, action_mask: np.ndarray) -> dict[str, ObservationT | np.ndarray]:
    """Wrap perception and legality once in the canonical policy dictionary."""
    return {"observation": observation, "action_mask": action_mask}


class AresEnvironment(Generic[ObservationT, ActionT]):
    """Run one rover through the registered framework-neutral RL composition."""

    possible_agents = ("rover_0",)

    def __init__(
        self,
        engine_config: EngineConfig,
        observation: ObservationBuilder[ObservationT],
        actions: ActionAdapter[ActionT],
        reward: RewardFunction,
        task: TaskEvaluator,
        *,
        scenario_id: str = "phase1_default_v1",
    ) -> None:
        engine_config.validate()
        if not scenario_id:
            raise ValueError("scenario identifier cannot be empty")
        self.engine_config = engine_config
        self.scenario_id = scenario_id
        self.observation_builder = observation
        self.action_adapter = actions
        self.reward_profile = reward
        self.task = task
        self.engine = AresEngine(engine_config)
        self._reset = False
        self._done = False

    @property
    def observation_space(self):
        """Gymnasium space for the perception portion of one rover input."""
        return self.observation_builder.space

    @property
    def action_space(self):
        """Policy action space declared by the configured action adapter."""
        return self.action_adapter.space

    @property
    def action_mask_space(self):
        """Legal-mask space declared by the configured action adapter."""
        return self.action_adapter.mask_space

    @property
    def world_state(self) -> WorldState:
        """Return a defensive state copy for privileged audit and replay recording."""
        return self.engine.state

    def _base_info(self, seed: int, checksum: str) -> dict[str, object]:
        return {
            "agent_id": self.possible_agents[0],
            "seed": seed,
            "scenario_id": self.scenario_id,
            "task_id": self.task.task_id,
            "observation_schema": self.observation_builder.schema,
            "action_schema": self.action_adapter.schema,
            "reward_profile": self.reward_profile.profile,
            "state_checksum": checksum,
        }

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> EnvironmentReset[ObservationT, ActionT]:
        """Reset deterministically and return one local observation plus legal mask."""
        if options:
            raise ValueError("open exploration does not accept reset options")
        resolved_seed = self.engine_config.world.default_seed if seed is None else seed
        state = self.engine.reset(resolved_seed)
        agent_ids = tuple(rover.id for rover in state.rovers)
        if agent_ids != self.possible_agents:
            raise ValueError("section-3 environment requires exactly rover_0")
        self.task.reset(state)
        observation = self.observation_builder.reset(state, self.engine_config)
        action_mask = self.action_adapter.mask(state, self.engine_config)
        self._reset = True
        self._done = False
        return EnvironmentReset(
            agent_id=self.possible_agents[0],
            observation=observation,
            action_mask=action_mask,
            info=self._base_info(resolved_seed, state_checksum(state)),
        )

    def step(self, action_id: ActionT) -> EnvironmentStep[ObservationT, ActionT]:
        """Decode and apply one action, then project its RL observation and reward."""
        if not self._reset:
            raise RuntimeError("environment has not been reset")
        if self._done:
            raise RuntimeError("episode has ended; reset before stepping again")
        before = self.engine.state
        command = self.action_adapter.decode(before, action_id)
        transition = self.engine.step(command, Actor.AGENT)
        outcome = self.task.evaluate(before, transition)
        breakdown = self.reward_profile.calculate(before, transition, outcome)
        observation = self.observation_builder.build(transition.state, self.engine_config)
        action_mask = self.action_adapter.mask(transition.state, self.engine_config)
        info = self._base_info(transition.state.seed, transition.after_checksum)
        info.update({
            "before_checksum": transition.before_checksum,
            "effective_action": transition.effective_action.value,
            "events": list(transition.events),
            "terminal_reason": outcome.terminal_reason,
            "reward_breakdown": breakdown.as_dict(),
            "engine_reward": transition.reward,
            "engine_reward_terms": dict(transition.reward_terms),
        })
        self._done = outcome.terminated
        return EnvironmentStep(
            agent_id=self.possible_agents[0],
            observation=observation,
            action_mask=action_mask,
            reward=breakdown.total,
            terminated=outcome.terminated,
            truncated=False,
            info=info,
            transition=transition,
            reward_breakdown=breakdown,
        )


class AresTimeLimit(Generic[ObservationT, ActionT]):
    """Apply an external episode cutoff without changing simulator termination."""

    def __init__(self, environment: Environment[ObservationT, ActionT], max_episode_steps: int) -> None:
        if not isinstance(max_episode_steps, int) or isinstance(max_episode_steps, bool) or max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be a positive integer")
        self.environment = environment
        self.max_episode_steps = max_episode_steps
        self.possible_agents = tuple(environment.possible_agents)
        self._episode_step = 0
        self._reset = False
        self._done = False

    @property
    def observation_space(self) -> Space[ObservationT]:
        """Return the wrapped observation space unchanged."""
        return self.environment.observation_space

    @property
    def action_space(self) -> Space[ActionT]:
        """Return the wrapped action space unchanged."""
        return self.environment.action_space

    @property
    def action_mask_space(self) -> Space[np.ndarray]:
        """Return the wrapped action-mask space unchanged."""
        return self.environment.action_mask_space

    @property
    def world_state(self) -> WorldState:
        """Expose the wrapped defensive state copy without changing policy inputs."""
        return self.environment.world_state

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> EnvironmentReset[ObservationT, ActionT]:
        """Reset the wrapped environment and external step counter."""
        result = self.environment.reset(seed=seed, options=options)
        self._episode_step = 0
        self._reset = True
        self._done = False
        info = {**result.info, "episode_step": 0, "max_episode_steps": self.max_episode_steps}
        return replace(result, info=info)

    def step(self, action_id: ActionT) -> EnvironmentStep[ObservationT, ActionT]:
        """Return one real transition and truncate it exactly at the cutoff."""
        if not self._reset:
            raise RuntimeError("environment has not been reset")
        if self._done:
            raise RuntimeError("episode has ended; reset before stepping again")
        result = self.environment.step(action_id)
        self._episode_step += 1
        reached_limit = self._episode_step >= self.max_episode_steps
        truncated = False if result.terminated else result.truncated or reached_limit
        info = {
            **result.info,
            "episode_step": self._episode_step,
            "max_episode_steps": self.max_episode_steps,
        }
        if reached_limit and not result.terminated and not result.truncated:
            info["truncation_reason"] = "max_episode_steps"
        self._done = result.terminated or truncated
        return replace(result, truncated=truncated, info=info)
