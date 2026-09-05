"""Explicit typed registry for RL environment components and baseline agents.

Registers observation, action, reward, task, and agent factories by semantic name.
Also re-exports the training registry (:func:`create_training_registry`) lazily.
No automatic discovery or third-party entry points.

**Last updated:** September 1, 2026

**Contains:** ``ComponentRegistry``, ``ComponentBuildContext``,
:func:`create_default_registry`, lazy training registry re-exports.

**Built-in component names:** ``local``, ``discrete``, ``shaped_train``,
``sparse_eval``, ``open_exploration``; agents ``random``, ``random_valid``,
``wait``, ``scripted``.

**See also:** :mod:`aresim.algorithms.registry` (learned-policy registration),
:mod:`aresim.factory` (construction helpers).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from gymnasium import Space

from .algorithms.base import Agent
from .components.base import ActionAdapter, ObservationBuilder, RewardFunction, TaskEvaluator
from .config import EngineConfig, EnvironmentConfig


@dataclass(frozen=True)
class ComponentBuildContext:
    """Resolved environment and engine configs passed uniformly to all factories."""

    environment_config: EnvironmentConfig
    engine_config: EngineConfig


ObservationFactory = Callable[[ComponentBuildContext], ObservationBuilder[Any]]
ActionFactory = Callable[[ComponentBuildContext], ActionAdapter[Any]]
RewardFactory = Callable[[ComponentBuildContext], RewardFunction]
TaskFactory = Callable[[ComponentBuildContext], TaskEvaluator]
AgentFactory = Callable[[ComponentBuildContext], Agent[Any, Any]]

FactoryT = TypeVar("FactoryT")


def _require_identifier(component: object, attribute: str, category: str) -> None:
    value = getattr(component, attribute, None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{category} factory returned an empty {attribute}")


def _require_space(component: object, attribute: str, category: str) -> None:
    if not isinstance(getattr(component, attribute, None), Space):
        raise TypeError(f"{category} factory returned an invalid {attribute}")


class ComponentRegistry:
    """Store and resolve explicit factories for observations, actions, rewards, tasks, and agents.

    Duplicate names and invalid built instances raise at registration or build time.
    """

    def __init__(self) -> None:
        self._observations: dict[str, ObservationFactory] = {}
        self._actions: dict[str, ActionFactory] = {}
        self._rewards: dict[str, RewardFactory] = {}
        self._tasks: dict[str, TaskFactory] = {}
        self._agents: dict[str, AgentFactory] = {}

    @staticmethod
    def _register(table: dict[str, FactoryT], category: str, name: str, factory: FactoryT) -> None:
        if not name:
            raise ValueError(f"{category} component name cannot be empty")
        if name in table:
            raise ValueError(f"duplicate {category} component: {name}")
        if not callable(factory):
            raise TypeError(f"{category} component factory must be callable")
        table[name] = factory

    @staticmethod
    def _resolve(table: dict[str, FactoryT], category: str, name: str) -> FactoryT:
        try:
            return table[name]
        except KeyError as error:
            raise ValueError(f"unknown {category} component: {name}") from error

    def register_observation(self, name: str, factory: ObservationFactory) -> None:
        """Register one observation factory under a unique semantic name."""
        self._register(self._observations, "observation", name, factory)

    def register_action(self, name: str, factory: ActionFactory) -> None:
        """Register one action factory under a unique semantic name."""
        self._register(self._actions, "action", name, factory)

    def register_reward(self, name: str, factory: RewardFactory) -> None:
        """Register one reward factory under a unique semantic name."""
        self._register(self._rewards, "reward", name, factory)

    def register_task(self, name: str, factory: TaskFactory) -> None:
        """Register one task factory under a unique semantic name."""
        self._register(self._tasks, "task", name, factory)

    def register_agent(self, name: str, factory: AgentFactory) -> None:
        """Register one agent factory under a unique semantic name."""
        self._register(self._agents, "agent", name, factory)

    def observation(self, name: str) -> ObservationFactory:
        """Resolve a registered observation factory."""
        return self._resolve(self._observations, "observation", name)

    def action(self, name: str) -> ActionFactory:
        """Resolve a registered action factory."""
        return self._resolve(self._actions, "action", name)

    def reward(self, name: str) -> RewardFactory:
        """Resolve a registered reward factory."""
        return self._resolve(self._rewards, "reward", name)

    def task(self, name: str) -> TaskFactory:
        """Resolve a registered task factory."""
        return self._resolve(self._tasks, "task", name)

    def agent(self, name: str) -> AgentFactory:
        """Resolve a registered agent factory."""
        return self._resolve(self._agents, "agent", name)

    def build_observation(self, name: str, context: ComponentBuildContext) -> ObservationBuilder[Any]:
        """Build and validate one observation implementation."""
        component = self.observation(name)(context)
        if not isinstance(component, ObservationBuilder):
            raise TypeError("observation factory result does not implement ObservationBuilder")
        _require_identifier(component, "schema", "observation")
        _require_space(component, "space", "observation")
        return component

    def build_action(self, name: str, context: ComponentBuildContext) -> ActionAdapter[Any]:
        """Build and validate one action implementation."""
        component = self.action(name)(context)
        if not isinstance(component, ActionAdapter):
            raise TypeError("action factory result does not implement ActionAdapter")
        _require_identifier(component, "schema", "action")
        _require_space(component, "space", "action")
        _require_space(component, "mask_space", "action")
        return component

    def build_reward(self, name: str, context: ComponentBuildContext) -> RewardFunction:
        """Build and validate one reward implementation."""
        component = self.reward(name)(context)
        if not isinstance(component, RewardFunction):
            raise TypeError("reward factory result does not implement RewardFunction")
        _require_identifier(component, "profile", "reward")
        return component

    def build_task(self, name: str, context: ComponentBuildContext) -> TaskEvaluator:
        """Build and validate one task implementation."""
        component = self.task(name)(context)
        if not isinstance(component, TaskEvaluator):
            raise TypeError("task factory result does not implement TaskEvaluator")
        _require_identifier(component, "task_id", "task")
        return component

    def build_agent(self, name: str, context: ComponentBuildContext) -> Agent[Any, Any]:
        """Build and validate one policy implementation."""
        component = self.agent(name)(context)
        if not isinstance(component, Agent):
            raise TypeError("agent factory result does not implement Agent")
        _require_identifier(component, "policy_id", "agent")
        _require_identifier(component, "action_schema", "agent")
        observation_schema = component.observation_schema
        if observation_schema is not None and (not isinstance(observation_schema, str) or not observation_schema.strip()):
            raise ValueError("agent factory returned an empty observation_schema")
        return component


def create_default_registry() -> ComponentRegistry:
    """Return a fresh registry populated with all built-in RL components."""
    from .components import DiscreteActions, LocalObservation, OpenExplorationTask, ShapedTrainReward, SparseEvalReward
    from .algorithms import RandomValidAgent, ScriptedAgent, UniformRandomAgent, WaitAgent

    registry = ComponentRegistry()
    registry.register_observation(
        "local",
        lambda context: LocalObservation(context.environment_config.observation_config),
    )
    registry.register_action("discrete", lambda context: DiscreteActions())
    registry.register_reward(
        "shaped_train",
        lambda context: ShapedTrainReward(context.environment_config.reward_config),
    )
    registry.register_reward(
        "sparse_eval",
        lambda context: SparseEvalReward(context.environment_config.reward_config),
    )
    registry.register_task("open_exploration", lambda context: OpenExplorationTask())
    registry.register_agent("random", lambda context: UniformRandomAgent())
    registry.register_agent("random_valid", lambda context: RandomValidAgent())
    registry.register_agent("wait", lambda context: WaitAgent())
    registry.register_agent("scripted", lambda context: ScriptedAgent())
    return registry


def create_training_registry():
    """Return a fresh :class:`~aresim.algorithms.registry.TrainingRegistry` with built-in PPO entries."""
    from .algorithms.registry import create_training_registry as _create

    return _create()


def __getattr__(name: str):
    if name in {
        "AlgorithmFactory",
        "CheckpointLoader",
        "ModelFactory",
        "TrainingContext",
        "TrainingRegistry",
    }:
        from .algorithms.registry import (
            AlgorithmFactory,
            CheckpointLoader,
            ModelFactory,
            TrainingContext,
            TrainingRegistry,
        )

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ActionFactory",
    "AgentFactory",
    "AlgorithmFactory",
    "CheckpointLoader",
    "ComponentBuildContext",
    "ComponentRegistry",
    "ModelFactory",
    "ObservationFactory",
    "RewardFactory",
    "TaskFactory",
    "TrainingContext",
    "TrainingRegistry",
    "create_default_registry",
    "create_training_registry",
]
