"""Public construction helpers for the engine, agents, and RL environments.

Resolves registered component and agent names through :mod:`aresim.registry` and
wires them into :class:`~aresim.envs.environment.AresEnvironment` or thin framework
adapters. Requires the optional ``env`` extra (NumPy, Gymnasium, PettingZoo).

**Last updated:** September 1, 2026

**Contains:** :func:`make_engine`, :func:`make_agent`, :func:`make_env`,
:func:`make_gym_env`, :func:`make_parallel_env`.

**See also:** :mod:`aresim.registry` (registration),
:mod:`aresim.algorithms` (baseline policies).
"""

from __future__ import annotations

from typing import Any

from .algorithms import Agent
from .config import EngineConfig, EnvironmentConfig
from .core.engine import AresEngine
from .defaults import DEFAULT_ENGINE_CONFIG, DEFAULT_ENVIRONMENT_CONFIG
from .envs import AresEnvironment, AresGymEnv, AresParallelEnv, AresTimeLimit, Environment
from .registry import ComponentBuildContext, ComponentRegistry, create_default_registry


def make_engine(config: EngineConfig = DEFAULT_ENGINE_CONFIG) -> AresEngine:
    """Construct a deterministic :class:`~aresim.core.engine.AresEngine` from config."""
    return AresEngine(config)


def make_agent(
    name: str,
    config: EnvironmentConfig = DEFAULT_ENVIRONMENT_CONFIG,
    registry: ComponentRegistry | None = None,
) -> Agent[Any, Any]:
    """Resolve one registered baseline or checkpoint policy by semantic name."""
    config.validate()
    components = create_default_registry() if registry is None else registry
    return components.build_agent(name, ComponentBuildContext(config, config.engine))


def make_env(
    config: EnvironmentConfig = DEFAULT_ENVIRONMENT_CONFIG,
    registry: ComponentRegistry | None = None,
    max_episode_steps: int | None = None,
) -> Environment[object, object]:
    """Build a framework-neutral composed environment from registered components.

    When ``max_episode_steps`` is set, wraps the result in :class:`~aresim.envs.environment.AresTimeLimit`.
    """
    config.validate()
    components = create_default_registry() if registry is None else registry
    context = ComponentBuildContext(config, config.engine)
    observation = components.build_observation(config.observation, context)
    actions = components.build_action(config.action, context)
    reward = components.build_reward(config.reward, context)
    task = components.build_task(config.task, context)
    environment = AresEnvironment(
        config.engine,
        observation,
        actions,
        reward,
        task,
        scenario_id=config.scenario_id,
    )
    if max_episode_steps is None:
        return environment
    return AresTimeLimit(environment, max_episode_steps)


def make_gym_env(
    config: EnvironmentConfig = DEFAULT_ENVIRONMENT_CONFIG,
    registry: ComponentRegistry | None = None,
    max_episode_steps: int | None = None,
) -> AresGymEnv:
    """Construct the exactly-one-rover Gymnasium adapter over :func:`make_env`."""
    return AresGymEnv(make_env(config, registry=registry, max_episode_steps=max_episode_steps))


def make_parallel_env(
    config: EnvironmentConfig = DEFAULT_ENVIRONMENT_CONFIG,
    registry: ComponentRegistry | None = None,
    max_episode_steps: int | None = None,
) -> AresParallelEnv:
    """Construct the canonical one-rover PettingZoo Parallel adapter over :func:`make_env`."""
    return AresParallelEnv(make_env(config, registry=registry, max_episode_steps=max_episode_steps))
