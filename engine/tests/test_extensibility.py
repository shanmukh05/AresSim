"""Reusable contract checks and end-to-end examples for public RL extensions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np
import pytest
from gymnasium import spaces

from aresim.components.actions import DiscreteActions
from aresim.components.base import ActionAdapter, ObservationBuilder, RewardFunction, TaskEvaluator
from aresim.components.observations import LocalObservation
from aresim.components.rewards import RewardBreakdown, RewardTerm, ShapedTrainReward
from aresim.components.tasks import OpenExplorationTask, TaskOutcome
from aresim.config import EngineConfig, EnvironmentConfig
from aresim.core.engine import AresEngine
from aresim.core.rules import validate_action
from aresim.defaults import DEFAULT_ENGINE_CONFIG, DEFAULT_ENVIRONMENT_CONFIG
from aresim.factory import make_env, make_gym_env, make_parallel_env
from aresim.registry import ComponentBuildContext, ComponentRegistry, create_default_registry
from aresim.types import ActionCommand, ActionType, EngineTransition, Position, WorldState


@dataclass(frozen=True)
class CustomObservationConfig:
    """Example extension-owned config with no dependency on built-in config shapes."""

    step_scale: float
    reset_marker: float


class CustomObservation:
    """Small stateful observation used to exercise the public reset contract."""

    schema = "example.obs.step.v1"

    def __init__(self, config: CustomObservationConfig) -> None:
        self.config = config
        self.space = spaces.Box(0, 1, shape=(2,), dtype=np.float32)
        self.reset_calls = 0
        self.marker = 0.0

    def reset(self, state: WorldState, engine_config: EngineConfig) -> np.ndarray:
        self.reset_calls += 1
        self.marker = self.config.reset_marker
        return self.build(state, engine_config)

    def build(self, state: WorldState, engine_config: EngineConfig) -> np.ndarray:
        return np.array(
            [min(1, state.step / self.config.step_scale), self.marker],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class CustomActionConfig:
    """Example action config captured by a registry closure."""

    scan_action: int


class CustomActions:
    """Tuple-valued policy action proving masks do not depend on ``Discrete.n``."""

    schema = "example.action.wait_scan.v1"

    def __init__(self, config: CustomActionConfig) -> None:
        self.config = config
        self.space = spaces.Tuple((spaces.Discrete(2),))
        self.mask_space = spaces.Box(0, 1, shape=(2,), dtype=np.int8)

    def decode(self, state: WorldState, action: tuple[int]) -> ActionCommand:
        if not self.space.contains(action):
            raise ValueError("action must belong to the custom Tuple space")
        if int(action[0]) != self.config.scan_action:
            return ActionCommand(ActionType.WAIT)
        rover = state.rovers[0]
        return ActionCommand(ActionType.SCAN, Position(rover.x, rover.y))

    def mask(self, state: WorldState, engine_config: EngineConfig) -> np.ndarray:
        commands = (
            ActionCommand(ActionType.WAIT),
            self.decode(state, (self.config.scan_action,)),
        )
        return np.array(
            [validate_action(state, command, engine_config).valid for command in commands],
            dtype=np.int8,
        )


@dataclass(frozen=True)
class CustomRewardConfig:
    """Example reward config unrelated to ``RewardProfileConfig``."""

    value: float


class CustomReward:
    """Return one fixed auditable term for every transition."""

    profile = "example.reward.fixed.v1"

    def __init__(self, config: CustomRewardConfig) -> None:
        self.config = config

    def calculate(
        self,
        before: WorldState,
        transition: EngineTransition,
        outcome: TaskOutcome,
    ) -> RewardBreakdown:
        term = RewardTerm(raw=1, weight=self.config.value, value=self.config.value)
        return RewardBreakdown(
            schema_version="example.reward.breakdown.v1",
            profile=self.profile,
            terms={"fixed": term},
            total_unclipped=self.config.value,
            total=self.config.value,
        )


@dataclass(frozen=True)
class CustomTaskConfig:
    """Example task-owned termination setting."""

    success_after: int


class CustomTask:
    """Track task progress independently and clear it at every reset."""

    task_id = "example.task.short.v1"

    def __init__(self, config: CustomTaskConfig) -> None:
        self.config = config
        self.reset_calls = 0
        self.transitions = 0

    def reset(self, state: WorldState) -> None:
        self.reset_calls += 1
        self.transitions = 0

    def evaluate(self, before: WorldState, transition: EngineTransition) -> TaskOutcome:
        self.transitions += 1
        success = self.transitions >= self.config.success_after
        return TaskOutcome(success, success, "custom_success" if success else None)


def _custom_composition() -> tuple[ComponentRegistry, EnvironmentConfig]:
    observation_config = CustomObservationConfig(step_scale=10, reset_marker=0.25)
    action_config = CustomActionConfig(scan_action=1)
    reward_config = CustomRewardConfig(value=0.375)
    task_config = CustomTaskConfig(success_after=2)
    registry = create_default_registry()
    registry.register_observation("example_step", lambda context: CustomObservation(observation_config))
    registry.register_action("example_wait_scan", lambda context: CustomActions(action_config))
    registry.register_reward("example_fixed", lambda context: CustomReward(reward_config))
    registry.register_task("example_short", lambda context: CustomTask(task_config))
    config = replace(
        DEFAULT_ENVIRONMENT_CONFIG,
        scenario_id="example_scenario_v1",
        observation="example_step",
        action="example_wait_scan",
        reward="example_fixed",
        task="example_short",
    )
    return registry, config


def _assert_value_equal(first: object, second: object) -> None:
    if isinstance(first, dict) and isinstance(second, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_value_equal(first[key], second[key])
        return
    np.testing.assert_array_equal(first, second)


def _assert_observation_contract(builder: ObservationBuilder[object]) -> None:
    state = AresEngine().reset(1447)
    initial = builder.reset(state, DEFAULT_ENGINE_CONFIG)
    repeated = builder.build(deepcopy(state), DEFAULT_ENGINE_CONFIG)
    assert builder.schema.strip()
    assert isinstance(builder.space, spaces.Space)
    assert builder.space.contains(initial)
    _assert_value_equal(initial, repeated)


def _assert_action_contract(adapter: ActionAdapter[object], actions: tuple[object, ...]) -> None:
    state = AresEngine().reset(1447)
    mask = adapter.mask(state, DEFAULT_ENGINE_CONFIG)
    assert adapter.schema.strip()
    assert isinstance(adapter.space, spaces.Space)
    assert isinstance(adapter.mask_space, spaces.Space)
    assert adapter.mask_space.contains(mask)
    assert mask.dtype == np.int8
    for index, action in enumerate(actions):
        assert adapter.space.contains(action)
        command = adapter.decode(state, action)
        assert bool(mask[index]) == validate_action(state, command, DEFAULT_ENGINE_CONFIG).valid


def test_built_in_and_custom_components_satisfy_public_contracts() -> None:
    _assert_observation_contract(LocalObservation(DEFAULT_ENVIRONMENT_CONFIG.observation_config))
    _assert_observation_contract(CustomObservation(CustomObservationConfig(10, 0.25)))
    _assert_action_contract(DiscreteActions(), tuple(range(10)))
    _assert_action_contract(
        CustomActions(CustomActionConfig(1)),
        ((0,), (1,)),
    )
    assert isinstance(ShapedTrainReward(DEFAULT_ENVIRONMENT_CONFIG.reward_config), RewardFunction)
    assert isinstance(OpenExplorationTask(), TaskEvaluator)


def test_stateful_observation_and_task_reset_hooks_run_each_episode() -> None:
    registry, config = _custom_composition()
    environment = make_env(config, registry=registry)
    first = environment.reset(seed=1447)
    environment.step((0,))
    second = environment.reset(seed=1447)

    observation = environment.observation_builder
    task = environment.task
    assert isinstance(observation, CustomObservation)
    assert isinstance(task, CustomTask)
    assert observation.reset_calls == 2
    assert task.reset_calls == 2
    assert task.transitions == 0
    np.testing.assert_array_equal(first.observation, second.observation)


def test_custom_components_work_through_all_factories_with_transition_parity() -> None:
    registry, config = _custom_composition()
    direct = make_env(config, registry=registry)
    gym = make_gym_env(config, registry=registry)
    parallel = make_parallel_env(config, registry=registry)

    direct_reset = direct.reset(seed=2468)
    gym_reset, gym_info = gym.reset(seed=2468)
    parallel_reset, parallel_info = parallel.reset(seed=2468)
    assert not hasattr(gym.action_space, "n")
    assert gym.observation_space.contains(gym_reset)
    assert parallel.observation_space("rover_0").contains(parallel_reset["rover_0"])
    np.testing.assert_array_equal(direct_reset.observation, gym_reset["observation"])
    np.testing.assert_array_equal(gym_reset["observation"], parallel_reset["rover_0"]["observation"])
    np.testing.assert_array_equal(direct_reset.action_mask, gym_reset["action_mask"])
    assert direct_reset.info["scenario_id"] == "example_scenario_v1"
    assert direct_reset.info["state_checksum"] == gym_info["state_checksum"]
    assert gym_info["state_checksum"] == parallel_info["rover_0"]["state_checksum"]

    for value in (0, 1):
        action = (value,)
        direct_step = direct.step(action)
        gym_step = gym.step(action)
        parallel_step = parallel.step({"rover_0": action})
        np.testing.assert_array_equal(direct_step.observation, gym_step[0]["observation"])
        np.testing.assert_array_equal(gym_step[0]["observation"], parallel_step[0]["rover_0"]["observation"])
        np.testing.assert_array_equal(direct_step.action_mask, gym_step[0]["action_mask"])
        assert direct_step.reward == gym_step[1] == parallel_step[1]["rover_0"] == 0.375
        assert direct_step.terminated == gym_step[2] == parallel_step[2]["rover_0"]
        assert direct_step.truncated == gym_step[3] == parallel_step[3]["rover_0"]
        assert direct_step.info["reward_breakdown"] == gym_step[4]["reward_breakdown"]
        assert gym_step[4]["reward_breakdown"] == parallel_step[4]["rover_0"]["reward_breakdown"]
        assert direct_step.info["events"] == gym_step[4]["events"]
        assert gym_step[4]["events"] == parallel_step[4]["rover_0"]["events"]
        assert direct_step.info["effective_action"] == gym_step[4]["effective_action"]
        assert gym_step[4]["effective_action"] == parallel_step[4]["rover_0"]["effective_action"]
        assert direct_step.info["terminal_reason"] == gym_step[4]["terminal_reason"]
        assert gym_step[4]["terminal_reason"] == parallel_step[4]["rover_0"]["terminal_reason"]
        assert direct_step.info["state_checksum"] == gym_step[4]["state_checksum"]
        assert gym_step[4]["state_checksum"] == parallel_step[4]["rover_0"]["state_checksum"]
    assert direct_step.terminated is True
    assert direct_step.info["terminal_reason"] == "custom_success"


class _BadObservation:
    schema = "bad.obs.v1"
    space = object()

    def reset(self, state: WorldState, engine_config: EngineConfig) -> int:
        return 0

    def build(self, state: WorldState, engine_config: EngineConfig) -> int:
        return 0


class _EmptyReward:
    profile = ""

    def calculate(
        self,
        before: WorldState,
        transition: EngineTransition,
        outcome: TaskOutcome,
    ) -> RewardBreakdown:
        raise AssertionError("validation should reject this component before use")


class _BadAction:
    schema = "bad.action.v1"
    space = spaces.Discrete(1)
    mask_space = object()

    def decode(self, state: WorldState, action: int) -> ActionCommand:
        return ActionCommand(ActionType.WAIT)

    def mask(self, state: WorldState, engine_config: EngineConfig) -> np.ndarray:
        return np.ones(1, dtype=np.int8)


@pytest.mark.parametrize(
    ("category", "name", "component", "message"),
    [
        ("observation", "missing_methods", object(), "ObservationBuilder"),
        ("observation", "bad_space", _BadObservation(), "invalid space"),
        ("action", "bad_mask_space", _BadAction(), "invalid mask_space"),
        ("reward", "empty_profile", _EmptyReward(), "empty profile"),
    ],
)
def test_invalid_factory_results_fail_during_composition(category, name, component, message) -> None:
    registry = create_default_registry()
    register = getattr(registry, f"register_{category}")
    register(name, lambda context: component)
    config = replace(DEFAULT_ENVIRONMENT_CONFIG, **{category: name})
    with pytest.raises((TypeError, ValueError), match=message):
        make_env(config, registry=registry)


def test_context_is_uniform_and_registrations_remain_explicit() -> None:
    seen: list[ComponentBuildContext] = []
    registry = create_default_registry()

    def build_reward(context: ComponentBuildContext) -> CustomReward:
        seen.append(context)
        return CustomReward(CustomRewardConfig(0.5))

    registry.register_reward("captured_reward", build_reward)
    config = replace(DEFAULT_ENVIRONMENT_CONFIG, reward="captured_reward")
    environment = make_env(config, registry=registry)
    environment.reset(seed=1447)
    result = environment.step(0)

    assert seen == [ComponentBuildContext(config, config.engine)]
    assert result.reward_breakdown.total == 0.5
    with pytest.raises(ValueError, match="duplicate reward"):
        registry.register_reward("captured_reward", build_reward)
    with pytest.raises(ValueError, match="unknown task"):
        registry.task("not_registered")
