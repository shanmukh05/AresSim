"""Compliance and transition-parity tests for every section-3 environment path."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from pettingzoo.test import parallel_api_test, parallel_seed_test

from aresim.core.engine import AresEngine
from aresim.defaults import DEFAULT_ENVIRONMENT_CONFIG
from aresim.envs.gymnasium import AresGymEnv
from aresim.factory import make_env, make_gym_env, make_parallel_env
from aresim.types import Actor


def assert_policy_inputs_equal(first, second) -> None:
    """Compare nested policy dictionaries without ambiguous NumPy equality."""
    assert first.keys() == second.keys()
    for key in first:
        if isinstance(first[key], dict):
            assert_policy_inputs_equal(first[key], second[key])
        else:
            np.testing.assert_array_equal(first[key], second[key])


def test_framework_neutral_environment_is_deterministic_and_audits_engine_reward() -> None:
    first = make_env()
    second = make_env()
    reset_a = first.reset(seed=1447)
    reset_b = second.reset(seed=1447)
    np.testing.assert_array_equal(reset_a.action_mask, reset_b.action_mask)

    step_a = first.step(0)
    step_b = second.step(0)
    assert step_a.reward == step_b.reward
    assert step_a.info["state_checksum"] == step_b.info["state_checksum"]
    assert step_a.info["engine_reward"] == step_a.transition.reward
    assert step_a.info["engine_reward_terms"] == step_a.transition.reward_terms
    assert step_a.reward != step_a.info["engine_reward"]
    assert step_a.truncated is False


def test_framework_neutral_transition_matches_direct_engine() -> None:
    composed = make_env()
    engine = AresEngine()
    composed.reset(seed=2468)
    engine.reset(seed=2468)

    for action_id in (0, 1, 2, 5, 0):
        command = composed.action_adapter.decode(engine.state, action_id)
        direct = engine.step(command, Actor.AGENT)
        result = composed.step(action_id)
        assert result.transition.after_checksum == direct.after_checksum
        assert result.transition.reward == direct.reward
        assert result.transition.reward_terms == direct.reward_terms


def test_framework_neutral_environment_rejects_invalid_lifecycle_calls() -> None:
    environment = make_env()
    with pytest.raises(RuntimeError, match="not been reset"):
        environment.step(0)
    with pytest.raises(ValueError, match="reset options"):
        environment.reset(options={"scenario": "unknown"})
    with pytest.raises(ValueError, match="scenario component name"):
        make_env(replace(DEFAULT_ENVIRONMENT_CONFIG, scenario_id=""))


@pytest.mark.parametrize(
    ("initial_changes", "action_id", "reason"),
    [
        ({"rover_battery": 0.01}, 1, "battery_depleted"),
        ({"rover_health": 0.01}, 0, "rover_health_depleted"),
        ({"livability": 0.01}, 0, "livability_depleted"),
    ],
)
def test_environment_terminates_for_all_engine_failures(initial_changes, action_id: int, reason: str) -> None:
    engine_config = replace(
        DEFAULT_ENVIRONMENT_CONFIG.engine,
        initial=replace(DEFAULT_ENVIRONMENT_CONFIG.engine.initial, **initial_changes),
    )
    if reason == "battery_depleted":
        engine_config = replace(
            engine_config,
            power=replace(engine_config.power, solar_panel_output=0),
        )
    environment = make_env(replace(DEFAULT_ENVIRONMENT_CONFIG, engine=engine_config))
    environment.reset(seed=1447)
    result = environment.step(action_id)
    assert result.terminated is True
    assert result.truncated is False
    assert result.info["terminal_reason"] == reason
    with pytest.raises(RuntimeError, match="episode has ended"):
        environment.step(0)


def test_gymnasium_environment_passes_checker() -> None:
    check_env(make_gym_env(), skip_render_check=True)


def test_gymnasium_rejects_non_single_rover_composition() -> None:
    environment = make_env()
    environment.possible_agents = ("rover_0", "rover_1")
    with pytest.raises(ValueError, match="exactly one"):
        AresGymEnv(environment)


def test_pettingzoo_parallel_environment_passes_contract_and_seed_tests() -> None:
    parallel_api_test(make_parallel_env(), num_cycles=100)
    parallel_seed_test(lambda: make_parallel_env(), num_cycles=20)


def test_pettingzoo_requires_exact_live_agent_actions() -> None:
    environment = make_parallel_env()
    environment.reset(seed=1447)
    with pytest.raises(ValueError, match="exactly every live agent"):
        environment.step({})
    with pytest.raises(ValueError, match="exactly every live agent"):
        environment.step({"rover_0": 0, "rover_1": 0})


def test_direct_gymnasium_and_pettingzoo_transitions_are_identical() -> None:
    direct = make_env()
    gym = make_gym_env()
    parallel = make_parallel_env()
    direct_reset = direct.reset(seed=9876)
    gym_reset, gym_info = gym.reset(seed=9876)
    parallel_reset, parallel_info = parallel.reset(seed=9876)

    assert_policy_inputs_equal(
        {"observation": direct_reset.observation, "action_mask": direct_reset.action_mask},
        gym_reset,
    )
    assert_policy_inputs_equal(gym_reset, parallel_reset["rover_0"])
    assert direct_reset.info["state_checksum"] == gym_info["state_checksum"] == parallel_info["rover_0"]["state_checksum"]

    for action_id in (0, 1, 2, 0, 7):
        direct_step = direct.step(action_id)
        gym_observation, gym_reward, gym_terminated, gym_truncated, gym_step_info = gym.step(action_id)
        observations, rewards, terminations, truncations, infos = parallel.step({"rover_0": action_id})
        assert_policy_inputs_equal(
            {"observation": direct_step.observation, "action_mask": direct_step.action_mask},
            gym_observation,
        )
        assert_policy_inputs_equal(gym_observation, observations["rover_0"])
        assert direct_step.reward == gym_reward == rewards["rover_0"]
        assert direct_step.terminated == gym_terminated == terminations["rover_0"]
        assert direct_step.truncated == gym_truncated == truncations["rover_0"]
        assert direct_step.info["reward_breakdown"] == gym_step_info["reward_breakdown"] == infos["rover_0"]["reward_breakdown"]
        assert direct_step.info["events"] == gym_step_info["events"] == infos["rover_0"]["events"]
        assert direct_step.info["effective_action"] == gym_step_info["effective_action"] == infos["rover_0"]["effective_action"]
        assert direct_step.info["state_checksum"] == gym_step_info["state_checksum"] == infos["rover_0"]["state_checksum"]
