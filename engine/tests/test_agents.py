"""Contract and decision-priority tests for registered baseline agents."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from aresim.algorithms import Agent, RandomValidAgent, ScriptedAgent, UniformRandomAgent, WaitAgent
from aresim.factory import make_agent, make_env
from aresim.registry import create_default_registry


def _policy_input():
    reset = make_env().reset(seed=1447)
    return deepcopy(reset.observation), reset.action_mask.copy()


def _mask(*actions: int) -> np.ndarray:
    result = np.zeros(10, dtype=np.int8)
    result[list(actions)] = 1
    return result


def test_registered_agents_satisfy_contract_and_reject_invalid_factories() -> None:
    registry = create_default_registry()
    for name in ("random", "random_valid", "wait", "scripted"):
        assert isinstance(make_agent(name, registry=registry), Agent)
    with pytest.raises(ValueError, match="duplicate agent"):
        registry.register_agent("wait", lambda context: WaitAgent())
    with pytest.raises(ValueError, match="unknown agent"):
        make_agent("missing", registry=registry)
    registry.register_agent("invalid", lambda context: object())
    with pytest.raises(TypeError, match="Agent"):
        make_agent("invalid", registry=registry)

    class EmptyPolicy(WaitAgent):
        policy_id = ""

    registry.register_agent("empty", lambda context: EmptyPolicy())
    with pytest.raises(ValueError, match="empty policy_id"):
        make_agent("empty", registry=registry)


def test_random_agents_are_reset_deterministic_and_respect_their_legality_contracts() -> None:
    observation, _ = _policy_input()
    restrictive_mask = _mask(0, 3, 7)
    first = UniformRandomAgent()
    second = UniformRandomAgent()
    first.reset(91)
    second.reset(91)
    first_actions = [first.act(observation, restrictive_mask) for _ in range(40)]
    second_actions = [second.act(observation, restrictive_mask) for _ in range(40)]
    assert first_actions == second_actions
    assert any(restrictive_mask[action] == 0 for action in first_actions)

    valid = RandomValidAgent()
    valid.reset(91)
    valid_actions = [valid.act(observation, restrictive_mask) for _ in range(40)]
    assert set(valid_actions) <= {0, 3, 7}
    with pytest.raises(ValueError, match="no legal"):
        valid.act(observation, _mask())


def test_agents_require_reset_and_wait_requires_legal_wait() -> None:
    observation, action_mask = _policy_input()
    with pytest.raises(RuntimeError, match="not been reset"):
        UniformRandomAgent().act(observation, action_mask)
    wait = WaitAgent()
    wait.reset(0)
    assert wait.act(observation, action_mask) == 0
    with pytest.raises(ValueError, match="Wait"):
        wait.act(observation, _mask(1))


def test_scripted_agent_uses_documented_action_priorities() -> None:
    observation, _ = _policy_input()
    agent = ScriptedAgent()
    agent.reset(1)

    unload = deepcopy(observation)
    unload["self"][7] = 0.5
    assert agent.act(unload, _mask(0, 9)) == 9

    service = deepcopy(observation)
    service["colony"][12] = 1
    assert agent.act(service, _mask(0, 8)) == 8

    build = deepcopy(observation)
    build["colony"][8] = 0.5
    assert agent.act(build, _mask(0, 7)) == 7

    recharge = deepcopy(observation)
    recharge["self"][2] = 0.5
    assert agent.act(recharge, _mask(0, 1)) == 0

    assert agent.act(observation, _mask(0, 6)) == 6
    assert agent.act(observation, _mask(0, 5)) == 5


def test_scripted_agent_remembers_home_and_moves_toward_visible_resources() -> None:
    observation, _ = _policy_input()
    observation["colony"][8] = 1
    agent = ScriptedAgent()
    agent.reset(4)
    assert agent.act(observation, _mask(0)) == 0

    away = deepcopy(observation)
    away["self"][0] += 0.1
    away["self"][2] = 0.2
    away["pad_proximity"] = 0
    assert agent.act(away, _mask(0, 4)) == 4

    agent.reset(4)
    resource = deepcopy(observation)
    resource["terrain_type"][:] = 1
    resource["terrain_type"][3, 5] = 3
    resource["pad_proximity"] = 0
    assert agent.act(resource, _mask(0, 2)) == 2
    fallback = deepcopy(observation)
    fallback["terrain_type"][:] = 1
    fallback["pad_proximity"] = 0
    assert agent.act(fallback, _mask(0, 1)) == 1
