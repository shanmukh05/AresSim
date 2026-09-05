"""Time-limit, rollout lifecycle, and seeding tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from aresim.defaults import DEFAULT_ENVIRONMENT_CONFIG
from aresim.factory import make_env
from aresim.training import EpisodeSpec, RolloutConfig, RolloutRunner


def test_external_time_limit_returns_one_real_truncated_transition() -> None:
    environment = make_env(max_episode_steps=3)
    reset = environment.reset(seed=1447)
    assert reset.info["episode_step"] == 0
    first = environment.step(0)
    second = environment.step(0)
    final = environment.step(0)

    assert first.truncated is second.truncated is False
    assert final.terminated is False
    assert final.truncated is True
    assert final.info["truncation_reason"] == "max_episode_steps"
    assert final.info["episode_step"] == 3
    assert final.transition.after_checksum == final.info["state_checksum"]
    with pytest.raises(RuntimeError, match="episode has ended"):
        environment.step(0)
    environment.reset(seed=1447)
    assert environment.step(0).truncated is False


def test_authoritative_termination_takes_precedence_at_the_limit() -> None:
    engine_config = replace(
        DEFAULT_ENVIRONMENT_CONFIG.engine,
        initial=replace(DEFAULT_ENVIRONMENT_CONFIG.engine.initial, rover_health=0.01),
    )
    environment = make_env(
        replace(DEFAULT_ENVIRONMENT_CONFIG, engine=engine_config),
        max_episode_steps=1,
    )
    environment.reset(seed=1447)
    result = environment.step(0)
    assert result.terminated is True
    assert result.truncated is False
    assert result.info["terminal_reason"] == "rover_health_depleted"
    assert "truncation_reason" not in result.info


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_external_time_limit_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        make_env(max_episode_steps=value)


def test_rollout_runner_is_deterministic_and_preserves_reward_audit_fields() -> None:
    config = RolloutConfig(
        (EpisodeSpec("episode-a", 1447, 71), EpisodeSpec("episode-b", 2468, 72)),
        max_episode_steps=8,
    )
    first = RolloutRunner(config, "random_valid").run()
    second = RolloutRunner(config, "random_valid").run()

    assert first.summaries == second.summaries
    assert first.transition_count == second.transition_count == 16
    for first_episode, second_episode in zip(first.episodes, second.episodes, strict=True):
        assert first_episode.actions == second_episode.actions
        assert first_episode.rewards == second_episode.rewards
        assert first_episode.state_checksums == second_episode.state_checksums
        assert all(first_episode.action_legal)
        assert first_episode.episode_return != first_episode.engine_return
        assert first_episode.truncated[-1] is True
        assert first_episode.ending_reason == "max_episode_steps"


def test_rollout_agent_seed_is_independent_from_environment_seed() -> None:
    config = RolloutConfig(
        (EpisodeSpec("first", 1447, 1), EpisodeSpec("second", 1447, 2)),
        max_episode_steps=20,
    )
    result = RolloutRunner(config, "random").run()
    assert result.episodes[0].state_checksums[0] == result.episodes[1].state_checksums[0]
    assert result.episodes[0].actions != result.episodes[1].actions


def test_rollout_config_rejects_duplicate_ids_and_out_of_range_seeds() -> None:
    with pytest.raises(ValueError, match="unique"):
        RolloutConfig((EpisodeSpec("same", 1, 1), EpisodeSpec("same", 2, 2))).validate()
    with pytest.raises(ValueError, match="outside configured bounds"):
        RolloutRunner(RolloutConfig((EpisodeSpec("bad", 100_000, 1),)), "wait")


def test_rollout_rejects_agent_schema_mismatch() -> None:
    class WrongSchemaAgent:
        policy_id = "example.agent.wrong.v1"
        observation_schema = None
        action_schema = "example.action.wrong.v1"

        def reset(self, seed: int) -> None:
            pass

        def act(self, observation, action_mask) -> int:
            return 0

    runner = RolloutRunner(
        RolloutConfig((EpisodeSpec("mismatch", 1447, 1),), max_episode_steps=1),
        WrongSchemaAgent(),
    )
    with pytest.raises(ValueError, match="action schema"):
        runner.run()
