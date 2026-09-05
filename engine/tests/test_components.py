"""Contract tests for RL observations, actions, rewards, and tasks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from aresim.components.actions import ACTION_COUNT, DiscreteActions
from aresim.components.observations import LocalObservation
from aresim.components.rewards import ShapedTrainReward, SparseEvalReward
from aresim.components.tasks import OpenExplorationTask, TaskOutcome
from aresim.config import ObservationConfig
from aresim.core.engine import AresEngine, state_checksum
from aresim.core.rules import validate_action
from aresim.defaults import DEFAULT_ENGINE_CONFIG, DEFAULT_ENVIRONMENT_CONFIG
from aresim.types import (
    ActionCommand,
    ActionType,
    Actor,
    EngineTransition,
    GameRule,
    GameStatus,
    Position,
    RuleStatus,
    TerrainType,
)


def initial_state(seed: int = 1447):
    """Return a canonical initialized state for component-level inspection."""
    engine = AresEngine()
    return engine.reset(seed)


def test_local_observation_has_canonical_shapes_types_and_empty_objectives() -> None:
    state = initial_state()
    builder = LocalObservation(DEFAULT_ENVIRONMENT_CONFIG.observation_config)
    observation = builder.build(state, DEFAULT_ENGINE_CONFIG)

    assert builder.anchor == 3
    assert builder.space.contains(observation)
    assert observation["terrain_type"].shape == (8, 8)
    assert observation["terrain_type"].dtype == np.uint8
    assert observation["spatial"].shape == (5, 8, 8)
    assert observation["spatial"].dtype == np.float32
    assert observation["cell_flags"].shape == (4, 8, 8)
    assert observation["self"].shape == (10,)
    assert observation["colony"].shape == (14,)
    assert observation["colony"][12] == int(state.build_pad_state.service_needed)
    assert np.count_nonzero(observation["objective_type"]) == 0
    assert np.count_nonzero(observation["objectives"]) == 0
    assert np.count_nonzero(observation["objective_mask"]) == 0


def test_local_observation_keeps_edge_padding_unknown_and_does_not_shift() -> None:
    state = initial_state()
    state.rovers[0].x = 0
    state.rovers[0].y = 0
    builder = LocalObservation(DEFAULT_ENVIRONMENT_CONFIG.observation_config)
    observation = builder.build(state, DEFAULT_ENGINE_CONFIG)

    known = observation["cell_flags"][0]
    assert known.sum() == 25
    assert observation["terrain_type"][3, 3] != 0
    assert np.all(observation["terrain_type"][:3] == 0)
    assert np.all(observation["spatial"][:, :3] == 0)


def test_local_observation_is_configurable_and_ignores_cells_outside_crop() -> None:
    state = initial_state()
    config = replace(DEFAULT_ENVIRONMENT_CONFIG.observation_config, window_size=6)
    builder = LocalObservation(config)
    first = builder.build(state, DEFAULT_ENGINE_CONFIG)
    changed = deepcopy(state)
    rover = state.rovers[0]
    far_cell = changed.terrain[(rover.y + 10) % changed.terrain_height][(rover.x + 10) % changed.terrain_width]
    far_cell.terrain = TerrainType.CRATER
    far_cell.ice = 1
    second = builder.build(changed, DEFAULT_ENGINE_CONFIG)

    assert builder.anchor == 2
    assert first["terrain_type"].shape == (6, 6)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_local_observation_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="window size"):
        ObservationConfig(0, 8, 50, 130, 190).validate()
    with pytest.raises(ValueError, match="normalization"):
        ObservationConfig(8, 8, 0, 130, 190).validate()


def test_discrete_action_ids_decode_to_explicit_local_commands() -> None:
    state = initial_state()
    adapter = DiscreteActions()
    rover = state.rovers[0]
    commands = [adapter.decode(state, action_id) for action_id in range(ACTION_COUNT)]

    assert commands[0] == ActionCommand(ActionType.WAIT)
    assert commands[1].target == Position(rover.x, rover.y - 1)
    assert commands[2].target == Position(rover.x + 1, rover.y)
    assert commands[3].target == Position(rover.x, rover.y + 1)
    assert commands[4].target == Position(rover.x - 1, rover.y)
    assert [command.type for command in commands[5:]] == [
        ActionType.SCAN,
        ActionType.EXTRACT,
        ActionType.BUILD,
        ActionType.SERVICE,
        ActionType.UNLOAD,
    ]
    assert all(command.target == Position(rover.x, rover.y) for command in commands[5:])
    with pytest.raises(ValueError, match="integer"):
        adapter.decode(state, 10)


def test_action_mask_matches_authoritative_validation_for_every_id() -> None:
    state = initial_state()
    adapter = DiscreteActions()
    mask = adapter.mask(state, DEFAULT_ENGINE_CONFIG)

    assert mask.shape == (10,)
    assert mask.dtype == np.int8
    assert mask[0] == 1
    for action_id, allowed in enumerate(mask):
        command = adapter.decode(state, action_id)
        assert bool(allowed) == validate_action(state, command, DEFAULT_ENGINE_CONFIG).valid


def test_action_mask_uses_current_payload_and_cell_rules() -> None:
    state = initial_state()
    adapter = DiscreteActions()
    rover = state.rovers[0]
    cell = state.terrain[rover.y][rover.x]

    cell.terrain = TerrainType.ROCK
    assert adapter.mask(state, DEFAULT_ENGINE_CONFIG)[5] == 1
    rover.cargo_ice = rover.cargo_capacity_kg
    assert adapter.mask(state, DEFAULT_ENGINE_CONFIG)[5] == 0
    rover.cargo_ice = 0
    cell.terrain = TerrainType.ICE
    cell.ice = 1
    assert adapter.mask(state, DEFAULT_ENGINE_CONFIG)[6] == 1


def test_action_mask_covers_build_service_unload_edges_without_hidden_leakage() -> None:
    state = initial_state()
    adapter = DiscreteActions()
    rover = state.rovers[0]
    assert adapter.mask(state, DEFAULT_ENGINE_CONFIG)[7] == 1
    assert adapter.mask(state, DEFAULT_ENGINE_CONFIG)[8] == 1
    assert adapter.mask(state, DEFAULT_ENGINE_CONFIG)[9] == 0

    state.rovers[0].cargo_ice = 1
    assert adapter.mask(state, DEFAULT_ENGINE_CONFIG)[9] == 1
    state.objective_stats.habitat_build_progress = 100
    assert adapter.mask(state, DEFAULT_ENGINE_CONFIG)[7] == 0

    first_mask = adapter.mask(state, DEFAULT_ENGINE_CONFIG)
    hidden = deepcopy(state)
    far_y = (rover.y + 10) % hidden.terrain_height
    far_x = (rover.x + 10) % hidden.terrain_width
    hidden.terrain[far_y][far_x].terrain = TerrainType.CRATER
    hidden.terrain[far_y][far_x].ice = 1
    np.testing.assert_array_equal(first_mask, adapter.mask(hidden, DEFAULT_ENGINE_CONFIG))


def _transition(before, after, effective_action: ActionType) -> EngineTransition:
    command = ActionCommand(effective_action if effective_action != ActionType.INVALID else ActionType.SCAN)
    return EngineTransition(
        command=command,
        effective_action=effective_action,
        actor=Actor.AGENT,
        before_checksum=state_checksum(before),
        after_checksum=state_checksum(after),
        reward=7.5,
        reward_terms={"engine": 7.5},
        events=(),
        state=after,
    )


def test_shaped_reward_uses_targetless_physical_deltas_and_clips_nonterminal() -> None:
    before = initial_state()
    after = deepcopy(before)
    after.objective_stats.terrain_scanned += 1
    after.objective_stats.ice_delivered += 6
    after.objective_stats.samples_delivered += 3
    after.objective_stats.habitat_build_progress += 10
    transition = _transition(before, after, ActionType.SCAN)
    reward_config = replace(DEFAULT_ENVIRONMENT_CONFIG.reward_config, new_scan=20)
    breakdown = ShapedTrainReward(reward_config).calculate(before, transition, TaskOutcome(False, False, None))

    assert breakdown.terms["objective_progress"].raw == 0
    assert breakdown.terms["ice_delivered"].raw == 0.5
    assert breakdown.terms["samples_delivered"].raw == 0.25
    assert breakdown.total_unclipped > 2
    assert breakdown.total == 2


def test_service_recovery_only_rewards_an_effective_service_action() -> None:
    before = initial_state()
    after = deepcopy(before)
    after.dust_intensity = max(0, before.dust_intensity - 0.2)
    for structure in after.structures:
        structure.health = 100
    profile = ShapedTrainReward(DEFAULT_ENVIRONMENT_CONFIG.reward_config)

    wait = profile.calculate(before, _transition(before, after, ActionType.WAIT), TaskOutcome(False, False, None))
    service = profile.calculate(before, _transition(before, after, ActionType.SERVICE), TaskOutcome(False, False, None))
    assert wait.terms["service_recovery"].raw == 0
    assert service.terms["service_recovery"].raw > 0


def test_sparse_reward_uses_only_failure_and_invalid_terms() -> None:
    before = initial_state()
    after = deepcopy(before)
    after.objective_stats.terrain_scanned += 1
    breakdown = SparseEvalReward(DEFAULT_ENVIRONMENT_CONFIG.reward_config).calculate(
        before,
        _transition(before, after, ActionType.INVALID),
        TaskOutcome(True, False, "battery_depleted"),
    )

    assert breakdown.terms["new_scan"].weight == 0
    assert breakdown.terms["terminal_failure"].value == -5
    assert breakdown.terms["invalid_action"].value == -0.10
    assert breakdown.total == pytest.approx(-5.1)


@pytest.mark.parametrize(
    ("rule_id", "reason"),
    [
        ("battery", "battery_depleted"),
        ("health", "rover_health_depleted"),
        ("livability", "livability_depleted"),
    ],
)
def test_open_exploration_delegates_all_engine_failures(rule_id: str, reason: str) -> None:
    before = initial_state()
    after = deepcopy(before)
    after.game_status = GameStatus.GAME_OVER
    after.rules = [GameRule(rule_id, rule_id, "failed", RuleStatus.FAILED, "0")]
    outcome = OpenExplorationTask().evaluate(before, _transition(before, after, ActionType.WAIT))
    assert outcome == TaskOutcome(True, False, reason)
