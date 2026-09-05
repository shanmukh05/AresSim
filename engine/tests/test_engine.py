"""Determinism, payload, terrain, and terminal-rule tests for `AresEngine`.

If a core rule changes, add the regression here rather than only in the UI.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from aresim.config import EngineConfig
from aresim.core import AresEngine, state_checksum
from aresim.core.rules import payload_used_kg
from aresim.defaults import DEFAULT_ENGINE_CONFIG
from aresim.integrations.ui import snapshot_from_state
from aresim.types import ActionCommand, ActionType, Actor, Position, TerrainType


def make_engine(config: EngineConfig = DEFAULT_ENGINE_CONFIG, seed: int = 1447) -> AresEngine:
    """Reset an engine on a known seed so tests share a stable map."""
    engine = AresEngine(config)
    engine.reset(seed)
    return engine


def find_cell(engine: AresEngine, terrain: TerrainType) -> Position:
    """First cell of `terrain` on the current map, or fail the test."""
    for row in engine.state.terrain:
        for cell in row:
            if cell.terrain == terrain:
                return Position(cell.x, cell.y)
    raise AssertionError(f"seed has no {terrain.value} cell")


def test_default_config_is_valid_and_overrides_are_isolated() -> None:
    DEFAULT_ENGINE_CONFIG.validate()
    overridden = replace(
        DEFAULT_ENGINE_CONFIG,
        payload=replace(DEFAULT_ENGINE_CONFIG.payload, capacity_kg=7),
    )
    overridden.validate()
    assert overridden.payload.capacity_kg == 7
    assert DEFAULT_ENGINE_CONFIG.payload.capacity_kg == 12
    with pytest.raises(TypeError):
        DEFAULT_ENGINE_CONFIG.action.base_drain[ActionType.MOVE] = 99  # type: ignore[index]


def test_seeded_generation_is_deterministic_and_has_safe_pad() -> None:
    first = make_engine(seed=1451).state
    second = make_engine(seed=1451).state
    other = make_engine(seed=1452).state
    assert state_checksum(first) == state_checksum(second)
    assert state_checksum(first) != state_checksum(other)
    assert snapshot_from_state(first) == snapshot_from_state(second)
    rover = first.rovers[0]
    pad = [
        first.terrain[y][x]
        for y in range(rover.y - 2, rover.y + 3)
        for x in range(rover.x - 2, rover.x + 3)
    ]
    assert len(pad) == 25
    assert all(cell.terrain == TerrainType.BUILD_PAD for cell in pad)
    assert all(cell.ice == 0 and cell.roughness <= DEFAULT_ENGINE_CONFIG.landing.max_pad_roughness for cell in pad)


def test_same_seed_and_commands_produce_the_same_final_checksum() -> None:
    first = make_engine(seed=9021)
    second = make_engine(seed=9021)
    commands = [
        ActionCommand(ActionType.WAIT),
        ActionCommand(ActionType.SCAN),
        ActionCommand(ActionType.EXTRACT),
        ActionCommand(ActionType.BUILD),
        ActionCommand(ActionType.SERVICE),
    ]
    first_rewards: list[float] = []
    second_rewards: list[float] = []
    for command in commands:
        first_rewards.append(first.step(command, Actor.AGENT).reward)
        second_rewards.append(second.step(command, Actor.AGENT).reward)
    assert first_rewards == second_rewards
    assert state_checksum(first.state) == state_checksum(second.state)


def test_every_public_action_and_invalid_transition() -> None:
    engine = make_engine()
    pad = find_cell(engine, TerrainType.BUILD_PAD)
    rock = find_cell(engine, TerrainType.ROCK)
    ice = find_cell(engine, TerrainType.ICE)
    regolith = find_cell(engine, TerrainType.REGOLITH)

    assert engine.step(ActionCommand(ActionType.MOVE, regolith), Actor.PLAYER).effective_action == ActionType.MOVE
    assert engine.step(ActionCommand(ActionType.SCAN, rock), Actor.PLAYER).effective_action == ActionType.SCAN
    assert engine.step(ActionCommand(ActionType.EXTRACT, ice), Actor.PLAYER).effective_action == ActionType.EXTRACT
    assert engine.step(ActionCommand(ActionType.BUILD, pad), Actor.PLAYER).effective_action == ActionType.BUILD
    assert engine.step(ActionCommand(ActionType.SERVICE, pad), Actor.PLAYER).effective_action == ActionType.SERVICE
    engine.step(ActionCommand(ActionType.MOVE, pad), Actor.PLAYER)
    assert engine.step(ActionCommand(ActionType.UNLOAD), Actor.PLAYER).effective_action == ActionType.UNLOAD
    assert engine.step(ActionCommand(ActionType.WAIT), Actor.AGENT).effective_action == ActionType.WAIT

    before = engine.state
    invalid = engine.step(ActionCommand(ActionType.SCAN, regolith), Actor.PLAYER)
    after = invalid.state
    assert invalid.effective_action == ActionType.INVALID
    assert invalid.reward == pytest.approx(-1.8)
    assert after.step == before.step + 1
    assert after.terrain[regolith.y][regolith.x] == before.terrain[regolith.y][regolith.x]
    assert after.rovers[0].cargo_samples == before.rovers[0].cargo_samples


def test_payload_capacity_blocks_atomic_scan_and_extract_and_unload_empties_it() -> None:
    engine = make_engine()
    extracted_positions: list[Position] = []
    for _ in range(6):
        target = find_cell(engine, TerrainType.ICE)
        extracted_positions.append(target)
        transition = engine.step(ActionCommand(ActionType.EXTRACT, target), Actor.PLAYER)
        assert transition.effective_action == ActionType.EXTRACT
    assert payload_used_kg(engine.state.rovers[0]) == 12
    rock = find_cell(engine, TerrainType.ROCK)
    before = deepcopy(engine.state.terrain[rock.y][rock.x])
    blocked_scan = engine.step(ActionCommand(ActionType.SCAN, rock), Actor.PLAYER)
    assert blocked_scan.effective_action == ActionType.INVALID
    assert blocked_scan.state.terrain[rock.y][rock.x] == before
    assert blocked_scan.state.rovers[0].cargo_samples == 0
    blocked_extract = engine.step(ActionCommand(ActionType.EXTRACT, find_cell(engine, TerrainType.ICE)), Actor.PLAYER)
    assert blocked_extract.effective_action == ActionType.INVALID
    pad = find_cell(engine, TerrainType.BUILD_PAD)
    engine.step(ActionCommand(ActionType.MOVE, pad), Actor.PLAYER)
    unloaded = engine.step(ActionCommand(ActionType.UNLOAD), Actor.PLAYER)
    assert unloaded.effective_action == ActionType.UNLOAD
    assert payload_used_kg(unloaded.state.rovers[0]) == 0
    assert unloaded.state.objective_stats.ice_delivered == 12
    for target in extracted_positions:
        assert unloaded.state.terrain[target.y][target.x].terrain == TerrainType.REGOLITH
        assert unloaded.state.terrain[target.y][target.x].extracted is True


def test_service_latches_until_explicit_service_and_livability_can_end_run() -> None:
    frequent_service = replace(
        DEFAULT_ENGINE_CONFIG,
        service=replace(DEFAULT_ENGINE_CONFIG.service, dust_threshold=0.1),
    )
    engine = make_engine(frequent_service)
    assert engine.state.build_pad_state.service_needed is True
    pad = find_cell(engine, TerrainType.BUILD_PAD)
    serviced = engine.step(ActionCommand(ActionType.SERVICE, pad), Actor.PLAYER).state
    assert serviced.build_pad_state.service_needed is False

    fragile = replace(
        DEFAULT_ENGINE_CONFIG,
        initial=replace(DEFAULT_ENGINE_CONFIG.initial, livability=0.01),
    )
    ending = make_engine(fragile).step(ActionCommand(ActionType.WAIT), Actor.PLAYER).state
    assert ending.game_status.value == "game_over"
    assert ending.resources.livability == 0


def test_pause_and_resume_do_not_advance_state() -> None:
    engine = make_engine()
    step = engine.state.step
    assert engine.pause().step == step
    assert engine.pause().game_status.value == "paused"
    assert engine.resume().step == step
    assert engine.resume().game_status.value == "running"
