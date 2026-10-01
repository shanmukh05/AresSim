"""Builds the bounded rover-centered ``aresim.obs.local.v1`` policy observation.

Projects canonical world state into a fixed local crop plus telemetry vectors.
Does not mutate state or leak full-map UI camera memory. Objective slots are a
mission progress table (how far ice/samples/pad/battery are); they do not
encode bearings or off-crop locations.

**Last updated:** September 26, 2026

**Contains:** ``LocalObservation``, ``OBSERVATION_SCHEMA``, terrain/weather ID tables.

**Registry name:** ``local``.

**Output keys:** ``self``, ``colony``, ``terrain_type``, ``spatial``, ``cell_flags``,
``pad_proximity``, ``weather_type``, objective tensors.
"""

from __future__ import annotations

import math

import numpy as np
from gymnasium import spaces

from ..config import EngineConfig, ObservationConfig
from ..core.rules import near_build_pad, rover_on_build_pad
from ..types import Position, TerrainType, WeatherState, WorldState
from .tasks import ICE_DELIVERY_TARGET_KG, SAMPLE_DELIVERY_TARGET_KG


OBSERVATION_SCHEMA = "aresim.obs.local.v1"

TERRAIN_IDS = {
    TerrainType.REGOLITH: 1,
    TerrainType.ROCK: 2,
    TerrainType.ICE: 3,
    TerrainType.CRATER: 4,
    TerrainType.DUNE: 5,
    TerrainType.BUILD_PAD: 6,
    TerrainType.RIDGE: 7,
}

WEATHER_IDS = {
    WeatherState.CLEAR: 1,
    WeatherState.DUSTY: 2,
    WeatherState.DUST_FRONT: 3,
    WeatherState.SEVERE_STORM: 4,
    WeatherState.COLD_NIGHT: 5,
}

# Declared ``aresim.obs.local.v1`` type IDs. Rows are progress only.
OBJECTIVE_TYPE_SCAN = 2
OBJECTIVE_TYPE_EXTRACT_ICE = 3
OBJECTIVE_TYPE_DELIVER_ICE = 4
OBJECTIVE_TYPE_SERVICE = 6
OBJECTIVE_TYPE_SURVIVE = 7


def _bounded(value: float, low: float = 0, high: float = 1) -> float:
    return min(high, max(low, value))


def _progress_row(current: float) -> np.ndarray:
    done = _bounded(current)
    return np.array((done, 1.0, 1.0 - done, 1.0), dtype=np.float32)


def _objective_progress_table(state: WorldState, slot_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill live slots with mission progress. No coordinates or bearings."""
    types = np.zeros(slot_count, dtype=np.uint8)
    rows = np.zeros((slot_count, 4), dtype=np.float32)
    mask = np.zeros(slot_count, dtype=np.uint8)
    stats = state.objective_stats
    slots = (
        (OBJECTIVE_TYPE_EXTRACT_ICE, stats.ice_collected / ICE_DELIVERY_TARGET_KG),
        (OBJECTIVE_TYPE_DELIVER_ICE, stats.ice_delivered / ICE_DELIVERY_TARGET_KG),
        (OBJECTIVE_TYPE_SCAN, stats.samples_delivered / SAMPLE_DELIVERY_TARGET_KG),
        (OBJECTIVE_TYPE_SERVICE, 0.0 if state.build_pad_state.service_needed else 1.0),
        (OBJECTIVE_TYPE_SURVIVE, state.rovers[0].battery / 100),
    )
    for index, (kind, current) in enumerate(slots[:slot_count]):
        types[index] = kind
        rows[index] = _progress_row(current)
        mask[index] = 1
    return types, rows, mask


class LocalObservation:
    """Build the configured local crop and its declared Gymnasium space."""

    schema = OBSERVATION_SCHEMA

    def __init__(self, config: ObservationConfig) -> None:
        config.validate()
        self.config = config
        size = config.window_size
        objectives = config.max_objectives
        self.space = spaces.Dict({
            "terrain_type": spaces.Box(0, 7, shape=(size, size), dtype=np.uint8),
            "spatial": spaces.Box(0, 1, shape=(5, size, size), dtype=np.float32),
            "cell_flags": spaces.Box(0, 1, shape=(4, size, size), dtype=np.uint8),
            "self": spaces.Box(-1, 1, shape=(10,), dtype=np.float32),
            "pad_proximity": spaces.Discrete(3),
            "colony": spaces.Box(-1, 1, shape=(14,), dtype=np.float32),
            "weather_type": spaces.Discrete(6),
            "objective_type": spaces.Box(0, 8, shape=(objectives,), dtype=np.uint8),
            "objectives": spaces.Box(0, 1, shape=(objectives, 4), dtype=np.float32),
            "objective_mask": spaces.Box(0, 1, shape=(objectives,), dtype=np.uint8),
        })

    @property
    def anchor(self) -> int:
        """Local index occupied by the rover on both axes."""
        return (self.config.window_size - 1) // 2

    def reset(self, state: WorldState, engine_config: EngineConfig) -> dict[str, np.ndarray | int]:
        """Return the initial observation; this built-in keeps no episode memory."""
        return self.build(state, engine_config)

    def build(self, state: WorldState, engine_config: EngineConfig) -> dict[str, np.ndarray | int]:
        """Return one bounded observation; arrays are newly allocated for the caller."""
        size = self.config.window_size
        rover = state.rovers[0]
        origin_x = rover.x - self.anchor
        origin_y = rover.y - self.anchor
        terrain_type = np.zeros((size, size), dtype=np.uint8)
        spatial = np.zeros((5, size, size), dtype=np.float32)
        cell_flags = np.zeros((4, size, size), dtype=np.uint8)

        for local_y in range(size):
            world_y = origin_y + local_y
            for local_x in range(size):
                world_x = origin_x + local_x
                if not (0 <= world_x < state.terrain_width and 0 <= world_y < state.terrain_height):
                    continue
                cell = state.terrain[world_y][world_x]
                terrain_type[local_y, local_x] = TERRAIN_IDS[cell.terrain]
                spatial[:, local_y, local_x] = (
                    _bounded(cell.height),
                    _bounded(cell.roughness),
                    _bounded(cell.ice),
                    _bounded(cell.ore),
                    _bounded(cell.dust),
                )
                cell_flags[0, local_y, local_x] = 1
                cell_flags[1, local_y, local_x] = 1
                cell_flags[2, local_y, local_x] = int(cell.scanned)
                cell_flags[3, local_y, local_x] = int(cell.extracted)

        capacity = rover.cargo_capacity_kg
        carried = rover.cargo_ice + rover.cargo_ore + rover.cargo_samples
        hours, minutes = (int(part) for part in state.local_time.split(":"))
        day_fraction = (hours * 60 + minutes) / (24 * 60)
        self_vector = np.array([
            rover.x / max(1, state.terrain_width - 1),
            rover.y / max(1, state.terrain_height - 1),
            _bounded(rover.battery / 100),
            _bounded(rover.health / 100),
            _bounded(rover.cargo_ice / capacity),
            _bounded(rover.cargo_ore / capacity),
            _bounded(rover.cargo_samples / capacity),
            _bounded((capacity - carried) / capacity),
            math.sin(2 * math.pi * day_fraction),
            math.cos(2 * math.pi * day_fraction),
        ], dtype=np.float32)

        resources = state.resources
        margin = resources.power_generated - resources.power_consumed
        colony = np.zeros(14, dtype=np.float32)
        colony[:9] = (
            _bounded(resources.power_generated / self.config.power_scale),
            _bounded(resources.power_consumed / self.config.power_scale),
            _bounded(margin / self.config.power_scale, -1, 1),
            _bounded(resources.battery / 100),
            _bounded(resources.water / self.config.water_scale),
            _bounded(resources.oxygen / self.config.oxygen_scale),
            _bounded(resources.livability / 100),
            _bounded(state.dust_intensity),
            _bounded(state.objective_stats.habitat_build_progress / 100),
        )
        colony[12] = int(state.build_pad_state.service_needed)

        if rover_on_build_pad(state):
            pad_proximity = 2
        elif near_build_pad(state, Position(rover.x, rover.y), engine_config.service.service_radius):
            pad_proximity = 1
        else:
            pad_proximity = 0

        objective_type, objectives, objective_mask = _objective_progress_table(
            state, self.config.max_objectives,
        )
        return {
            "terrain_type": terrain_type,
            "spatial": spatial,
            "cell_flags": cell_flags,
            "self": self_vector,
            "pad_proximity": pad_proximity,
            "colony": colony,
            "weather_type": WEATHER_IDS[state.weather],
            "objective_type": objective_type,
            "objectives": objectives,
            "objective_mask": objective_mask,
        }
