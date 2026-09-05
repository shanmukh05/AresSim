"""Builds the bounded rover-centered ``aresim.obs.local.v1`` policy observation.

Projects canonical world state into a fixed local crop plus telemetry vectors.
Does not mutate state or leak full-map UI camera memory.

**Last updated:** September 1, 2026

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


def _bounded(value: float, low: float = 0, high: float = 1) -> float:
    return min(high, max(low, value))


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

        objective_count = self.config.max_objectives
        return {
            "terrain_type": terrain_type,
            "spatial": spatial,
            "cell_flags": cell_flags,
            "self": self_vector,
            "pad_proximity": pad_proximity,
            "colony": colony,
            "weather_type": WEATHER_IDS[state.weather],
            "objective_type": np.zeros(objective_count, dtype=np.uint8),
            "objectives": np.zeros((objective_count, 4), dtype=np.float32),
            "objective_mask": np.zeros(objective_count, dtype=np.uint8),
        }
