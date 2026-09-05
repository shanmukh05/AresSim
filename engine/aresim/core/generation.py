"""Seeded procedural world generation for a deterministic Mars map.

Builds terrain, landing pad, starting rover, and initial colony state from an
integer seed and :class:`~aresim.config.EngineConfig`. No wall-clock or unseeded RNG.

**Last updated:** September 1, 2026

**Entry point:** :func:`create_world` (used by :meth:`~aresim.core.engine.AresEngine.reset`).

**Contains:** terrain placement helpers, landing-pad search, deterministic PRNG utilities.

**Constraints:** must remain importable without NumPy or RL dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from ..config import EngineConfig
from ..types import (
    ActionType,
    Actor,
    BuildPadState,
    BuildPadStatus,
    GameStatus,
    HistoryEntry,
    MissionState,
    ObjectiveStats,
    Position,
    ResourceDelta,
    ResourceState,
    RoverState,
    StructureState,
    StructureType,
    TerrainCell,
    TerrainType,
    WorldState,
)

UINT32_MASK = 0xFFFFFFFF


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return min(maximum, max(minimum, value))


def _imul(left: int, right: int) -> int:
    value = ((left & UINT32_MASK) * (right & UINT32_MASK)) & UINT32_MASK
    return value if value < 0x80000000 else value - 0x100000000


def mulberry32(seed: int) -> Callable[[], float]:
    """Return a seeded 0–1 generator. Keep this algorithm stable; the UI mock uses the same one."""
    value = seed & UINT32_MASK

    def random_value() -> float:
        nonlocal value
        value = (value + 0x6D2B79F5) & UINT32_MASK
        mixed = value
        mixed = _imul(mixed ^ (mixed >> 15), mixed | 1) & UINT32_MASK
        mixed ^= (mixed + _imul(mixed ^ (mixed >> 7), mixed | 61)) & UINT32_MASK
        return ((mixed ^ (mixed >> 14)) & UINT32_MASK) / 4294967296

    return random_value


def _base36(value: int) -> str:
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    value = abs(value)
    if value == 0:
        return "0"
    encoded = ""
    while value:
        value, remainder = divmod(value, 36)
        encoded = digits[remainder] + encoded
    return encoded


def _smooth_noise(x: int, y: int, seed: int) -> float:
    base = math.sin((x * 12.9898 + y * 78.233 + seed * 0.121) * 0.055) * 43758.5453
    wave = math.sin((x + seed) * 0.21) * 0.18 + math.cos((y - seed) * 0.17) * 0.14
    return clamp((base - math.floor(base)) * 0.72 + 0.18 + wave)


def _cluster_value(x: int, y: int, center: Position, radius: float) -> float:
    return clamp(1 - math.hypot(x - center.x, y - center.y) / radius)


def _cluster_max(x: int, y: int, centers: list[Position], radius: float) -> float:
    return max((_cluster_value(x, y, center, radius) for center in centers), default=0)


def _make_feature_centers(
    random_value: Callable[[], float],
    count: int,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
) -> list[Position]:
    return [
        Position(
            x=x_min + math.floor(random_value() * (x_max - x_min + 1)),
            y=y_min + math.floor(random_value() * (y_max - y_min + 1)),
        )
        for _ in range(count)
    ]


def _terrain_for(
    ice: float,
    ore: float,
    crater: float,
    ridge: float,
    dune: float,
    roughness: float,
    height: float,
) -> TerrainType:
    if crater > 0.72:
        return TerrainType.CRATER
    if ridge > 0.68:
        return TerrainType.RIDGE
    if ice > 0.55:
        return TerrainType.ICE
    if ore > 0.48:
        return TerrainType.ROCK
    if dune > 0.56:
        return TerrainType.DUNE
    if height > 0.78:
        return TerrainType.RIDGE
    if roughness > 0.68:
        return TerrainType.DUNE
    return TerrainType.REGOLITH


def generate_terrain(seed: int, config: EngineConfig) -> list[list[TerrainCell]]:
    """Fill a square grid with ice, ore, craters, ridges, and dunes. Does not place the pad."""
    size = config.world.size
    generation = config.generation
    random_value = mulberry32(seed)
    ice_centers = _make_feature_centers(random_value, generation.ice_center_count, 3, size - 5, 6, size - 4)
    ore_centers = _make_feature_centers(random_value, generation.ore_center_count, 4, size - 4, 3, size - 5)
    crater_centers = _make_feature_centers(random_value, generation.crater_center_count, 4, size - 5, 5, size - 5)
    ridge_y = generation.ridge_y_min + math.floor(random_value() * generation.ridge_y_span)
    dune_offset = generation.dune_offset_min + math.floor(random_value() * generation.dune_offset_span)
    terrain: list[list[TerrainCell]] = []
    for y in range(size):
        row: list[TerrainCell] = []
        for x in range(size):
            noise = _smooth_noise(x, y, seed)
            ridge_axis = ridge_y + math.sin((x + seed) * 0.28) * 2.2
            ridge = clamp(1 - abs(y - ridge_axis) / 3.1)
            dune = clamp(1 - abs(x + y * 0.42 - dune_offset) / 4.2)
            ice = _cluster_max(x, y, ice_centers, generation.ice_radius) * (0.78 + random_value() * 0.14)
            ore_cluster = _cluster_max(x, y, ore_centers, generation.ore_radius) * 0.84
            ore_ridge = ridge * 0.46 * _cluster_max(x, y, ore_centers, generation.ore_ridge_radius)
            ore = max(ore_cluster, ore_ridge)
            crater = _cluster_max(x, y, crater_centers, generation.crater_radius)
            height = clamp(0.38 + noise * 0.32 + ridge * 0.35 - crater * 0.32)
            roughness = clamp(0.14 + abs(noise - 0.5) * 0.42 + crater * 0.44 + ridge * 0.36 + dune * 0.24)
            dust = clamp(0.2 + _smooth_noise(x + 19, y - 7, seed + 77) * 0.6)
            row.append(
                TerrainCell(
                    x=x,
                    y=y,
                    terrain=_terrain_for(ice, ore, crater, ridge, dune, roughness, height),
                    height=height,
                    roughness=roughness,
                    ice=ice,
                    ore=ore,
                    dust=dust,
                )
            )
        terrain.append(row)
    return terrain


def _is_safe_landing_patch(terrain: list[list[TerrainCell]], center: Position, config: EngineConfig) -> bool:
    landing = config.landing
    for y in range(center.y - landing.radius, center.y + landing.radius + 1):
        for x in range(center.x - landing.radius, center.x + landing.radius + 1):
            cell = terrain[y][x]
            if cell.terrain in {TerrainType.CRATER, TerrainType.RIDGE, TerrainType.ICE}:
                return False
            if cell.ice > landing.max_ice or cell.roughness > landing.max_roughness:
                return False
            if cell.height < landing.min_height or cell.height > landing.max_height:
                return False
    return True


def _score_landing_cell(cell: TerrainCell) -> float:
    return (1 - cell.roughness) * 2 - cell.ice * 1.5 - abs(cell.height - 0.48)


def _find_safe_landing_site(terrain: list[list[TerrainCell]], seed: int, config: EngineConfig) -> Position | None:
    margin = config.landing.safe_margin
    size = config.world.size
    candidates = [
        cell
        for row in terrain
        for cell in row
        if margin <= cell.x < size - margin
        and margin <= cell.y < size - margin
        and _is_safe_landing_patch(terrain, Position(cell.x, cell.y), config)
    ]
    candidates.sort(key=_score_landing_cell, reverse=True)
    if not candidates:
        return None
    chosen = candidates[abs(seed) % min(len(candidates), 8)]
    return Position(chosen.x, chosen.y)


def _sanitize_landing_patch(terrain: list[list[TerrainCell]], center: Position, config: EngineConfig) -> None:
    landing = config.landing
    for y in range(center.y - landing.radius, center.y + landing.radius + 1):
        for x in range(center.x - landing.radius, center.x + landing.radius + 1):
            cell = terrain[y][x]
            cell.terrain = TerrainType.REGOLITH
            cell.height = landing.sanitized_height
            cell.roughness = landing.sanitized_roughness
            cell.ice = 0
            cell.ore = 0
            cell.scanned = False
            cell.extracted = False


def _apply_landing_build_pad(terrain: list[list[TerrainCell]], center: Position, config: EngineConfig) -> None:
    _sanitize_landing_patch(terrain, center, config)
    landing = config.landing
    for y in range(center.y - landing.radius, center.y + landing.radius + 1):
        for x in range(center.x - landing.radius, center.x + landing.radius + 1):
            cell = terrain[y][x]
            cell.terrain = TerrainType.BUILD_PAD
            cell.roughness = min(cell.roughness, landing.max_pad_roughness)
            cell.dust = min(cell.dust, landing.max_pad_dust)


def create_terrain_with_landing(seed: int, config: EngineConfig) -> tuple[list[list[TerrainCell]], Position]:
    """Generate terrain, find a safe landing patch, and stamp the 5×5 build pad.

    Retries with offset seeds when no safe site exists, then falls back to the
    map center so reset never fails.
    """
    for attempt in range(config.landing.retry_count):
        attempt_seed = seed + attempt * config.landing.retry_seed_offset
        terrain = generate_terrain(attempt_seed, config)
        center = _find_safe_landing_site(terrain, seed + attempt, config)
        if center is not None:
            _apply_landing_build_pad(terrain, center, config)
            return terrain, center

    terrain = generate_terrain(seed + config.landing.fallback_seed_offset, config)
    center = Position(config.world.size // 2, config.world.size // 2)
    _apply_landing_build_pad(terrain, center, config)
    return terrain, center


def create_world(seed: int, config: EngineConfig) -> WorldState:
    """Build a complete initial `WorldState`. Call `rules.initialize_state` before the first step."""
    config.validate()
    if seed < config.world.seed_min or seed > config.world.seed_max:
        raise ValueError(f"seed must be between {config.world.seed_min} and {config.world.seed_max}")
    random_value = mulberry32(seed)
    terrain, rover_start = create_terrain_with_landing(seed, config)
    initial = config.initial
    structures = [
        StructureState("hab_0", StructureType.HABITAT, "Ares Habitat", rover_start.x, rover_start.y, initial.habitat_health, True, "Pressure shell incomplete"),
        StructureState("solar_0", StructureType.SOLAR, "Solar Wing Alpha", rover_start.x + 1, rover_start.y - 1, initial.solar_health, True, "Dust accumulation 18%"),
        StructureState("battery_0", StructureType.BATTERY, "Charging Mast", rover_start.x - 1, rover_start.y + 1, initial.battery_health, True, "Prototype rover charging point"),
        StructureState("store_0", StructureType.STORAGE, "Supply Cache", rover_start.x + 1, rover_start.y + 1, initial.storage_health, True, "Holding spare parts"),
    ]
    ice_sites = sum(cell.terrain == TerrainType.ICE and cell.ice >= config.payload.extract_min_ice_signal for row in terrain for cell in row)
    rock_sites = sum(cell.terrain == TerrainType.ROCK for row in terrain for cell in row)
    history = [
        HistoryEntry(
            id="hist_0_event",
            step=0,
            actor=Actor.SYSTEM,
            action=ActionType.EVENT,
            target=rover_start,
            result="Mission initialized at landing zone",
            reward=0,
            reward_terms={"progress": 0, "safety": 0},
            resource_delta=ResourceDelta(),
            events=["Landing telemetry synchronized"],
        )
    ]
    return WorldState(
        session_id=f"session_{_base36(seed)}",
        seed=seed,
        step=0,
        sol=1,
        local_time="08:00",
        mode="Play",
        game_status=GameStatus.RUNNING,
        status_reason="Open exploration active",
        terrain_width=config.world.size,
        terrain_height=config.world.size,
        weather=config.world.weather_cycle[math.floor(random_value() * len(config.world.weather_cycle))],
        dust_intensity=initial.dust_min + random_value() * initial.dust_range,
        resources=ResourceState(
            power_generated=initial.initial_power_generated,
            power_consumed=initial.initial_power_consumed,
            battery=initial.rover_battery,
            water=initial.water_min + round(random_value() * initial.water_range),
            oxygen=initial.oxygen_min + round(random_value() * initial.oxygen_range),
            livability=initial.livability,
        ),
        mission=MissionState(
            title="Open Mars Exploration",
            objective="Build a livable Mars base before rover or habitat failure.",
            reward_objectives=["Collecting Ice", "Returning Payload", "Scanning Terrain", "Building Habitat", "Servicing Build Pad"],
            alerts=["Solar output stable", "Build pad service check pending"],
        ),
        objective_stats=ObjectiveStats(ice_sites_total=ice_sites, rock_sites_total=rock_sites),
        build_pad_state=BuildPadState(False, BuildPadStatus.NORMAL),
        rules=[],
        terrain=terrain,
        rovers=[
            RoverState(
                id="rover_0",
                name="Rover Ares-01",
                x=rover_start.x,
                y=rover_start.y,
                battery=initial.rover_battery,
                health=initial.rover_health,
                cargo_ice=0,
                cargo_ore=0,
                cargo_samples=0,
                cargo_capacity_kg=config.payload.capacity_kg,
                current_task="Awaiting command",
            )
        ],
        structures=structures,
        history=history,
    )
