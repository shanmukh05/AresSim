"""Frozen configuration dataclasses for the simulator and RL composition.

Defines typed knob groups consumed by :mod:`aresim.core`, :mod:`aresim.components`,
and :mod:`aresim.envs`. Numbers live in :mod:`aresim.defaults`; behavior lives in
the owning module for each config section.

**Last updated:** September 1, 2026

**Contains:** ``EngineConfig``, ``EnvironmentConfig``, and nested world/generation/
landing/reward/observation profile configs with ``validate()`` methods.

**See also:** :mod:`aresim.defaults` (canonical Phase 1 values),
:mod:`aresim.factory` (wiring configs into environments).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .types import ActionType, TerrainType, WeatherState


@dataclass(frozen=True)
class WorldConfig:
    """Map size, allowed seeds, clock, and the repeating weather cycle."""
    size: int
    default_seed: int
    seed_min: int
    seed_max: int
    steps_per_sol: int
    start_time_minutes: int
    minutes_per_step: int
    weather_cycle: tuple[WeatherState, ...]


@dataclass(frozen=True)
class GenerationConfig:
    """Counts and radii for ice, ore, crater, ridge, and dune placement."""
    ice_center_count: int
    ore_center_count: int
    crater_center_count: int
    ice_radius: float
    ore_radius: float
    ore_ridge_radius: float
    crater_radius: float
    ridge_y_min: int
    ridge_y_span: int
    dune_offset_min: int
    dune_offset_span: int


@dataclass(frozen=True)
class LandingConfig:
    """Search and sanitization of the contiguous 5×5 landing / build-pad patch."""
    radius: int
    retry_count: int
    retry_seed_offset: int
    fallback_seed_offset: int
    safe_margin: int
    max_ice: float
    max_roughness: float
    min_height: float
    max_height: float
    sanitized_height: float
    sanitized_roughness: float
    max_pad_roughness: float
    max_pad_dust: float


@dataclass(frozen=True)
class InitialConfig:
    """Starting rover, structure, colony-reserve, and dust values after reset."""
    rover_battery: float
    rover_health: float
    habitat_health: float
    solar_health: float
    battery_health: float
    storage_health: float
    initial_power_generated: float
    initial_power_consumed: float
    water_min: int
    water_range: int
    oxygen_min: int
    oxygen_range: int
    livability: float
    dust_min: float
    dust_range: float


@dataclass(frozen=True)
class PayloadConfig:
    """Shared cargo capacity, Scan/Extract masses, and Unload conversion rates."""
    capacity_kg: float
    scan_sample_kg: float
    ice_extraction_kg: float
    ice_channel_reduction: float
    extract_min_ice_signal: float
    unload_water_per_ice: float
    unload_oxygen_per_ice: float
    unload_livability_per_ice: float


@dataclass(frozen=True)
class PowerConfig:
    """Solar generation, base loads, deficit drain, and Wait/pad recharge."""
    base_load: float
    habitat_load: float
    storage_load: float
    charger_load: float
    solar_panel_output: float
    weather_generation_factor: Mapping[WeatherState, float]
    dust_generation_multiplier: float
    dust_generation_floor: float
    deficit_battery_cap: float
    deficit_battery_rate: float
    wait_charge_per_kw: float
    wait_max_charge: float
    pad_charge_per_kw: float
    pad_max_charge: float


@dataclass(frozen=True)
class ServiceConfig:
    """When the pad needs Service, and how weather/dust degrade infrastructure."""
    dust_threshold: float
    health_threshold: float
    power_margin_threshold: float
    power_dust_threshold: float
    service_radius: float
    serviced_dust: float
    restored_structure_health: float
    weather_damage: Mapping[WeatherState, float]
    weather_dust_delta: Mapping[WeatherState, float]
    dust_damage_rate: float
    dust_floor: float
    dust_ceiling: float


@dataclass(frozen=True)
class LifeSupportConfig:
    """Water, oxygen, livability, and rover-health drain after each valid action."""
    water_base_drain: float
    water_livability_rate: float
    water_dust_rate: float
    oxygen_base_drain: float
    oxygen_livability_rate: float
    oxygen_deficit_rate: float
    livability_base_decay: float
    empty_reserve_penalty: float
    power_deficit_cap: float
    power_deficit_rate: float
    service_bonus: float
    build_bonus: float
    low_battery_health_threshold: float
    critical_battery_health_threshold: float
    normal_health_decay: float
    low_battery_health_decay: float
    critical_battery_health_decay: float


@dataclass(frozen=True)
class ActionConfig:
    """Per-action battery drain, grid load, terrain/weather stress, and Build costs."""
    base_drain: Mapping[ActionType, float]
    action_load: Mapping[ActionType, float]
    action_stress: Mapping[ActionType, float]
    terrain_stress: Mapping[TerrainType, float]
    weather_stress: Mapping[WeatherState, float]
    roughness_stress_rate: float
    cell_dust_stress_rate: float
    cargo_stress_rate: float
    global_dust_stress_rate: float
    stress_exponent_rate: float
    habitat_build_steps: int
    build_water_cost: float
    build_oxygen_cost: float
    habitat_health_gain: float
    extract_ice_priority_weight: float


@dataclass(frozen=True)
class WarningConfig:
    """HUD warning thresholds. Crossing them does not end the run by itself."""
    water_reserve: float
    oxygen_reserve: float
    rover_battery: float
    rover_health: float
    livability: float


@dataclass(frozen=True)
class RewardConfig:
    """Named reward terms summed into the step total. Invalid actions use `invalid`."""
    energy_rate: float
    normal_traversal: float
    pad_traversal: float
    safe_move: float
    hazardous_move: float
    scan_information: float
    scan_exploration: float
    ice_resource_per_kg: float
    ore_resource_per_kg: float
    build_infrastructure: float
    build_livability: float
    service_base: float
    service_needed: float
    service_efficiency: float
    ice_delivery_per_kg: float
    ore_delivery_per_kg: float
    sample_delivery_per_kg: float
    unload_logistics: float
    wait_surplus: float
    wait_deficit: float
    invalid: float
    invalid_safety: float


@dataclass(frozen=True)
class ReplayConfig:
    """Gameplay schema version, checkpoint spacing, and upload size limit."""
    schema_version: str
    app_version: str
    checkpoint_interval: int
    max_upload_bytes: int


@dataclass(frozen=True)
class EngineConfig:
    """Complete simulator config. Call `validate()` before constructing `AresEngine`."""

    world: WorldConfig
    generation: GenerationConfig
    landing: LandingConfig
    initial: InitialConfig
    payload: PayloadConfig
    power: PowerConfig
    service: ServiceConfig
    life_support: LifeSupportConfig
    action: ActionConfig
    warning: WarningConfig
    reward: RewardConfig
    replay: ReplayConfig

    def _validate_world(self) -> None:
        if self.world.size <= 0:
            raise ValueError("world size must be positive")
        if self.landing.radius < 0 or self.landing.radius * 2 + 1 > self.world.size:
            raise ValueError("landing pad must fit inside the world")
        if self.world.seed_min > self.world.seed_max:
            raise ValueError("seed range is invalid")
        if not self.world.weather_cycle:
            raise ValueError("weather cycle cannot be empty")

    def _validate_payload(self) -> None:
        if self.payload.capacity_kg <= 0:
            raise ValueError("payload capacity must be positive")
        if self.payload.scan_sample_kg <= 0 or self.payload.ice_extraction_kg <= 0:
            raise ValueError("payload additions must be positive")
        if not 0 < self.payload.extract_min_ice_signal <= 1:
            raise ValueError("extractable ice threshold must be in (0, 1]")

    def _validate_action_and_service(self) -> None:
        if self.action.habitat_build_steps <= 0:
            raise ValueError("habitat build steps must be positive")
        if self.action.build_water_cost < 0 or self.action.build_oxygen_cost < 0:
            raise ValueError("build resource costs cannot be negative")
        if self.service.dust_floor < 0 or self.service.dust_floor > self.service.dust_ceiling or self.service.dust_ceiling > 1:
            raise ValueError("dust bounds must be ordered within [0, 1]")

    def _validate_lookup_tables(self) -> None:
        required_weather = set(WeatherState)
        for name, values in (
            ("power weather factors", self.power.weather_generation_factor),
            ("service weather damage", self.service.weather_damage),
            ("weather dust deltas", self.service.weather_dust_delta),
            ("weather stress", self.action.weather_stress),
        ):
            if set(values) != required_weather:
                raise ValueError(f"{name} must define every weather state")
        if set(self.action.base_drain) != set(ActionType) or set(self.action.action_load) != set(ActionType) or set(self.action.action_stress) != set(ActionType):
            raise ValueError("action tuning must define every action and history category")
        if set(self.action.terrain_stress) != set(TerrainType):
            raise ValueError("terrain stress must define every terrain type")

    def _validate_warning_and_replay(self) -> None:
        if min(self.warning.water_reserve, self.warning.oxygen_reserve, self.warning.rover_battery, self.warning.rover_health, self.warning.livability) < 0:
            raise ValueError("warning thresholds cannot be negative")
        if self.replay.checkpoint_interval <= 0 or self.replay.max_upload_bytes <= 0:
            raise ValueError("replay limits must be positive")

    def validate(self) -> None:
        """Reject configs that would generate an illegal world or incomplete lookup tables."""
        self._validate_world()
        self._validate_payload()
        self._validate_action_and_service()
        self._validate_lookup_tables()
        self._validate_warning_and_replay()


@dataclass(frozen=True)
class ObservationConfig:
    """Shape and normalization scales for the local numerical observation."""

    window_size: int
    max_objectives: int
    power_scale: float
    water_scale: float
    oxygen_scale: float

    def validate(self) -> None:
        """Reject shapes or scales that cannot produce a bounded observation."""
        if self.window_size <= 0:
            raise ValueError("observation window size must be positive")
        if self.max_objectives <= 0:
            raise ValueError("maximum objective count must be positive")
        if min(self.power_scale, self.water_scale, self.oxygen_scale) <= 0:
            raise ValueError("observation normalization scales must be positive")


@dataclass(frozen=True)
class RewardProfileConfig:
    """Weights and clipping bounds for RL-facing reward profiles."""

    mission_success: float
    terminal_failure: float
    objective_progress: float
    new_scan: float
    ice_delivered: float
    samples_delivered: float
    build_progress: float
    service_recovery: float
    hazard_damage: float
    energy_used: float
    invalid_action: float
    time_cost: float
    clip_min: float
    clip_max: float

    def validate(self) -> None:
        """Reject reversed shaped-reward clipping bounds."""
        if self.clip_min > self.clip_max:
            raise ValueError("reward clipping bounds are reversed")


@dataclass(frozen=True)
class EnvironmentConfig:
    """Select the engine and registered components for one composed RL environment."""

    engine: EngineConfig
    scenario_id: str
    observation: str
    observation_config: ObservationConfig
    action: str
    reward: str
    reward_config: RewardProfileConfig
    task: str

    def validate(self) -> None:
        """Validate nested values; component names are resolved by the local registry."""
        self.engine.validate()
        self.observation_config.validate()
        self.reward_config.validate()
        for label, value in (
            ("scenario", self.scenario_id),
            ("observation", self.observation),
            ("action", self.action),
            ("reward", self.reward),
            ("task", self.task),
        ):
            if not value:
                raise ValueError(f"{label} component name cannot be empty")
