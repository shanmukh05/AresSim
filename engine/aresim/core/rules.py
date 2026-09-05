"""Authoritative gameplay: legality, transitions, rewards, warnings, and terminals.

Single source of simulator rules. UI, frontend, and RL adapters must not duplicate
validation or mutation logic found here. Tunables live in :mod:`aresim.defaults`.

**Last updated:** September 1, 2026

**Key entry points:** :func:`validate_action`, :func:`apply_action`,
:func:`initialize_state`.

**Contains:** payload/battery/power helpers, action resolution, terminal failure
detection, engine reward terms for history.

**See also:** :mod:`aresim.components.actions` (RL action decode + mask projection).
"""

from __future__ import annotations

import math
from dataclasses import fields

from ..config import EngineConfig
from ..types import (
    ActionCommand,
    ActionType,
    ActionValidation,
    Actor,
    BuildPadStatus,
    GameRule,
    GameStatus,
    HistoryEntry,
    Position,
    ResourceDelta,
    RoverState,
    RuleStatus,
    StructureType,
    TerrainCell,
    TerrainType,
    WeatherState,
    WorldState,
)
from .generation import clamp


def payload_used_kg(rover: RoverState) -> float:
    """Ice + ore + samples currently on the rover. Capacity is shared, not per-resource."""
    return round(rover.cargo_ice + rover.cargo_ore + rover.cargo_samples, 2)


def can_fit_payload(rover: RoverState, addition_kg: float) -> bool:
    """True only if the entire addition fits. Scan and Extract never take a partial load."""
    return math.isfinite(addition_kg) and addition_kg >= 0 and payload_used_kg(rover) + addition_kg <= rover.cargo_capacity_kg + 1e-9


def _cell_at(state: WorldState, target: Position | None) -> TerrainCell | None:
    if target is None or target.x < 0 or target.y < 0:
        return None
    if target.y >= state.terrain_height or target.x >= state.terrain_width:
        return None
    return state.terrain[target.y][target.x]


def rover_on_build_pad(state: WorldState) -> bool:
    """Unload and pad trickle-charge require the rover's cell to be `BUILD_PAD`."""
    rover = state.rovers[0]
    return state.terrain[rover.y][rover.x].terrain == TerrainType.BUILD_PAD


def near_build_pad(state: WorldState, target: Position, radius: float) -> bool:
    """Service range check. The rover may stand on or beside the pad, not anywhere on the map."""
    return any(
        cell.terrain == TerrainType.BUILD_PAD and math.hypot(cell.x - target.x, cell.y - target.y) <= radius
        for row in state.terrain
        for cell in row
    )


def needs_service(state: WorldState, config: EngineConfig) -> bool:
    """True when dust, structure health, or a dusty power deficit latches the pad warning."""
    margin = state.resources.power_generated - state.resources.power_consumed
    sustained_power_stress = margin < config.service.power_margin_threshold and state.dust_intensity > config.service.power_dust_threshold
    return (
        state.dust_intensity > config.service.dust_threshold
        or sustained_power_stress
        or any(structure.health < config.service.health_threshold for structure in state.structures)
    )


def _validate_unload(state: WorldState) -> ActionValidation:
    if not rover_on_build_pad(state):
        return ActionValidation(False, "Unload requires the rover to be on the build pad")
    if payload_used_kg(state.rovers[0]) <= 0:
        return ActionValidation(False, "Rover payload is already empty")
    return ActionValidation(True)


def _validate_move(cell: TerrainCell) -> ActionValidation:
    if cell.terrain == TerrainType.CRATER:
        return ActionValidation(False, "Rover cannot move on crater")
    if cell.terrain == TerrainType.RIDGE:
        return ActionValidation(True, warning="Ridge traversal allowed, but battery drain is high")
    if cell.terrain == TerrainType.DUNE:
        return ActionValidation(True, warning="Dune traversal allowed with extra battery drain")
    return ActionValidation(True)


def _validate_scan(state: WorldState, cell: TerrainCell, config: EngineConfig) -> ActionValidation:
    if cell.scanned:
        return ActionValidation(False, "Terrain block has already been scanned")
    if cell.terrain != TerrainType.ROCK:
        return ActionValidation(False, "Scan requires a rock or ore outcrop")
    sample_mass = config.payload.scan_sample_kg
    if not can_fit_payload(state.rovers[0], sample_mass):
        return ActionValidation(False, f"Scan requires {sample_mass:.2f} kg free payload; return to the build pad and Unload")
    return ActionValidation(True)


def _validate_extract(state: WorldState, cell: TerrainCell, config: EngineConfig) -> ActionValidation:
    if cell.terrain != TerrainType.ICE or cell.ice < config.payload.extract_min_ice_signal:
        return ActionValidation(False, "Extract requires an ice deposit")
    extraction_mass = config.payload.ice_extraction_kg
    if not can_fit_payload(state.rovers[0], extraction_mass):
        return ActionValidation(False, f"Extract requires {extraction_mass:.2f} kg free payload; return to the build pad and Unload")
    return ActionValidation(True)


def _validate_build(state: WorldState, cell: TerrainCell) -> ActionValidation:
    if state.objective_stats.habitat_build_progress >= 100:
        return ActionValidation(False, "Habitat construction is already complete")
    if cell.terrain != TerrainType.BUILD_PAD:
        return ActionValidation(False, "Build actions require the landing-zone build pad")
    return ActionValidation(True)


def _validate_service(state: WorldState, target: Position, config: EngineConfig) -> ActionValidation:
    if not near_build_pad(state, target, config.service.service_radius):
        return ActionValidation(False, "Service requires the rover to be at or near the build pad")
    return ActionValidation(True)


def _validate_targeted_action(
    state: WorldState,
    command: ActionCommand,
    config: EngineConfig,
) -> ActionValidation:
    if command.target is None:
        return ActionValidation(False, "Select a target cell first")
    cell = _cell_at(state, command.target)
    if cell is None:
        return ActionValidation(False, "Target is outside the environment")
    action = command.type
    if action == ActionType.MOVE:
        return _validate_move(cell)
    if action == ActionType.SCAN:
        return _validate_scan(state, cell, config)
    if action == ActionType.EXTRACT:
        return _validate_extract(state, cell, config)
    if action == ActionType.BUILD:
        return _validate_build(state, cell)
    if action == ActionType.SERVICE:
        return _validate_service(state, command.target, config)
    return ActionValidation(False, "Unknown action")


def validate_action(state: WorldState, command: ActionCommand, config: EngineConfig) -> ActionValidation:
    """Return whether `command` may run. Does not mutate state or infer a missing target."""
    action = command.type
    if action in {ActionType.INVALID, ActionType.EVENT}:
        return ActionValidation(False, "Unsupported public action")
    if action == ActionType.WAIT:
        return ActionValidation(True)
    if action == ActionType.UNLOAD:
        return _validate_unload(state)
    return _validate_targeted_action(state, command, config)


def _suggest_target(state: WorldState, action: ActionType, config: EngineConfig) -> Position | None:
    if action == ActionType.UNLOAD:
        rover = state.rovers[0]
        return Position(rover.x, rover.y)
    if action not in {ActionType.SCAN, ActionType.EXTRACT, ActionType.BUILD, ActionType.SERVICE, ActionType.MOVE}:
        return None
    rover = state.rovers[0]
    candidates: list[TerrainCell] = []
    for row in state.terrain:
        for cell in row:
            target = Position(cell.x, cell.y)
            if validate_action(state, ActionCommand(action, target), config).valid:
                candidates.append(cell)
    if not candidates:
        return None
    if action == ActionType.EXTRACT:
        candidates.sort(key=lambda cell: (-cell.ice * config.action.extract_ice_priority_weight - cell.ore, math.hypot(cell.x - rover.x, cell.y - rover.y)))
    else:
        candidates.sort(key=lambda cell: math.hypot(cell.x - rover.x, cell.y - rover.y))
    return Position(candidates[0].x, candidates[0].y)


def resolve_command(state: WorldState, command: ActionCommand, config: EngineConfig) -> ActionCommand:
    """Fill in a missing target from the nearest legal cell. Wait needs no target."""
    if command.target is not None or command.type == ActionType.WAIT:
        return command
    return ActionCommand(command.type, _suggest_target(state, command.type, config))


def _wait_recharge(margin: float, config: EngineConfig) -> float:
    return round(max(0, min(config.power.wait_max_charge, margin * config.power.wait_charge_per_kw)), 2)


def _pad_recharge(margin: float, config: EngineConfig) -> float:
    return round(max(0, min(config.power.pad_max_charge, margin * config.power.pad_charge_per_kw)), 2)


def _battery_drain(state: WorldState, action: ActionType, target: Position | None, config: EngineConfig) -> float:
    rover = state.rovers[0]
    cell = _cell_at(state, target) or state.terrain[rover.y][rover.x]
    action_config = config.action
    terrain_stress = action_config.terrain_stress[cell.terrain]
    cargo_stress = payload_used_kg(rover) * action_config.cargo_stress_rate
    weather_stress = action_config.weather_stress[state.weather] + state.dust_intensity * action_config.global_dust_stress_rate
    stress = (
        terrain_stress
        + cell.roughness * action_config.roughness_stress_rate
        + cell.dust * action_config.cell_dust_stress_rate
        + weather_stress
        + action_config.action_stress[action]
        + cargo_stress
    )
    return round(action_config.base_drain[action] * math.exp(stress * action_config.stress_exponent_rate), 2)


def _resource_delta_for(state: WorldState, action: ActionType, target: Position | None, config: EngineConfig) -> ResourceDelta:
    drain = -_battery_drain(state, action, target, config)
    load = -config.action.action_load[action]
    if action == ActionType.MOVE:
        return ResourceDelta(battery=drain, power=load)
    if action == ActionType.SCAN:
        return ResourceDelta(samples=config.payload.scan_sample_kg, battery=drain, power=load)
    if action == ActionType.EXTRACT:
        cell = _cell_at(state, target)
        ice = config.payload.ice_extraction_kg if cell is not None and cell.ice >= config.payload.extract_min_ice_signal else 0
        return ResourceDelta(ice=ice, ore=0, battery=drain, power=load)
    if action == ActionType.BUILD:
        return ResourceDelta(power=load, battery=drain, water=-config.action.build_water_cost, oxygen=-config.action.build_oxygen_cost)
    if action in {ActionType.SERVICE, ActionType.UNLOAD}:
        return ResourceDelta(power=load, battery=drain)
    if action == ActionType.WAIT:
        return ResourceDelta(power=0, battery=0)
    return ResourceDelta()


def _apply_service(state: WorldState, config: EngineConfig) -> str:
    state.objective_stats.service_count += 1
    for structure in state.structures:
        structure.health = config.service.restored_structure_health
        structure.powered = True
        structure.status = "Build pad solar and dust systems serviced" if structure.type == StructureType.SOLAR else "Build pad infrastructure serviced"
    state.dust_intensity = config.service.serviced_dust
    state.build_pad_state.service_needed = False
    return "Servicing build pad infrastructure"


def _apply_build(state: WorldState, config: EngineConfig) -> str:
    stats = state.objective_stats
    stats.habitat_build_count += 1
    stats.habitat_build_progress = min(100, stats.habitat_build_count * (100 / config.action.habitat_build_steps))
    for structure in state.structures:
        if structure.type == StructureType.HABITAT:
            structure.health = min(100, structure.health + config.action.habitat_health_gain)
            structure.powered = True
            structure.status = "Habitat shell upgraded"
    return "Building habitat capacity at build pad"


def _powered_count(state: WorldState, structure_type: StructureType) -> int:
    return sum(structure.type == structure_type and structure.powered for structure in state.structures)


def _solar_generation(state: WorldState, config: EngineConfig) -> float:
    solar_panels = [structure for structure in state.structures if structure.type == StructureType.SOLAR and structure.powered]
    solar_health = sum(panel.health / 100 for panel in solar_panels) / len(solar_panels) if solar_panels else 0
    weather_factor = config.power.weather_generation_factor[state.weather]
    dust_factor = clamp(1 - state.dust_intensity * config.power.dust_generation_multiplier, config.power.dust_generation_floor, 1)
    return max(0, round(len(solar_panels) * config.power.solar_panel_output * solar_health * weather_factor * dust_factor, 1))


def _consumed_power(state: WorldState, action_load: float, config: EngineConfig) -> float:
    return (
        config.power.base_load
        + _powered_count(state, StructureType.HABITAT) * config.power.habitat_load
        + _powered_count(state, StructureType.STORAGE) * config.power.storage_load
        + _powered_count(state, StructureType.BATTERY) * config.power.charger_load
        + action_load
    )


def _apply_battery_from_margin(state: WorldState, action: ActionType, margin: float, config: EngineConfig) -> None:
    if margin < 0:
        drain = min(config.power.deficit_battery_cap, abs(margin) * config.power.deficit_battery_rate)
        state.resources.battery = clamp(state.resources.battery - drain, 0, 100)
        return
    if action == ActionType.WAIT:
        state.resources.battery = clamp(state.resources.battery + _wait_recharge(margin, config), 0, 100)
        return
    if rover_on_build_pad(state):
        state.resources.battery = clamp(state.resources.battery + _pad_recharge(margin, config), 0, 100)


def _apply_power_model(state: WorldState, action: ActionType, power_delta: float, config: EngineConfig) -> None:
    generated = _solar_generation(state, config)
    consumed = _consumed_power(state, max(0, -power_delta), config)
    state.resources.power_generated = generated
    state.resources.power_consumed = round(consumed, 1)
    _apply_battery_from_margin(state, action, generated - consumed, config)


def _apply_life_support(state: WorldState, action: ActionType, config: EngineConfig) -> None:
    life = config.life_support
    power_deficit = max(0, state.resources.power_consumed - state.resources.power_generated)
    water_drain = life.water_base_drain + state.resources.livability * life.water_livability_rate + state.dust_intensity * life.water_dust_rate
    oxygen_drain = life.oxygen_base_drain + state.resources.livability * life.oxygen_livability_rate + power_deficit * life.oxygen_deficit_rate
    state.resources.water = max(0, state.resources.water - water_drain)
    state.resources.oxygen = max(0, state.resources.oxygen - oxygen_drain)
    if action == ActionType.BUILD:
        state.resources.livability = clamp(state.resources.livability + life.build_bonus, 0, 100)


def _process_unload(state: WorldState, action: ActionType, config: EngineConfig) -> ResourceDelta:
    rover = state.rovers[0]
    if action != ActionType.UNLOAD or not rover_on_build_pad(state):
        return ResourceDelta()
    ice, ore, samples = rover.cargo_ice, rover.cargo_ore, rover.cargo_samples
    water = round(ice * config.payload.unload_water_per_ice, 2)
    oxygen = round(ice * config.payload.unload_oxygen_per_ice, 2)
    rover.cargo_ice = rover.cargo_ore = rover.cargo_samples = 0
    state.objective_stats.ice_delivered += ice
    state.objective_stats.samples_delivered += samples
    state.objective_stats.unload_count += 1
    state.resources.water += water
    state.resources.oxygen += oxygen
    state.resources.livability = clamp(state.resources.livability + ice * config.payload.unload_livability_per_ice, 0, 100)
    return ResourceDelta(ice=-ice, ore=-ore, samples=-samples, water=water, oxygen=oxygen)


def _merge_resource_delta(first: ResourceDelta, second: ResourceDelta) -> ResourceDelta:
    values: dict[str, float | None] = {}
    for descriptor in fields(ResourceDelta):
        left = getattr(first, descriptor.name)
        right = getattr(second, descriptor.name)
        values[descriptor.name] = None if left is None and right is None else round((left or 0) + (right or 0), 2)
    return ResourceDelta(**values)


def _livability_delta(state: WorldState, action: ActionType, config: EngineConfig) -> float:
    life = config.life_support
    deficit = max(0, state.resources.power_consumed - state.resources.power_generated)
    reserve_penalty = -life.empty_reserve_penalty if state.resources.water <= 0 or state.resources.oxygen <= 0 else 0
    power_penalty = -min(life.power_deficit_cap, deficit * life.power_deficit_rate) if deficit > 0 else 0
    service_bonus = life.service_bonus if action == ActionType.SERVICE else 0
    return -life.livability_base_decay + reserve_penalty + power_penalty + service_bonus


def _rover_health_delta(state: WorldState, config: EngineConfig) -> float:
    life = config.life_support
    if state.resources.battery <= life.critical_battery_health_threshold:
        return -life.critical_battery_health_decay
    if state.resources.battery <= life.low_battery_health_threshold:
        return -life.low_battery_health_decay
    return -life.normal_health_decay


def _degrade_infrastructure(state: WorldState, config: EngineConfig) -> None:
    damage = config.service.weather_damage[state.weather] + state.dust_intensity * config.service.dust_damage_rate
    for structure in state.structures:
        structure.health = clamp(structure.health - damage, 0, 100)
        if structure.health < config.service.health_threshold:
            structure.status = "Build pad service needed"


def _next_dust_intensity(state: WorldState, config: EngineConfig) -> float:
    weather_delta = config.service.weather_dust_delta[state.weather]
    return clamp(state.dust_intensity + weather_delta, config.service.dust_floor, config.service.dust_ceiling)


def _energy_term(battery_cost: float, config: EngineConfig) -> float:
    return -battery_cost * config.reward.energy_rate


def _move_reward(state: WorldState, target: Position | None, delta: ResourceDelta, config: EngineConfig) -> dict[str, float]:
    cell = _cell_at(state, target)
    on_pad = cell is not None and cell.terrain == TerrainType.BUILD_PAD
    hazardous = cell is not None and cell.terrain in {TerrainType.RIDGE, TerrainType.DUNE}
    reward = config.reward
    return {
        "traversal": reward.pad_traversal if on_pad else reward.normal_traversal,
        "energy": _energy_term(abs(delta.battery or 0), config),
        "safety": reward.hazardous_move if hazardous else reward.safe_move,
    }


def _scan_reward(state: WorldState, target: Position | None, delta: ResourceDelta, config: EngineConfig) -> dict[str, float]:
    reward = config.reward
    return {"information": reward.scan_information, "exploration": reward.scan_exploration, "energy": _energy_term(abs(delta.battery or 0), config)}


def _extract_reward(state: WorldState, target: Position | None, delta: ResourceDelta, config: EngineConfig) -> dict[str, float]:
    reward = config.reward
    return {
        "resource": (delta.ice or 0) * reward.ice_resource_per_kg + (delta.ore or 0) * reward.ore_resource_per_kg,
        "energy": _energy_term(abs(delta.battery or 0), config),
    }


def _build_reward(state: WorldState, target: Position | None, delta: ResourceDelta, config: EngineConfig) -> dict[str, float]:
    reward = config.reward
    return {"infrastructure": reward.build_infrastructure, "livability": reward.build_livability, "energy": _energy_term(abs(delta.battery or 0), config)}


def _service_reward(state: WorldState, target: Position | None, delta: ResourceDelta, config: EngineConfig) -> dict[str, float]:
    reward = config.reward
    return {
        "service": reward.service_needed if needs_service(state, config) else reward.service_base,
        "efficiency": reward.service_efficiency,
        "energy": _energy_term(abs(delta.battery or 0), config),
    }


def _unload_reward(state: WorldState, target: Position | None, delta: ResourceDelta, config: EngineConfig) -> dict[str, float]:
    reward = config.reward
    return {
        "delivery": abs(delta.ice or 0) * reward.ice_delivery_per_kg + abs(delta.ore or 0) * reward.ore_delivery_per_kg + abs(delta.samples or 0) * reward.sample_delivery_per_kg,
        "logistics": reward.unload_logistics,
        "energy": _energy_term(abs(delta.battery or 0), config),
    }


def _wait_reward(state: WorldState, target: Position | None, delta: ResourceDelta, config: EngineConfig) -> dict[str, float]:
    surplus = state.resources.power_generated >= state.resources.power_consumed
    return {"recovery": config.reward.wait_surplus if surplus else config.reward.wait_deficit, "energy": 0}


_REWARD_FOR_ACTION = {
    ActionType.MOVE: _move_reward,
    ActionType.SCAN: _scan_reward,
    ActionType.EXTRACT: _extract_reward,
    ActionType.BUILD: _build_reward,
    ActionType.SERVICE: _service_reward,
    ActionType.UNLOAD: _unload_reward,
}


def _reward_for(
    state: WorldState,
    action: ActionType,
    target: Position | None,
    delta: ResourceDelta,
    config: EngineConfig,
) -> dict[str, float]:
    builder = _REWARD_FOR_ACTION.get(action, _wait_reward)
    return builder(state, target, delta, config)


_REWARD_TOTAL_FIELDS = {
    ActionType.EXTRACT: "ice_collected",
    ActionType.SCAN: "terrain_scanned",
    ActionType.BUILD: "habitat_built",
    ActionType.SERVICE: "serviced",
    ActionType.UNLOAD: "delivered",
    ActionType.MOVE: "traversal",
    ActionType.INVALID: "blocked_penalty",
}


def _add_reward_total(state: WorldState, action: ActionType, reward: float) -> None:
    totals = state.objective_stats.reward_totals
    rounded_reward = round(reward, 2)
    field = _REWARD_TOTAL_FIELDS.get(action)
    if field is not None:
        setattr(totals, field, round(getattr(totals, field) + rounded_reward, 2))
    totals.total = round(totals.total + rounded_reward, 2)


def _action_label(action: ActionType) -> str:
    return action.value.replace("_", " ")


def _wait_events(state: WorldState, action: ActionType, config: EngineConfig) -> list[str]:
    if action != ActionType.WAIT:
        return []
    recharge = _wait_recharge(state.resources.power_generated - state.resources.power_consumed, config)
    if recharge > 0:
        return [f"System: Wait charging cycle restored {recharge:.2f}% battery from surplus power"]
    return ["System: Wait could not recharge because power margin is not positive"]


def _processed_ice_reserves(unload_delta: ResourceDelta) -> bool:
    return (unload_delta.water or 0) > 0 or (unload_delta.oxygen or 0) > 0


def _unload_events(action: ActionType, unload_delta: ResourceDelta) -> list[str]:
    events: list[str] = []
    if _processed_ice_reserves(unload_delta):
        events.append(f"Base processed ice cargo into {round(unload_delta.water or 0)} L water and {round(unload_delta.oxygen or 0)} L O2")
    if action == ActionType.UNLOAD:
        events.append(
            f"Payload delivered: {abs(unload_delta.ice or 0)} kg ice, {abs(unload_delta.samples or 0)} kg samples, and {abs(unload_delta.ore or 0)} kg ore"
        )
    return events


def _system_events(state: WorldState, config: EngineConfig) -> list[str]:
    events: list[str] = []
    if needs_service(state, config):
        events.append("System: Build pad service needed")
    if state.resources.power_generated < state.resources.power_consumed:
        events.append(
            f"System: Power deficit is draining rover battery ({state.resources.power_generated} generated / {state.resources.power_consumed} consumed)"
        )
    return events


def _events_for(
    action: ActionType,
    warning: str | None,
    state: WorldState,
    unload_delta: ResourceDelta,
    config: EngineConfig,
) -> list[str]:
    events = [f"{_action_label(action)} completed"]
    if "Dust" in state.weather.value:
        events.append("Dust reduced solar efficiency")
    if warning:
        events.append(f"Warning: {warning}")
    events.extend(_wait_events(state, action, config))
    events.extend(_unload_events(action, unload_delta))
    events.extend(_system_events(state, config))
    return events


def _sync_build_pad_state(state: WorldState, config: EngineConfig, service_needed: bool | None = None) -> None:
    needed = needs_service(state, config) if service_needed is None else service_needed
    habitat_built = state.objective_stats.habitat_build_progress >= 100
    state.build_pad_state.service_needed = needed
    state.build_pad_state.status = (
        BuildPadStatus.HABITAT_BUILT_NEEDS_SERVICE
        if habitat_built and needed
        else BuildPadStatus.HABITAT_BUILT
        if habitat_built
        else BuildPadStatus.NEEDS_SERVICE
        if needed
        else BuildPadStatus.NORMAL
    )


def _rule_status(*, failed: bool = False, warning: bool = False) -> RuleStatus:
    if failed:
        return RuleStatus.FAILED
    if warning:
        return RuleStatus.WARNING
    return RuleStatus.STABLE


def _game_over_reason(battery_failed: bool, health_failed: bool) -> str:
    if battery_failed:
        return "Exploration ended..... Rover battery depleted"
    if health_failed:
        return "Exploration ended..... Rover can no longer operate"
    return "Exploration ended..... Habitat livability collapsed"


def _build_game_rules(state: WorldState, config: EngineConfig, service_needed: bool) -> list[GameRule]:
    rover = state.rovers[0]
    used_payload = payload_used_kg(rover)
    warning = config.warning
    return [
        GameRule("oxygen", "Colony O2 reserve", "Base life-support oxygen belongs to the colony system, not the unmanned rover.", _rule_status(warning=state.resources.oxygen < warning.oxygen_reserve), f"{state.resources.oxygen:.2f} L"),
        GameRule("water", "Colony water reserve", "Water is stored and processed by base systems for habitat readiness.", _rule_status(warning=state.resources.water < warning.water_reserve), f"{state.resources.water:.2f} L"),
        GameRule("battery", "Battery survival", "The rover can no longer operate when stored power is exhausted.", _rule_status(failed=state.resources.battery <= 0, warning=state.resources.battery < warning.rover_battery), f"{state.resources.battery:.2f}%"),
        GameRule("health", "Rover health", "Dust, low power, terrain hazards, and system stress damage the active rover.", _rule_status(failed=rover.health <= 0, warning=rover.health < warning.rover_health), f"{rover.health:.2f}%"),
        GameRule("livability", "Habitat livability", "Open exploration continues until rover battery, rover health, or habitat livability fails.", _rule_status(failed=state.resources.livability <= 0, warning=state.resources.livability < warning.livability), f"{state.resources.livability:.2f}% livability"),
        GameRule("payload", "Rover payload", "Scan and Extract require free payload capacity. Return to the build pad and use Unload when full.", _rule_status(warning=used_payload >= rover.cargo_capacity_kg), f"{used_payload:.2f} / {rover.cargo_capacity_kg:.2f} kg"),
        GameRule("service", "Build pad service", "Service is needed when dust, damaged infrastructure, or poor power margin threatens the build pad.", _rule_status(warning=service_needed), "Needed" if service_needed else "Nominal"),
    ]


def _apply_terminal_status(state: WorldState, battery_failed: bool, health_failed: bool, livability_failed: bool) -> None:
    if battery_failed or health_failed or livability_failed:
        state.game_status = GameStatus.GAME_OVER
        state.status_reason = _game_over_reason(battery_failed, health_failed)
        return
    if state.game_status != GameStatus.PAUSED:
        state.game_status = GameStatus.RUNNING
        state.status_reason = "Open exploration active"


def _mission_alerts(state: WorldState) -> list[str]:
    warnings = [f"{rule.label}: {rule.value}" for rule in state.rules if rule.status in {RuleStatus.WARNING, RuleStatus.FAILED}]
    visible_warnings = [warning for warning in warnings if not warning.startswith("Build pad service:")]
    if state.game_status == GameStatus.GAME_OVER:
        return [state.status_reason, *visible_warnings[:2]]
    baseline = "Solar output stable" if state.resources.power_generated >= state.resources.power_consumed else "Power demand exceeds generation"
    fallback = visible_warnings or ["Explore until battery or livability fails"]
    return [baseline, *fallback][:3]


def _apply_game_rules(state: WorldState, config: EngineConfig, action: ActionType | None = None) -> None:
    rover = state.rovers[0]
    battery_failed = state.resources.battery <= 0
    health_failed = rover.health <= 0
    livability_failed = state.resources.livability <= 0
    service_needed = False if action == ActionType.SERVICE else state.build_pad_state.service_needed or needs_service(state, config)
    _sync_build_pad_state(state, config, service_needed)
    state.rules = _build_game_rules(state, config, service_needed)
    _apply_terminal_status(state, battery_failed, health_failed, livability_failed)
    state.mission.alerts = _mission_alerts(state)


def initialize_state(state: WorldState, config: EngineConfig) -> None:
    """Fill power meters and HUD rules on a freshly generated world before the first step."""
    _apply_power_model(state, ActionType.EVENT, 0, config)
    _apply_game_rules(state, config)


def _execute_move(state: WorldState, command: ActionCommand, config: EngineConfig) -> str:
    rover = state.rovers[0]
    if command.target is not None:
        rover.x, rover.y = command.target.x, command.target.y
    return "Traversing planned route"


def _execute_scan(state: WorldState, command: ActionCommand, config: EngineConfig) -> str:
    rover = state.rovers[0]
    rover.cargo_samples = round(rover.cargo_samples + config.payload.scan_sample_kg, 2)
    cell = _cell_at(state, command.target)
    if cell is not None:
        cell.scanned = True
    state.objective_stats.terrain_scanned += 1
    state.objective_stats.samples_collected += config.payload.scan_sample_kg
    return "Surveying and storing a geological sample"


def _execute_extract(state: WorldState, command: ActionCommand, config: EngineConfig) -> str:
    rover = state.rovers[0]
    rover.cargo_ice = round(rover.cargo_ice + config.payload.ice_extraction_kg, 2)
    cell = _cell_at(state, command.target)
    if cell is not None:
        cell.ice = clamp(cell.ice - config.payload.ice_channel_reduction, 0, 1)
        cell.extracted = True
        cell.terrain = TerrainType.REGOLITH
    state.objective_stats.ice_collected += config.payload.ice_extraction_kg
    state.objective_stats.ice_sites_extracted += 1
    return "Extracting resource sample"


def _execute_build(state: WorldState, command: ActionCommand, config: EngineConfig) -> str:
    return _apply_build(state, config)


def _execute_service(state: WorldState, command: ActionCommand, config: EngineConfig) -> str:
    return _apply_service(state, config)


def _execute_unload(state: WorldState, command: ActionCommand, config: EngineConfig) -> str:
    return "Unloading payload at build pad"


_TARGETED_EXECUTORS = {
    ActionType.MOVE: _execute_move,
    ActionType.SCAN: _execute_scan,
    ActionType.EXTRACT: _execute_extract,
}
_PAD_EXECUTORS = {
    ActionType.BUILD: _execute_build,
    ActionType.SERVICE: _execute_service,
    ActionType.UNLOAD: _execute_unload,
}


def _apply_rover_effects(state: WorldState, command: ActionCommand, validation: ActionValidation, config: EngineConfig) -> None:
    rover = state.rovers[0]
    if not validation.valid:
        rover.current_task = f"Blocked: {validation.reason}"
        return
    action = command.type
    targeted = _TARGETED_EXECUTORS.get(action)
    if targeted is not None and command.target is not None:
        rover.current_task = targeted(state, command, config)
        return
    pad_executor = _PAD_EXECUTORS.get(action)
    if pad_executor is not None:
        rover.current_task = pad_executor(state, command, config)
        return
    rover.current_task = _action_label(action)


def _advance_world_clock(state: WorldState, config: EngineConfig) -> None:
    state.step += 1
    state.sol = 1 + state.step // config.world.steps_per_sol
    minutes = config.world.start_time_minutes + state.step * config.world.minutes_per_step
    state.local_time = f"{(minutes // 60) % 24:02d}:{minutes % 60:02d}"
    state.weather = config.world.weather_cycle[abs(state.seed + state.step) % len(config.world.weather_cycle)]
    state.dust_intensity = _next_dust_intensity(state, config)
    _degrade_infrastructure(state, config)


def _apply_valid_tick(
    state: WorldState,
    action: ActionType,
    delta: ResourceDelta,
    config: EngineConfig,
) -> ResourceDelta:
    state.resources.battery = clamp(state.resources.battery + (delta.battery or 0), 0, 100)
    state.resources.water = max(0, state.resources.water + (delta.water or 0))
    state.resources.oxygen = max(0, state.resources.oxygen + (delta.oxygen or 0))
    _apply_power_model(state, action, delta.power or 0, config)
    _apply_life_support(state, action, config)
    unload_delta = _process_unload(state, action, config)
    state.resources.livability = clamp(state.resources.livability + _livability_delta(state, action, config), 0, 100)
    state.rovers[0].health = clamp(state.rovers[0].health + _rover_health_delta(state, config), 0, 100)
    return unload_delta


def _append_history(
    state: WorldState,
    command: ActionCommand,
    actor: Actor,
    effective_action: ActionType,
    validation: ActionValidation,
    reward: float,
    reward_terms: dict[str, float],
    final_delta: ResourceDelta,
    events: list[str],
) -> None:
    target_suffix = f" at {command.target.x}, {command.target.y}" if command.target is not None else ""
    result = f"{_action_label(command.type)}{target_suffix}" if validation.valid else f"{_action_label(command.type)} blocked: {validation.reason}"
    state.history.insert(
        0,
        HistoryEntry(
            id=f"hist_{state.step}_{effective_action.value}",
            step=state.step,
            actor=actor,
            action=effective_action,
            target=command.target,
            result=result,
            reward=reward,
            reward_terms=reward_terms,
            resource_delta=final_delta,
            events=events,
        ),
    )


def apply_action(state: WorldState, command: ActionCommand, actor: Actor, config: EngineConfig) -> tuple[ActionType, float, dict[str, float], list[str]]:
    """Mutate `state` by one command and return effective action, reward, terms, and events.

    Always advances the clock, weather, dust, and history. Blocked commands become
    `INVALID` with no resource spend. Game-over sessions ignore further commands.
    """
    if state.game_status == GameStatus.GAME_OVER:
        return command.type, 0, {}, []
    command = resolve_command(state, command, config)
    validation = validate_action(state, command, config)
    action = command.type
    effective_action = action if validation.valid else ActionType.INVALID
    delta = _resource_delta_for(state, action, command.target, config) if validation.valid else ResourceDelta()
    _apply_rover_effects(state, command, validation, config)
    _advance_world_clock(state, config)
    unload_delta = _apply_valid_tick(state, action, delta, config) if validation.valid else ResourceDelta()
    final_delta = _merge_resource_delta(delta, unload_delta)
    state.rovers[0].battery = state.resources.battery
    reward_terms = _reward_for(state, action, command.target, final_delta, config) if validation.valid else {"invalid": config.reward.invalid, "safety": config.reward.invalid_safety}
    reward = sum(reward_terms.values())
    _add_reward_total(state, effective_action, reward)
    events = _events_for(action, validation.warning, state, unload_delta, config) if validation.valid else ["Action blocked by environment rules", validation.reason]
    _append_history(state, command, actor, effective_action, validation, reward, reward_terms, final_delta, events)
    _apply_game_rules(state, config, action if validation.valid else effective_action)
    return effective_action, reward, reward_terms, events
