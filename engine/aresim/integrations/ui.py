"""Map authoritative :class:`~aresim.types.WorldState` into camelCase JSON for the React app.

Field names must stay aligned with ``web/src/types/sim.ts``. This is a projection,
not a second simulator.

**Last updated:** September 1, 2026

**Contains:** :func:`snapshot_from_state`.

**Consumers:** :mod:`aresim.service`, :mod:`aresim.api`, trajectory replay builders.

**Constraint:** no gameplay semantics; display-only adaptation of engine numbers.
"""

from __future__ import annotations

from ..types import (
    GameRule,
    HistoryEntry,
    ObjectiveStats,
    ResourceDelta,
    ResourceState,
    RoverState,
    StructureState,
    TerrainCell,
    WorldState,
)


def _resource_delta(delta: ResourceDelta) -> dict[str, float]:
    values = {
        "power": delta.power,
        "battery": delta.battery,
        "water": delta.water,
        "oxygen": delta.oxygen,
        "ice": delta.ice,
        "ore": delta.ore,
        "samples": delta.samples,
    }
    return {key: value for key, value in values.items() if value is not None}


def _history_entry(entry: HistoryEntry) -> dict[str, object]:
    result: dict[str, object] = {
        "id": entry.id,
        "step": entry.step,
        "actor": entry.actor.value,
        "action": entry.action.value,
        "result": entry.result,
        "reward": entry.reward,
        "rewardTerms": dict(entry.reward_terms),
        "resourceDelta": _resource_delta(entry.resource_delta),
        "events": list(entry.events),
    }
    if entry.target is not None:
        result["target"] = {"x": entry.target.x, "y": entry.target.y}
    return result


def _terrain_cell(cell: TerrainCell) -> dict[str, object]:
    return {
        "x": cell.x,
        "y": cell.y,
        "terrain": cell.terrain.value,
        "height": cell.height,
        "roughness": cell.roughness,
        "ice": cell.ice,
        "ore": cell.ore,
        "dust": cell.dust,
        "scanned": cell.scanned,
        "extracted": cell.extracted,
    }


def _rover(rover: RoverState) -> dict[str, object]:
    return {
        "id": rover.id,
        "name": rover.name,
        "x": rover.x,
        "y": rover.y,
        "battery": rover.battery,
        "health": rover.health,
        "cargoIce": rover.cargo_ice,
        "cargoOre": rover.cargo_ore,
        "cargoSamples": rover.cargo_samples,
        "cargoCapacityKg": rover.cargo_capacity_kg,
        "currentTask": rover.current_task,
    }


def _structure(structure: StructureState) -> dict[str, object]:
    return {
        "id": structure.id,
        "type": structure.type.value,
        "name": structure.name,
        "x": structure.x,
        "y": structure.y,
        "health": structure.health,
        "powered": structure.powered,
        "status": structure.status,
    }


def _resources(resources: ResourceState) -> dict[str, float]:
    return {
        "powerGenerated": resources.power_generated,
        "powerConsumed": resources.power_consumed,
        "battery": resources.battery,
        "water": resources.water,
        "oxygen": resources.oxygen,
        "livability": resources.livability,
    }


def _objective_stats(stats: ObjectiveStats) -> dict[str, object]:
    totals = stats.reward_totals
    return {
        "iceCollected": stats.ice_collected,
        "iceDelivered": stats.ice_delivered,
        "samplesCollected": stats.samples_collected,
        "samplesDelivered": stats.samples_delivered,
        "unloadCount": stats.unload_count,
        "iceSitesExtracted": stats.ice_sites_extracted,
        "iceSitesTotal": stats.ice_sites_total,
        "terrainScanned": stats.terrain_scanned,
        "rockSitesTotal": stats.rock_sites_total,
        "habitatBuildProgress": stats.habitat_build_progress,
        "habitatBuildCount": stats.habitat_build_count,
        "serviceCount": stats.service_count,
        "rewardTotals": {
            "iceCollected": totals.ice_collected,
            "terrainScanned": totals.terrain_scanned,
            "habitatBuilt": totals.habitat_built,
            "serviced": totals.serviced,
            "delivered": totals.delivered,
            "traversal": totals.traversal,
            "blockedPenalty": totals.blocked_penalty,
            "total": totals.total,
        },
    }


def _rule(rule: GameRule) -> dict[str, str]:
    return {
        "id": rule.id,
        "label": rule.label,
        "description": rule.description,
        "status": rule.status.value,
        "value": rule.value,
    }


def snapshot_from_state(state: WorldState) -> dict[str, object]:
    """Build one UI snapshot. Callers own copying; this walks the current state as-is."""
    return {
        "sessionId": state.session_id,
        "seed": state.seed,
        "step": state.step,
        "sol": state.sol,
        "localTime": state.local_time,
        "mode": state.mode,
        "gameStatus": state.game_status.value,
        "statusReason": state.status_reason,
        "terrainSize": {"width": state.terrain_width, "height": state.terrain_height},
        "weather": state.weather.value,
        "dustIntensity": state.dust_intensity,
        "resources": _resources(state.resources),
        "mission": {
            "title": state.mission.title,
            "objective": state.mission.objective,
            "rewardObjectives": list(state.mission.reward_objectives),
            "alerts": list(state.mission.alerts),
        },
        "objectiveStats": _objective_stats(state.objective_stats),
        "buildPadState": {
            "serviceNeeded": state.build_pad_state.service_needed,
            "status": state.build_pad_state.status.value,
        },
        "rules": [_rule(rule) for rule in state.rules],
        "terrain": [[_terrain_cell(cell) for cell in row] for row in state.terrain],
        "rovers": [_rover(rover) for rover in state.rovers],
        "structures": [_structure(structure) for structure in state.structures],
        "history": [_history_entry(entry) for entry in state.history],
    }

