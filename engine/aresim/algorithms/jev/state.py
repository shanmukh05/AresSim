"""Encode local observations into named ``aresim.llm.ops.v1`` JSON for Jev.

Projects the rover-centered ``8 x 8`` policy crop into a 5x5 list of named cells
with rover-relative ``(dx, dy)`` and per-cell percents. Does not read canonical
``WorldState``.

**Last updated:** September 20, 2026

**Contains:** action-name tables, ``OpsScales``, :func:`encode_ops_state`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TypeAlias

import numpy as np

from ...components.actions import ACTION_COUNT
from ...components.observations import OBSERVATION_SCHEMA, TERRAIN_IDS, WEATHER_IDS
from ...defaults import DEFAULT_ENVIRONMENT_CONFIG
from ..common.masks import require_mask


OPS_SCHEMA = "aresim.llm.ops.v1"
DEFAULT_NEIGHBORHOOD_SIZE = 5

ACTION_NAMES: tuple[str, ...] = (
    "wait",
    "move_north",
    "move_east",
    "move_south",
    "move_west",
    "scan",
    "extract",
    "build",
    "service",
    "unload",
)
ACTION_IDS: dict[str, int] = {name: index for index, name in enumerate(ACTION_NAMES)}
MOVE_OFFSETS: dict[str, tuple[int, int]] = {
    "north": (0, -1),
    "east": (1, 0),
    "south": (0, 1),
    "west": (-1, 0),
}
MOVE_DIRECTIONS: dict[tuple[int, int], str] = {offset: name for name, offset in MOVE_OFFSETS.items()}

TERRAIN_NAMES: dict[int, str] = {
    0: "unknown",
    **{code: terrain.value for terrain, code in TERRAIN_IDS.items()},
}
WEATHER_NAMES: dict[int, str] = {
    0: "unknown",
    **{code: weather.value.lower().replace(" ", "_") for weather, code in WEATHER_IDS.items()},
}
PAD_NAMES: dict[int, str] = {
    0: "outside_service_range",
    1: "within_service_range",
    2: "on_build_pad",
}
OBJECTIVE_NAMES: dict[int, str] = {
    0: "padding",
    1: "navigate",
    2: "scan",
    3: "extract_ice",
    4: "deliver_ice",
    5: "build",
    6: "service",
    7: "survive",
    8: "maintain_resource",
}

MISSION_PRIORITIES: tuple[str, ...] = (
    "Survive: battery/health/livability",
    "Unload on the pad when the bay is full or the rover is already on the pad with cargo",
    "Service pad if it needs service and rover is in range",
    "Build on pad",
    "Extract ice / scan rocks when standing on them and the bay has room",
    "Move toward prospects.best_ice or best_ore using first_move; do not camp the pad while deposits are visible",
    "If recent actions are repeating, leave the region and explore a new heading",
)

PolicyObservation: TypeAlias = Mapping[str, np.ndarray | int]


@dataclass(frozen=True)
class OpsScales:
    """Observation denormalization constants copied from environment config."""

    payload_capacity_kg: float
    power_scale: float
    water_scale: float
    oxygen_scale: float
    neighborhood_size: int = DEFAULT_NEIGHBORHOOD_SIZE


def default_ops_scales(neighborhood_size: int = DEFAULT_NEIGHBORHOOD_SIZE) -> OpsScales:
    """Return Phase 1 default scales for tests and fallback construction."""
    observation = DEFAULT_ENVIRONMENT_CONFIG.observation_config
    return OpsScales(
        payload_capacity_kg=DEFAULT_ENVIRONMENT_CONFIG.engine.payload.capacity_kg,
        power_scale=observation.power_scale,
        water_scale=observation.water_scale,
        oxygen_scale=observation.oxygen_scale,
        neighborhood_size=neighborhood_size,
    )


def encode_ops_state(
    observation: PolicyObservation,
    action_mask: np.ndarray,
    *,
    recent_actions: tuple[str, ...] = (),
    remembered_pad: tuple[int, int] | None = None,
    scales: OpsScales | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Return ``(state, legal_criteria)`` for one Jev Choice call.

    ``legal_criteria`` maps action names to structured option descriptions and
    contains only mask-legal actions.
    """
    mask = require_mask(action_mask)
    resolved = default_ops_scales() if scales is None else scales
    arrays = _policy_arrays(observation)
    cells = _grid_cells(arrays, resolved.neighborhood_size)
    here = _cell_at(cells, 0, 0)
    colony = _colony(arrays["colony"], observation.get("weather_type"), resolved)
    rover = _rover(arrays["self"], here, observation.get("pad_proximity"), resolved, remembered_pad)
    wait_recharges = float(colony["power_margin"]) > 0
    rover["wait_recharges"] = wait_recharges
    cargo = rover["cargo"]
    free_kg = float(cargo["free_kg"]) if isinstance(cargo, dict) else 0.0
    rover["should_unload"] = _cargo_mass_kg(rover) > 0 and (rover["pad"] == "on_build_pad" or free_kg <= 0)
    prospects = _prospects(cells)
    state: dict[str, object] = {
        "schema_version": OPS_SCHEMA,
        "observation_schema": OBSERVATION_SCHEMA,
        "mission": {"priorities": list(MISSION_PRIORITIES)},
        "rover": rover,
        "colony": colony,
        "objectives": _objectives(observation),
        "grid": {"size": resolved.neighborhood_size, "rover": {"dx": 0, "dy": 0}, "cells": cells},
        "adjacent": _adjacent(mask),
        "recent_actions": list(recent_actions),
        "recent_repeating": _recent_repeating(recent_actions),
        "prospects": prospects,
    }
    return state, _legal_criteria(mask, here, cells, wait_recharges, prospects)


def _policy_arrays(observation: PolicyObservation) -> dict[str, np.ndarray]:
    self_vector = observation.get("self")
    colony = observation.get("colony")
    terrain = observation.get("terrain_type")
    spatial = observation.get("spatial")
    flags = observation.get("cell_flags")
    if not isinstance(self_vector, np.ndarray) or self_vector.shape != (10,):
        raise ValueError("jev encoder requires self float[10]")
    if not isinstance(colony, np.ndarray) or colony.shape != (14,):
        raise ValueError("jev encoder requires colony float[14]")
    if not isinstance(terrain, np.ndarray) or terrain.ndim != 2 or terrain.shape[0] != terrain.shape[1]:
        raise ValueError("jev encoder requires square terrain_type")
    if not isinstance(spatial, np.ndarray) or spatial.shape != (5, *terrain.shape):
        raise ValueError("jev encoder requires spatial float[5,H,W]")
    if not isinstance(flags, np.ndarray) or flags.shape != (4, *terrain.shape):
        raise ValueError("jev encoder requires cell_flags uint8[4,H,W]")
    return {"self": self_vector, "colony": colony, "terrain": terrain, "spatial": spatial, "flags": flags}


def _percent(value: float) -> int:
    return int(round(float(np.clip(value, 0.0, 1.0) * 100)))


def _grid_cells(arrays: dict[str, np.ndarray], neighborhood_size: int) -> list[dict[str, object]]:
    window = int(arrays["terrain"].shape[0])
    anchor = (window - 1) // 2
    radius = neighborhood_size // 2
    cells: list[dict[str, object]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            cells.append(_one_cell(arrays, anchor + dx, anchor + dy, dx, dy, window))
    return cells


def _one_cell(arrays: dict[str, np.ndarray], local_x: int, local_y: int, dx: int, dy: int, window: int) -> dict[str, object]:
    in_window = 0 <= local_x < window and 0 <= local_y < window
    known = bool(in_window and arrays["flags"][0, local_y, local_x] == 1)
    cell: dict[str, object] = {
        "dx": dx,
        "dy": dy,
        "steps": abs(dx) + abs(dy),
        "in_bounds": known,
        "terrain": "unknown",
        "ice_pct": 0,
        "ore_pct": 0,
        "dust_pct": 0,
        "height_pct": 0,
        "roughness_pct": 0,
        "scanned": False,
        "extracted": False,
        "is_rover": dx == 0 and dy == 0,
    }
    if known:
        cell["terrain"] = TERRAIN_NAMES.get(int(arrays["terrain"][local_y, local_x]), "unknown")
        cell["height_pct"] = _percent(float(arrays["spatial"][0, local_y, local_x]))
        cell["roughness_pct"] = _percent(float(arrays["spatial"][1, local_y, local_x]))
        cell["ice_pct"] = _percent(float(arrays["spatial"][2, local_y, local_x]))
        cell["ore_pct"] = _percent(float(arrays["spatial"][3, local_y, local_x]))
        cell["dust_pct"] = _percent(float(arrays["spatial"][4, local_y, local_x]))
        cell["scanned"] = bool(arrays["flags"][2, local_y, local_x])
        cell["extracted"] = bool(arrays["flags"][3, local_y, local_x])
    direction = MOVE_DIRECTIONS.get((dx, dy))
    if direction is not None:
        cell["move_direction"] = direction
    return cell


def _cell_at(cells: list[dict[str, object]], dx: int, dy: int) -> dict[str, object]:
    for cell in cells:
        if cell["dx"] == dx and cell["dy"] == dy:
            return cell
    raise KeyError(f"missing grid cell dx={dx} dy={dy}")


def _rover(
    self_vector: np.ndarray,
    here: dict[str, object],
    pad_proximity: object,
    scales: OpsScales,
    remembered_pad: tuple[int, int] | None,
) -> dict[str, object]:
    capacity = scales.payload_capacity_kg
    ice_kg = round(float(self_vector[4]) * capacity, 2)
    ore_kg = round(float(self_vector[5]) * capacity, 2)
    samples_kg = round(float(self_vector[6]) * capacity, 2)
    rover: dict[str, object] = {
        "relative_origin": {"dx": 0, "dy": 0, "note": "negative dy is north, positive dx is east"},
        "here": {"terrain": here["terrain"], "ice_pct": here["ice_pct"], "ore_pct": here["ore_pct"], "scanned": here["scanned"], "extracted": here["extracted"]},
        "battery_pct": _percent(float(self_vector[2])),
        "health_pct": _percent(float(self_vector[3])),
        "cargo": {
            "ice_kg": ice_kg,
            "ore_kg": ore_kg,
            "samples_kg": samples_kg,
            "free_kg": round(float(self_vector[7]) * capacity, 2),
            "capacity_kg": capacity,
        },
        "pad": PAD_NAMES.get(int(pad_proximity) if isinstance(pad_proximity, (int, np.integer)) else 0, "outside_service_range"),
    }
    if remembered_pad is not None:
        rover["remembered_pad"] = {"dx": remembered_pad[0], "dy": remembered_pad[1]}
    return rover


def _colony(colony: np.ndarray, weather_type: object, scales: OpsScales) -> dict[str, object]:
    weather_id = int(weather_type) if isinstance(weather_type, (int, np.integer)) else 0
    return {
        "power_generated": round(float(colony[0]) * scales.power_scale, 2),
        "power_consumed": round(float(colony[1]) * scales.power_scale, 2),
        "power_margin": round(float(colony[2]) * scales.power_scale, 2),
        "battery_pct": _percent(float(colony[3])),
        "water_pct": _percent(float(colony[4])),
        "oxygen_pct": _percent(float(colony[5])),
        "livability_pct": _percent(float(colony[6])),
        "dust_intensity_pct": _percent(float(colony[7])),
        "habitat_progress_pct": _percent(float(colony[8])),
        "pad_needs_service": bool(colony[12] >= 0.5),
        "weather": WEATHER_NAMES.get(weather_id, "unknown"),
    }


def _objectives(observation: PolicyObservation) -> list[dict[str, object]]:
    types = observation.get("objective_type")
    rows = observation.get("objectives")
    mask = observation.get("objective_mask")
    if not isinstance(types, np.ndarray) or not isinstance(rows, np.ndarray) or not isinstance(mask, np.ndarray):
        return []
    encoded: list[dict[str, object]] = []
    for index, present in enumerate(mask.tolist()):
        if int(present) != 1:
            continue
        encoded.append(
            {
                "type": OBJECTIVE_NAMES.get(int(types[index]), "padding"),
                "current_pct": _percent(float(rows[index, 0])),
                "target_pct": _percent(float(rows[index, 1])),
                "remaining_pct": _percent(float(rows[index, 2])),
                "required": bool(rows[index, 3] >= 0.5),
            }
        )
    return encoded


def _cargo_mass_kg(rover: dict[str, object]) -> float:
    cargo = rover.get("cargo")
    if not isinstance(cargo, dict):
        return 0.0
    return float(cargo.get("ice_kg") or 0) + float(cargo.get("ore_kg") or 0) + float(cargo.get("samples_kg") or 0)


def _recent_repeating(recent_actions: tuple[str, ...]) -> bool:
    if len(recent_actions) >= 3 and recent_actions[-1] == recent_actions[-2] == recent_actions[-3]:
        return True
    return len(recent_actions) >= 4 and recent_actions[-4:-2] == recent_actions[-2:]


def _adjacent(mask: np.ndarray) -> dict[str, object]:
    return {
        name: {"dx": dx, "dy": dy, "legal": bool(mask[ACTION_IDS[f"move_{name}"]] == 1)}
        for name, (dx, dy) in MOVE_OFFSETS.items()
    }


def _first_move(dx: int, dy: int) -> str | None:
    """Name the adjacent step that reduces Manhattan distance to (dx, dy)."""
    if dx == 0 and dy == 0:
        return None
    if abs(dx) >= abs(dy):
        return "move_east" if dx > 0 else "move_west"
    return "move_south" if dy > 0 else "move_north"


def _best_resource(cells: list[dict[str, object]], key: str) -> dict[str, object] | None:
    best: dict[str, object] | None = None
    best_score = (0, 0)
    for cell in cells:
        if not cell.get("in_bounds") or cell.get("extracted"):
            continue
        pct = int(cell.get(key) or 0)
        if pct <= 0:
            continue
        score = (pct, -int(cell["steps"]))
        if best is None or score > best_score:
            best = cell
            best_score = score
    return best


def _prospect_entry(cell: dict[str, object]) -> dict[str, object]:
    dx = int(cell["dx"])
    dy = int(cell["dy"])
    return {
        "dx": dx,
        "dy": dy,
        "ice_pct": cell["ice_pct"],
        "ore_pct": cell["ore_pct"],
        "terrain": cell["terrain"],
        "steps": cell["steps"],
        "first_move": _first_move(dx, dy),
        "extract_here": dx == 0 and dy == 0,
    }


def _prospects(cells: list[dict[str, object]]) -> dict[str, object]:
    """Name the richest ice and ore cells so Jev does not have to search the 5x5."""
    named: dict[str, object] = {}
    ice = _best_resource(cells, "ice_pct")
    ore = _best_resource(cells, "ore_pct")
    if ice is not None:
        named["best_ice"] = _prospect_entry(ice)
    if ore is not None:
        named["best_ore"] = _prospect_entry(ore)
    return named


def _legal_criteria(
    mask: np.ndarray,
    here: dict[str, object],
    cells: list[dict[str, object]],
    wait_recharges: bool,
    prospects: Mapping[str, object],
) -> dict[str, object]:
    ice = prospects.get("best_ice") if isinstance(prospects, Mapping) else None
    ore = prospects.get("best_ore") if isinstance(prospects, Mapping) else None
    ice_move = ice.get("first_move") if isinstance(ice, dict) else None
    ore_move = ore.get("first_move") if isinstance(ore, dict) else None
    criteria: dict[str, object] = {}
    for name, (dx, dy) in MOVE_OFFSETS.items():
        action_name = f"move_{name}"
        if mask[ACTION_IDS[action_name]] != 1:
            continue
        destination = _cell_at(cells, dx, dy)
        criteria[action_name] = {
            "effect": f"Move one cell {name}.",
            "destination": destination,
            "toward_best_ice": action_name == ice_move,
            "toward_best_ore": action_name == ore_move,
        }
    work = {
        "scan": "Scan the rock under the rover and pick up a geological sample.",
        "extract": "Extract ice from the cell under the rover.",
        "build": "Advance habitat build progress on the build pad.",
        "service": "Service nearby build-pad infrastructure.",
        "unload": "Unload the entire payload on the build pad: ice, ore, and samples together.",
    }
    for name, effect in work.items():
        if mask[ACTION_IDS[name]] == 1:
            criteria[name] = {"effect": effect, "here": here}
    if mask[0] == 1:
        if wait_recharges:
            wait_effect = "Stay in place. Colony power_margin is positive, so Wait charges the colony battery."
        else:
            wait_effect = (
                "Stay in place. Colony power_margin is not positive, so Wait does not recharge. "
                "Prefer any other legal action."
            )
        criteria["wait"] = {"effect": wait_effect, "here": here, "recharges": wait_recharges}
    if not criteria:
        raise ValueError("action mask contains no legal actions")
    if len(criteria) > ACTION_COUNT:
        raise ValueError("legal criteria exceeded action count")
    return criteria


__all__ = [
    "ACTION_IDS",
    "ACTION_NAMES",
    "DEFAULT_NEIGHBORHOOD_SIZE",
    "OPS_SCHEMA",
    "OpsScales",
    "default_ops_scales",
    "encode_ops_state",
]
