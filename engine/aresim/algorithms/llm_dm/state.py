"""Encode permitted rover observations and public game requirements as named JSON.

The encoder has no access to WorldState, replay snapshots, seeds or checksums.
Normalization and mission thresholds come from the resolved environment config.
"""

from __future__ import annotations

import numpy as np

from ...components.actions import ACTION_NAMES, ACTION_SCHEMA
from ...components.observations import (
    OBSERVATION_SCHEMA,
    MEMORY_OBSERVATION_SCHEMA,
    TERRAIN_IDS,
    WEATHER_IDS,
)
from ...config import EnvironmentConfig
from .contracts import Json

TERRAINS = {
    0: "unknown",
    **{code: terrain.value for terrain, code in TERRAIN_IDS.items()},
}
WEATHERS = {
    0: "unknown",
    **{
        code: kind.value.lower().replace(" ", "_") for kind, code in WEATHER_IDS.items()
    },
}
PADS = ("outside_service_range", "within_service_range", "on_build_pad")
OBJECTIVES = {2: "scan", 3: "extract_ice", 4: "deliver_ice", 6: "service", 7: "survive"}


def observation_schema(config: EnvironmentConfig) -> str:
    return (
        MEMORY_OBSERVATION_SCHEMA
        if config.observation == "local_memory"
        else OBSERVATION_SCHEMA
    )


def _number(value: object, scale: float = 1) -> float:
    result = float(value) * scale
    if not np.isfinite(result):
        raise ValueError("nonfinite observation channel")
    return round(result, 6)


def require_mask(value: np.ndarray) -> np.ndarray:
    """Reject malformed/no-choice masks before any paid request."""
    mask = np.asarray(value)
    if (
        mask.shape != (len(ACTION_NAMES),)
        or not np.all((mask == 0) | (mask == 1))
        or not np.any(mask)
    ):
        raise ValueError("expected a nonempty binary ten-action mask")
    return mask.astype(np.int8, copy=True)


def encode_state(observation: dict, config: EnvironmentConfig) -> Json:
    """Project all observable channels with explicit padding and remembered-field provenance."""
    oc = config.observation_config
    size = oc.window_size
    shapes = {
        "self": (10,),
        "colony": (14,),
        "terrain_type": (size, size),
        "spatial": (5, size, size),
        "cell_flags": (4, size, size),
        "objective_type": (oc.max_objectives,),
        "objectives": (oc.max_objectives, 4),
        "objective_mask": (oc.max_objectives,),
    }
    for key, shape in shapes.items():
        array = observation.get(key)
        if (
            not isinstance(array, np.ndarray)
            or array.shape != shape
            or not np.all(np.isfinite(array))
        ):
            raise ValueError(f"invalid {key} observation; expected finite {shape}")
    rover, colony = observation["self"], observation["colony"]
    maximum = config.engine.world.size - 1
    position = {
        "x": int(round(float(rover[0]) * maximum)),
        "y": int(round(float(rover[1]) * maximum)),
    }
    for axis, index in (("x", 0), ("y", 1)):
        if (
            not 0 <= position[axis] <= maximum
            or abs(position[axis] / maximum - float(rover[index])) > 1e-5
        ):
            raise ValueError("invalid normalized rover coordinate")
    capacity = config.engine.payload.capacity_kg
    pad = observation.get("pad_proximity")
    weather = observation.get("weather_type")
    if (
        not isinstance(pad, (int, np.integer))
        or int(pad) not in range(3)
        or int(weather) not in WEATHERS
    ):
        raise ValueError("unknown pad/weather enum")
    anchor = (size - 1) // 2
    cells = []
    for row in range(size):
        for col in range(size):
            flags = observation["cell_flags"][:, row, col]
            if not np.all((flags == 0) | (flags == 1)):
                raise ValueError("cell flags must be binary")
            in_bounds, visible = bool(flags[0]), bool(flags[1])
            if visible and not in_bounds:
                raise ValueError("visible out-of-bounds cell")
            kind = int(observation["terrain_type"][row, col])
            if kind not in TERRAINS:
                raise ValueError("unknown terrain ID")
            cell = {
                "row": row,
                "column": col,
                "dx": col - anchor,
                "dy": row - anchor,
                "in_bounds": in_bounds,
                "visible": visible,
                "terrain": TERRAINS[kind] if visible else "unknown",
            }
            for index, name in enumerate(("height", "roughness", "ice", "ore", "dust")):
                cell[name] = (
                    _number(observation["spatial"][index, row, col])
                    if visible
                    else None
                )
            cell.update(
                scanned=bool(flags[2]) if visible else None,
                extracted=bool(flags[3]) if visible else None,
            )
            cells.append(cell)
    objectives = []
    for slot, present in enumerate(observation["objective_mask"]):
        if int(present) != 1:
            continue
        kind = int(observation["objective_type"][slot])
        if kind not in OBJECTIVES:
            raise ValueError("unknown live objective type")
        row = observation["objectives"][slot]
        objectives.append(
            {
                "slot": slot,
                "type_id": kind,
                "type_name": OBJECTIVES[kind],
                "current_fraction": _number(row[0]),
                "target_fraction": _number(row[1]),
                "remaining_fraction": _number(row[2]),
                "required": bool(row[3] >= 0.5),
            }
        )
    navigation = None
    if config.observation == "local_memory":
        nav, visited = observation.get("navigation"), observation.get("visited")
        if (
            not isinstance(nav, np.ndarray)
            or nav.shape != (13,)
            or not np.all(np.isfinite(nav))
        ):
            raise ValueError("expected finite navigation[13]")
        if not isinstance(visited, np.ndarray) or visited.shape != (1, size, size):
            raise ValueError("expected visited[1,H,W]")
        navigation = {}
        for key, offset in (("home", 0), ("ice", 5), ("unscanned_rock", 8)):
            valid = bool(nav[offset + 2] >= 0.5)
            navigation[key] = {
                "dx": int(round(float(nav[offset]) * maximum)) if valid else None,
                "dy": int(round(float(nav[offset + 1]) * maximum)) if valid else None,
                "valid": valid,
                "age_steps": None,
                "source": "observation_memory",
            }
        navigation["previous_displacement"] = {
            "dx": int(round(float(nav[3]) * maximum)),
            "dy": int(round(float(nav[4]) * maximum)),
        }
        navigation["home_position_cells"] = (
            {
                "x": int(round(float(nav[11]) * maximum)),
                "y": int(round(float(nav[12]) * maximum)),
            }
            if nav[2] >= 0.5
            else None
        )
        navigation["visited"] = visited[0].astype(bool).tolist()
    return {
        "schema_version": "aresim.llm.state.v1",
        "observation_schema": observation_schema(config),
        "rover": {
            "position_cells": position,
            "battery_pct": _number(rover[2], 100),
            "health_pct": _number(rover[3], 100),
            "cargo": {
                name: _number(rover[index], capacity)
                for index, name in enumerate(
                    ("ice_kg", "ore_kg", "samples_kg", "free_kg"), 4
                )
            }
            | {"capacity_kg": capacity},
            "pad_proximity": PADS[int(pad)],
        },
        "colony": {
            "power": {
                name: _number(colony[index], oc.power_scale)
                for index, name in enumerate(("generated", "consumed", "margin"))
            },
            "battery_pct": _number(colony[3], 100),
            "water_reserve": _number(colony[4], oc.water_scale),
            "oxygen_reserve": _number(colony[5], oc.oxygen_scale),
            "livability_pct": _number(colony[6], 100),
            "dust_intensity_pct": _number(colony[7], 100),
            "habitat_progress_pct": _number(colony[8], 100),
            "pad_needs_service": bool(colony[12] >= 0.5),
        },
        "time": {"day_sin": _number(rover[8]), "day_cos": _number(rover[9])},
        "weather": WEATHERS[int(weather)],
        "objectives": objectives,
        "local_grid": {
            "width": size,
            "height": size,
            "anchor_row": anchor,
            "anchor_column": anchor,
            "cells": cells,
        },
        "navigation": navigation,
    }


def public_requirements(config: EnvironmentConfig) -> Json:
    """Public rules and configured quantities shared by Actor, Learner, logs and bundle identity."""
    payload, action = config.engine.payload, config.engine.action
    return {
        "schema_version": "aresim.llm.rules.v1",
        "task": config.task,
        "task_id": "phase1_resource_mission_v2"
        if config.task == "resource_mission"
        else "phase1_open_exploration_v1",
        "world": {
            "width": config.engine.world.size,
            "height": config.engine.world.size,
            "rovers": 1,
            "coordinates": "x increases east; y increases south; north is dy=-1; crop is row-major",
        },
        "mission": {
            "task_id": "phase1_resource_mission_v2"
            if config.task == "resource_mission"
            else "phase1_open_exploration_v1",
            "ice_target_kg": config.task_config.ice_target_kg,
            "sample_target_kg": config.task_config.sample_target_kg,
            "require_pad_safe": config.task_config.require_pad_safe,
            "success": "Deliver both enabled quotas and meet pad safety while surviving."
            if config.task == "resource_mission"
            else "No victory condition; explore and survive until external cutoff or failure.",
            "failure": "Engine battery, rover health, or colony livability depleted. Failure takes precedence over mission completion.",
        },
        "actions": [
            {"id": i, "name": name, "effect": effect}
            for i, (name, effect) in enumerate(
                zip(
                    ACTION_NAMES,
                    (
                        "Stay for one tick. Charging requires ON the pad and positive power margin; next-tick weather can change the margin. Away from pad, surplus alone does not charge.",
                        "Move one cell north (dy=-1).",
                        "Move one cell east (dx=1).",
                        "Move one cell south (dy=1).",
                        "Move one cell west (dx=-1).",
                        f"Scan the current unscanned ROCK and collect {payload.scan_sample_kg:g} kg samples if that much cargo space is free. Does not reveal the hidden map.",
                        f"Extract {payload.ice_extraction_kg:g} kg ICE at the current eligible cell with enough free cargo. Minimum ice signal {payload.extract_min_ice_signal:g}. Cell becomes regolith and extracted=true. Does not mine ore.",
                        f"Build on the current build-pad cell until habitat progress reaches 100%; requires {action.habitat_build_steps} successful build steps from zero. Uses reserves. Habitat completion alone is not mission victory.",
                        f"Service pad infrastructure within {config.engine.service.service_radius:g} cells of the pad; can clear service-needed latch. Legal service need not be useful.",
                        "Unload ALL ice, ore and samples while ON a build-pad cell with nonempty cargo. Only delivered ice/samples count toward quotas.",
                    ),
                )
            )
        ],
        "cargo_capacity_kg": payload.capacity_kg,
        "operating_requirements": {
            "minutes_per_tick": config.engine.world.minutes_per_step,
            "build_water_cost": action.build_water_cost,
            "build_oxygen_cost": action.build_oxygen_cost,
            "ice_unload_conversion_per_kg": {
                "water": payload.unload_water_per_ice,
                "oxygen": payload.unload_oxygen_per_ice,
                "livability": payload.unload_livability_per_ice,
            },
            "wait_charge_on_pad": {
                "per_positive_power_unit": config.engine.power.wait_charge_per_kw,
                "maximum_percent_per_tick": config.engine.power.wait_max_charge,
            },
            "other_action_pad_charge_max_percent": config.engine.power.pad_max_charge,
            "warning_thresholds": {
                "battery_pct": config.engine.warning.rover_battery,
                "rover_health_pct": config.engine.warning.rover_health,
                "livability_pct": config.engine.warning.livability,
                "water_reserve": config.engine.warning.water_reserve,
                "oxygen_reserve": config.engine.warning.oxygen_reserve,
            },
            "service": "Use the observed service-needed latch; do not invent hidden structure health. Repeated service while safe spends time without delivering resources.",
            "limits": "Battery, health and reserves are bounded telemetry. Life support, health and infrastructure can deteriorate as time advances, even while waiting. Weather and dust affect power and traversal costs.",
        },
        "survival": "All actions advance weather, dust, power and life support. Craters block movement; ridges/dunes may be legal but costly. Legal does not mean safe. Wait is not a guaranteed recovery action.",
        "observation": "Only the configured local crop plus public telemetry is current. Unknown/padded cell values are null, not empty terrain. No off-crop resource/pad location is known unless previously observed in episode memory. Remembered resources may be stale. Current visibility supersedes old facts.",
        "telemetry": "Battery/health/livability/dust/habitat are percentages. Water and oxygen are bounded denormalized reserves, NOT percentages. Power uses engine units. Day sine/cosine is periodic local time, not an elapsed-sol counter.",
        "objectives": "extract_ice tracks collection, deliver_ice tracks delivery, scan tracks DELIVERED sample quota, service tracks latch safety, survive tracks current battery. Progress is clipped; it does not prove exact unbounded totals.",
        "history": "Only completed transitions precede this decision. Proposed moves do not confirm movement. Clear coordinates/history on episode reset; never transfer map coordinates across seeds.",
        "reward": "RL reward and engine/UI reward are different. Positive shaped reward is not proof of mission success. Acquiring cargo is not delivering it. External step cutoff is not a hidden gameplay deadline.",
        "action_schema": ACTION_SCHEMA,
    }
