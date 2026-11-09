"""Own the provider-independent packet shared by LLM and decision-model actors.

Encoders and episode memory supply permitted data; this module only assembles it.
No simulator state, provider client or output-selection policy belongs here.
"""

from __future__ import annotations

import numpy as np

from ...components.actions import ACTION_NAMES
from .masks import require_mask


def _legal_actions(
    mask: np.ndarray,
    rules: dict[str, object],
    state: dict[str, object],
    history: dict[str, object],
) -> list[dict[str, object]]:
    """Describe choices using visible cells and measured history without ranking them."""
    effects = {row["name"]: row["effect"] for row in rules["actions"]}
    summary = history.get("summary") or {}
    home = summary.get("observed_home")
    trace = summary.get("recent_trace", [])
    position = state["rover"]["position_cells"]
    cells = {(cell["dx"], cell["dy"]): cell for cell in state["local_grid"]["cells"]}
    offsets = {
        "move_north": (0, -1),
        "move_east": (1, 0),
        "move_south": (0, 1),
        "move_west": (-1, 0),
    }
    actions = []
    for index, name in enumerate(ACTION_NAMES):
        if not mask[index]:
            continue
        entry = {"id": index, "name": name, "effect": effects[name]}
        if name in offsets:
            dx, dy = offsets[name]
            destination = {"x": position["x"] + dx, "y": position["y"] + dy}
            cell = cells.get((dx, dy))
            entry["destination"] = destination
            entry["observed_destination"] = (
                {
                    key: cell[key]
                    for key in (
                        "visible",
                        "terrain",
                        "ice",
                        "ore",
                        "scanned",
                        "extracted",
                    )
                }
                if cell is not None
                else None
            )
            entry["recent_arrivals_at_destination"] = sum(
                row["to"] == destination and row["from"] != row["to"] for row in trace
            )
            last = trace[-1] if trace else None
            entry["immediately_reverses_displacement"] = bool(
                last and last["from"] == destination and last["from"] != last["to"]
            )
            if home:
                point = home["position_cells"]
                entry["observed_home_manhattan_distance_after"] = abs(
                    point["x"] - destination["x"]
                ) + abs(point["y"] - destination["y"])
                entry["observed_home_distance_change"] = (
                    entry["observed_home_manhattan_distance_after"]
                    - home["manhattan_distance"]
                )
        elif name == "wait":
            entry["on_pad"] = state["rover"]["pad_proximity"] == "on_build_pad"
            entry["positive_power_margin_now"] = state["colony"]["power"]["margin"] > 0
            entry["battery_headroom_pct"] = round(
                max(0, 100 - state["rover"]["battery_pct"]), 6
            )
        actions.append(entry)
    return actions


def decision_packet(
    *,
    episode_id: str,
    decision_id: str,
    step_index: int,
    policy_bundle_id: str,
    rules: dict[str, object],
    state: dict[str, object],
    action_schema: str,
    action_mask: np.ndarray,
    history: dict[str, object],
    lessons: list[dict[str, object]],
    learnings: list[dict[str, object]],
) -> dict[str, object]:
    """Keep the same named state, legal actions and past-only context for both actors."""
    mask = require_mask(action_mask).astype(np.int8, copy=False)
    if not np.any(mask):
        raise ValueError("decision packet requires at least one legal action")
    return {
        "schema_version": "aresim.llm.packet.v1",
        "episode_id": episode_id,
        "decision_id": decision_id,
        "step_index": step_index,
        "policy_bundle_id": policy_bundle_id,
        "mission": rules["mission"],
        "state": state,
        "action_schema": action_schema,
        "action_mask": mask.tolist(),
        "legal_actions": _legal_actions(mask, rules, state, history),
        "history": history,
        "lessons": lessons,
        "learnings": learnings,
    }
