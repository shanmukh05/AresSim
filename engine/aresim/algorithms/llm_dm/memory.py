"""Deterministic observed-cell facts and bounded completed-transition history.

All facts retain episode/source identity and observation time. Proposed actions
never update the map; only observed rover positions and cells do.
"""

from __future__ import annotations

from copy import deepcopy

from .config import HistoryConfig
from .contracts import Json, canonical


def telemetry(state: Json) -> Json:
    """Small named before/after projection for prompts, evidence and human timelines."""
    return {
        key: deepcopy(state[key])
        for key in ("rover", "colony", "weather", "time", "objectives")
    }


class EpisodeMemory:
    """An exact episode-local map, with independent recent and fact retrieval limits."""

    def __init__(self, config: HistoryConfig) -> None:
        self.config = config
        self.reset("")

    def reset(self, episode_id: str) -> None:
        self.episode_id = episode_id
        self.facts: dict[tuple[int, int], Json] = {}
        self.recent: list[Json] = []
        self.through = -1
        self.last_observation = -1
        self.stationary_steps = 0
        self.wait_steps = 0
        self.no_collection_or_delivery_steps = 0

    def observe(self, state: Json, index: int) -> None:
        """Apply a permitted observation once; current cells supersede old sightings."""
        if index == self.last_observation:
            return
        if index != self.last_observation + 1:
            raise ValueError("out-of-order memory observation")
        for fact in self.facts.values():
            fact["currently_visible"] = False
        pos = state["rover"]["position_cells"]
        for cell in state["local_grid"]["cells"]:
            if not cell["visible"] or not cell["in_bounds"]:
                continue
            point = (pos["x"] + cell["dx"], pos["y"] + cell["dy"])
            previous = self.facts.get(point)
            value = {
                k: cell[k]
                for k in (
                    "terrain",
                    "height",
                    "roughness",
                    "ice",
                    "ore",
                    "dust",
                    "scanned",
                    "extracted",
                )
            }
            self.facts[point] = {
                "id": f"{self.episode_id}:cell:{point[0]}:{point[1]}",
                "episode_id": self.episode_id,
                "kind": "cell_observation",
                "position_cells": {"x": point[0], "y": point[1]},
                "value": value,
                "first_seen_step": previous["first_seen_step"] if previous else index,
                "last_seen_step": index,
                "last_visited_step": previous["last_visited_step"]
                if previous
                else None,
                "source_observation_id": f"{self.episode_id}:observation:{index}",
                "currently_visible": True,
            }
        self.facts[(pos["x"], pos["y"])]["last_visited_step"] = index
        self.last_observation = index

    def complete(
        self,
        transition_id: str,
        index: int,
        before: Json,
        after: Json,
        action: str,
        fallback: str | None,
        reward: float | None,
        terminated: bool,
        truncated: bool,
        reason: str | None,
    ) -> None:
        """Commit one real transition; preserve only the configured recent window in memory."""
        if index != self.through + 1:
            raise ValueError("out-of-order completed transition")
        self.stationary_steps = (
            self.stationary_steps + 1
            if before["rover"]["position_cells"] == after["rover"]["position_cells"]
            else 0
        )
        self.wait_steps = self.wait_steps + 1 if action == "wait" else 0
        previous = {
            row["type_name"]: row["current_fraction"] for row in before["objectives"]
        }
        delivered = any(
            row["current_fraction"]
            > previous.get(row["type_name"], row["current_fraction"])
            for row in after["objectives"]
            if row["type_name"] in {"deliver_ice", "scan"}
        )
        collected = any(
            after["rover"]["cargo"][key] > before["rover"]["cargo"][key]
            for key in ("ice_kg", "samples_kg")
        )
        self.no_collection_or_delivery_steps = (
            0 if collected or delivered else self.no_collection_or_delivery_steps + 1
        )
        self.recent.append(
            {
                "source_transition_id": transition_id,
                "step_index": index,
                "issued_action": action,
                "fallback": fallback,
                "rl_reward": reward,
                "before": telemetry(before),
                "after": telemetry(after),
                "terminated": terminated,
                "truncated": truncated,
                "ending_reason": reason,
            }
        )
        self.recent = self.recent[-self.config.recent_limit :]
        self.through = index
        self.observe(after, index + 1)

    def _summary(self, state: Json) -> Json:
        """Expose compact measured progress and observed home coordinates, never a chosen action."""
        position = state["rover"]["position_cells"]
        pads = (
            [
                fact
                for fact in self.facts.values()
                if fact["value"]["terrain"] == "build_pad"
            ]
            if self.config.mode == "facts_recent"
            else []
        )

        def distance(fact: Json) -> int:
            return abs(fact["position_cells"]["x"] - position["x"]) + abs(
                fact["position_cells"]["y"] - position["y"]
            )

        pad = (
            min(
                pads,
                key=lambda fact: (
                    distance(fact),
                    fact["position_cells"]["y"],
                    fact["position_cells"]["x"],
                ),
            )
            if pads
            else None
        )
        home = (
            None
            if pad is None
            else {
                "position_cells": deepcopy(pad["position_cells"]),
                "dx": pad["position_cells"]["x"] - position["x"],
                "dy": pad["position_cells"]["y"] - position["y"],
                "manhattan_distance": distance(pad),
                "last_seen_step": pad["last_seen_step"],
                "source_observation_id": pad["source_observation_id"],
                "note": "Nearest observed pad cell; straight-line grid distance is not a verified route or energy estimate.",
            }
        )
        trace = [
            {
                "source_transition_id": row["source_transition_id"],
                "step_index": row["step_index"],
                "action": row["issued_action"],
                "from": deepcopy(row["before"]["rover"]["position_cells"]),
                "to": deepcopy(row["after"]["rover"]["position_cells"]),
                "battery_before_pct": row["before"]["rover"]["battery_pct"],
                "battery_after_pct": row["after"]["rover"]["battery_pct"],
            }
            for row in self.recent
        ]
        reversals = sum(
            a["from"] == b["to"] and a["to"] == b["from"] and a["from"] != a["to"]
            for a, b in zip(trace, trace[1:])
        )
        return {
            "recent_trace": trace,
            "recent_trace_omitted": self.through + 1 - len(trace),
            "immediate_reversals_in_recent_trace": reversals,
            "consecutive_stationary_steps": self.stationary_steps,
            "consecutive_waits": self.wait_steps,
            "turns_without_collection_or_delivery": self.no_collection_or_delivery_steps,
            "observed_home": home,
            "known_resource_cells": {
                "ice": sum(
                    f["value"]["terrain"] == "ice" and not f["value"]["extracted"]
                    for f in self.facts.values()
                ),
                "unscanned_rock": sum(
                    f["value"]["terrain"] == "rock" and not f["value"]["scanned"]
                    for f in self.facts.values()
                ),
            }
            if self.config.mode == "facts_recent"
            else None,
        }

    def retrieve(self, state: Json, *, max_bytes: int | None = None) -> Json:
        """Rank deterministically; disclose every omission instead of truncating JSON text."""
        byte_limit = (
            self.config.max_bytes
            if max_bytes is None
            else min(self.config.max_bytes, max_bytes)
        )
        result = {
            "schema_version": "aresim.llm.history.v1",
            "episode_id": self.episode_id,
            "mode": self.config.mode,
            "through_step_index": self.through,
            "recent_transitions": [],
            "facts": [],
            "summary": self._summary(state) if self.config.mode != "none" else None,
            "omitted": {
                "transitions": max(0, self.through + 1),
                "facts": len(self.facts),
            },
        }
        if self.config.mode != "none":
            result["recent_transitions"] = deepcopy(self.recent)
        if self.config.mode == "facts_recent":
            pos = state["rover"]["position_cells"]

            def rank(fact: Json) -> tuple:
                point = fact["position_cells"]
                return (
                    abs(point["x"] - pos["x"]) + abs(point["y"] - pos["y"]),
                    -fact["last_seen_step"],
                    point["y"],
                    point["x"],
                )

            # Visible cells are already represented exactly in state.local_grid.
            facts = sorted(
                (fact for fact in self.facts.values() if not fact["currently_visible"]),
                key=rank,
            )
            groups = [
                ([f for f in facts if f["value"]["terrain"] == "build_pad"], 1),
                (
                    [
                        f
                        for f in facts
                        if f["value"]["terrain"] == "ice"
                        and not f["value"]["extracted"]
                    ],
                    4,
                ),
                (
                    [
                        f
                        for f in facts
                        if f["value"]["terrain"] == "rock" and not f["value"]["scanned"]
                    ],
                    4,
                ),
                (
                    sorted(
                        [f for f in facts if f["last_visited_step"] is not None],
                        key=lambda f: (-f["last_visited_step"], rank(f)),
                    ),
                    self.config.fact_limit,
                ),
            ]
            selected = {}
            for candidates, limit in groups:
                for fact in candidates[:limit]:
                    selected[fact["id"]] = fact
            result["facts"] = deepcopy(
                list(selected.values())[: self.config.fact_limit]
            )
        while True:
            result["omitted"] = {
                "transitions": self.through + 1 - len(result["recent_transitions"]),
                "facts": len(self.facts) - len(result["facts"]),
            }
            if len(canonical(result).encode()) <= byte_limit:
                return result
            if result["recent_transitions"]:
                result["recent_transitions"].pop(0)
            elif result["facts"]:
                result["facts"].pop()
            elif result["summary"] and result["summary"]["recent_trace"]:
                result["summary"]["recent_trace"].pop(0)
                result["summary"]["recent_trace_omitted"] += 1
                trace = result["summary"]["recent_trace"]
                result["summary"]["immediate_reversals_in_recent_trace"] = sum(
                    a["from"] == b["to"]
                    and a["to"] == b["from"]
                    and a["from"] != a["to"]
                    for a, b in zip(trace, trace[1:])
                )
            elif result["summary"] is not None:
                result["summary"] = None
            else:
                raise ValueError(
                    "history byte allowance cannot hold its mandatory envelope"
                )
