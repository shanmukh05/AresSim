"""Observed-only goal working memory; the Actor chooses goals and primitive actions.

The complete episode map and learning provenance remain outside the action prompt.
Routing measures observed connectivity, never hidden terrain or energy feasibility.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy

from ...components.actions import ACTION_NAMES
from .contracts import Json, canonical, digest
from .knowledge import conditions_match
from .memory import EpisodeMemory, telemetry

DIRECTIONS = {
    "move_north": (0, -1),
    "move_east": (1, 0),
    "move_south": (0, 1),
    "move_west": (-1, 0),
}


def point(state: Json) -> tuple[int, int]:
    pos = state["rover"]["position_cells"]
    return pos["x"], pos["y"]


class FocusedContext:
    """Project exact facts into a small goal menu and goal-conditioned action view."""

    def __init__(
        self, memory: EpisodeMemory, rules: Json, learnings: list[Json]
    ) -> None:
        self.memory, self.rules, self.learnings = memory, rules, learnings
        self.reset()

    def reset(self) -> None:
        self.goal = None
        self.started = self.reviewed = self.last_progress = 0
        self.best_distance = None
        self.last_reason = None
        self.selections = 0
        self.reviewed_opportunities = set()
        self.reviewed_battery_warning = False

    def _distances(self, origin: tuple[int, int]) -> dict[tuple[int, int], int]:
        fact = self.memory.facts.get(origin)
        if not fact or fact["value"]["terrain"] in {"unknown", "crater"}:
            return {}
        distances = {origin: 0}
        queue = deque([origin])
        while queue:
            x, y = queue.popleft()
            for dx, dy in DIRECTIONS.values():
                adjacent = (x + dx, y + dy)
                fact = self.memory.facts.get(adjacent)
                if (
                    adjacent not in distances
                    and fact
                    and fact["value"]["terrain"] not in {"unknown", "crater"}
                ):
                    distances[adjacent] = distances[(x, y)] + 1
                    queue.append(adjacent)
        return distances

    def route(self, origin: tuple[int, int], target: tuple[int, int]) -> Json:
        distances = self._distances(target)
        distance = distances.get(origin)
        return {
            "distance_steps": distance,
            "first_hops": [
                name
                for name, (dx, dy) in DIRECTIONS.items()
                if distance is not None
                and distances.get((origin[0] + dx, origin[1] + dy)) == distance - 1
            ],
            "basis": "observed_passable_cells; energy_cost_unknown",
        }

    def deficits(self, state: Json) -> Json:
        remaining = {
            row["type_name"]: row["remaining_fraction"] for row in state["objectives"]
        }
        cargo = state["rover"]["cargo"]
        ice = self.rules["mission"]["ice_target_kg"] * remaining.get("deliver_ice", 0)
        samples = self.rules["mission"]["sample_target_kg"] * remaining.get("scan", 0)
        return {
            "delivery_remaining_kg": {
                "ice": round(ice, 4),
                "samples": round(samples, 4),
            },
            "additional_collection_needed_kg": {
                "ice": round(max(0, ice - cargo["ice_kg"]), 4),
                "samples": round(max(0, samples - cargo["samples_kg"]), 4),
            },
        }

    def _useful(self, goal: Json, state: Json) -> bool:
        target = tuple(goal["position"])
        fact = self.memory.facts.get(target)
        if not fact:
            return False
        value, kind = fact["value"], goal["kind"]
        cargo, needs = (
            state["rover"]["cargo"],
            self.deficits(state)["additional_collection_needed_kg"],
        )
        if kind == "collect_ice":
            return (
                needs["ice"] > 0
                and value["terrain"] == "ice"
                and not value["extracted"]
                and cargo["free_kg"] >= goal["collection_kg"]
            )
        if kind == "collect_samples":
            return (
                needs["samples"] > 0
                and value["terrain"] == "rock"
                and not value["scanned"]
                and cargo["free_kg"] >= goal["collection_kg"]
            )
        if kind == "deliver":
            return any(cargo[key] > 0 for key in ("ice_kg", "ore_kg", "samples_kg"))
        if kind == "service":
            return state["colony"]["pad_needs_service"]
        if kind == "recover":
            return state["rover"]["battery_pct"] < 85
        if kind == "explore":
            return target != point(state) and fact["last_visited_step"] is None
        return True

    def candidates(self, state: Json, payload: object) -> list[Json]:
        """Bound and diversify reachable targets without choosing the Actor's objective."""
        distances = self._distances(point(state))
        groups: dict[str, list[Json]] = {}
        width, height = self.rules["world"]["width"], self.rules["world"]["height"]
        for pos, fact in self.memory.facts.items():
            if pos not in distances:
                continue
            value, terrain = fact["value"], fact["value"]["terrain"]
            kinds = []
            if terrain == "ice" and value["ice"] >= payload.extract_min_ice_signal:
                kinds.append(("collect_ice", payload.ice_extraction_kg))
            if terrain == "rock":
                kinds.append(("collect_samples", payload.scan_sample_kg))
            if terrain == "build_pad":
                kinds.extend((kind, None) for kind in ("deliver", "service", "recover"))
            unknown_edges = sum(
                0 <= pos[0] + dx < width
                and 0 <= pos[1] + dy < height
                and (pos[0] + dx, pos[1] + dy) not in self.memory.facts
                for dx, dy in DIRECTIONS.values()
            )
            if unknown_edges:
                kinds.append(("explore", None))
            for kind, amount in kinds:
                goal = {
                    "id": f"{kind}:{pos[0]},{pos[1]}",
                    "kind": kind,
                    "position": list(pos),
                }
                if amount is not None:
                    goal["collection_kg"] = amount
                if kind == "explore":
                    goal["unknown_adjacent_cells"] = unknown_edges
                if self._useful(goal, state):
                    groups.setdefault(kind, []).append(goal)
        # Round-robin keeps e.g. nearby ice from displacing every exploration option.
        for values in groups.values():
            values.sort(
                key=lambda g: (
                    distances[tuple(g["position"])],
                    -g.get("unknown_adjacent_cells", 0),
                    g["id"],
                )
            )
        selected = []
        while groups and len(selected) < self.memory.config.target_limit:
            for kind in list(groups):
                if len(selected) == self.memory.config.target_limit:
                    break
                selected.append(groups[kind].pop(0))
                if not groups[kind]:
                    del groups[kind]
        if (
            self.goal
            and self._useful(self.goal, state)
            and tuple(self.goal["position"]) in distances
            and self.goal["id"] not in {g["id"] for g in selected}
        ):
            selected = selected[: max(0, self.memory.config.target_limit - 1)] + [
                deepcopy(self.goal)
            ]
        if not selected:
            selected = [
                {
                    "id": "observe_survive",
                    "kind": "observe_survive",
                    "position": list(point(state)),
                }
            ]
        return [
            {
                **goal,
                "route": self.route(point(state), tuple(goal["position"])),
                "last_seen_steps_ago": self.memory.last_observation
                - self.memory.facts[tuple(goal["position"])]["last_seen_step"],
            }
            for goal in selected
        ]

    def review_reason(self, state: Json, candidates: list[Json]) -> str | None:
        step = self.memory.last_observation
        if self.goal is None:
            return "initial_goal"
        if not self._useful(self.goal, state):
            return "completed_or_no_longer_useful"
        distance = self.route(point(state), tuple(self.goal["position"]))[
            "distance_steps"
        ]
        if distance is None:
            return "observed_route_unavailable"
        if self.best_distance is None or distance < self.best_distance:
            self.best_distance, self.last_progress = distance, step
        # Local collection/unloading/service counts as progress even with no movement.
        if self.memory.recent:
            before = self.memory.recent[-1]["before"]
            if (
                before["rover"]["cargo"] != state["rover"]["cargo"]
                or before["colony"]["pad_needs_service"]
                != state["colony"]["pad_needs_service"]
                or (
                    self.goal["kind"] == "recover"
                    and state["rover"]["battery_pct"] > before["rover"]["battery_pct"]
                )
            ):
                self.last_progress = step
        opportunities = {
            g["id"] for g in candidates if g["kind"].startswith("collect_")
        }
        if (
            self.goal["kind"] == "explore"
            and opportunities - self.reviewed_opportunities
        ):
            return "resource_opportunity_discovered"
        warning = self.rules["operating_requirements"]["warning_thresholds"][
            "battery_pct"
        ]
        if (
            state["rover"]["battery_pct"] <= warning
            and not self.reviewed_battery_warning
            and self.goal["kind"] not in {"recover", "deliver", "service"}
        ):
            return "battery_warning"
        if (
            step - self.last_progress >= self.memory.config.stall_steps
            and step - self.reviewed >= self.memory.config.stall_steps
        ):
            return "goal_stalled"
        if step - self.reviewed >= self.memory.config.goal_horizon:
            return "review_horizon"
        return None

    def select(
        self, goal: Json, reason: str, state: Json, candidates: list[Json]
    ) -> None:
        if self.goal is None or self.goal["id"] != goal["id"]:
            self.started = self.last_progress = self.memory.last_observation
            self.best_distance = goal["route"]["distance_steps"]
        self.goal = {
            k: deepcopy(v)
            for k, v in goal.items()
            if k not in {"route", "last_seen_steps_ago"}
        }
        self.reviewed, self.last_reason = self.memory.last_observation, reason
        self.selections += 1
        self.reviewed_opportunities = {
            g["id"] for g in candidates if g["kind"].startswith("collect_")
        }
        self.reviewed_battery_warning = (
            state["rover"]["battery_pct"]
            <= self.rules["operating_requirements"]["warning_thresholds"]["battery_pct"]
        )

    def advice(
        self, state: Json, *, planning: bool = False
    ) -> tuple[list[Json], list[str]]:
        entries = [
            {
                "id": f"L{i + 1}",
                "text": row["text"],
                "conditions": row.get("conditions", []),
            }
            for i, row in enumerate(self.learnings)
            if (
                not row.get("scope", {}).get("rules_hash")
                or row["scope"]["rules_hash"] == digest(self.rules)
            )
            and (
                not row.get("scope", {}).get("observation_schema")
                or row["scope"]["observation_schema"] == state["observation_schema"]
            )
        ]
        if planning:
            return entries, []
        condition_state = {**state, "decision": self.decision_state()}
        matching = [
            i
            for i, row in enumerate(entries)
            if conditions_match(row["conditions"], condition_state)
        ]
        # Prefer specific conditions, then newer advice. Source provenance stays in the bundle.
        matching.sort(key=lambda i: (-len(entries[i]["conditions"]), -i))
        chosen = set(matching[: self.memory.config.advice_limit])
        return [row for i, row in enumerate(entries) if i in chosen], [
            row["id"] for i, row in enumerate(entries) if i not in chosen
        ]

    def decision_state(self) -> Json:
        return {
            "goal_kind": self.goal["kind"] if self.goal else None,
            "goal_steps_without_progress": self.memory.last_observation
            - self.last_progress
            if self.goal
            else None,
        }

    def state_view(self, state: Json) -> Json:
        current = self.memory.facts[point(state)]
        return {
            **telemetry(state),
            "mission": self.deficits(state),
            "current_cell": deepcopy(current["value"]),
        }

    def working_memory(self, state: Json) -> Json:
        origin, step = point(state), self.memory.last_observation
        goal = (
            None
            if self.goal is None
            else {
                **self.goal,
                "route": self.route(origin, tuple(self.goal["position"])),
                "age_steps": step - self.started,
                "steps_without_progress": step - self.last_progress,
            }
        )
        cargo = state["rover"]["cargo"]
        return_needed = (
            any(cargo[k] > 0 for k in ("ice_kg", "ore_kg", "samples_kg"))
            or state["rover"]["battery_pct"]
            <= self.rules["operating_requirements"]["warning_thresholds"]["battery_pct"]
            or state["colony"]["pad_needs_service"]
        )
        pads = (
            [
                pos
                for pos, fact in self.memory.facts.items()
                if fact["value"]["terrain"] == "build_pad"
            ]
            if return_needed
            else []
        )
        routes = [{"position": list(pos), **self.route(origin, pos)} for pos in pads]
        reachable = [route for route in routes if route["distance_steps"] is not None]
        home = min(reachable, key=lambda r: r["distance_steps"]) if reachable else None
        recent = [
            {
                "action": row["issued_action"],
                "from": row["before"]["rover"]["position_cells"],
                "to": row["after"]["rover"]["position_cells"],
                "battery_change_pct": round(
                    row["after"]["rover"]["battery_pct"]
                    - row["before"]["rover"]["battery_pct"],
                    4,
                ),
                "fallback": row["fallback"],
            }
            for row in self.memory.recent[-2:]
        ]
        result = {
            "goal": goal,
            "return_route": home,
            "return_route_needed": return_needed,
            "recent_outcomes": recent,
            "stationary_steps": self.memory.stationary_steps,
            "wait_steps": self.memory.wait_steps,
            "steps_without_collection_or_delivery": self.memory.no_collection_or_delivery_steps,
            "known_cells": len(self.memory.facts),
            "older_transitions_omitted": max(0, step - len(recent)),
        }
        while (
            len(canonical(result).encode()) > self.memory.config.max_bytes
            and result["recent_outcomes"]
        ):
            result["recent_outcomes"].pop(0)
            result["older_transitions_omitted"] += 1
        if len(canonical(result).encode()) > self.memory.config.max_bytes:
            from ...algorithms.base import EpisodeStop

            raise EpisodeStop("focused_memory_budget_exceeded")
        return result

    def options(self, state: Json, mask: object) -> list[Json]:
        origin = point(state)
        route = (
            self.route(origin, tuple(self.goal["position"]))
            if self.goal
            else {"first_hops": []}
        )
        result = []
        for index, name in enumerate(ACTION_NAMES):
            if not mask[index]:
                continue
            dx, dy = DIRECTIONS.get(name, (0, 0))
            destination = (origin[0] + dx, origin[1] + dy)
            fact = self.memory.facts.get(destination)
            entry = {"name": name}
            if fact and name in DIRECTIONS:
                entry["cell"] = {
                    key: fact["value"][key]
                    for key in (
                        "terrain",
                        "roughness",
                        "ice",
                        "scanned",
                        "extracted",
                        "dust",
                    )
                }
                entry["cell"]["seen_steps_ago"] = (
                    self.memory.last_observation - fact["last_seen_step"]
                )
            elif name in DIRECTIONS:
                entry["cell"] = {"terrain": "unknown"}
            if name in DIRECTIONS:
                entry.update(
                    destination=list(destination),
                    follows_goal_route=name in route["first_hops"],
                    last_visited_steps_ago=None
                    if not fact or fact["last_visited_step"] is None
                    else self.memory.last_observation - fact["last_visited_step"],
                    reverses_last_move=bool(
                        self.memory.recent
                        and destination == point(self.memory.recent[-1]["before"])
                        and destination != origin
                    ),
                )
            result.append(entry)
        return result
