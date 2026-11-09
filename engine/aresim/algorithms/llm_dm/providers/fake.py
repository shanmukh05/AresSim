"""Deterministic offline transport for end-to-end simulator smoke runs.

It consumes exactly the same permitted packets as a live provider. Its canned
learning proposal is explicitly diagnostic evidence, never a measured LLM result.
"""

from __future__ import annotations

from collections import deque
from threading import Event

from ..config import RoleConfig
from ..contracts import Json, canonical, strict_json
from .base import ProviderRequest, ProviderResult


class FakeProvider:
    """A small reproducible policy plus injectable normalized outcomes for manual faults."""

    def __init__(
        self, config: RoleConfig, outcomes: tuple[ProviderResult, ...] = ()
    ) -> None:
        self.config = config
        self.outcomes = deque(outcomes)

    def preflight(self) -> Json:
        return {
            "provider": "fake",
            "model": "offline-smoke-v1",
            "native_schema": True,
            "network": False,
        }

    def generate(
        self, request: ProviderRequest, timeout: float, cancellation: Event
    ) -> ProviderResult:
        if cancellation.is_set():
            return ProviderResult("error", error_code="cancelled")
        if self.outcomes:
            return self.outcomes.popleft()
        packet = strict_json(request.user)
        if request.role == "learner":
            refs = [
                row["transition_id"]
                for episode in packet["episodes"]
                for row in episode["transitions"]
            ]
            payload = {
                "schema_version": "aresim.llm.proposal.v1",
                "parent_bundle_id": packet["parent_bundle_id"],
                "outcome": "change",
                "target": "lesson",
                "text": "Plan collection and delivery together using current observations; check cargo, battery and remembered pad locations before taking another collection trip.",
                "reason": "Offline pipeline probe; this proposal is canned and is not an inferred or validated LLM finding.",
                "expected_benefit": "Exercise scoped advice, independent paired simulation and rejection/publication recording.",
                "failure_case": "A remembered route may be blocked or energy-expensive; advice cannot guarantee mission completion.",
                "support": refs[:1],
                "counterexamples": [],
                "conditions": [
                    {
                        "field": "rover.pad_proximity",
                        "operator": "eq",
                        "value": packet["episodes"][0]["transitions"][0]["before"][
                            "rover"
                        ]["pad_proximity"],
                    }
                ],
            }
        elif packet.get("schema_version") == "aresim.llm_dm.goal_packet.v1":
            low = packet["state"]["rover"]["battery_pct"] < 35
            order = (["recover"] if low else []) + [
                "deliver",
                "collect_samples",
                "collect_ice",
                "explore",
                "service",
                "recover",
                "observe_survive",
            ]
            choice = min(
                packet["goals"],
                key=lambda g: (
                    order.index(g["kind"]),
                    g["route"]["distance_steps"] or 0,
                ),
            )
            payload = {
                "schema_version": "aresim.llm_dm.goal.v1",
                **{k: packet[k] for k in ("episode_id", "decision_id", "step_index")},
                "goal_id": choice["id"],
            }
        else:
            legal = {entry["name"] for entry in packet["legal_actions"]}
            preferred = ["unload", "extract", "scan"]
            directions = ["move_north", "move_east", "move_south", "move_west"]
            offset = (packet["step_index"] // 8) % 4
            preferred += (
                directions[offset:] + directions[:offset] + ["service", "build", "wait"]
            )
            if packet.get("schema_version") == "aresim.llm_dm.packet.v2":
                goal = packet["memory"]["goal"]
                local_action = {
                    "deliver": "unload",
                    "collect_ice": "extract",
                    "collect_samples": "scan",
                    "recover": "wait",
                    "service": "service",
                }.get(goal["kind"])
                preferred = (
                    (
                        [local_action]
                        if local_action and goal["route"]["distance_steps"] == 0
                        else []
                    )
                    + goal["route"]["first_hops"]
                    + ["wait"]
                )
            choice = next(name for name in preferred if name in legal)
            payload = {
                "schema_version": "aresim.llm.action.v1",
                **{k: packet[k] for k in ("episode_id", "decision_id", "step_index")},
                "action": choice,
            }
        text = canonical(payload)
        return ProviderResult(
            "ok",
            text=text,
            model="offline-smoke-v1",
            request_id=request.request_id,
            finish_reason="stop",
            usage={
                "input_tokens": len(request.system.encode())
                + len(request.user.encode()),
                "output_tokens": len(text.encode()),
            },
        )

    def close(self) -> None:
        pass
