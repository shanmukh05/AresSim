"""One-step Decision Model or LLM Actor with explicit decisions and shared Memory.

Implements Agent and the optional outcome lifecycle; it never steps the simulator
or reads canonical state. Calls, packets, evidence and readable timelines share IDs.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from copy import deepcopy
from uuid import uuid4

import numpy as np

from ...algorithms.base import EpisodeStop
from ...components.actions import ACTION_IDS, ACTION_SCHEMA
from ...config import EnvironmentConfig
from .artifacts import RunArtifacts
from .calls import ModelCalls
from .config import HistoryConfig
from .contracts import (
    Json,
    action_schema,
    canonical,
    digest,
    strict_json,
    validate,
    goal_schema,
)
from .budget import estimate_input_tokens
from .knowledge import check_bundle, retrieve_lessons, run_learnings
from ..common.decision_context import decision_packet
from .memory import EpisodeMemory, telemetry
from .focus import FocusedContext
from .state import encode_state, require_mask

logger = logging.getLogger(__name__)


class LLMDMAgent:
    """One instance per active episode, sharing a run budget but no mutable episode memory."""

    policy_id = "aresim.agent.llm_dm.v1"
    action_schema = ACTION_SCHEMA

    def __init__(
        self,
        bundle: Json,
        environment: EnvironmentConfig,
        calls: ModelCalls,
        artifacts: RunArtifacts,
        *,
        evaluating: bool = False,
        phase: str = "rollout",
    ) -> None:
        self.bundle = check_bundle(bundle, environment)
        self.environment_config, self.calls, self.artifacts = (
            environment,
            calls,
            artifacts,
        )
        self.observation_schema = bundle["observation_schema"]
        self.memory = EpisodeMemory(HistoryConfig(**bundle["history"]))
        self.evaluating = evaluating
        self.learnings = run_learnings(self.bundle, include_tentative=evaluating)
        self.focus = (
            FocusedContext(self.memory, bundle["rules"], self.learnings)
            if self.memory.config.mode == "focused"
            else None
        )
        self.phase = phase
        self._ready = False
        self.last_info: Json | None = None
        self.completed_evidence: list[Json] = []
        self.metrics: list[Json] = []

    @property
    def episode_open(self) -> bool:
        return self._ready and not self._closed

    def reset(self, seed: int) -> None:
        """Clear every episode fact/pending decision; do not reset shared spend or bundle."""
        self.episode_id = uuid4().hex
        self.source_episode_id = self.episode_id
        self.memory.reset(self.episode_id)
        if self.focus:
            self.focus.reset()
        self.pending = None
        self.state = None
        self.step_index = 0
        self._closed = False
        self._ready = True
        self._feedback_digests: dict[str, str] = {}
        self._finish_digest = None
        self._evidence = []
        self._latencies = []
        self._fallbacks = 0
        self._advice_exposed = 0
        self._candidate_advice_exposed = 0
        self._learning_context_decisions = 0
        self._history_budget_reductions = 0
        self._input_bytes = []
        self._deferred_advice = 0
        self._episode_return = 0.0
        self._budget_start = deepcopy(self.calls.budget.snapshot())
        self._next_mask = None
        self.last_info = None
        self.calls.cancellation.clear()
        self._inflight_decision = None

    def begin_episode(self, source_episode_id: str) -> None:
        """Audit IDs may be descriptive; the model receives only its opaque episode ID."""
        self.source_episode_id = source_episode_id

    def act(self, observation: dict, action_mask: np.ndarray) -> int:
        if not self._ready or self._closed or self.pending is not None:
            raise RuntimeError(
                "act requires reset, an open episode and no pending transition"
            )
        self.calls.budget.check()
        if self.calls.cancellation.is_set():
            raise EpisodeStop("cancelled")
        mask = require_mask(action_mask)
        if self._next_mask is not None and not np.array_equal(mask, self._next_mask):
            raise RuntimeError("act mask differs from delivered feedback")
        state = encode_state(observation, self.environment_config)
        if self.state is not None and digest(state) != digest(self.state):
            raise RuntimeError("act observation differs from delivered feedback")
        if self.state is None:
            self.state = state
            self.memory.observe(state, 0)
            self.artifacts.episode_record(
                self.episode_id,
                {
                    "event": "reset",
                    "source_episode_id": self.source_episode_id,
                    "bundle_id": self.bundle["bundle_id"],
                    "phase": self.phase,
                    "state": state,
                    "step_index": 0,
                },
            )
        decision_id = uuid4().hex
        start = time.monotonic()
        deadline = start + self.calls.config.decision_deadline_seconds
        overflow = 0
        if self.focus:
            packet = self._focused_packet(state, mask, decision_id, deadline)
        else:
            packet = decision_packet(
                episode_id=self.episode_id,
                decision_id=decision_id,
                step_index=self.step_index,
                policy_bundle_id=self.bundle["bundle_id"],
                rules=self.bundle["rules"],
                state=state,
                action_schema=self.action_schema,
                action_mask=mask,
                history=self.memory.retrieve(state),
                lessons=retrieve_lessons(
                    self.bundle, state, evaluating=self.evaluating
                ),
                learnings=self.learnings,
            )
            # History is optional retrieved context. Keep telemetry, rules and every
            # learning intact, and fit this view to the remaining whole-request limit.
            overflow = (
                estimate_input_tokens(
                    self.bundle["actor_system_prompt"], canonical(packet)
                )
                - self.calls.config.max_input_tokens
            )
            if overflow > 0:
                allowance = len(canonical(packet["history"]).encode()) - overflow
                try:
                    fitted = self.memory.retrieve(state, max_bytes=allowance)
                except ValueError:
                    # Use the smallest valid history view for an actionable hard-limit
                    # failure if even the mandatory context cannot fit.
                    fitted = {
                        **packet["history"],
                        "recent_transitions": [],
                        "facts": [],
                        "summary": None,
                        "omitted": {
                            "transitions": self.memory.through + 1,
                            "facts": len(self.memory.facts),
                        },
                    }
                packet["history"] = fitted
                if self._history_budget_reductions == 0:
                    logger.info(
                        "%s: fitting optional history to the Actor input limit; all %d learnings retained; omitted %d transitions and %d facts",
                        self.source_episode_id,
                        len(self.learnings),
                        fitted["omitted"]["transitions"],
                        fitted["omitted"]["facts"],
                    )
        packet_ref = self.artifacts.content("packets", packet)
        candidate_ids = {
            item["id"]
            for item in self.bundle["knowledge"]
            if item["status"] == "evaluation"
        }
        self.artifacts.decision_record(
            {
                "schema_version": "aresim.llm.decision.v1",
                "event": "decision_started",
                "episode_id": self.episode_id,
                "decision_id": decision_id,
                "step_index": self.step_index,
                "bundle_id": self.bundle["bundle_id"],
                "packet_reference": packet_ref,
            },
        )
        self._inflight_decision = decision_id
        logger.info(
            "%s turn %d: calling %s/%s",
            self.source_episode_id,
            self.step_index + 1,
            self.calls.config.provider,
            self.calls.config.model,
        )
        attempts_before = self.calls.budget.attempts
        try:
            result = self.calls.generate(
                self.bundle["actor_system_prompt"],
                packet,
                action_schema(),
                deadline_at=deadline,
            )
        finally:
            if self.calls.budget.attempts > attempts_before:
                self._deferred_advice += packet.get("deferred_advice_count", 0)
                self._history_budget_reductions += int(overflow > 0)
                self._learning_context_decisions += int(
                    bool(packet.get("advice", self.learnings))
                )
                self._advice_exposed += int(
                    bool(packet.get("advice", packet.get("lessons", [])))
                )
                self._candidate_advice_exposed += int(
                    any(
                        (
                            item["id"]
                            if not self.focus
                            else item.get("source", {}).get("lesson_id")
                        )
                        in candidate_ids
                        for item in (
                            packet.get("lessons", [])
                            if not self.focus
                            else [
                                self.learnings[int(row["id"][1:]) - 1]
                                for row in packet["advice"]
                            ]
                        )
                    )
                )
        self._input_bytes.append(
            len(self.bundle["actor_system_prompt"].encode())
            + len(canonical(packet).encode())
        )
        latency = time.monotonic() - start
        self._latencies.append(latency)
        proposed, fallback = None, None
        if result.status == "error" and not result.retryable:
            logger.info(
                "%s turn %d: %s in %.1fs",
                self.source_episode_id,
                self.step_index + 1,
                result.error_code or result.status,
                latency,
            )
            raise EpisodeStop("provider_error")
        if result.status != "ok":
            fallback = result.error_code or result.status
        else:
            try:
                response = validate(strict_json(result.text), action_schema())
                for key in ("episode_id", "decision_id", "step_index"):
                    if response[key] != packet[key]:
                        raise ValueError("wrong_decision_identity")
                proposed = response["action"]
                if not mask[ACTION_IDS[proposed]]:
                    raise ValueError("mask_illegal_action")
            except ValueError as error:
                fallback = str(error)
        action = "wait" if fallback else proposed
        if action is None or not mask[ACTION_IDS[action]]:
            raise ValueError("fallback action not legal under authoritative mask")
        self._fallbacks += int(fallback is not None)
        self.pending = {
            "decision_id": decision_id,
            "step_index": self.step_index,
            "state": state,
            "issued_action": action,
            "proposed_action": proposed,
            "fallback": fallback,
            "packet_reference": packet_ref,
            "legal_actions": [entry["name"] for entry in packet["legal_actions"]],
            "actor_details": result.details,
            "history_omitted": packet.get("history", {}).get("omitted"),
            "decision_context": (
                {
                    "goal": packet["memory"]["goal"],
                    "legal_actions": packet["legal_actions"],
                    "advice_ids": [row["id"] for row in packet["advice"]],
                    "deferred_advice_count": packet["deferred_advice_count"],
                }
                if self.focus
                else None
            ),
            "history_trace_steps": len(
                (packet.get("history", {}).get("summary") or {}).get("recent_trace", [])
            ),
        }
        self.last_info = {
            "decision_id": decision_id,
            "episode_id": self.episode_id,
            "bundle_id": self.bundle["bundle_id"],
            "action": action,
            "fallback": fallback,
            "provider": self.calls.config.provider,
            "model": result.model,
            "latency_seconds": latency,
            "usage": result.usage,
            "actor_details": result.details,
            "summary": f"llm_dm actor={self.calls.config.provider} action={action}; fallback={bool(fallback)}; decision={decision_id}",
        }
        place = state["rover"]["position_cells"]
        note = f" (fallback {fallback})" if fallback else ""
        logger.info(
            "%s turn %d at (%s,%s): %s in %.1fs%s",
            self.source_episode_id,
            self.step_index + 1,
            place["x"],
            place["y"],
            action,
            latency,
            note,
        )
        self.artifacts.decision_record(
            {
                "schema_version": "aresim.llm.decision.v1",
                "event": "decision_intent",
                "episode_id": self.episode_id,
                **{k: v for k, v in self.pending.items() if k != "state"},
                "latency_seconds": latency,
                "usage": result.usage,
            },
        )
        return ACTION_IDS[action]

    def _focused_packet(
        self, state: Json, mask: np.ndarray, decision_id: str, deadline: float
    ) -> Json:
        """Select a persistent goal only when needed, under the same tick deadline."""
        focus = self.focus
        candidates = focus.candidates(state, self.environment_config.engine.payload)
        reason = focus.review_reason(state, candidates)
        if reason:
            goal_decision = uuid4().hex
            planning_advice, _ = focus.advice(state, planning=True)
            goal_packet = {
                "schema_version": "aresim.llm_dm.goal_packet.v1",
                "episode_id": self.episode_id,
                "decision_id": goal_decision,
                "step_index": self.step_index,
                "state": focus.state_view(state),
                "memory": focus.working_memory(state),
                "review_reason": reason,
                "goals": candidates,
                "learnings": planning_advice,
            }
            self._inflight_decision = goal_decision
            schema = goal_schema([goal["id"] for goal in candidates])
            result = self.calls.generate(
                self.bundle["goal_system_prompt"],
                goal_packet,
                schema,
                deadline_at=deadline,
            )
            try:
                if result.status != "ok":
                    raise ValueError(result.error_code or result.status)
                choice = validate(strict_json(result.text), schema)
                if any(
                    choice[key] != goal_packet[key]
                    for key in ("episode_id", "decision_id", "step_index")
                ):
                    raise ValueError("wrong_goal_identity")
                selected = next(
                    goal for goal in candidates if goal["id"] == choice["goal_id"]
                )
            except (ValueError, StopIteration) as error:
                self.artifacts.record(
                    "games.jsonl",
                    {
                        "event": "goal_selection_failed",
                        "episode_id": self.episode_id,
                        "step_index": self.step_index,
                        "reason": str(error),
                    },
                )
                raise EpisodeStop("goal_selection_failed") from error
            focus.select(selected, reason, state, candidates)
            self.artifacts.record(
                "games.jsonl",
                {
                    "event": "goal_selected",
                    "episode_id": self.episode_id,
                    "step_index": self.step_index,
                    "goal": selected,
                    "review_reason": reason,
                    "usage": result.usage,
                },
            )
            logger.info(
                "%s turn %d: goal %s (%s)",
                self.source_episode_id,
                self.step_index + 1,
                selected["id"],
                reason,
            )
        advice, deferred = focus.advice(state)
        return {
            "schema_version": "aresim.llm_dm.packet.v2",
            "episode_id": self.episode_id,
            "decision_id": decision_id,
            "step_index": self.step_index,
            "state": focus.state_view(state),
            "memory": focus.working_memory(state),
            "legal_actions": focus.options(state, mask),
            "advice": advice,
            "deferred_advice_count": len(deferred),
        }

    def cancel(self, reason: str = "cancelled") -> None:
        """Signal an in-flight transport without changing its pending memory concurrently."""
        self.calls.cancellation.set()

    def observe_transition(self, feedback: Json) -> None:
        """Validate identity and complete the pending decision with an observed outcome."""
        if (
            feedback.get("schema_version") != "aresim.llm.feedback.v1"
            or feedback.get("episode_id") != self.episode_id
        ):
            raise ValueError("feedback schema or episode identity mismatch")
        fingerprint = digest(feedback)
        decision_id = feedback["decision_id"]
        previous = self._feedback_digests.get(decision_id)
        if previous is not None:
            if previous != fingerprint:
                raise ValueError("inconsistent duplicate feedback")
            return
        if (
            self.pending is None
            or self.pending["decision_id"] != decision_id
            or feedback["step_index"] != self.step_index
        ):
            raise ValueError("feedback does not match pending decision")
        if feedback["issued_action_id"] != ACTION_IDS[self.pending["issued_action"]]:
            raise ValueError("feedback issued action disagrees with dispatch")
        after = encode_state(feedback["observation"], self.environment_config)
        transition_id = f"{self.episode_id}:transition:{self.step_index}"
        reward = feedback["reward"] if self.bundle["include_rl_reward"] else None
        self.memory.complete(
            transition_id,
            self.step_index,
            self.state,
            after,
            self.pending["issued_action"],
            self.pending["fallback"],
            reward,
            feedback["terminated"],
            feedback["truncated"],
            feedback["ending_reason"],
        )
        record = {
            "event": "transition",
            "actor_details": self.pending["actor_details"],
            "history_omitted": self.pending["history_omitted"],
            "observation_index_after": self.step_index + 1,
            "transition_id": transition_id,
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            **{
                k: v
                for k, v in self.pending.items()
                if k not in {"state", "step_index"}
            },
            "effective_action": feedback["effective_action"],
            "state_before": self.state,
            "state_after": after,
            "rl_reward": reward,
            "terminated": feedback["terminated"],
            "truncated": feedback["truncated"],
            "ending_reason": feedback["ending_reason"],
        }
        self.artifacts.episode_record(self.episode_id, record)
        self.artifacts.decision_record(
            {
                "event": "decision_completed",
                "episode_id": self.episode_id,
                "decision_id": decision_id,
                "transition_id": transition_id,
                "step_index": self.step_index,
            },
        )
        # The Learner receives bounded telemetry evidence, not privileged replay or a private reasoning trace.
        evidence_before = telemetry(self.state)
        if self.focus:
            goal = self.pending["decision_context"]["goal"]
            evidence_before["decision"] = {
                "goal_kind": goal["kind"],
                "goal_steps_without_progress": goal["steps_without_progress"],
            }
        self._evidence.append(
            {
                **{
                    k: record[k]
                    for k in (
                        "transition_id",
                        "step_index",
                        "proposed_action",
                        "issued_action",
                        "fallback",
                        "legal_actions",
                        "rl_reward",
                        "terminated",
                        "truncated",
                        "ending_reason",
                    )
                },
                "before": evidence_before,
                "after": telemetry(after),
                "packet_reference": self.pending["packet_reference"],
                "decision_context": self.pending["decision_context"],
            }
        )
        self.state = after
        self._next_mask = require_mask(feedback["action_mask"])
        self._episode_return += float(feedback["reward"])
        self.step_index += 1
        self.pending = None
        self._inflight_decision = None
        self._closed = feedback["terminated"] or feedback["truncated"]
        self._feedback_digests[decision_id] = fingerprint

    def finish_episode(self, outcome: Json) -> None:
        fingerprint = digest(outcome)
        if self._finish_digest is not None:
            if self._finish_digest != fingerprint:
                raise ValueError("inconsistent duplicate episode closure")
            return
        if self.state is None:
            self.artifacts.episode_record(
                self.episode_id,
                {
                    "event": "reset",
                    "source_episode_id": self.source_episode_id,
                    "bundle_id": self.bundle["bundle_id"],
                    "phase": self.phase,
                    "state": None,
                    "step_index": 0,
                },
            )
        if self.pending or self._inflight_decision:
            self.artifacts.decision_record(
                {
                    "event": "decision_unexecuted",
                    "episode_id": self.episode_id,
                    "decision_id": self.pending["decision_id"]
                    if self.pending
                    else self._inflight_decision,
                    "reason": outcome["ending_reason"],
                },
            )
        self.artifacts.episode_record(
            self.episode_id,
            {"event": "episode_closed", "step_index": self.step_index, **outcome},
        )
        evidence = {
            "episode_id": self.episode_id,
            "bundle_id": self.bundle["bundle_id"],
            "outcome": outcome,
            "transitions": deepcopy(self._evidence),
        }
        self.completed_evidence.append(evidence)
        states = ([self._evidence[0]["before"]] if self._evidence else []) + [
            row["after"] for row in self._evidence
        ]
        end_budget = self.calls.budget.snapshot()
        start_role = self._budget_start["roles"].get("actor", {})
        end_role = end_budget["roles"].get("actor", {})
        operational = {
            "action_counts": dict(
                Counter(row["issued_action"] for row in self._evidence)
            ),
            "fallback_reasons": dict(
                Counter(row["fallback"] for row in self._evidence if row["fallback"])
            ),
            "visited_cells": len(
                {tuple(s["rover"]["position_cells"].values()) for s in states}
            ),
            "first_unload_step": next(
                (
                    row["step_index"]
                    for row in self._evidence
                    if row["issued_action"] == "unload"
                ),
                None,
            ),
            "minimum_battery_pct": min(
                (s["rover"]["battery_pct"] for s in states), default=None
            ),
            "minimum_health_pct": min(
                (s["rover"]["health_pct"] for s in states), default=None
            ),
            "minimum_livability_pct": min(
                (s["colony"]["livability_pct"] for s in states), default=None
            ),
            "actor_usage": {
                key: end_role.get(key, 0) - start_role.get(key, 0)
                for key in ("attempts", "tokens", "usd", "unknown_usage_attempts")
            },
        }
        self.metrics.append(
            {
                **operational,
                "episode_id": self.episode_id,
                "steps": self.step_index,
                "success": outcome["ending_reason"] == "mission_complete",
                "survival_failure": outcome["ending_reason"]
                in {
                    "battery_depleted",
                    "rover_health_depleted",
                    "livability_depleted",
                    "environment_failure",
                },
                "fallbacks": self._fallbacks,
                "advice_exposed_decisions": self._advice_exposed,
                "candidate_advice_exposed_decisions": self._candidate_advice_exposed,
                "learning_context_decisions": self._learning_context_decisions,
                "history_budget_reductions": self._history_budget_reductions,
                "goal_selections": self.focus.selections if self.focus else 0,
                "context_profile": self.bundle.get(
                    "context_profile", "full-context-v1"
                ),
                "action_input_bytes_mean": sum(self._input_bytes)
                / len(self._input_bytes)
                if self._input_bytes
                else None,
                "deferred_advice_exposures": self._deferred_advice,
                "learning_statement_count": len(self.learnings),
                "learning_statement_ids": [entry["id"] for entry in self.learnings],
                "latencies": self._latencies,
                "ending_reason": outcome["ending_reason"],
                "final_state": telemetry(self.state) if self.state else None,
                "episode_return": self._episode_return,
            }
        )
        self.artifacts.game_summary(
            {
                "episode_id": self.episode_id,
                "source_episode_id": self.source_episode_id,
                "phase": self.phase,
                "bundle_id": self.bundle["bundle_id"],
                "outcome": outcome,
                "metrics": self.metrics[-1],
            },
        )
        self._finish_digest = fingerprint
        self._closed = True
        self.pending = None
        self._inflight_decision = None
        self.artifacts.budget_updated(self.calls.budget.snapshot())


def build_llm_dm_agent(context):
    """Explicit component factory, configured by a backend-only run/bundle config path."""
    import os
    from pathlib import Path
    from .rollout import build_frozen_agent

    path = os.environ.get("ARESIM_LLM_DM_CONFIG")
    if not path:
        raise ValueError("set ARESIM_LLM_DM_CONFIG to a frozen-evaluation LLM run YAML")
    return build_frozen_agent(Path(path), environment=context.environment_config)
