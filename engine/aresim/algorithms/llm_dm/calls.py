"""One bounded structured call with auditable retries and shared cost accounting.

Actor and Learner use identical transport authority; their callers decide whether
an invalid model result becomes a gameplay fallback or a failed learning run.
"""

from __future__ import annotations

import logging
import time
from threading import Event
from dataclasses import replace
from uuid import uuid4

from ...algorithms.base import EpisodeStop
from .artifacts import RunArtifacts
from .budget import BudgetLedger
from .config import RoleConfig
from .contracts import Json, canonical, json_value
from .providers import LLMProvider, ProviderRequest, ProviderResult

logger = logging.getLogger(__name__)


class ModelCalls:
    """A role-specific requester over a run-owned ledger and append-only audit log."""

    def __init__(
        self,
        role: str,
        config: RoleConfig,
        provider: LLMProvider,
        budget: BudgetLedger,
        artifacts: RunArtifacts,
        cancellation: Event | None = None,
    ) -> None:
        self.role, self.config, self.provider = role, config, provider
        self.budget, self.artifacts = budget, artifacts
        self.cancellation = cancellation or Event()

    def generate(
        self,
        system: str,
        packet: Json,
        schema: Json,
        *,
        deadline_at: float | None = None,
    ) -> ProviderResult:
        deadline = min(
            time.monotonic() + self.config.decision_deadline_seconds,
            self.budget.started + self.budget.config.max_wall_seconds,
            deadline_at if deadline_at is not None else float("inf"),
        )
        user = canonical(packet)
        request_kind = (
            "goal"
            if packet.get("schema_version") == "aresim.llm_dm.goal_packet.v1"
            else "action"
            if self.role == "actor"
            else "learning"
        )
        request_ref = self.artifacts.content(
            "requests",
            {
                "role": self.role,
                "system": system,
                "user": packet,
                "output_schema": schema,
            },
        )
        result = ProviderResult("error", error_code="deadline_exceeded")
        for attempt in range(self.config.transient_retries + 1):
            if self.cancellation.is_set():
                raise EpisodeStop("cancelled")
            if time.monotonic() >= deadline:
                break
            if attempt:
                logger.info("%s retry %d", self.role, attempt + 1)
            reservation = self.budget.reserve(self.role, self.config, system, user)
            request = ProviderRequest(self.role, uuid4().hex, system, user, schema)
            self.artifacts.call_record(
                {
                    "event": "attempt_started",
                    "role": self.role,
                    "request_kind": request_kind,
                    "attempt_id": request.request_id,
                    "attempt_index": attempt,
                    "request_reference": request_ref,
                    "requested_model": self.config.model,
                    "provider": self.config.provider,
                    "route": self.config.route,
                    "context": {
                        key: packet[key]
                        for key in (
                            "episode_id",
                            "decision_id",
                            "step_index",
                            "parent_bundle_id",
                        )
                        if key in packet
                    },
                    "reservation": json_value(reservation),
                },
            )
            start = time.monotonic()
            result = self.provider.generate(
                request,
                min(self.config.attempt_timeout_seconds, deadline - start),
                self.cancellation,
            )
            self.budget.reconcile(reservation, result.usage)
            self.artifacts.call_record(
                {
                    "event": "attempt_completed",
                    "role": self.role,
                    "request_kind": request_kind,
                    "input_bytes": len(system.encode()) + len(user.encode()),
                    "attempt_id": request.request_id,
                    "context": {
                        key: packet[key]
                        for key in (
                            "episode_id",
                            "decision_id",
                            "step_index",
                            "parent_bundle_id",
                        )
                        if key in packet
                    },
                    "result": json_value(result),
                    "elapsed_seconds": time.monotonic() - start,
                    "request_reference": request_ref,
                    "budget": self.budget.snapshot(),
                },
            )
            if self.cancellation.is_set():
                raise EpisodeStop("cancelled")
            self.budget.check()
            if time.monotonic() > deadline:
                return replace(
                    result,
                    status="incomplete",
                    error_code="decision_deadline_exceeded",
                    retryable=False,
                )
            if not result.retryable or attempt == self.config.transient_retries:
                return result
            wait = max(self.config.retry_backoff_seconds, result.retry_after_seconds)
            if time.monotonic() + wait >= deadline:
                return result
            if self.cancellation.wait(wait):
                raise EpisodeStop("cancelled")
        return result
