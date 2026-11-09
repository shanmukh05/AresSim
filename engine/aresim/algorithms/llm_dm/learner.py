"""Mandatory between-batch proposals from bounded permitted development evidence.

Selection is deterministic and disclosed; exact game logs remain unfiltered.
The Learner stages advice but never judges, publishes, or recursively learns from
its parent/candidate comparison episodes.
"""

from __future__ import annotations

import logging
from copy import deepcopy

from ...algorithms.base import EpisodeStop
from .artifacts import RunArtifacts
from .calls import ModelCalls
from .contracts import Json, canonical, digest, proposal_schema, strict_json
from .knowledge import stage_proposal

logger = logging.getLogger(__name__)


def _timeline_row(row: Json) -> Json:
    after = row["after"]
    cargo = after["rover"]["cargo"]
    progress = {o["type_name"]: o["current_fraction"] for o in after["objectives"]}
    return {
        "source_transition_id": row["transition_id"],
        "step_index": row["step_index"],
        "issued_action": row["issued_action"],
        "position_cells": after["rover"]["position_cells"],
        "battery_pct": after["rover"]["battery_pct"],
        "cargo_kg": {key: cargo[key] for key in ("ice_kg", "samples_kg", "free_kg")},
        "delivered_fractions": {
            "ice": progress.get("deliver_ice"),
            "samples": progress.get("scan"),
        },
        "pad_needs_service": after["colony"]["pad_needs_service"],
        "weather": after["weather"],
        "fallback": row["fallback"],
        "goal": ((row.get("decision_context") or {}).get("goal") or {}).get("id"),
        "goal_steps_without_progress": (
            (row.get("decision_context") or {}).get("goal") or {}
        ).get("steps_without_progress"),
    }


def development_evidence(
    episodes: list[Json], parent: Json, system: str, max_input: int
) -> Json:
    """Select endpoints and consequential decisions under the role's input ceiling."""
    selected = []
    for episode in episodes:
        rows = episode["transitions"]
        if not rows:
            continue
        indices = {0, len(rows) - 1}
        # One example of each event avoids drowning the Learner in repeated loops.
        seen = set()
        for index, row in enumerate(rows):
            context = row.get("decision_context") or {}
            goal = context.get("goal") or {}
            event = None
            if row["issued_action"] in {"scan", "extract", "unload"}:
                event = row["issued_action"]
            elif goal.get("steps_without_progress", 0) >= 6:
                event = "goal_stalled"
            elif goal.get("age_steps") == 0:
                event = "goal_started"
            elif row["after"]["rover"]["cargo"]["free_kg"] == 0:
                event = "cargo_full"
            if event and event not in seen and len(indices) < 8:
                indices.add(index)
                seen.add(event)
        selected.append(
            {
                "episode_id": episode["episode_id"],
                "bundle_id": episode["bundle_id"],
                "outcome": {
                    k: v
                    for k, v in episode["outcome"].items()
                    if k != "source_episode_id"
                },
                "transitions": [deepcopy(rows[index]) for index in sorted(indices)],
                "available_transitions": len(rows),
                "timeline": [_timeline_row(row) for row in rows],
            }
        )
    packet = {
        "schema_version": "aresim.llm.development_evidence.v1",
        "parent_bundle_id": parent["bundle_id"],
        "active_knowledge": deepcopy(parent["knowledge"]),
        "imported_learnings": deepcopy(parent.get("learnings", [])),
        "episodes": selected,
        "selection": "Every turn has a compact timeline; up to eight endpoints and first collection/delivery/full-bay/goal-start/stall events include before/after telemetry and the Actor's observed decision context. Details then interior timeline entries may be omitted to fit. No hidden replay state or private reasoning is supplied.",
        "omitted_transitions": 0,
        "omitted_timeline_transitions": 0,
    }
    total = sum(len(e["transitions"]) for e in episodes)
    while True:
        packet["omitted_transitions"] = total - sum(
            len(e["transitions"]) for e in selected
        )
        packet["omitted_timeline_transitions"] = total - sum(
            len(e["timeline"]) for e in selected
        )
        if len(system.encode()) + len(canonical(packet).encode()) + 1024 <= max_input:
            return packet
        candidate = next((e for e in selected if len(e["transitions"]) > 1), None)
        if candidate:
            candidate["transitions"].pop(0)
        else:
            removable = None
            for episode in selected:
                ordinary = [
                    row
                    for row in episode["timeline"][1:-1]
                    if row["issued_action"]
                    not in {"scan", "extract", "unload", "service"}
                    and not row["fallback"]
                ]
                options = ordinary or episode["timeline"][1:-1]
                if options:
                    removable = (episode, options[len(options) // 2])
                    break
            if removable:
                removable[0]["timeline"].remove(removable[1])
            elif len(selected) > 1:
                selected.pop(0)
            else:
                raise EpisodeStop("input_budget_exceeded")


def evidence_schema(evidence: Json) -> Json:
    """Constrain native citation generation to IDs actually supplied to this Learner."""
    references = set()
    for episode in evidence["episodes"]:
        references.add(episode["episode_id"])
        references.update(row["transition_id"] for row in episode["transitions"])
        references.update(
            row["source_transition_id"] for row in episode.get("timeline", [])
        )
    schema = proposal_schema()
    schema["properties"]["parent_bundle_id"]["enum"] = [evidence["parent_bundle_id"]]
    for field in ("support", "counterexamples"):
        schema["properties"][field]["items"]["enum"] = sorted(references)
    return schema


def propose(
    calls: ModelCalls,
    system: str,
    parent: Json,
    episodes: list[Json],
    artifacts: RunArtifacts,
    batch: int,
) -> tuple[Json | None, Json]:
    """Actually invoke the configured Learner; failure stops the learning run."""
    evidence = development_evidence(
        episodes, parent, system, calls.config.max_input_tokens - 4096
    )
    evidence_ref = artifacts.content("evidence", evidence)
    artifacts.learning_record(
        {
            "event": "learner_started",
            "batch": batch,
            "parent_bundle_id": parent["bundle_id"],
            "evidence_reference": evidence_ref,
            "system_prompt_reference": "manifest.json#learner_system_prompt",
            "selection": evidence["selection"],
            "omitted_transitions": evidence["omitted_transitions"],
            "omitted_timeline_transitions": evidence["omitted_timeline_transitions"],
            "episode_outcomes": [
                {
                    "episode_id": e["episode_id"],
                    "outcome": e["outcome"],
                    "selected_transitions": [
                        t["transition_id"] for t in e["transitions"]
                    ],
                }
                for e in evidence["episodes"]
            ],
        }
    )
    logger.info(
        "Learner batch %d: calling %s/%s",
        batch + 1,
        calls.config.provider,
        calls.config.model,
    )
    packet = evidence
    for attempt in range(2):
        result = calls.generate(system, packet, evidence_schema(evidence))
        if result.status != "ok":
            logger.info(
                "Learner batch %d: %s", batch + 1, result.error_code or result.status
            )
            artifacts.learning_record(
                {
                    "event": "learner_failed",
                    "batch": batch,
                    "status": result.status,
                    "error_code": result.error_code,
                    "parent_retained": parent["bundle_id"],
                }
            )
            raise EpisodeStop("learner_error")
        proposal = None
        try:
            proposal = strict_json(result.text)
            item = stage_proposal(proposal, parent, evidence, source_episodes=episodes)
            break
        except ValueError as error:
            reason = str(error)
            logger.info("Learner batch %d: %s", batch + 1, reason)
            artifacts.learning_record(
                {
                    "event": "learner_rejected_invalid_output",
                    "batch": batch,
                    "validation_attempt": attempt + 1,
                    "reason": reason,
                    "raw_output": result.text,
                    "parent_retained": parent["bundle_id"],
                }
            )
            if attempt:
                raise EpisodeStop("learner_error") from error
            rejected = (
                canonical(proposal.get("conditions", []))[:512]
                if isinstance(proposal, dict)
                else "unparseable"
            )
            feedback = (
                reason[:512]
                + " Rejected conditions: "
                + rejected
                + ". Correct the proposal using cited pre-action states, or return no_change. "
                "mission.ice_remaining_fraction and mission.samples_remaining_fraction are undelivered quota fractions."
            )
            packet = {**evidence, "validation_feedback": feedback}
            artifacts.learning_record(
                {
                    "event": "learner_retry_started",
                    "batch": batch,
                    "reason": reason,
                    "validation_feedback": feedback,
                    "evidence_reference": evidence_ref,
                }
            )
    artifacts.learning_record(
        {
            "event": "learner_proposed" if item else "learner_no_change",
            "batch": batch,
            "parent_bundle_id": parent["bundle_id"],
            "proposal": proposal,
            "proposal_id": digest(proposal),
            "candidate_item": item,
            "evidence_reference": evidence_ref,
            "usage": result.usage,
        }
    )
    return item, proposal
