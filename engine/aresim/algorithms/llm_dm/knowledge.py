"""Immutable policy bundles and evidence-linked candidate knowledge.

The Learner may stage bounded advice; only independent episode evaluation can
activate it. Applicability is deterministic and missing evidence stays unknown.
"""

from __future__ import annotations

import operator
from copy import deepcopy
from pathlib import Path

from ...config import EnvironmentConfig
from ..common.learnings import (
    LEARNING_CONTEXT_INSTRUCTIONS,
    learning_statement,
    merge_learnings,
)
from .config import HistoryConfig, RoleConfig
from .contracts import (
    Json,
    canonical,
    digest,
    json_value,
    proposal_schema,
    strict_json,
    validate,
)
from .state import observation_schema, public_requirements


def make_bundle(
    actor: RoleConfig,
    history: HistoryConfig,
    environment: EnvironmentConfig,
    prompt: str,
    include_rl_reward: bool,
    *,
    knowledge: list[Json] | None = None,
    parent: str | None = None,
    learnings: list[Json] | None = None,
) -> Json:
    """Freeze all acting semantics; run budgets, seeds and live credentials stay external."""
    bundle = {
        "schema_version": "aresim.llm.bundle.v1",
        "policy_id": "aresim.agent.llm_dm.v1",
        "observation_schema": observation_schema(environment),
        "action_schema": "aresim.action.rover.v1",
        "environment": json_value(environment),
        "environment_hash": digest(environment),
        "rules": public_requirements(environment),
        "rules_hash": digest(public_requirements(environment)),
        "actor": json_value(actor),
        "actor_system_prompt": prompt,
        "history": json_value(history),
        "include_rl_reward": include_rl_reward,
        "knowledge": deepcopy(knowledge or []),
        "parent_bundle_id": parent,
        "learnings": deepcopy(learnings or []),
        "action_schema_hash": None,
    }
    from .contracts import action_schema

    bundle["action_schema_hash"] = digest(action_schema())
    if history.mode == "focused":
        from .prompts import render_prompt

        bundle["context_profile"] = "focused-v2"
        bundle["goal_system_prompt"] = render_prompt(
            "goal", actor, environment, focused=True
        )
    return {"bundle_id": digest(bundle), **bundle}


def check_bundle(bundle: Json, environment: EnvironmentConfig) -> Json:
    """Reject mutation or incompatible environment before any provider request."""
    content = {k: v for k, v in bundle.items() if k != "bundle_id"}
    if bundle.get("schema_version") != "aresim.llm.bundle.v1" or digest(
        content
    ) != bundle.get("bundle_id"):
        raise ValueError("invalid policy bundle schema/content digest")
    if bundle.get("environment_hash") != digest(environment):
        raise ValueError("bundle environment does not match this run")
    from .contracts import action_schema

    rules = public_requirements(environment)
    if (
        bundle.get("rules_hash") != digest(rules)
        or bundle.get("rules") != rules
        or bundle.get("observation_schema") != observation_schema(environment)
        or bundle.get("action_schema") != "aresim.action.rover.v1"
        or bundle.get("action_schema_hash") != digest(action_schema())
    ):
        raise ValueError("bundle rules or schemas differ from this runtime")
    RoleConfig(**bundle["actor"]).validate()
    HistoryConfig(**bundle["history"]).validate()
    if bundle["history"]["mode"] == "focused" and (
        bundle.get("context_profile") != "focused-v2"
        or not isinstance(bundle.get("goal_system_prompt"), str)
        or not bundle["goal_system_prompt"].strip()
    ):
        raise ValueError(
            "focused bundles require their pinned goal prompt and context profile"
        )
    if (
        not isinstance(bundle.get("actor_system_prompt"), str)
        or not bundle["actor_system_prompt"].strip()
    ):
        raise ValueError("bundle requires its exact actor system prompt")
    return deepcopy(bundle)


def load_bundle(path: Path, environment: EnvironmentConfig) -> Json:
    return check_bundle(strict_json(path.read_text()), environment)


def stage_proposal(
    proposal: Json,
    parent: Json,
    evidence: Json,
    *,
    source_episodes: list[Json] | None = None,
) -> Json | None:
    """Validate identity, actual source references, bounds and observable predicates."""
    validate(proposal, proposal_schema())
    if proposal["parent_bundle_id"] != parent["bundle_id"]:
        raise ValueError("proposal parent identity mismatch")
    allowed = {episode["episode_id"] for episode in evidence["episodes"]}
    allowed.update(
        row["transition_id"]
        for episode in evidence["episodes"]
        for row in episode["transitions"]
    )
    allowed.update(
        row["source_transition_id"]
        for episode in evidence["episodes"]
        for row in episode.get("timeline", [])
    )
    if not proposal["support"] or any(
        ref not in allowed for ref in proposal["support"] + proposal["counterexamples"]
    ):
        raise ValueError("proposal cites missing or withheld evidence")
    if not proposal["reason"].strip() or len(proposal["reason"]) > 4000:
        raise ValueError("proposal requires a bounded reason")
    if proposal["outcome"] == "no_change":
        if (
            proposal["target"] is not None
            or proposal["text"] is not None
            or proposal["conditions"]
        ):
            raise ValueError("no_change must have null target/text and no conditions")
        return None
    limit = 512 if proposal["target"] == "lesson" else 1024
    if (
        proposal["target"] not in {"lesson", "actor_guidance"}
        or not isinstance(proposal["text"], str)
        or not 0 < len(proposal["text"].strip()) <= limit
    ):
        raise ValueError("change requires bounded lesson or actor guidance")
    if (
        not proposal["expected_benefit"].strip()
        or not proposal["failure_case"].strip()
        or len(proposal["conditions"]) > 8
    ):
        raise ValueError(
            "change requires benefit, falsifiable failure case and at most eight conditions"
        )
    for cond in proposal["conditions"]:
        if cond["operator"] != "eq" and (
            isinstance(cond["value"], bool)
            or not isinstance(cond["value"], (int, float))
        ):
            raise ValueError("ordered condition comparisons require a finite number")
    support = set(proposal["support"])
    source_rows = [
        row
        for episode in (
            source_episodes if source_episodes is not None else evidence["episodes"]
        )
        for row in episode["transitions"]
        if episode["episode_id"] in support or row["transition_id"] in support
    ]
    matches = [
        row["transition_id"]
        for row in source_rows
        if conditions_match(proposal["conditions"], row["before"])
    ]
    if proposal["conditions"] and not matches:
        raise ValueError(
            "proposal conditions never match a cited pre-action state; mission remaining fractions measure DELIVERY, not collection"
        )
    item = {
        "schema_version": "aresim.llm.lesson.v1",
        "version": 1,
        "target": proposal["target"],
        "text": proposal["text"],
        "conditions": proposal["conditions"],
        "scope": {
            "rules_hash": parent["rules_hash"],
            "observation_schema": parent["observation_schema"],
        },
        "support": proposal["support"],
        "counterexamples": proposal["counterexamples"],
        "status": "candidate",
        "priority": len(parent["knowledge"]),
        "validation_references": [],
        "evidence_validation": "References and predicates checked; natural-language claim remains a hypothesis until evaluated.",
        "condition_support": {
            "cited_decisions": len(source_rows),
            "matching_decisions": len(matches),
            "example_transition_ids": matches[:4],
        },
    }
    return {"id": digest(item), **item}


def candidate_bundle(parent: Json, item: Json) -> Json:
    """Stage the candidate's tentative knowledge separately from the published parent."""
    bundle = deepcopy(parent)
    bundle.pop("bundle_id")
    bundle["parent_bundle_id"] = parent["bundle_id"]
    candidate_item = {**deepcopy(item), "status": "evaluation"}
    bundle["knowledge"].append(candidate_item)
    return {"bundle_id": digest(bundle), **bundle}


def publish_bundle(candidate: Json, evaluation_reference: str) -> Json:
    """Publish by creating another content identity; never rewrite an evaluated artifact."""
    result = deepcopy(candidate)
    result.pop("bundle_id")
    for item in result["knowledge"]:
        if item["status"] == "evaluation":
            item["status"] = "active"
            item["validation_references"] = [evaluation_reference]
    return {"bundle_id": digest(result), **result}


def with_learnings(bundle: Json, statements: list[Json]) -> Json:
    """Derive a new pinned policy identity for explicitly imported advice."""
    combined = merge_learnings(bundle.get("learnings", []), statements)
    if combined == bundle.get("learnings", []):
        return bundle
    result = deepcopy(bundle)
    result.pop("bundle_id")
    result["parent_bundle_id"] = bundle["bundle_id"]
    result["learnings"] = combined
    if (
        result.get("context_profile") != "focused-v2"
        and "PERSISTENT LEARNING CONTEXT" not in result["actor_system_prompt"]
    ):
        result["actor_system_prompt"] += (
            "\n\nPERSISTENT LEARNING CONTEXT\n" + LEARNING_CONTEXT_INSTRUCTIONS
        )
    return {"bundle_id": digest(result), **result}


def run_learnings(bundle: Json, *, include_tentative: bool = False) -> list[Json]:
    """Provide all prior advice as persistent context, independent of conditional retrieval."""
    knowledge = [
        learning_statement(item, source={"lesson_id": item["id"], "kind": "within_run"})
        for item in bundle["knowledge"]
        if item["status"]
        in ({"active", "evaluation"} if include_tentative else {"active"})
    ]
    return merge_learnings(bundle.get("learnings", []), knowledge)


def _field(state: Json, path: str) -> object:
    objective = {
        "mission.ice_remaining_fraction": "deliver_ice",
        "mission.samples_remaining_fraction": "scan",
    }.get(path)
    if objective:
        return next(
            (
                row["remaining_fraction"]
                for row in state["objectives"]
                if row["type_name"] == objective
            ),
            None,
        )
    value: object = state
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def retrieve_lessons(
    bundle: Json, state: Json, *, evaluating: bool = False
) -> list[Json]:
    """Keep only compatible, applicable items under the shared four-item/2048-byte cap."""
    selected = []
    for item in sorted(bundle["knowledge"], key=lambda k: (k["priority"], k["id"])):
        if item["status"] not in (
            {"active", "evaluation"} if evaluating else {"active"}
        ):
            continue
        if item["scope"] != {
            "rules_hash": bundle["rules_hash"],
            "observation_schema": bundle["observation_schema"],
        }:
            continue
        if conditions_match(item["conditions"], state):
            entry = {
                key: deepcopy(item[key])
                for key in (
                    "id",
                    "target",
                    "text",
                    "conditions",
                    "support",
                    "counterexamples",
                )
            }
            if len(canonical(selected + [entry]).encode()) <= 2048:
                selected.append(entry)
            if len(selected) == 4:
                break
    return selected


def conditions_match(conditions: list[Json], state: Json) -> bool:
    """Use identical observable predicates for evidence support and acting-time retrieval."""
    predicates = {
        "eq": operator.eq,
        "lt": operator.lt,
        "lte": operator.le,
        "gt": operator.gt,
        "gte": operator.ge,
    }
    for condition in conditions:
        current, expected = _field(state, condition["field"]), condition["value"]
        comparable = type(current) is type(expected) or (
            type(current) in (int, float) and type(expected) in (int, float)
        )
        if (
            current is None
            or not comparable
            or not predicates[condition["operator"]](current, expected)
        ):
            return False
    return True
