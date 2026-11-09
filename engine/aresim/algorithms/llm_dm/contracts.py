"""Versioned JSON boundaries and canonical serialization for LLM agents.

All provider output is untrusted and validated here before it can influence an
issued action or candidate bundle. No world state or provider SDK belongs here.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import fields, is_dataclass
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import TypeAlias

Json: TypeAlias = dict[str, object]


def json_value(value: object) -> object:
    """Convert dataclasses/arrays to finite portable JSON without losing array order."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: json_value(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, Enum):
        return json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON keys must be strings")
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON numbers must be finite")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported JSON type: {type(value).__name__}")


def canonical(value: object) -> str:
    """One canonical encoding for content IDs, provider packets and audit records."""
    return json.dumps(
        json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value: object) -> str:
    """Content identity, independent of timestamps or the current working directory."""
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> Json:
    result: Json = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(text: str) -> Json:
    """Read exactly one JSON object, rejecting duplicate keys and nonfinite numbers."""

    def invalid_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant: {value}")

    result = json.loads(
        text, object_pairs_hook=_unique_object, parse_constant=invalid_constant
    )
    if not isinstance(result, dict):
        raise ValueError("expected one JSON object")
    return result


def object_schema(properties: Json) -> Json:
    """Use the conservative common structured-output subset: all properties required."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def validate(value: object, schema: Json) -> Json:
    """Apply complete host validation even when a provider claims strict decoding."""
    from jsonschema import Draft202012Validator

    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda error: str(error.path),
    )
    if errors:
        raise ValueError(
            f"invalid model output at {list(errors[0].path)}: {errors[0].message}"
        )
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    canonical(value)
    return value


def action_schema() -> Json:
    """Action syntax is fixed across masks; freshness and legality are checked locally."""
    from ...components.actions import ACTION_NAMES

    return object_schema(
        {
            "schema_version": {"type": "string", "enum": ["aresim.llm.action.v1"]},
            "episode_id": {"type": "string"},
            "decision_id": {"type": "string"},
            "step_index": {"type": "integer"},
            "action": {"type": "string", "enum": list(ACTION_NAMES)},
        }
    )


def goal_schema(goal_ids: list[str]) -> Json:
    """Only the Actor can select one of the currently observed, episode-local goals."""
    return object_schema(
        {
            "schema_version": {"type": "string", "enum": ["aresim.llm_dm.goal.v1"]},
            "episode_id": {"type": "string"},
            "decision_id": {"type": "string"},
            "step_index": {"type": "integer"},
            "goal_id": {"type": "string", "enum": goal_ids},
        }
    )


CONDITION_FIELDS = (
    "rover.pad_proximity",
    "rover.battery_pct",
    "rover.health_pct",
    "rover.cargo.free_kg",
    "rover.cargo.ice_kg",
    "rover.cargo.samples_kg",
    "colony.pad_needs_service",
    "colony.power.margin",
    "mission.ice_remaining_fraction",
    "mission.samples_remaining_fraction",
    "decision.goal_kind",
    "decision.goal_steps_without_progress",
)


def proposal_schema() -> Json:
    """A bounded edit or an explicit no-change, with checkable evidence references."""
    condition = object_schema(
        {
            "field": {"type": "string", "enum": list(CONDITION_FIELDS)},
            "operator": {"type": "string", "enum": ["eq", "lt", "lte", "gt", "gte"]},
            "value": {"type": ["string", "number", "boolean"]},
        }
    )
    return object_schema(
        {
            "schema_version": {"type": "string", "enum": ["aresim.llm.proposal.v1"]},
            "parent_bundle_id": {"type": "string"},
            "outcome": {"type": "string", "enum": ["change", "no_change"]},
            "target": {
                "type": ["string", "null"],
                "enum": ["lesson", "actor_guidance", None],
            },
            "text": {"type": ["string", "null"]},
            "reason": {"type": "string"},
            "expected_benefit": {"type": "string"},
            "failure_case": {"type": "string"},
            "support": {"type": "array", "items": {"type": "string"}},
            "counterexamples": {"type": "array", "items": {"type": "string"}},
            "conditions": {"type": "array", "items": condition},
        }
    )
