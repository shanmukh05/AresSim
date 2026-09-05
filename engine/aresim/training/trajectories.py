"""Record and validate transition-aligned datasets and replayable trajectories.

Owns ``aresim.trajectory.v1`` JSONL shards and standalone
``aresim.trajectory.episode.v1`` artifacts with UI replay projections. The writer
never steps environments or chooses actions.

**Last updated:** September 1, 2026

**Contains:** ``TrajectoryWriter``, ``EpisodeTrajectory``, validation helpers,
Gymnasium space (de)serializers.

**Schemas:** ``aresim.trajectory.v1``, ``aresim.trajectory.episode.v1``.

**See also:** :mod:`aresim.gameplay` (replay projection builders).
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Generic, Literal, TypeVar

import numpy as np
from gymnasium import Space, spaces

from ..defaults import DEFAULT_ENGINE_CONFIG
from ..gameplay import (
    TRAJECTORY_EPISODE_SCHEMA,
    TRAJECTORY_REPLAY_SCHEMA,
    JsonObject,
    create_trajectory_episode,
    normalize_gameplay_payload,
    normalize_trajectory_episode,
)


TRAJECTORY_SCHEMA = "aresim.trajectory.v1"
ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")
Compression = Literal["none", "gzip"]
SpaceDescriptor = dict[str, object]


@dataclass(frozen=True)
class EpisodeTrajectory(Generic[ObservationT, ActionT]):
    """One complete episode with `T+1` policy inputs and `T` transitions."""

    episode_id: str
    agent_id: str
    policy_id: str
    scenario_id: str
    task_id: str
    observation_schema: str
    action_schema: str
    reward_profile: str
    environment_seed: int
    agent_seed: int
    environment_config: object
    observation_space: Space[ObservationT]
    action_space: Space[ActionT]
    action_mask_space: Space[np.ndarray]
    observations: tuple[ObservationT, ...]
    action_masks: tuple[np.ndarray, ...]
    actions: tuple[ActionT, ...]
    rewards: tuple[float, ...]
    reward_breakdowns: tuple[dict[str, object], ...]
    engine_rewards: tuple[float, ...]
    engine_reward_terms: tuple[dict[str, float], ...]
    terminated: tuple[bool, ...]
    truncated: tuple[bool, ...]
    action_legal: tuple[bool | None, ...]
    effective_actions: tuple[str, ...]
    events: tuple[tuple[str, ...], ...]
    terminal_reasons: tuple[str | None, ...]
    state_checksums: tuple[str, ...]
    replay: JsonObject | None = None

    @property
    def length(self) -> int:
        """Number of authoritative environment transitions in this episode."""
        return len(self.actions)

    @property
    def episode_return(self) -> float:
        """Sum of selected RL rewards."""
        return float(sum(self.rewards))

    @property
    def engine_return(self) -> float:
        """Sum of authoritative engine/UI rewards for audit."""
        return float(sum(self.engine_rewards))

    @property
    def ending_reason(self) -> str | None:
        """Final task or truncation reason."""
        return self.terminal_reasons[-1] if self.terminal_reasons else None


@dataclass(frozen=True)
class TrajectoryManifest:
    """Validated dataset metadata and immutable shard inventory."""

    payload: dict[str, object]

    @property
    def episode_count(self) -> int:
        """Declared number of complete episode records."""
        return int(self.payload["episode_count"])

    @property
    def transition_count(self) -> int:
        """Declared number of environment transitions."""
        return int(self.payload["transition_count"])


class TrajectoryValidationError(ValueError):
    """Raised when a trajectory dataset violates its declared contract."""


def _json_mapping(value: Mapping[object, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, (str, Enum)):
            raise TypeError("JSON mappings require string or enum keys")
        result[str(key.value if isinstance(key, Enum) else key)] = _json_value(item)
    return result


def _json_scalar(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("trajectory JSON cannot contain NaN or infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported trajectory JSON value: {type(value).__name__}")


def _json_structured(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Enum):
        return _json_value(value.value)
    return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}


def _json_value(value: object) -> object:
    if isinstance(value, (np.ndarray, np.generic, Enum)) or (is_dataclass(value) and not isinstance(value, type)):
        return _json_structured(value)
    if isinstance(value, Mapping):
        return _json_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return _json_scalar(value)


def _canonical_json(value: object) -> str:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _parse_json(text: str) -> object:
    try:
        return json.loads(
            text,
            parse_constant=lambda value: (_ for _ in ()).throw(
                TrajectoryValidationError(f"invalid numeric constant: {value}")
            ),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise TrajectoryValidationError("trajectory contains malformed JSON") from error


def _array_or_scalar(value: np.ndarray) -> object:
    raw = value.item() if value.ndim == 0 else value.tolist()

    def encode(item: object) -> object:
        if isinstance(item, list):
            return [encode(child) for child in item]
        if isinstance(item, float) and math.isinf(item):
            return "Infinity" if item > 0 else "-Infinity"
        if isinstance(item, float) and math.isnan(item):
            raise ValueError("Gymnasium space bounds cannot contain NaN")
        return item

    return encode(raw)


def _space_bound(value: object) -> object:
    if isinstance(value, list):
        return [_space_bound(item) for item in value]
    if value == "Infinity":
        return math.inf
    if value == "-Infinity":
        return -math.inf
    return value


def describe_space(space: Space[Any]) -> SpaceDescriptor:
    """Return a JSON-compatible recursive descriptor for supported Gymnasium spaces."""
    if isinstance(space, spaces.Box):
        return {
            "type": "box",
            "shape": list(space.shape),
            "dtype": space.dtype.str,
            "low": _array_or_scalar(space.low),
            "high": _array_or_scalar(space.high),
        }
    if isinstance(space, spaces.Discrete):
        return {"type": "discrete", "n": int(space.n), "start": int(space.start)}
    if isinstance(space, spaces.Dict):
        return {"type": "dict", "spaces": {key: describe_space(item) for key, item in space.spaces.items()}}
    if isinstance(space, spaces.Tuple):
        return {"type": "tuple", "spaces": [describe_space(item) for item in space.spaces]}
    if isinstance(space, spaces.MultiBinary):
        return {"type": "multi_binary", "n": _json_value(space.n), "dtype": space.dtype.str}
    if isinstance(space, spaces.MultiDiscrete):
        return {
            "type": "multi_discrete",
            "nvec": space.nvec.tolist(),
            "start": space.start.tolist(),
            "dtype": space.dtype.str,
        }
    raise TypeError(f"unsupported Gymnasium space: {type(space).__name__}")


def _box_from_descriptor(descriptor: SpaceDescriptor) -> Space[Any]:
    dtype = np.dtype(descriptor["dtype"])
    shape = tuple(int(value) for value in descriptor["shape"])
    low = np.asarray(_space_bound(descriptor["low"]), dtype=dtype)
    high = np.asarray(_space_bound(descriptor["high"]), dtype=dtype)
    return spaces.Box(low=low, high=high, shape=shape, dtype=dtype)


def _discrete_from_descriptor(descriptor: SpaceDescriptor) -> Space[Any]:
    return spaces.Discrete(int(descriptor["n"]), start=int(descriptor["start"]))


def _dict_from_descriptor(descriptor: SpaceDescriptor) -> Space[Any]:
    children = descriptor["spaces"]
    if not isinstance(children, dict):
        raise TypeError
    return spaces.Dict({str(key): space_from_descriptor(value) for key, value in children.items()})


def _tuple_from_descriptor(descriptor: SpaceDescriptor) -> Space[Any]:
    children = descriptor["spaces"]
    if not isinstance(children, list):
        raise TypeError
    return spaces.Tuple(tuple(space_from_descriptor(value) for value in children))


def _multi_binary_from_descriptor(descriptor: SpaceDescriptor) -> Space[Any]:
    n = descriptor["n"]
    resolved_n = tuple(int(value) for value in n) if isinstance(n, list) else int(n)
    return spaces.MultiBinary(resolved_n)


def _multi_discrete_from_descriptor(descriptor: SpaceDescriptor) -> Space[Any]:
    dtype = np.dtype(descriptor["dtype"])
    return spaces.MultiDiscrete(
        np.asarray(descriptor["nvec"], dtype=dtype),
        start=np.asarray(descriptor["start"], dtype=dtype),
        dtype=dtype,
    )


_SPACE_FROM_DESCRIPTOR = {
    "box": _box_from_descriptor,
    "discrete": _discrete_from_descriptor,
    "dict": _dict_from_descriptor,
    "tuple": _tuple_from_descriptor,
    "multi_binary": _multi_binary_from_descriptor,
    "multi_discrete": _multi_discrete_from_descriptor,
}


def space_from_descriptor(descriptor: SpaceDescriptor) -> Space[Any]:
    """Reconstruct a supported Gymnasium space from validated JSON metadata."""
    kind = descriptor.get("type")
    builder = _SPACE_FROM_DESCRIPTOR.get(kind) if isinstance(kind, str) else None
    if builder is None:
        raise TrajectoryValidationError(f"unsupported Gymnasium space descriptor: {kind}")
    try:
        return builder(descriptor)
    except (KeyError, TypeError, ValueError) as error:
        raise TrajectoryValidationError("invalid Gymnasium space descriptor") from error


def _decode_discrete(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TrajectoryValidationError("discrete action must be an integer")
    return int(value)


def _decode_dict(space: spaces.Dict, value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != set(space.spaces):
        raise TrajectoryValidationError("observation dictionary does not match its space")
    return {key: _decode_space_value(child, value[key]) for key, child in space.spaces.items()}


def _decode_tuple(space: spaces.Tuple, value: object) -> tuple[object, ...]:
    if not isinstance(value, list) or len(value) != len(space.spaces):
        raise TrajectoryValidationError("tuple action does not match its space")
    return tuple(_decode_space_value(child, item) for child, item in zip(space.spaces, value, strict=True))


def _decode_space_value(space: Space[Any], value: object) -> object:
    if isinstance(space, spaces.Box):
        result: object = np.asarray(value, dtype=space.dtype)
    elif isinstance(space, spaces.Discrete):
        result = _decode_discrete(value)
    elif isinstance(space, spaces.Dict):
        result = _decode_dict(space, value)
    elif isinstance(space, spaces.Tuple):
        result = _decode_tuple(space, value)
    elif isinstance(space, (spaces.MultiBinary, spaces.MultiDiscrete)):
        result = np.asarray(value, dtype=space.dtype)
    else:
        raise TrajectoryValidationError(f"unsupported Gymnasium space: {type(space).__name__}")
    if not space.contains(result):
        raise TrajectoryValidationError("trajectory value is outside its declared Gymnasium space")
    return result


def _require_episode_identifiers(episode: EpisodeTrajectory[Any, Any]) -> None:
    identifiers = (
        episode.episode_id,
        episode.agent_id,
        episode.policy_id,
        episode.scenario_id,
        episode.task_id,
        episode.observation_schema,
        episode.action_schema,
        episode.reward_profile,
    )
    if any(not isinstance(identifier, str) or not identifier.strip() for identifier in identifiers):
        raise TrajectoryValidationError("trajectory identifiers cannot be empty")


def _require_episode_lengths(episode: EpisodeTrajectory[Any, Any]) -> None:
    transition_count = episode.length
    if transition_count <= 0:
        raise TrajectoryValidationError("trajectory episodes cannot be empty")
    next_aligned = (episode.observations, episode.action_masks, episode.state_checksums)
    if any(len(values) != transition_count + 1 for values in next_aligned):
        raise TrajectoryValidationError("observations, masks, and checksums must have length T+1")
    transition_aligned = (
        episode.rewards,
        episode.reward_breakdowns,
        episode.engine_rewards,
        episode.engine_reward_terms,
        episode.terminated,
        episode.truncated,
        episode.action_legal,
        episode.effective_actions,
        episode.events,
        episode.terminal_reasons,
    )
    if any(len(values) != transition_count for values in transition_aligned):
        raise TrajectoryValidationError("transition fields must have length T")


def _require_single_ending(episode: EpisodeTrajectory[Any, Any]) -> None:
    endings = [terminated or truncated for terminated, truncated in zip(episode.terminated, episode.truncated, strict=True)]
    if any(endings[:-1]) or not endings[-1]:
        raise TrajectoryValidationError("only the final trajectory step may end the episode")
    if episode.terminated[-1] and episode.truncated[-1]:
        raise TrajectoryValidationError("final transition cannot be both terminated and truncated")


def _require_space_membership(episode: EpisodeTrajectory[Any, Any]) -> None:
    for observation in episode.observations:
        if not episode.observation_space.contains(observation):
            raise TrajectoryValidationError("observation is outside its declared space")
    for action_mask in episode.action_masks:
        if not episode.action_mask_space.contains(action_mask):
            raise TrajectoryValidationError("action mask is outside its declared space")
    for action in episode.actions:
        if not episode.action_space.contains(action):
            raise TrajectoryValidationError("action is outside its declared space")


def _require_action_legality(episode: EpisodeTrajectory[Any, Any]) -> None:
    if not isinstance(episode.action_space, spaces.Discrete):
        return
    for action, mask, legal in zip(episode.actions, episode.action_masks[:-1], episode.action_legal, strict=True):
        expected = bool(mask[int(action)])
        if legal is None or legal != expected:
            raise TrajectoryValidationError("action legality does not match its selection mask")


def _require_checksum_format(episode: EpisodeTrajectory[Any, Any]) -> None:
    for checksum in episode.state_checksums:
        if len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum):
            raise TrajectoryValidationError("state checksum must be lowercase SHA-256")


def _require_reward_breakdowns(episode: EpisodeTrajectory[Any, Any]) -> None:
    for reward, breakdown in zip(episode.rewards, episode.reward_breakdowns, strict=True):
        if not math.isfinite(reward):
            raise TrajectoryValidationError("reward must be finite")
        try:
            total = float(breakdown["total"])
            total_unclipped = float(breakdown["total_unclipped"])
            terms = breakdown["terms"]
            term_total = sum(float(term["value"]) for term in terms.values())
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise TrajectoryValidationError("invalid reward breakdown") from error
        if not math.isclose(reward, total, rel_tol=0, abs_tol=1e-9):
            raise TrajectoryValidationError("selected reward does not match its breakdown")
        if not math.isclose(total_unclipped, term_total, rel_tol=0, abs_tol=1e-9):
            raise TrajectoryValidationError("reward terms do not sum to total_unclipped")


def _require_engine_rewards(episode: EpisodeTrajectory[Any, Any]) -> None:
    for reward, terms in zip(episode.engine_rewards, episode.engine_reward_terms, strict=True):
        if not math.isfinite(reward) or not all(math.isfinite(value) for value in terms.values()):
            raise TrajectoryValidationError("engine reward values must be finite")
        if not math.isclose(reward, sum(terms.values()), rel_tol=0, abs_tol=1e-9):
            raise TrajectoryValidationError("engine reward terms do not sum to engine reward")


def _require_replay_alignment(episode: EpisodeTrajectory[Any, Any]) -> None:
    if episode.replay is None:
        return
    try:
        replay = episode.replay
        if replay.get("schemaVersion") == TRAJECTORY_REPLAY_SCHEMA:
            metadata = replay["metadata"]
            integrity = replay["integrity"]
        else:
            normalized = normalize_gameplay_payload(
                replay,
                str(replay.get("fileName", f"{episode.episode_id}.json")),
                DEFAULT_ENGINE_CONFIG.replay,
            )
            metadata = normalized["metadata"]
            integrity = normalized["integrity"]
    except (TypeError, ValueError, KeyError) as error:
        raise TrajectoryValidationError("episode gameplay replay is invalid") from error
    if metadata.get("seed") != episode.environment_seed:
        raise TrajectoryValidationError("gameplay replay seed does not match the trajectory")
    if integrity.get("stepCount") != episode.length:
        raise TrajectoryValidationError("gameplay replay step count does not match the trajectory")
    if metadata.get("totalSteps") != episode.length:
        raise TrajectoryValidationError("gameplay replay final step does not match the trajectory")


def validate_episode(episode: EpisodeTrajectory[Any, Any]) -> None:
    """Reject an in-memory episode that violates trajectory alignment or totals."""
    _require_episode_identifiers(episode)
    _require_episode_lengths(episode)
    _require_single_ending(episode)
    _require_space_membership(episode)
    _require_action_legality(episode)
    _require_checksum_format(episode)
    _require_reward_breakdowns(episode)
    _require_engine_rewards(episode)
    _require_replay_alignment(episode)


def _episode_payload(
    episode: EpisodeTrajectory[Any, Any],
    episode_artifact: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "episode_id": episode.episode_id,
        "agent_id": episode.agent_id,
        "policy_id": episode.policy_id,
        "scenario_id": episode.scenario_id,
        "task_id": episode.task_id,
        "observation_schema": episode.observation_schema,
        "action_schema": episode.action_schema,
        "reward_profile": episode.reward_profile,
        "environment_seed": episode.environment_seed,
        "agent_seed": episode.agent_seed,
        "observations": episode.observations,
        "action_masks": episode.action_masks,
        "actions": episode.actions,
        "rewards": episode.rewards,
        "reward_breakdowns": episode.reward_breakdowns,
        "engine_rewards": episode.engine_rewards,
        "engine_reward_terms": episode.engine_reward_terms,
        "terminated": episode.terminated,
        "truncated": episode.truncated,
        "action_legal": episode.action_legal,
        "effective_actions": episode.effective_actions,
        "events": episode.events,
        "terminal_reasons": episode.terminal_reasons,
        "state_checksums": episode.state_checksums,
        "length": episode.length,
        "episode_return": episode.episode_return,
        "engine_return": episode.engine_return,
        "ending_reason": episode.ending_reason,
    }
    if episode_artifact is not None:
        payload["episode_artifact"] = episode_artifact
    return payload


def _require_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TrajectoryValidationError(f"{label} must be a JSON object")
    return value


def _boolean_tuple(value: object, label: str) -> tuple[bool, ...]:
    if not isinstance(value, list) or not all(isinstance(item, bool) for item in value):
        raise TrajectoryValidationError(f"{label} must contain only booleans")
    return tuple(value)


def _legality_tuple(value: object) -> tuple[bool | None, ...]:
    if not isinstance(value, list) or not all(item is None or isinstance(item, bool) for item in value):
        raise TrajectoryValidationError("action_legal must contain only booleans or null")
    return tuple(value)


def _decode_episode_sequences(
    record: dict[str, object],
    observation_space: Space[Any],
    action_space: Space[Any],
    action_mask_space: Space[np.ndarray],
) -> tuple[tuple[object, ...], tuple[object, ...], tuple[object, ...]]:
    observations = tuple(_decode_space_value(observation_space, item) for item in record["observations"])
    action_masks = tuple(_decode_space_value(action_mask_space, item) for item in record["action_masks"])
    actions = tuple(_decode_space_value(action_space, item) for item in record["actions"])
    return observations, action_masks, actions


def _engine_reward_terms(record: dict[str, object]) -> tuple[dict[str, float], ...]:
    return tuple(
        {str(key): float(value) for key, value in _require_mapping(terms, "engine reward terms").items()}
        for terms in record["engine_reward_terms"]
    )


def _episode_from_record(
    record: dict[str, object],
    environment_config: object,
    observation_space: Space[Any],
    action_space: Space[Any],
    action_mask_space: Space[np.ndarray],
    gameplay: JsonObject | None,
) -> EpisodeTrajectory[Any, Any]:
    observations, action_masks, actions = _decode_episode_sequences(record, observation_space, action_space, action_mask_space)
    return EpisodeTrajectory(
        episode_id=str(record["episode_id"]),
        agent_id=str(record["agent_id"]),
        policy_id=str(record["policy_id"]),
        scenario_id=str(record["scenario_id"]),
        task_id=str(record["task_id"]),
        observation_schema=str(record["observation_schema"]),
        action_schema=str(record["action_schema"]),
        reward_profile=str(record["reward_profile"]),
        environment_seed=int(record["environment_seed"]),
        agent_seed=int(record["agent_seed"]),
        environment_config=environment_config,
        observation_space=observation_space,
        action_space=action_space,
        action_mask_space=action_mask_space,
        observations=observations,
        action_masks=action_masks,
        actions=actions,
        rewards=tuple(float(value) for value in record["rewards"]),
        reward_breakdowns=tuple(_require_mapping(value, "reward breakdown") for value in record["reward_breakdowns"]),
        engine_rewards=tuple(float(value) for value in record["engine_rewards"]),
        engine_reward_terms=_engine_reward_terms(record),
        terminated=_boolean_tuple(record["terminated"], "terminated"),
        truncated=_boolean_tuple(record["truncated"], "truncated"),
        action_legal=_legality_tuple(record["action_legal"]),
        effective_actions=tuple(str(value) for value in record["effective_actions"]),
        events=tuple(tuple(str(event) for event in values) for values in record["events"]),
        terminal_reasons=tuple(value if value is None else str(value) for value in record["terminal_reasons"]),
        state_checksums=tuple(str(value) for value in record["state_checksums"]),
        replay=gameplay,
    )


def _require_episode_summaries(record: dict[str, object], episode: EpisodeTrajectory[Any, Any]) -> None:
    if int(record.get("length", -1)) != episode.length:
        raise TrajectoryValidationError("episode length summary does not match transitions")
    if not math.isclose(float(record.get("episode_return", math.nan)), episode.episode_return, rel_tol=0, abs_tol=1e-9):
        raise TrajectoryValidationError("episode return summary is invalid")
    if not math.isclose(float(record.get("engine_return", math.nan)), episode.engine_return, rel_tol=0, abs_tol=1e-9):
        raise TrajectoryValidationError("engine return summary is invalid")
    if record.get("ending_reason") != episode.ending_reason:
        raise TrajectoryValidationError("episode ending reason summary is invalid")


def _episode_from_payload(
    payload: object,
    environment_config: object,
    observation_space: Space[Any],
    action_space: Space[Any],
    action_mask_space: Space[np.ndarray],
    gameplay: JsonObject | None = None,
) -> EpisodeTrajectory[Any, Any]:
    record = _require_mapping(payload, "episode")
    try:
        episode = _episode_from_record(record, environment_config, observation_space, action_space, action_mask_space, gameplay)
    except (KeyError, TypeError, ValueError) as error:
        raise TrajectoryValidationError("episode is missing required typed fields") from error
    validate_episode(episode)
    _require_episode_summaries(record, episode)
    return episode


def _validate_trajectory_episode_payload(payload: object, file_name: str) -> JsonObject:
    """Validate one self-contained trajectory episode, including its policy view."""
    try:
        trajectory = normalize_trajectory_episode(payload, file_name, DEFAULT_ENGINE_CONFIG.replay)
        policy = trajectory.get("policy")
        if policy is None:
            return trajectory
        policy_record = _require_mapping(policy, "trajectory episode policy")
        observation_space = space_from_descriptor(
            _require_mapping(policy_record.get("observation_space"), "trajectory observation space")
        )
        action_space = space_from_descriptor(
            _require_mapping(policy_record.get("action_space"), "trajectory action space")
        )
        action_mask_space = space_from_descriptor(
            _require_mapping(policy_record.get("action_mask_space"), "trajectory mask space")
        )
        replay = _require_mapping(trajectory.get("replay"), "trajectory episode replay")
        episode = _episode_from_payload(
            policy_record,
            policy_record.get("environment_config"),
            observation_space,
            action_space,
            action_mask_space,
            replay,
        )
        metadata = _require_mapping(trajectory.get("metadata"), "trajectory episode metadata")
        if (
            metadata.get("episodeId") != episode.episode_id
            or metadata.get("environmentSeed") != episode.environment_seed
            or metadata.get("agentSeed") != episode.agent_seed
            or metadata.get("policyId") != episode.policy_id
        ):
            raise TrajectoryValidationError("trajectory episode metadata disagrees with its policy data")
        return trajectory
    except TrajectoryValidationError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise TrajectoryValidationError("trajectory episode is malformed or invalid") from error


class TrajectoryWriter:
    """Write atomic policy shards, replayable episode artifacts, and a manifest."""

    def __init__(
        self,
        output_directory: Path | str,
        dataset_id: str,
        *,
        compression: Compression = "none",
        episodes_per_shard: int = 100,
        include_episode_artifacts: bool = True,
    ) -> None:
        if not dataset_id.strip():
            raise ValueError("dataset_id cannot be empty")
        if compression not in {"none", "gzip"}:
            raise ValueError("compression must be 'none' or 'gzip'")
        if not isinstance(episodes_per_shard, int) or isinstance(episodes_per_shard, bool) or episodes_per_shard <= 0:
            raise ValueError("episodes_per_shard must be a positive integer")
        if not isinstance(include_episode_artifacts, bool):
            raise TypeError("include_episode_artifacts must be a boolean")
        self.output_directory = Path(output_directory)
        if self.output_directory.exists():
            raise FileExistsError(f"trajectory dataset already exists: {self.output_directory}")
        self.output_directory.mkdir(parents=True)
        self.dataset_id = dataset_id
        self.compression = compression
        self.episodes_per_shard = episodes_per_shard
        self.include_episode_artifacts = include_episode_artifacts
        self._episode_count = 0
        self._transition_count = 0
        self._maximum_episode_length = 0
        self._shard_episode_count = 0
        self._shard_transition_count = 0
        self._shards: list[dict[str, object]] = []
        self._episode_artifacts: list[dict[str, object]] = []
        self._stream: io.TextIOWrapper | None = None
        self._raw_stream: io.BufferedWriter | None = None
        self._partial_path: Path | None = None
        self._metadata: dict[str, object] | None = None
        self._identifiers: dict[str, set[str]] = {
            "agent_ids": set(),
            "policy_sources": set(),
            "scenario_ids": set(),
            "task_ids": set(),
            "observation_schemas": set(),
            "action_schemas": set(),
            "reward_profiles": set(),
        }
        self._finalized = False

    def _write_episode_artifact(self, episode: EpisodeTrajectory[Any, Any]) -> str | None:
        """Write one standalone, lossless, UI-loadable trajectory episode."""
        if not self.include_episode_artifacts:
            return None
        if episode.replay is None:
            raise ValueError("trajectory episode has no replay projection")
        episode_directory = self.output_directory / "episodes"
        episode_directory.mkdir(exist_ok=True)
        relative = PurePosixPath("episodes") / f"episode-{self._episode_count:06d}.json"
        final_path = self.output_directory.joinpath(*relative.parts)
        partial_path = final_path.with_suffix(".json.partial")
        policy = {
            **_episode_payload(episode),
            "environment_config": episode.environment_config,
            "observation_space": describe_space(episode.observation_space),
            "action_space": describe_space(episode.action_space),
            "action_mask_space": describe_space(episode.action_mask_space),
        }
        serialized_policy = _json_value(policy)
        if not isinstance(serialized_policy, dict):
            raise TypeError("trajectory episode policy must serialize to a JSON object")
        artifact = create_trajectory_episode(
            DEFAULT_ENGINE_CONFIG.replay,
            episode.replay,
            episode_id=episode.episode_id,
            source="rollout",
            policy=serialized_policy,
            policy_id=episode.policy_id,
            agent_seed=episode.agent_seed,
        )
        partial_path.write_text(_canonical_json(artifact) + "\n", encoding="utf-8")
        os.replace(partial_path, final_path)
        digest = hashlib.sha256(final_path.read_bytes()).hexdigest()
        self._episode_artifacts.append({
            "episode_id": episode.episode_id,
            "path": str(relative),
            "sha256": digest,
            "step_count": episode.length,
        })
        return str(relative)

    def _start_shard(self) -> None:
        index = len(self._shards)
        suffix = ".jsonl.gz" if self.compression == "gzip" else ".jsonl"
        final_name = f"episodes-{index:05d}{suffix}"
        self._partial_path = self.output_directory / f"{final_name}.partial"
        self._raw_stream = self._partial_path.open("wb")
        if self.compression == "gzip":
            binary_stream = gzip.GzipFile(filename="", mode="wb", fileobj=self._raw_stream, mtime=0)
            self._stream = io.TextIOWrapper(binary_stream, encoding="utf-8", newline="\n")
        else:
            self._stream = io.TextIOWrapper(self._raw_stream, encoding="utf-8", newline="\n")

    def _set_metadata(self, episode: EpisodeTrajectory[Any, Any]) -> None:
        self._metadata = {
            "environment_config": _json_value(episode.environment_config),
            "observation_space": describe_space(episode.observation_space),
            "action_space": describe_space(episode.action_space),
            "action_mask_space": describe_space(episode.action_mask_space),
        }

    def _check_metadata(self, episode: EpisodeTrajectory[Any, Any]) -> None:
        descriptors = (
            ("observation_space", describe_space(episode.observation_space)),
            ("action_space", describe_space(episode.action_space)),
            ("action_mask_space", describe_space(episode.action_mask_space)),
        )
        if self._metadata is None:
            self._set_metadata(episode)
        if self._metadata["environment_config"] != _json_value(episode.environment_config):
            raise ValueError("episode environment configuration differs from earlier shards")
        for key, value in descriptors:
            if self._metadata[key] != value:
                raise ValueError(f"episode {key} differs from earlier shards")

    def write_episode(self, episode: EpisodeTrajectory[Any, Any]) -> None:
        """Validate and append one complete episode to the active shard."""
        if self._finalized:
            raise RuntimeError("trajectory writer has already been finalized")
        validate_episode(episode)
        episode_artifact = self._write_episode_artifact(episode)
        self._check_metadata(episode)
        for key, value in (
            ("agent_ids", episode.agent_id),
            ("policy_sources", episode.policy_id),
            ("scenario_ids", episode.scenario_id),
            ("task_ids", episode.task_id),
            ("observation_schemas", episode.observation_schema),
            ("action_schemas", episode.action_schema),
            ("reward_profiles", episode.reward_profile),
        ):
            self._identifiers[key].add(value)
        if self._stream is None:
            self._start_shard()
        if self._shard_episode_count >= self.episodes_per_shard:
            self._finish_shard()
            self._start_shard()
        assert self._stream is not None
        self._stream.write(_canonical_json(_episode_payload(episode, episode_artifact)))
        self._stream.write("\n")
        self._episode_count += 1
        self._transition_count += episode.length
        self._maximum_episode_length = max(self._maximum_episode_length, episode.length)
        self._shard_episode_count += 1
        self._shard_transition_count += episode.length

    def _finish_shard(self) -> None:
        if self._stream is None or self._partial_path is None:
            return
        raw_stream = self._raw_stream
        self._stream.flush()
        self._stream.close()
        if raw_stream is not None and not raw_stream.closed:
            raw_stream.flush()
            raw_stream.close()
        self._stream = None
        self._raw_stream = None
        final_name = self._partial_path.name.removesuffix(".partial")
        final_path = self.output_directory / final_name
        os.replace(self._partial_path, final_path)
        digest = hashlib.sha256(final_path.read_bytes()).hexdigest()
        self._shards.append({
            "path": final_name,
            "episode_count": self._shard_episode_count,
            "transition_count": self._shard_transition_count,
            "sha256": digest,
        })
        self._partial_path = None
        self._shard_episode_count = 0
        self._shard_transition_count = 0

    def finalize(self, *, max_episode_steps: int) -> Path:
        """Close the last shard and atomically publish the dataset manifest."""
        if self._finalized:
            raise RuntimeError("trajectory writer has already been finalized")
        if self._episode_count == 0 or self._metadata is None:
            raise ValueError("cannot finalize an empty trajectory dataset")
        if not isinstance(max_episode_steps, int) or isinstance(max_episode_steps, bool) or max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be a positive integer")
        if self._maximum_episode_length > max_episode_steps:
            raise ValueError("recorded episode exceeds max_episode_steps")
        self._finish_shard()
        self._finalized = True
        environment_config = self._metadata["environment_config"]
        manifest = {
            "schema_version": TRAJECTORY_SCHEMA,
            "dataset_id": self.dataset_id,
            "created_at": datetime.now(UTC).isoformat(),
            "application_version": _application_version(environment_config),
            "environment_config": environment_config,
            "observation_space": self._metadata["observation_space"],
            "action_space": self._metadata["action_space"],
            "action_mask_space": self._metadata["action_mask_space"],
            **{key: sorted(values) for key, values in self._identifiers.items()},
            "max_episode_steps": max_episode_steps,
            "compression": self.compression,
            "episodes_per_shard": self.episodes_per_shard,
            "include_episode_artifacts": self.include_episode_artifacts,
            "episode_count": self._episode_count,
            "transition_count": self._transition_count,
            "shards": self._shards,
            "episode_artifacts": self._episode_artifacts,
        }
        partial_manifest = self.output_directory / "manifest.json.partial"
        partial_manifest.write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
        manifest_path = self.output_directory / "manifest.json"
        os.replace(partial_manifest, manifest_path)
        return manifest_path


def _application_version(environment_config: object) -> object:
    if not isinstance(environment_config, dict):
        return None
    engine = environment_config.get("engine")
    if not isinstance(engine, dict):
        return None
    replay = engine.get("replay")
    if not isinstance(replay, dict):
        return None
    return replay.get("app_version")


def _require_positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TrajectoryValidationError(f"{label} must be a positive integer")
    return value


def _normalize_artifact_flags(payload: dict[str, object]) -> None:
    include_episode_artifacts = payload.get("include_episode_artifacts")
    if not isinstance(include_episode_artifacts, bool):
        raise TrajectoryValidationError("manifest include_episode_artifacts must be a boolean")
    episode_artifacts = payload.get("episode_artifacts")
    if not isinstance(episode_artifacts, list):
        raise TrajectoryValidationError("manifest episode_artifacts must be a list")


def _require_manifest_fields(payload: dict[str, object]) -> None:
    for field in (
        "dataset_id",
        "environment_config",
        "observation_space",
        "action_space",
        "action_mask_space",
        "agent_ids",
        "policy_sources",
        "scenario_ids",
        "task_ids",
        "observation_schemas",
        "action_schemas",
        "reward_profiles",
        "shards",
        "include_episode_artifacts",
        "episode_artifacts",
    ):
        if field not in payload:
            raise TrajectoryValidationError(f"trajectory manifest is missing {field}")
    if not isinstance(payload.get("dataset_id"), str) or not str(payload["dataset_id"]).strip():
        raise TrajectoryValidationError("trajectory dataset_id cannot be empty")
    for count_field in ("episode_count", "transition_count", "episodes_per_shard"):
        _require_positive_int(payload.get(count_field), f"manifest {count_field}")


def _manifest_payload(directory: Path) -> dict[str, object]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise TrajectoryValidationError("trajectory manifest.json is missing")
    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise TrajectoryValidationError("trajectory manifest is unreadable") from error
    payload = _require_mapping(_parse_json(manifest_text), "manifest")
    if payload.get("schema_version") != TRAJECTORY_SCHEMA:
        raise TrajectoryValidationError("unsupported trajectory schema version")
    if payload.get("compression") not in {"none", "gzip"}:
        raise TrajectoryValidationError("unsupported trajectory compression")
    _normalize_artifact_flags(payload)
    _require_manifest_fields(payload)
    return payload


def _shard_path(directory: Path, value: object) -> Path:
    if not isinstance(value, str):
        raise TrajectoryValidationError("shard path must be a string")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
        raise TrajectoryValidationError("shard path must be a safe dataset-relative filename")
    path = directory / value
    if not path.is_file() or not path.resolve().is_relative_to(directory.resolve()):
        raise TrajectoryValidationError("trajectory shard is missing or unsafe")
    return path


def _episode_artifact_path(directory: Path, value: object) -> Path:
    if not isinstance(value, str):
        raise TrajectoryValidationError("trajectory episode artifact path must be a string")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2 or relative.parts[0] != "episodes":
        raise TrajectoryValidationError("trajectory episode artifact path must be safe and dataset-relative")
    path = directory.joinpath(*relative.parts)
    if not path.is_file() or not path.resolve().is_relative_to(directory.resolve()):
        raise TrajectoryValidationError("trajectory episode artifact is missing or unsafe")
    return path


def _require_artifact_identity(entry: dict[str, object], artifacts: dict[str, tuple[str, JsonObject]], seen_paths: set[str]) -> tuple[str, str, str, int]:
    episode_id = entry.get("episode_id")
    relative_path = entry.get("path")
    digest = entry.get("sha256")
    step_count = entry.get("step_count")
    if not isinstance(episode_id, str) or not episode_id:
        raise TrajectoryValidationError("trajectory episode artifact episode_id is invalid")
    if episode_id in artifacts or not isinstance(relative_path, str) or relative_path in seen_paths:
        raise TrajectoryValidationError("trajectory episode artifact inventory contains duplicate entries")
    if not isinstance(digest, str) or len(digest) != 64:
        raise TrajectoryValidationError("trajectory episode artifact sha256 is invalid")
    return episode_id, relative_path, digest, _require_positive_int(step_count, "trajectory episode artifact step_count")


def _load_artifact_entry(
    directory: Path,
    entry: dict[str, object],
    artifacts: dict[str, tuple[str, JsonObject]],
    seen_paths: set[str],
) -> None:
    episode_id, relative_path, digest, step_count = _require_artifact_identity(entry, artifacts, seen_paths)
    path = _episode_artifact_path(directory, relative_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise TrajectoryValidationError("trajectory episode artifact checksum mismatch")
    try:
        payload = _require_mapping(_parse_json(path.read_text(encoding="utf-8")), "trajectory episode artifact")
        normalized = _validate_trajectory_episode_payload(payload, path.name)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise TrajectoryValidationError("trajectory episode artifact is malformed or invalid") from error
    replay = _require_mapping(normalized.get("replay"), "trajectory episode replay")
    integrity = _require_mapping(replay.get("integrity"), "trajectory episode replay integrity")
    if integrity.get("stepCount") != step_count:
        raise TrajectoryValidationError("trajectory episode artifact step count does not match its inventory")
    artifacts[episode_id] = (relative_path, normalized)
    seen_paths.add(relative_path)


def _load_episode_artifacts(
    directory: Path,
    manifest: dict[str, object],
) -> dict[str, tuple[str, JsonObject]]:
    include_artifacts = manifest["include_episode_artifacts"]
    inventory = manifest["episode_artifacts"]
    if not isinstance(inventory, list):
        raise TrajectoryValidationError("manifest episode_artifacts must be a list")
    if not include_artifacts:
        if inventory:
            raise TrajectoryValidationError("manifest disables episode artifacts but declares files")
        return {}
    if len(inventory) != manifest["episode_count"]:
        raise TrajectoryValidationError("trajectory episode artifact count does not match episode count")
    artifacts: dict[str, tuple[str, JsonObject]] = {}
    seen_paths: set[str] = set()
    for value in inventory:
        _load_artifact_entry(directory, _require_mapping(value, "trajectory episode artifact inventory entry"), artifacts, seen_paths)
    return artifacts


def _open_shard(path: Path, compression: object) -> io.TextIOBase:
    if compression == "gzip":
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return path.open(mode="rt", encoding="utf-8", newline="")


_IDENTIFIER_FIELDS = {
    "agent_ids": "agent_id",
    "policy_sources": "policy_id",
    "scenario_ids": "scenario_id",
    "task_ids": "task_id",
    "observation_schemas": "observation_schema",
    "action_schemas": "action_schema",
    "reward_profiles": "reward_profile",
}


def _declared_identifiers(manifest: dict[str, object]) -> dict[str, set[str]]:
    declared: dict[str, set[str]] = {}
    for manifest_key in _IDENTIFIER_FIELDS:
        values = manifest[manifest_key]
        if not isinstance(values, list) or not values or not all(isinstance(value, str) and value for value in values):
            raise TrajectoryValidationError(f"manifest {manifest_key} must contain identifiers")
        declared[manifest_key] = set(values)
    return declared


def _require_shard_header(directory: Path, shard_value: object) -> tuple[dict[str, object], Path]:
    shard = _require_mapping(shard_value, "shard")
    for count_field in ("episode_count", "transition_count"):
        _require_positive_int(shard.get(count_field), f"shard {count_field}")
    declared_digest = shard.get("sha256")
    if not isinstance(declared_digest, str) or len(declared_digest) != 64:
        raise TrajectoryValidationError("shard sha256 is invalid")
    path = _shard_path(directory, shard.get("path"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != declared_digest:
        raise TrajectoryValidationError("trajectory shard checksum mismatch")
    return shard, path


def _replay_from_artifact(
    manifest: dict[str, object],
    record: dict[str, object],
    episode_artifacts: dict[str, tuple[str, JsonObject]],
) -> JsonObject | None:
    episode_id = str(record.get("episode_id", ""))
    artifact_entry = episode_artifacts.get(episode_id)
    if manifest["include_episode_artifacts"]:
        if artifact_entry is None or record.get("episode_artifact") != artifact_entry[0]:
            raise TrajectoryValidationError("trajectory episode artifact reference is missing or inconsistent")
    elif "episode_artifact" in record:
        raise TrajectoryValidationError("episode references an undeclared trajectory artifact")
    if artifact_entry is None:
        return None
    return _require_mapping(artifact_entry[1].get("replay"), "trajectory episode replay")


def _require_policy_matches_shard(
    manifest: dict[str, object],
    record: dict[str, object],
    episode_artifacts: dict[str, tuple[str, JsonObject]],
) -> None:
    artifact_entry = episode_artifacts.get(str(record.get("episode_id", "")))
    if artifact_entry is None:
        return
    policy = _require_mapping(artifact_entry[1].get("policy"), "trajectory episode policy")
    expected_policy = dict(record)
    expected_policy.pop("episode_artifact", None)
    expected_policy.update({
        "environment_config": manifest["environment_config"],
        "observation_space": manifest["observation_space"],
        "action_space": manifest["action_space"],
        "action_mask_space": manifest["action_mask_space"],
    })
    if policy != expected_policy:
        raise TrajectoryValidationError("trajectory episode policy data disagrees with its shard")


def _require_episode_in_dataset(
    episode: EpisodeTrajectory[Any, Any],
    episode_ids: set[str],
    max_episode_steps: int,
) -> None:
    if episode.length > max_episode_steps:
        raise TrajectoryValidationError("episode exceeds the declared external time limit")
    if episode.episode_id in episode_ids:
        raise TrajectoryValidationError("trajectory episode IDs must be unique")
    episode_ids.add(episode.episode_id)


def _iter_shard_episodes(
    directory: Path,
    manifest: dict[str, object],
    shard_value: object,
    episode_artifacts: dict[str, tuple[str, JsonObject]],
    observation_space: Space[Any],
    action_space: Space[Any],
    action_mask_space: Space[np.ndarray],
    episode_ids: set[str],
    max_episode_steps: int,
) -> Iterator[EpisodeTrajectory[Any, Any]]:
    shard, path = _require_shard_header(directory, shard_value)
    episode_count = 0
    transition_count = 0
    try:
        with _open_shard(path, manifest["compression"]) as stream:
            for line in stream:
                if not line.strip():
                    raise TrajectoryValidationError("trajectory shard contains a blank record")
                record = _require_mapping(_parse_json(line), "episode")
                replay = _replay_from_artifact(manifest, record, episode_artifacts)
                episode = _episode_from_payload(
                    record,
                    manifest["environment_config"],
                    observation_space,
                    action_space,
                    action_mask_space,
                    replay,
                )
                _require_policy_matches_shard(manifest, record, episode_artifacts)
                _require_episode_in_dataset(episode, episode_ids, max_episode_steps)
                episode_count += 1
                transition_count += episode.length
                yield episode
    except (gzip.BadGzipFile, EOFError, OSError, UnicodeDecodeError) as error:
        raise TrajectoryValidationError("trajectory shard is truncated or unreadable") from error
    if episode_count != shard.get("episode_count") or transition_count != shard.get("transition_count"):
        raise TrajectoryValidationError("trajectory shard counts do not match its manifest")


def _iter_dataset(directory: Path) -> tuple[dict[str, object], Iterator[EpisodeTrajectory[Any, Any]]]:
    manifest = _manifest_payload(directory)
    observation_space = space_from_descriptor(_require_mapping(manifest["observation_space"], "observation space"))
    action_space = space_from_descriptor(_require_mapping(manifest["action_space"], "action space"))
    action_mask_space = space_from_descriptor(_require_mapping(manifest["action_mask_space"], "mask space"))
    episode_artifacts = _load_episode_artifacts(directory, manifest)
    shards = manifest["shards"]
    if not isinstance(shards, list) or not shards:
        raise TrajectoryValidationError("trajectory manifest must contain shards")
    declared_identifiers = _declared_identifiers(manifest)
    max_episode_steps = _require_positive_int(manifest.get("max_episode_steps"), "manifest max_episode_steps")

    def generate() -> Iterator[EpisodeTrajectory[Any, Any]]:
        dataset_episode_count = 0
        dataset_transition_count = 0
        episode_ids: set[str] = set()
        actual_identifiers = {manifest_key: set() for manifest_key in _IDENTIFIER_FIELDS}
        for shard_value in shards:
            for episode in _iter_shard_episodes(
                directory,
                manifest,
                shard_value,
                episode_artifacts,
                observation_space,
                action_space,
                action_mask_space,
                episode_ids,
                max_episode_steps,
            ):
                dataset_episode_count += 1
                dataset_transition_count += episode.length
                for manifest_key, episode_attribute in _IDENTIFIER_FIELDS.items():
                    actual_identifiers[manifest_key].add(getattr(episode, episode_attribute))
                yield episode
        if dataset_episode_count != manifest.get("episode_count") or dataset_transition_count != manifest.get("transition_count"):
            raise TrajectoryValidationError("dataset counts do not match the manifest")
        if manifest["include_episode_artifacts"] and set(episode_artifacts) != episode_ids:
            raise TrajectoryValidationError("trajectory episode artifact inventory does not match trajectory episodes")
        for field, values in actual_identifiers.items():
            if declared_identifiers[field] != values:
                raise TrajectoryValidationError(f"manifest {field} does not match recorded episodes")

    return manifest, generate()


def iter_trajectory_episodes(path: Path | str) -> Iterator[EpisodeTrajectory[Any, Any]]:
    """Yield validated episodes from a manifest-backed JSONL dataset."""
    _, episodes = _iter_dataset(Path(path))
    yield from episodes


def validate_trajectory_episode(path: Path | str) -> JsonObject:
    """Validate and return one standalone `aresim.trajectory.episode.v1` file."""
    episode_path = Path(path)
    try:
        payload = _require_mapping(
            _parse_json(episode_path.read_text(encoding="utf-8")),
            "trajectory episode",
        )
    except (OSError, UnicodeDecodeError) as error:
        raise TrajectoryValidationError("trajectory episode is unreadable") from error
    trajectory = _validate_trajectory_episode_payload(payload, episode_path.name)
    if trajectory.get("schemaVersion") != TRAJECTORY_EPISODE_SCHEMA:
        raise TrajectoryValidationError("unsupported trajectory episode schema version")
    return trajectory


def validate_trajectory_dataset(path: Path | str) -> TrajectoryManifest:
    """Validate every shard, episode, space, checksum, and declared count."""
    manifest, episodes = _iter_dataset(Path(path))
    for _ in episodes:
        pass
    return TrajectoryManifest(dict(manifest))


__all__ = [
    "EpisodeTrajectory",
    "TRAJECTORY_SCHEMA",
    "TrajectoryManifest",
    "TrajectoryValidationError",
    "TrajectoryWriter",
    "describe_space",
    "iter_trajectory_episodes",
    "space_from_descriptor",
    "validate_episode",
    "validate_trajectory_episode",
    "validate_trajectory_dataset",
]
