"""Trajectory replay projections, legacy gameplay import, and reconstruction.

New UI exports use ``aresim.trajectory.episode.v1`` with nested
``aresim.trajectory.replay.v1`` projections. Legacy ``aresim.gameplay.v1`` remains
import-only. Replay rebuilds snapshots from checkpoints plus step deltas without
re-running :func:`aresim.core.rules.apply_action`.

**Last updated:** September 1, 2026

**Contains:** ``TrajectoryRecorder``, ``ReplayCursor``, normalization helpers,
replay projection builders.

**Schemas:** ``aresim.trajectory.episode.v1``, ``aresim.trajectory.replay.v1``,
``aresim.gameplay.v1`` (legacy import).

**See also:** :mod:`aresim.training.trajectories` (RL dataset writer).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import PurePath
from typing import Literal

from .config import ReplayConfig

JsonObject = dict[str, object]

CHECKPOINT_REASON_PRIORITY = {"initial": 0, "interval": 1, "event": 2, "final": 3}
TRAJECTORY_EPISODE_SCHEMA = "aresim.trajectory.episode.v1"
TRAJECTORY_REPLAY_SCHEMA = "aresim.trajectory.replay.v1"


def _changed(before: object, after: object) -> bool:
    return before != after


def _changed_by_id(before: list[JsonObject], after: list[JsonObject]) -> list[JsonObject]:
    before_by_id = {str(item.get("id")): item for item in before}
    return [deepcopy(item) for item in after if before_by_id.get(str(item.get("id"))) != item]


def _changed_row_cells(previous_row: object, row: list[object]) -> list[JsonObject]:
    previous = previous_row if isinstance(previous_row, list) else []
    changed: list[JsonObject] = []
    for x, cell in enumerate(row):
        if not isinstance(cell, dict):
            continue
        prior = previous[x] if x < len(previous) else None
        if prior != cell:
            changed.append(deepcopy(cell))
    return changed


def _changed_terrain(before: JsonObject, after: JsonObject) -> list[JsonObject]:
    previous_rows = before["terrain"]
    current_rows = after["terrain"]
    if not isinstance(previous_rows, list) or not isinstance(current_rows, list):
        return []
    changed: list[JsonObject] = []
    for y, row in enumerate(current_rows):
        if not isinstance(row, list):
            continue
        previous_row = previous_rows[y] if y < len(previous_rows) else []
        changed.extend(_changed_row_cells(previous_row, row))
    return changed


def _snapshot_rover(after: JsonObject) -> JsonObject:
    rovers = after.get("rovers")
    if not isinstance(rovers, list) or not rovers or not isinstance(rovers[0], dict):
        raise ValueError("snapshot has no rover")
    return rovers[0]


def _snapshot_history_entry(after: JsonObject) -> JsonObject:
    history = after.get("history")
    if not isinstance(history, list) or not history or not isinstance(history[0], dict):
        raise ValueError("snapshot has no current history entry")
    return history[0]


def create_step_delta(before: JsonObject, after: JsonObject) -> JsonObject:
    """Compact diff between two UI snapshots. Used when recording a live session."""
    entry = _snapshot_history_entry(after)
    rover = _snapshot_rover(after)
    before_rovers = before.get("rovers") if isinstance(before.get("rovers"), list) else []
    after_rovers = after.get("rovers") if isinstance(after.get("rovers"), list) else []
    before_structures = before.get("structures") if isinstance(before.get("structures"), list) else []
    after_structures = after.get("structures") if isinstance(after.get("structures"), list) else []
    changes: JsonObject = {
        "terrainCells": _changed_terrain(before, after),
        "rovers": _changed_by_id(before_rovers, after_rovers),
        "structures": _changed_by_id(before_structures, after_structures),
        "status": {
            "step": after["step"],
            "sol": after["sol"],
            "localTime": after["localTime"],
            "gameStatus": after["gameStatus"],
            "statusReason": after["statusReason"],
            "weather": after["weather"],
            "dustIntensity": after["dustIntensity"],
        },
        "appendedHistoryEntry": deepcopy(entry),
    }
    for key in ("resources", "objectiveStats", "buildPadState", "mission", "rules"):
        if _changed(before.get(key), after.get(key)):
            changes[key] = deepcopy(after[key])
    result: JsonObject = {
        "step": entry["step"],
        "sol": after["sol"],
        "localTime": after["localTime"],
        "actor": entry["actor"],
        "action": entry["action"],
        "result": entry["result"],
        "events": deepcopy(entry["events"]),
        "reward": entry["reward"],
        "rewardTerms": deepcopy(entry["rewardTerms"]),
        "resourceDelta": deepcopy(entry["resourceDelta"]),
        "changes": changes,
        "after": {
            "rover": {"x": rover["x"], "y": rover["y"], "battery": rover["battery"], "health": rover["health"]},
            "gameStatus": after["gameStatus"],
            "totalReward": after["objectiveStats"]["rewardTotals"]["total"],
        },
    }
    target = entry.get("target")
    if target is not None:
        result["target"] = deepcopy(target)
    return result


def _apply_status_changes(current: JsonObject, delta: JsonObject, changes: JsonObject) -> None:
    status = changes.get("status")
    if isinstance(status, dict):
        for key in ("step", "sol", "localTime", "gameStatus", "statusReason", "weather", "dustIntensity"):
            if key in status:
                current[key] = deepcopy(status[key])
        return
    current["step"] = delta["step"]
    current["sol"] = delta["sol"]
    current["localTime"] = delta["localTime"]


def _apply_objective_stats(current: JsonObject, changes: JsonObject) -> None:
    if "objectiveStats" not in changes or not isinstance(changes["objectiveStats"], dict):
        return
    incoming = deepcopy(changes["objectiveStats"])
    existing = current.get("objectiveStats") if isinstance(current.get("objectiveStats"), dict) else {}
    incoming_totals = incoming.get("rewardTotals") if isinstance(incoming.get("rewardTotals"), dict) else {}
    existing_totals = existing.get("rewardTotals") if isinstance(existing.get("rewardTotals"), dict) else {}
    current["objectiveStats"] = {**existing, **incoming, "rewardTotals": {**existing_totals, **incoming_totals}}


def _apply_terrain_cells(current: JsonObject, changes: JsonObject) -> None:
    terrain = current.get("terrain")
    cells = changes.get("terrainCells", [])
    if not isinstance(terrain, list) or not isinstance(cells, list):
        return
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        x, y = cell.get("x"), cell.get("y")
        if isinstance(x, int) and isinstance(y, int) and 0 <= y < len(terrain) and isinstance(terrain[y], list) and 0 <= x < len(terrain[y]):
            terrain[y][x] = deepcopy(cell)


def _replace_entities_by_id(entities: object, changed_entities: object) -> None:
    if not isinstance(entities, list) or not isinstance(changed_entities, list):
        return
    for changed_entity in changed_entities:
        if not isinstance(changed_entity, dict):
            continue
        index = next((index for index, entity in enumerate(entities) if isinstance(entity, dict) and entity.get("id") == changed_entity.get("id")), -1)
        if index >= 0:
            entities[index] = deepcopy(changed_entity)


def _prepend_history_entry(current: JsonObject, changes: JsonObject) -> None:
    appended = changes.get("appendedHistoryEntry")
    history = current.get("history")
    if not isinstance(appended, dict) or not isinstance(history, list):
        return
    current["history"] = [deepcopy(appended), *[entry for entry in history if not isinstance(entry, dict) or entry.get("id") != appended.get("id")]]


def apply_step_delta(snapshot: JsonObject, delta: JsonObject) -> JsonObject:
    """Apply one recorded delta and return a new snapshot. Does not call the simulator."""
    current = deepcopy(snapshot)
    changes = delta.get("changes")
    if not isinstance(changes, dict):
        raise ValueError("replay delta has no changes")
    _apply_status_changes(current, delta, changes)
    for key in ("resources", "buildPadState", "mission", "rules"):
        if key in changes:
            current[key] = deepcopy(changes[key])
    _apply_objective_stats(current, changes)
    _apply_terrain_cells(current, changes)
    for key in ("rovers", "structures"):
        _replace_entities_by_id(current.get(key), changes.get(key))
    _prepend_history_entry(current, changes)
    current["mode"] = "Replay"
    return current


def _is_significant(delta: JsonObject) -> bool:
    action = delta.get("action")
    events = " ".join(str(event) for event in delta.get("events", [])).lower()
    return action in {"invalid", "extract", "unload", "build", "service"} or any(term in events for term in ("warning", "hazard", "game over", "exploration ended"))


def _checkpoint(snapshot: JsonObject, reason: str, label: str, summary: str, delta: JsonObject | None = None) -> JsonObject:
    step = int(snapshot["step"])
    result: JsonObject = {
        "id": f"checkpoint-{step}-{reason}",
        "step": step,
        "label": label,
        "reason": reason,
        "summary": summary,
        "snapshot": deepcopy(snapshot),
    }
    if delta is not None:
        if "target" in delta:
            result["target"] = deepcopy(delta["target"])
        result["reward"] = delta["reward"]
    return result


@dataclass
class TrajectoryRecorder:
    """Accumulate replay deltas and export unified trajectory episodes.

    `export` builds the downloadable JSON. Interval checkpoints fire every N
    steps; event checkpoints fire on extract, unload, build, service, invalid,
    and warning/game-over text.
    """
    replay_config: ReplayConfig
    initial_snapshot: JsonObject
    current_snapshot: JsonObject
    steps: list[JsonObject]
    checkpoints: list[JsonObject]
    checkpoint_mode: Literal["full", "endpoints"]

    @classmethod
    def create(
        cls,
        replay_config: ReplayConfig,
        snapshot: JsonObject,
        *,
        checkpoint_mode: Literal["full", "endpoints"] = "full",
    ) -> "TrajectoryRecorder":
        """Start recording, optionally retaining only initial/final checkpoints."""
        initial = deepcopy(snapshot)
        return cls(
            replay_config=replay_config,
            initial_snapshot=initial,
            current_snapshot=deepcopy(snapshot),
            steps=[],
            checkpoints=[_checkpoint(initial, "initial", "Initial", f"Seed {initial['seed']}")],
            checkpoint_mode=checkpoint_mode,
        )

    def record(self, before: JsonObject, after: JsonObject) -> None:
        """Append one step. Call only when the snapshot `step` actually advanced."""
        delta = create_step_delta(before, after)
        self.steps.append(delta)
        self.current_snapshot = deepcopy(after)
        if self.checkpoint_mode == "endpoints":
            return
        step = int(after["step"])
        if step % self.replay_config.checkpoint_interval == 0:
            self.checkpoints.append(_checkpoint(after, "interval", f"Step {step}", str(delta["result"]), delta))
        if _is_significant(delta):
            self.checkpoints.append(_checkpoint(after, "event", f"Step {step}", str(delta["result"]), delta))

    def export_replay_projection(
        self,
        file_name: str,
        run_mode: Literal["manual", "algorithm"],
        algorithm_id: str | None = None,
        checkpoint_path: str | None = None,
    ) -> JsonObject:
        """Build the nested replay projection, including a final checkpoint."""
        safe_name = sanitize_file_name(file_name, self.current_snapshot)
        final_snapshot = deepcopy(self.current_snapshot)
        checkpoints = [deepcopy(checkpoint) for checkpoint in self.checkpoints if checkpoint.get("reason") != "final"]
        final_delta = self.steps[-1] if self.steps else None
        checkpoints.append(_checkpoint(final_snapshot, "final", "Final", str(final_snapshot["statusReason"]), final_delta))
        metadata: JsonObject = {
            "sessionId": final_snapshot["sessionId"],
            "seed": final_snapshot["seed"],
            "runMode": run_mode,
            "totalSteps": final_snapshot["step"],
            "finalStatus": final_snapshot["gameStatus"],
        }
        if run_mode == "algorithm" and algorithm_id is not None:
            metadata["algorithmId"] = algorithm_id
        if run_mode == "algorithm" and checkpoint_path:
            metadata["checkpointPath"] = checkpoint_path
        return {
            # ReplayConfig.schema_version is the legacy import name (`aresim.gameplay.v1`).
            "schemaVersion": TRAJECTORY_REPLAY_SCHEMA,
            "savedAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "fileName": safe_name,
            "appVersion": self.replay_config.app_version,
            "metadata": metadata,
            "initialSnapshot": deepcopy(self.initial_snapshot),
            "steps": deepcopy(self.steps),
            "checkpoints": checkpoints,
            "finalSnapshot": final_snapshot,
            "integrity": {
                "finalStep": final_snapshot["step"],
                "stepCount": len(self.steps),
                "checkpointCount": len(checkpoints),
            },
        }

    def export_trajectory(
        self,
        file_name: str,
        run_mode: Literal["manual", "algorithm"],
        algorithm_id: str | None = None,
        checkpoint_path: str | None = None,
    ) -> JsonObject:
        """Build the public UI export directly as a unified trajectory episode."""
        replay = self.export_replay_projection(file_name, run_mode, algorithm_id, checkpoint_path=checkpoint_path)
        return create_trajectory_episode(
            self.replay_config,
            replay,
            episode_id=str(self.current_snapshot["sessionId"]),
            source="ui",
            policy=None,
            policy_id=algorithm_id,
        )


def create_trajectory_episode(
    replay_config: ReplayConfig,
    replay: JsonObject,
    *,
    episode_id: str,
    source: Literal["ui", "rollout"],
    policy: JsonObject | None,
    policy_id: str | None = None,
    agent_seed: int | None = None,
) -> JsonObject:
    """Combine complete replay state and optional policy tensors in one artifact."""
    if not episode_id.strip():
        raise ValueError("trajectory episode_id cannot be empty")
    if isinstance(replay, dict) and replay.get("schemaVersion") == TRAJECTORY_REPLAY_SCHEMA:
        normalized_replay = deepcopy(replay)
    else:
        normalized_replay = normalize_gameplay_payload(
            replay,
            str(replay.get("fileName", f"{episode_id}.json")),
            replay_config,
        )
    metadata: JsonObject = {
        "episodeId": episode_id,
        "source": source,
        "environmentSeed": normalized_replay["metadata"]["seed"],
        "policyId": policy_id,
        "agentSeed": agent_seed,
    }
    return {
        "schemaVersion": TRAJECTORY_EPISODE_SCHEMA,
        "savedAt": normalized_replay["savedAt"],
        "metadata": metadata,
        "policy": deepcopy(policy),
        "replay": normalized_replay,
    }


def sanitize_file_name(file_name: str, snapshot: JsonObject) -> str:
    """Keep only the basename, force `.json`, and cap length for downloads."""
    default_name = f"aresim-seed-{snapshot['seed']}-step-{snapshot['step']}.json"
    candidate = PurePath(file_name.strip()).name if file_name.strip() else default_name
    if not candidate.lower().endswith(".json"):
        candidate += ".json"
    return candidate[:180]


def _is_snapshot(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        isinstance(value.get("sessionId"), str)
        and isinstance(value.get("seed"), int)
        and isinstance(value.get("step"), int)
        and isinstance(value.get("terrain"), list)
        and isinstance(value.get("rovers"), list)
        and isinstance(value.get("history"), list)
    )


def _hydrate_snapshot(snapshot: JsonObject) -> JsonObject:
    current = deepcopy(snapshot)
    rovers = current.get("rovers")
    if isinstance(rovers, list):
        for rover in rovers:
            if isinstance(rover, dict):
                rover.setdefault("cargoIce", 0)
                rover.setdefault("cargoOre", 0)
                rover.setdefault("cargoSamples", 0)
                rover.setdefault("cargoCapacityKg", 12)
    stats = current.get("objectiveStats")
    if isinstance(stats, dict):
        for key, value in (
            ("iceDelivered", 0),
            ("samplesCollected", 0),
            ("samplesDelivered", 0),
            ("unloadCount", 0),
            ("iceSitesExtracted", 0),
        ):
            stats.setdefault(key, value)
        totals = stats.get("rewardTotals")
        if isinstance(totals, dict):
            totals.setdefault("traversal", 0)
            totals.setdefault("blockedPenalty", 0)
            totals.setdefault("delivered", 0)
    return current


def _require_metadata(payload: JsonObject) -> JsonObject:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("sessionId"), str) or not isinstance(metadata.get("seed"), int):
        raise ValueError("gameplay metadata is invalid")
    return metadata


def _require_integrity(payload: JsonObject) -> JsonObject:
    integrity = payload.get("integrity")
    if not isinstance(integrity, dict) or not all(isinstance(integrity.get(key), int) for key in ("finalStep", "stepCount", "checkpointCount")):
        raise ValueError("gameplay integrity metadata is invalid")
    return integrity


def _require_gameplay_envelope(payload: JsonObject) -> tuple[JsonObject, JsonObject, list[object], list[object], JsonObject, JsonObject]:
    metadata = _require_metadata(payload)
    integrity = _require_integrity(payload)
    steps = payload.get("steps")
    checkpoints = payload.get("checkpoints")
    initial = payload.get("initialSnapshot")
    final = payload.get("finalSnapshot")
    if not isinstance(steps, list) or not isinstance(checkpoints, list):
        raise ValueError("gameplay timeline is invalid")
    if not _is_snapshot(initial) or not _is_snapshot(final):
        raise ValueError("gameplay snapshots are invalid")
    return metadata, integrity, steps, checkpoints, initial, final


def _reconstruct_steps(initial: JsonObject, steps: list[object]) -> tuple[JsonObject, dict[int, JsonObject]]:
    previous_step = int(initial["step"])
    reconstructed = deepcopy(initial)
    reconstructed_by_step: dict[int, JsonObject] = {previous_step: deepcopy(reconstructed)}
    for delta in steps:
        if not isinstance(delta, dict) or not isinstance(delta.get("step"), int) or not isinstance(delta.get("changes"), dict):
            raise ValueError("gameplay contains an invalid step delta")
        if int(delta["step"]) != previous_step + 1:
            raise ValueError("gameplay step deltas must be contiguous")
        previous_step = int(delta["step"])
        reconstructed = apply_step_delta(reconstructed, delta)
        reconstructed_by_step[previous_step] = deepcopy(reconstructed)
    return reconstructed, reconstructed_by_step


def _require_checkpoint_shapes(checkpoints: list[object]) -> None:
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("step"), int) or not _is_snapshot(checkpoint.get("snapshot")):
            raise ValueError("gameplay checkpoint is invalid")


def _require_integrity_counts(
    metadata: JsonObject,
    integrity: JsonObject,
    steps: list[object],
    checkpoints: list[object],
    initial: JsonObject,
    final: JsonObject,
) -> None:
    if integrity["stepCount"] != len(steps) or integrity["checkpointCount"] != len(checkpoints):
        raise ValueError("gameplay integrity counts do not match its timeline")
    if integrity["finalStep"] != final["step"] or metadata.get("totalSteps") != final["step"]:
        raise ValueError("gameplay final-step metadata is inconsistent")
    if steps and steps[-1]["step"] != final["step"]:
        raise ValueError("gameplay final snapshot does not match the last step")
    _require_endpoint_checkpoints(checkpoints, initial, final)


def _require_matching_checkpoint(checkpoint: JsonObject, snapshot: JsonObject, label: str) -> None:
    if checkpoint["step"] != snapshot["step"] or checkpoint["snapshot"] != snapshot:
        raise ValueError(f"gameplay {label} checkpoint disagrees with its {label} snapshot")


def _require_endpoint_checkpoints(checkpoints: list[object], initial: JsonObject, final: JsonObject) -> None:
    initial_checkpoints = [item for item in checkpoints if isinstance(item, dict) and item.get("reason") == "initial"]
    final_checkpoints = [item for item in checkpoints if isinstance(item, dict) and item.get("reason") == "final"]
    if not initial_checkpoints or not final_checkpoints:
        raise ValueError("gameplay must contain initial and final checkpoints")
    _require_matching_checkpoint(initial_checkpoints[0], initial, "initial")
    _require_matching_checkpoint(final_checkpoints[-1], final, "final")


def _require_reconstructed_final(initial: JsonObject, final: JsonObject, steps: list[object], reconstructed: JsonObject) -> None:
    # Legacy checkpoint-only files can jump to a later final snapshot but do
    # not claim to contain invented intermediate deltas.
    if not steps and initial["step"] != final["step"]:
        return
    reconstructed_final = deepcopy(reconstructed)
    expected_final = deepcopy(final)
    reconstructed_final["mode"] = expected_final["mode"] = "Replay"
    if expected_final.get("gameStatus") == "paused":
        reconstructed_final["gameStatus"] = expected_final["gameStatus"]
        reconstructed_final["statusReason"] = expected_final["statusReason"]
    if reconstructed_final != expected_final:
        raise ValueError("gameplay final snapshot disagrees with reconstructed step data")


def _require_checkpoint_reconstruction(checkpoints: list[object], reconstructed_by_step: dict[int, JsonObject]) -> None:
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict) or checkpoint.get("reason") in {"initial", "final"}:
            continue
        expected = reconstructed_by_step.get(int(checkpoint["step"]))
        actual = deepcopy(checkpoint["snapshot"])
        if expected is None:
            raise ValueError("gameplay checkpoint does not reference a recorded step")
        expected["mode"] = "Replay"
        actual["mode"] = "Replay"
        if actual != expected:
            raise ValueError("gameplay checkpoint disagrees with reconstructed step data")


def _validate_canonical_gameplay(payload: JsonObject) -> None:
    metadata, integrity, steps, checkpoints, initial, final = _require_gameplay_envelope(payload)
    reconstructed, reconstructed_by_step = _reconstruct_steps(initial, steps)
    _require_checkpoint_shapes(checkpoints)
    _require_integrity_counts(metadata, integrity, steps, checkpoints, initial, final)
    _require_reconstructed_final(initial, final, steps, reconstructed)
    _require_checkpoint_reconstruction(checkpoints, reconstructed_by_step)


def _normalize_canonical_replay(payload: JsonObject) -> JsonObject:
    required = ("initialSnapshot", "steps", "checkpoints", "finalSnapshot", "metadata", "integrity")
    if not all(key in payload for key in required):
        raise ValueError("gameplay file is missing required fields")
    _validate_canonical_gameplay(payload)
    normalized = deepcopy(payload)
    # Import-only `aresim.gameplay.v1` files keep their payload; the public stamp is trajectory.replay.v1.
    normalized["schemaVersion"] = TRAJECTORY_REPLAY_SCHEMA
    normalized["initialSnapshot"] = _hydrate_snapshot(normalized["initialSnapshot"])
    normalized["finalSnapshot"] = _hydrate_snapshot(normalized["finalSnapshot"])
    for checkpoint in normalized["checkpoints"]:
        checkpoint["snapshot"] = _hydrate_snapshot(checkpoint["snapshot"])
    return normalized


def _legacy_snapshot_payload(payload: object) -> tuple[JsonObject, str] | None:
    saved_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    if _is_snapshot(payload):
        return _hydrate_snapshot(payload), saved_at
    if isinstance(payload, dict) and _is_snapshot(payload.get("snapshot")):
        snapshot = _hydrate_snapshot(payload["snapshot"])
        if isinstance(payload.get("savedAt"), str):
            saved_at = payload["savedAt"]
        return snapshot, saved_at
    return None


def _wrap_snapshot_as_replay(snapshot: JsonObject, file_name: str, saved_at: str, replay_config: ReplayConfig) -> JsonObject:
    safe_name = sanitize_file_name(file_name, snapshot)
    initial = deepcopy(snapshot)
    initial["mode"] = "Replay"
    final = deepcopy(initial)
    checkpoints = [
        _checkpoint(initial, "initial", "Initial", f"Seed {initial['seed']}"),
        _checkpoint(final, "final", "Final", str(final.get("statusReason", "Loaded snapshot"))),
    ]
    return {
        "schemaVersion": TRAJECTORY_REPLAY_SCHEMA,
        "savedAt": saved_at,
        "fileName": safe_name,
        "appVersion": replay_config.app_version,
        "metadata": {
            "sessionId": final["sessionId"],
            "seed": final["seed"],
            "runMode": "load",
            "totalSteps": final["step"],
            "finalStatus": final["gameStatus"],
        },
        "initialSnapshot": initial,
        "steps": [],
        "checkpoints": checkpoints,
        "finalSnapshot": final,
        "integrity": {"finalStep": final["step"], "stepCount": 0, "checkpointCount": len(checkpoints)},
    }


def normalize_gameplay_payload(payload: object, file_name: str, replay_config: ReplayConfig) -> JsonObject:
    """Accept canonical v1 files or a legacy raw snapshot, and fill missing cargo fields.

    Raises `UnsupportedGameplaySchema` for unknown schema versions and `ValueError`
    for malformed JSON structure.
    """
    if isinstance(payload, dict) and payload.get("schemaVersion") == TRAJECTORY_EPISODE_SCHEMA:
        replay = payload.get("replay")
        if not isinstance(replay, dict):
            raise ValueError("trajectory episode is missing its replay data")
        return normalize_gameplay_payload(replay, file_name, replay_config)
    supported_schemas = {None, TRAJECTORY_REPLAY_SCHEMA, replay_config.schema_version}
    if isinstance(payload, dict) and payload.get("schemaVersion") not in supported_schemas:
        raise UnsupportedGameplaySchema(str(payload.get("schemaVersion")))
    if isinstance(payload, dict) and payload.get("schemaVersion") in {TRAJECTORY_REPLAY_SCHEMA, replay_config.schema_version}:
        return _normalize_canonical_replay(payload)
    legacy = _legacy_snapshot_payload(payload)
    if legacy is None:
        raise ValueError("gameplay JSON must contain an AresSim gameplay or snapshot")
    snapshot, saved_at = legacy
    return _wrap_snapshot_as_replay(snapshot, file_name, saved_at, replay_config)


def _require_optional_policy_id(policy_id: object) -> None:
    if policy_id is not None and (not isinstance(policy_id, str) or not policy_id.strip()):
        raise ValueError("trajectory episode policy identifier is invalid")


def _require_optional_agent_seed(agent_seed: object) -> None:
    if agent_seed is not None and (not isinstance(agent_seed, int) or isinstance(agent_seed, bool)):
        raise ValueError("trajectory episode agent seed is invalid")


def _require_episode_metadata(payload: JsonObject) -> None:
    metadata = payload.get("metadata")
    policy = payload.get("policy")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("episodeId"), str) or not str(metadata["episodeId"]).strip():
        raise ValueError("trajectory episode metadata is invalid")
    if metadata.get("source") not in {"ui", "rollout"}:
        raise ValueError("trajectory episode source is invalid")
    _require_optional_policy_id(metadata.get("policyId"))
    _require_optional_agent_seed(metadata.get("agentSeed"))
    if policy is not None and not isinstance(policy, dict):
        raise ValueError("trajectory episode policy data must be an object or null")


def normalize_trajectory_episode(
    payload: object,
    file_name: str,
    replay_config: ReplayConfig,
) -> JsonObject:
    """Validate a unified trajectory episode or wrap a legacy gameplay file."""
    if not isinstance(payload, dict) or payload.get("schemaVersion") != TRAJECTORY_EPISODE_SCHEMA:
        replay = normalize_gameplay_payload(payload, file_name, replay_config)
        session_id = str(replay["metadata"]["sessionId"])
        return create_trajectory_episode(
            replay_config,
            replay,
            episode_id=session_id,
            source="ui",
            policy=None,
            policy_id=None,
            agent_seed=None,
        )
    _require_episode_metadata(payload)
    normalized = deepcopy(payload)
    normalized["replay"] = normalize_gameplay_payload(payload.get("replay"), file_name, replay_config)
    if payload["metadata"].get("environmentSeed") != normalized["replay"]["metadata"]["seed"]:
        raise ValueError("trajectory episode environment seed disagrees with its replay")
    normalized["savedAt"] = normalized["replay"]["savedAt"]
    return normalized


class UnsupportedGameplaySchema(ValueError):
    """The upload used a schema this engine build does not know how to read."""
    def __init__(self, schema: str) -> None:
        super().__init__(f"unsupported gameplay schema: {schema}")
        self.schema = schema


@dataclass
class ReplayCursor:
    """Navigate a loaded gameplay file by reconstructing snapshots from deltas.

    `jump` seeks the best checkpoint at or before the target step, then applies
    remaining deltas. The process is reconstruction, not a new simulation.
    """
    replay_id: str
    file_name: str
    gameplay: JsonObject
    snapshot: JsonObject
    cursor: int

    @classmethod
    def create(cls, gameplay: JsonObject, file_name: str) -> "ReplayCursor":
        """Open a normalized gameplay payload at its initial snapshot."""
        initial = deepcopy(gameplay["initialSnapshot"])
        initial["mode"] = "Replay"
        session_id = str(gameplay["metadata"]["sessionId"])
        return cls(f"replay_{session_id}", file_name, gameplay, initial, int(initial["step"]))

    def reset(self) -> JsonObject:
        """Return to the initial snapshot."""
        self.snapshot = deepcopy(self.gameplay["initialSnapshot"])
        self.snapshot["mode"] = "Replay"
        self.cursor = int(self.snapshot["step"])
        return deepcopy(self.snapshot)

    def step(self) -> JsonObject:
        """Advance one recorded step, or stay put at the end of the file."""
        steps = self.gameplay["steps"]
        if not isinstance(steps, list):
            return deepcopy(self.snapshot)
        delta = next((item for item in steps if isinstance(item, dict) and item.get("step") == self.cursor + 1), None)
        if delta is not None:
            self.snapshot = apply_step_delta(self.snapshot, delta)
            self.cursor += 1
        return deepcopy(self.snapshot)

    def jump(self, target_step: int) -> JsonObject:
        """Seek to `target_step`, clamped to the recorded range."""
        final_step = int(self.gameplay["integrity"]["finalStep"])
        target = max(0, min(final_step, target_step))
        checkpoints = [
            checkpoint
            for checkpoint in self.gameplay["checkpoints"]
            if isinstance(checkpoint, dict) and isinstance(checkpoint.get("step"), int) and checkpoint["step"] <= target
        ]
        if not checkpoints:
            raise ValueError("replay has no checkpoint before target step")
        checkpoint = max(
            checkpoints,
            key=lambda item: (int(item["step"]), CHECKPOINT_REASON_PRIORITY.get(str(item.get("reason")), -1)),
        )
        self.snapshot = deepcopy(checkpoint["snapshot"])
        self.snapshot["mode"] = "Replay"
        self.cursor = int(checkpoint["step"])
        steps = self.gameplay["steps"]
        if isinstance(steps, list):
            for delta in steps:
                if isinstance(delta, dict) and self.cursor < int(delta.get("step", -1)) <= target:
                    self.snapshot = apply_step_delta(self.snapshot, delta)
                    self.cursor = int(delta["step"])
        return deepcopy(self.snapshot)
