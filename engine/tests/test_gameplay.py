"""Save recording, legacy-snapshot import, and `ReplayCursor` reconstruction tests."""

from __future__ import annotations

from copy import deepcopy

import pytest

from aresim.core.engine import AresEngine
from aresim.defaults import DEFAULT_ENGINE_CONFIG
from aresim.gameplay import (
    TrajectoryRecorder,
    ReplayCursor,
    create_trajectory_episode,
    normalize_gameplay_payload,
    normalize_trajectory_episode,
)
from aresim.integrations.ui import snapshot_from_state
from aresim.types import ActionCommand, ActionType, Actor


def recorded_gameplay(step_count: int = 13) -> dict[str, object]:
    """Play Wait steps on seed 1447 and return an exported gameplay file."""
    engine = AresEngine()
    initial = snapshot_from_state(engine.reset(1447))
    recorder = TrajectoryRecorder.create(DEFAULT_ENGINE_CONFIG.replay, initial)
    for _ in range(step_count):
        before = snapshot_from_state(engine.state)
        after = snapshot_from_state(engine.step(ActionCommand(ActionType.WAIT), Actor.PLAYER).state)
        recorder.record(before, after)
    return recorder.export_replay_projection("mission", "manual")


def test_recorder_builds_deltas_event_interval_and_final_checkpoints() -> None:
    gameplay = recorded_gameplay()
    assert gameplay["schemaVersion"] == "aresim.trajectory.replay.v1"
    assert gameplay["fileName"] == "mission.json"
    assert len(gameplay["steps"]) == 13
    checkpoints = gameplay["checkpoints"]
    assert any(item["reason"] == "initial" for item in checkpoints)
    assert any(item["reason"] == "interval" and item["step"] == 10 for item in checkpoints)
    assert checkpoints[-1]["reason"] == "final"
    assert gameplay["integrity"] == {"finalStep": 13, "stepCount": 13, "checkpointCount": len(checkpoints)}
    assert "llmAgentId" not in gameplay["metadata"]


def test_endpoint_checkpoint_mode_preserves_complete_reconstruction() -> None:
    engine = AresEngine()
    initial = snapshot_from_state(engine.reset(1447))
    recorder = TrajectoryRecorder.create(
        DEFAULT_ENGINE_CONFIG.replay,
        initial,
        checkpoint_mode="endpoints",
    )
    expected_snapshots = [deepcopy(initial)]
    for _ in range(13):
        before = snapshot_from_state(engine.state)
        after = snapshot_from_state(engine.step(ActionCommand(ActionType.WAIT), Actor.AGENT).state)
        recorder.record(before, after)
        expected_snapshots.append(deepcopy(after))
    gameplay = recorder.export_replay_projection("agent-run", "algorithm", "aresim.agent.wait.v1")

    assert [checkpoint["reason"] for checkpoint in gameplay["checkpoints"]] == ["initial", "final"]
    replay = ReplayCursor.create(gameplay, "agent-run.json")
    for step, expected in enumerate(expected_snapshots):
        expected["mode"] = "Replay"
        assert replay.jump(step) == expected


def test_replay_step_jump_and_reset_reconstruct_exact_snapshots() -> None:
    gameplay = recorded_gameplay()
    replay = ReplayCursor.create(gameplay, "mission.json")
    replay.step()
    assert replay.cursor == 1
    assert replay.snapshot == gameplay["checkpoints"][0]["snapshot"] or replay.snapshot["step"] == 1
    jumped = replay.jump(13)
    expected = deepcopy(gameplay["finalSnapshot"])
    expected["mode"] = "Replay"
    assert jumped == expected
    reset = replay.reset()
    assert reset["step"] == 0
    assert replay.cursor == 0


def test_final_checkpoint_wins_when_interval_and_final_share_a_step() -> None:
    gameplay = recorded_gameplay(10)
    final = gameplay["finalSnapshot"]
    final["gameStatus"] = "paused"
    final["statusReason"] = "Simulation paused by player"
    gameplay["checkpoints"][-1]["snapshot"] = deepcopy(final)
    replay = ReplayCursor.create(gameplay, "mission.json")
    jumped = replay.jump(10)
    assert jumped["gameStatus"] == "paused"
    assert jumped["statusReason"] == "Simulation paused by player"


def test_canonical_legacy_llm_and_snapshot_wrappers_are_supported() -> None:
    gameplay = recorded_gameplay(2)
    gameplay["metadata"]["runMode"] = "llm"
    gameplay["metadata"]["llmAgentId"] = "legacy-agent"
    normalized = normalize_gameplay_payload(gameplay, "legacy.json", DEFAULT_ENGINE_CONFIG.replay)
    assert normalized["metadata"]["runMode"] == "llm"
    assert normalized["metadata"]["llmAgentId"] == "legacy-agent"

    snapshot = deepcopy(gameplay["finalSnapshot"])
    wrapped = normalize_gameplay_payload({"savedAt": "2025-01-01T00:00:00Z", "snapshot": snapshot}, "wrapped.json", DEFAULT_ENGINE_CONFIG.replay)
    raw = normalize_gameplay_payload(snapshot, "raw.json", DEFAULT_ENGINE_CONFIG.replay)
    assert wrapped["savedAt"] == "2025-01-01T00:00:00Z"
    assert raw["metadata"]["runMode"] == "load"


def test_unified_trajectory_preserves_and_exposes_the_complete_replay() -> None:
    gameplay = recorded_gameplay(3)
    policy = {"actions": [0, 0, 0], "environment_seed": 1447, "agent_seed": 91}
    trajectory = create_trajectory_episode(
        DEFAULT_ENGINE_CONFIG.replay,
        gameplay,
        episode_id="episode-1",
        source="rollout",
        policy=policy,
        policy_id="aresim.agent.wait.v1",
        agent_seed=91,
    )

    normalized = normalize_trajectory_episode(trajectory, "episode-1.json", DEFAULT_ENGINE_CONFIG.replay)
    imported_replay = normalize_gameplay_payload(trajectory, "episode-1.json", DEFAULT_ENGINE_CONFIG.replay)

    assert normalized["schemaVersion"] == "aresim.trajectory.episode.v1"
    assert normalized["metadata"]["environmentSeed"] == 1447
    assert normalized["policy"] == policy
    assert normalized["replay"] == gameplay
    assert normalized["replay"]["schemaVersion"] == "aresim.trajectory.replay.v1"
    assert imported_replay == gameplay
    assert ReplayCursor.create(imported_replay, "episode-1.json").jump(3)["rovers"] == gameplay["finalSnapshot"]["rovers"]

    legacy = deepcopy(trajectory)
    legacy["replay"] = deepcopy(gameplay)
    legacy["replay"]["schemaVersion"] = "aresim.gameplay.v1"
    rewritten = normalize_trajectory_episode(legacy, "episode-1.json", DEFAULT_ENGINE_CONFIG.replay)
    assert rewritten["replay"]["schemaVersion"] == "aresim.trajectory.replay.v1"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"schemaVersion": "aresim.gameplay.v99"}),
        lambda value: value["integrity"].update({"stepCount": 999}),
        lambda value: value["steps"].append({"step": 14}),
    ],
)
def test_invalid_gameplay_is_rejected(mutation) -> None:
    gameplay = recorded_gameplay(1)
    mutation(gameplay)
    with pytest.raises(ValueError):
        normalize_gameplay_payload(gameplay, "broken.json", DEFAULT_ENGINE_CONFIG.replay)


def test_validation_rejects_replay_data_that_cannot_reconstruct_saved_snapshots() -> None:
    broken_delta = recorded_gameplay(2)
    broken_delta["steps"][-1]["changes"]["rovers"][0]["battery"] = 99
    with pytest.raises(ValueError, match="reconstructed step data"):
        normalize_gameplay_payload(broken_delta, "broken-delta.json", DEFAULT_ENGINE_CONFIG.replay)

    broken_checkpoint = recorded_gameplay(13)
    interval = next(item for item in broken_checkpoint["checkpoints"] if item["reason"] == "interval")
    interval["snapshot"]["resources"]["water"] += 1
    with pytest.raises(ValueError, match="checkpoint disagrees"):
        normalize_gameplay_payload(broken_checkpoint, "broken-checkpoint.json", DEFAULT_ENGINE_CONFIG.replay)
