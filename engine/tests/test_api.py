"""HTTP tests for session, action, save, and replay routes on `create_app()`."""

from __future__ import annotations

import json
from dataclasses import replace

from fastapi.testclient import TestClient

from aresim.api import create_app
from aresim.defaults import DEFAULT_ENGINE_CONFIG


def test_session_action_pause_save_and_replay_routes() -> None:
    client = TestClient(create_app())
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["trajectoryEpisodeSchemaVersion"] == "aresim.trajectory.episode.v1"
    assert health.json()["trajectoryReplaySchemaVersion"] == "aresim.trajectory.replay.v1"
    assert health.json()["gameplaySchemaVersion"] == "aresim.gameplay.v1"

    created = client.post("/api/sessions", json={"seed": 1447})
    assert created.status_code == 200
    initial = created.json()["snapshot"]
    session_id = initial["sessionId"]
    assert client.get(f"/api/sessions/{session_id}").json()["snapshot"] == initial

    invalid = client.post(
        f"/api/sessions/{session_id}/actions",
        json={"action": {"type": "scan", "target": {"x": initial["rovers"][0]["x"], "y": initial["rovers"][0]["y"]}}, "actor": "Player"},
    )
    assert invalid.status_code == 200
    assert invalid.json()["snapshot"]["history"][0]["action"] == "invalid"

    paused = client.post(f"/api/sessions/{session_id}/pause").json()["snapshot"]
    resumed = client.post(f"/api/sessions/{session_id}/resume").json()["snapshot"]
    assert paused["step"] == resumed["step"] == 1
    assert paused["gameStatus"] == "paused"
    assert resumed["gameStatus"] == "running"

    saved_response = client.post(
        f"/api/sessions/{session_id}/save",
        json={"fileName": "api-run", "runMode": "algorithm", "algorithmId": "random_valid"},
    )
    assert saved_response.status_code == 200
    trajectory = saved_response.json()
    assert trajectory["schemaVersion"] == "aresim.trajectory.episode.v1"
    assert trajectory["metadata"]["policyId"] == "random_valid"
    assert trajectory["policy"] is None
    assert trajectory["replay"]["schemaVersion"] == "aresim.trajectory.replay.v1"
    assert trajectory["replay"]["metadata"]["algorithmId"] == "random_valid"

    loaded = client.post("/api/replays", json={"fileName": "api-run.json", "content": json.dumps(trajectory)})
    assert loaded.status_code == 200
    replay = loaded.json()
    replay_id = replay["replayId"]
    stepped = client.post(f"/api/replays/{replay_id}/step").json()
    assert stepped["cursor"] == 1
    reset = client.post(f"/api/replays/{replay_id}/reset").json()
    assert reset["cursor"] == 0
    jumped = client.post(f"/api/replays/{replay_id}/jump", json={"step": 1}).json()
    assert jumped["snapshot"]["step"] == 1


def test_api_errors_have_one_typed_shape() -> None:
    client = TestClient(create_app())
    missing = client.get("/api/sessions/missing")
    assert missing.status_code == 404
    assert missing.json() == {"error": {"code": "SESSION_NOT_FOUND", "message": "The requested session is not active."}}
    invalid_request = client.post("/api/sessions", json={"seed": "not-a-number", "extra": True})
    assert invalid_request.status_code == 422
    assert invalid_request.json()["error"]["code"] == "INVALID_REQUEST"
    malformed = client.post("/api/replays", json={"fileName": "bad.json", "content": "{"})
    assert malformed.status_code == 400
    assert malformed.json()["error"]["code"] == "INVALID_GAMEPLAY"


def test_raw_snapshot_upload_is_normalized() -> None:
    client = TestClient(create_app())
    snapshot = client.post("/api/sessions", json={"seed": 77}).json()["snapshot"]
    loaded = client.post("/api/replays", json={"fileName": "old.json", "content": json.dumps(snapshot)})
    assert loaded.status_code == 200
    assert loaded.json()["gameplay"]["schemaVersion"] == "aresim.trajectory.replay.v1"
    assert loaded.json()["snapshot"]["mode"] == "Replay"


def test_replay_size_schema_and_structure_fail_with_specific_codes() -> None:
    tiny_upload = replace(
        DEFAULT_ENGINE_CONFIG,
        replay=replace(DEFAULT_ENGINE_CONFIG.replay, max_upload_bytes=16),
    )
    client = TestClient(create_app(tiny_upload))
    oversized = client.post("/api/replays", json={"fileName": "large.json", "content": "{" + "x" * 30 + "}"})
    assert oversized.status_code == 413
    assert oversized.json()["error"]["code"] == "GAMEPLAY_TOO_LARGE"

    client = TestClient(create_app())
    unsupported = client.post(
        "/api/replays",
        json={"fileName": "future.json", "content": json.dumps({"schemaVersion": "aresim.gameplay.v2"})},
    )
    assert unsupported.status_code == 400
    assert unsupported.json()["error"]["code"] == "UNSUPPORTED_GAMEPLAY_SCHEMA"

    snapshot = client.post("/api/sessions", json={"seed": 1}).json()["snapshot"]
    malformed_canonical = {
        "schemaVersion": "aresim.gameplay.v1",
        "metadata": {"sessionId": snapshot["sessionId"], "seed": 1, "totalSteps": 0},
        "initialSnapshot": snapshot,
        "steps": [{"step": 1}],
        "checkpoints": [],
        "finalSnapshot": snapshot,
        "integrity": {"finalStep": 0, "stepCount": 1, "checkpointCount": 0},
    }
    invalid = client.post("/api/replays", json={"fileName": "broken.json", "content": json.dumps(malformed_canonical)})
    assert invalid.status_code == 400
    assert invalid.json()["error"]["code"] == "INVALID_GAMEPLAY"
