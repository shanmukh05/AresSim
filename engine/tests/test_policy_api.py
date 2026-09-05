"""HTTP and service tests for policy attach and agent-step routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aresim.api import create_app
from aresim.core.engine import AresEngine
from aresim.defaults import DEFAULT_ENGINE_CONFIG
from aresim.factory import make_agent
from aresim.integrations.policy import PolicyBridge, PolicyError, rllib_available, resolve_checkpoint_path


def _start_session(client: TestClient, seed: int = 1447) -> str:
    response = client.post("/api/sessions", json={"seed": seed})
    assert response.status_code == 200
    return response.json()["snapshot"]["sessionId"]


def test_policies_and_health_expose_rllib_capability() -> None:
    client = TestClient(create_app())
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["rllibAvailable"] is rllib_available()

    policies = client.get("/api/policies")
    assert policies.status_code == 200
    payload = policies.json()
    assert payload["capabilities"]["rllib"] is rllib_available()
    ids = {item["id"] for item in payload["policies"]}
    assert ids == {"random", "random_valid", "wait", "scripted", "masked_ppo"}


def test_attach_random_valid_and_agent_step_records_agent_actor() -> None:
    client = TestClient(create_app())
    session_id = _start_session(client)

    attached = client.post(
        f"/api/sessions/{session_id}/attach-policy",
        json={"algorithmId": "random_valid"},
    )
    assert attached.status_code == 200
    assert attached.json()["policyId"] == "aresim.agent.random_valid.v1"

    stepped = client.post(f"/api/sessions/{session_id}/agent-step")
    assert stepped.status_code == 200
    body = stepped.json()
    assert body["policyMeta"]["algorithmId"] == "random_valid"
    assert body["policyMeta"]["actionIndex"] >= 0
    assert body["snapshot"]["history"][0]["actor"] == "Agent"


def test_agent_step_without_attach_returns_policy_not_attached() -> None:
    client = TestClient(create_app())
    session_id = _start_session(client)
    response = client.post(f"/api/sessions/{session_id}/agent-step")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "POLICY_NOT_ATTACHED"


def test_attach_masked_ppo_without_path_fails() -> None:
    client = TestClient(create_app())
    session_id = _start_session(client)
    response = client.post(
        f"/api/sessions/{session_id}/attach-policy",
        json={"algorithmId": "masked_ppo"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "CHECKPOINT_NOT_FOUND"


def test_attach_masked_ppo_with_missing_file_fails() -> None:
    client = TestClient(create_app())
    session_id = _start_session(client)
    response = client.post(
        f"/api/sessions/{session_id}/attach-policy",
        json={"algorithmId": "masked_ppo", "checkpointPath": "/tmp/aresim-missing/checkpoints/final/checkpoint.json"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "CHECKPOINT_NOT_FOUND"


def test_resolve_checkpoint_path_accepts_absolute_sidecar() -> None:
    sidecar = Path("results/rllib_masked_ppo_smoke/seed_7/checkpoints/final/checkpoint.json")
    if not sidecar.is_file():
        pytest.skip("smoke checkpoint not present")
    resolved = resolve_checkpoint_path(str(sidecar.resolve()))
    assert resolved.is_file()
    assert resolved.name == "checkpoint.json"


def test_resolve_checkpoint_path_rejects_relative_path() -> None:
    with pytest.raises(PolicyError, match="absolute"):
        resolve_checkpoint_path("results/rllib_masked_ppo_smoke/seed_7/checkpoints/final/checkpoint.json")


def test_validate_checkpoint_reports_invalid_paths() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/api/policies/validate-checkpoint",
        json={"checkpointPath": "/tmp/aresim-missing/checkpoints/final/checkpoint.json"},
    )
    assert response.status_code == 200
    assert response.json()["valid"] is False


def test_wait_baseline_agent_step_is_deterministic_for_seed() -> None:
    client = TestClient(create_app())
    session_id = _start_session(client, seed=512)
    client.post(f"/api/sessions/{session_id}/attach-policy", json={"algorithmId": "wait"})
    first = client.post(f"/api/sessions/{session_id}/agent-step").json()["policyMeta"]["actionIndex"]
    second = client.post(f"/api/sessions/{session_id}/agent-step").json()["policyMeta"]["actionIndex"]
    assert first == second == 0


def test_policy_bridge_matches_wait_agent_actions() -> None:
    seed = 901
    engine = AresEngine(DEFAULT_ENGINE_CONFIG)
    engine.reset(seed)
    bridge = PolicyBridge.for_engine(DEFAULT_ENGINE_CONFIG)
    agent = make_agent("wait")
    agent.reset(seed)

    for _ in range(5):
        result = bridge.step(engine, agent, algorithm_id="wait")
        assert result.action_index == 0
        assert result.command.type.value == "wait"


@pytest.mark.rllib_slow
def test_attach_smoke_checkpoint_and_step_once() -> None:
    if not rllib_available():
        pytest.skip("rllib extra not installed")
    sidecar = Path("results/rllib_masked_ppo_smoke/seed_7/checkpoints/final/checkpoint.json")
    if not sidecar.is_file():
        pytest.skip("smoke checkpoint not present")

    client = TestClient(create_app())
    session_id = _start_session(client, seed=1447)
    attached = client.post(
        f"/api/sessions/{session_id}/attach-policy",
        json={"algorithmId": "masked_ppo", "checkpointPath": str(sidecar.resolve())},
    )
    assert attached.status_code == 200
    stepped = client.post(f"/api/sessions/{session_id}/agent-step")
    assert stepped.status_code == 200
    assert stepped.json()["snapshot"]["history"][0]["actor"] == "Agent"


def test_save_algorithm_run_accepts_new_algorithm_ids() -> None:
    client = TestClient(create_app())
    session_id = _start_session(client)
    client.post(f"/api/sessions/{session_id}/attach-policy", json={"algorithmId": "wait"})
    client.post(f"/api/sessions/{session_id}/agent-step")
    saved = client.post(
        f"/api/sessions/{session_id}/save",
        json={"fileName": "wait-run", "runMode": "algorithm", "algorithmId": "wait"},
    )
    assert saved.status_code == 200
    trajectory = saved.json()
    assert trajectory["metadata"]["policyId"] == "wait"
    assert trajectory["replay"]["metadata"]["algorithmId"] == "wait"
