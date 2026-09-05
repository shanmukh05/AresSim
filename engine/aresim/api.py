"""FastAPI REST surface used by the React UI.

Maps camelCase JSON to :class:`~aresim.service.AresService` calls and returns UI
snapshots. Errors use ``{ error: { code, message } }`` so the client switches on
``code``, not prose.

**Last updated:** September 1, 2026

**Contains:** route handlers, request models, validation error mapping.

**Run:** ``python -m aresim.api`` (127.0.0.1:8000). Vite proxies ``/api`` there.

**See also:** :mod:`aresim.integrations.ui` (snapshot projection).
"""

from __future__ import annotations

from typing import Literal

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .config import EngineConfig
from .defaults import DEFAULT_ENGINE_CONFIG
from .gameplay import CHECKPOINT_REASON_PRIORITY, TRAJECTORY_EPISODE_SCHEMA, TRAJECTORY_REPLAY_SCHEMA, JsonObject, ReplayCursor
from .integrations.policy import rllib_available
from .service import AresService, ServiceError
from .types import ActionCommand, ActionType, Actor, Position


class StrictModel(BaseModel):
    """Reject unknown JSON fields so typos fail at the HTTP boundary."""
    model_config = ConfigDict(extra="forbid")


class SessionRequest(StrictModel):
    """`POST /api/sessions`. Omit `seed` to let the service pick one."""
    seed: int | None = None


class PositionRequest(StrictModel):
    x: int
    y: int


class ActionPayload(StrictModel):
    type: Literal["move", "scan", "extract", "build", "service", "unload", "wait"]
    target: PositionRequest | None = None


class ActionRequest(StrictModel):
    """`POST /api/sessions/{id}/actions`. `actor` is Player (manual) or Agent (algorithm)."""
    action: ActionPayload
    actor: Literal["Player", "Agent"]


class SaveRequest(StrictModel):
    """`POST /api/sessions/{id}/save`. Downloads one unified trajectory JSON."""
    file_name: str = Field(alias="fileName", max_length=180)
    run_mode: Literal["manual", "algorithm"] = Field(alias="runMode")
    algorithm_id: Literal["random", "random_valid", "wait", "scripted", "masked_ppo"] | None = Field(default=None, alias="algorithmId")
    checkpoint_path: str | None = Field(default=None, alias="checkpointPath", max_length=500)


class AttachPolicyRequest(StrictModel):
    """`POST /api/sessions/{id}/attach-policy`."""
    algorithm_id: Literal["random", "random_valid", "wait", "scripted", "masked_ppo"] = Field(alias="algorithmId")
    checkpoint_path: str | None = Field(default=None, alias="checkpointPath", max_length=500)


class ValidateCheckpointRequest(StrictModel):
    """`POST /api/policies/validate-checkpoint`."""
    checkpoint_path: str = Field(alias="checkpointPath", max_length=500)


class ReplayLoadRequest(StrictModel):
    """`POST /api/replays`. `content` is trajectory or legacy gameplay JSON."""
    file_name: str = Field(alias="fileName", max_length=180)
    content: str


class ReplayJumpRequest(StrictModel):
    step: int = Field(ge=0)


def _error(code: str, message: str, status_code: int) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message}})


def _replay_response(replay: ReplayCursor, include_gameplay: bool = False) -> JsonObject:
    gameplay = replay.gameplay
    checkpoints = gameplay.get("checkpoints")
    active_checkpoint_id = None
    if isinstance(checkpoints, list):
        exact_matches = [checkpoint for checkpoint in checkpoints if isinstance(checkpoint, dict) and checkpoint.get("step") == replay.cursor]
        exact = max(exact_matches, key=lambda item: CHECKPOINT_REASON_PRIORITY.get(str(item.get("reason")), -1), default=None)
        if isinstance(exact, dict):
            active_checkpoint_id = exact.get("id")
    response: JsonObject = {
        "replayId": replay.replay_id,
        "fileName": gameplay.get("fileName", replay.file_name),
        "seed": gameplay["metadata"]["seed"],
        "totalSteps": gameplay["integrity"]["finalStep"],
        "cursor": replay.cursor,
        "schemaVersion": gameplay["schemaVersion"],
        "activeCheckpointId": active_checkpoint_id,
        "snapshot": replay.snapshot,
    }
    if include_gameplay:
        response["gameplay"] = gameplay
    return response


def create_app(config: EngineConfig = DEFAULT_ENGINE_CONFIG) -> FastAPI:
    """Build the API with one `AresService`. Tests pass a custom `config`."""
    service = AresService(config)
    app = FastAPI(title="AresSim Local API", version=config.replay.app_version)
    app.state.ares_service = service

    @app.exception_handler(ServiceError)
    async def handle_service_error(_request: Request, error: ServiceError) -> JSONResponse:
        return _error(error.code, error.message, error.status_code)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_request: Request, _error_value: RequestValidationError) -> JSONResponse:
        return _error("INVALID_REQUEST", "The request body or path parameters are invalid.", 422)

    @app.get("/api/health")
    def health() -> JsonObject:
        return {
            "status": "ok",
            "backendVersion": config.replay.app_version,
            "trajectoryEpisodeSchemaVersion": TRAJECTORY_EPISODE_SCHEMA,
            "trajectoryReplaySchemaVersion": TRAJECTORY_REPLAY_SCHEMA,
            "gameplaySchemaVersion": config.replay.schema_version,
            "rllibAvailable": rllib_available(),
        }

    @app.get("/api/policies")
    def list_policies() -> JsonObject:
        return service.list_policies()

    @app.post("/api/policies/validate-checkpoint")
    def validate_checkpoint(request: ValidateCheckpointRequest) -> JsonObject:
        return service.validate_checkpoint(request.checkpoint_path)

    @app.post("/api/sessions")
    def start_session(request: SessionRequest) -> JsonObject:
        return {"snapshot": service.start_session(request.seed)}

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> JsonObject:
        return {"snapshot": service.get_snapshot(session_id)}

    @app.post("/api/sessions/{session_id}/actions")
    def apply_session_action(session_id: str, request: ActionRequest) -> JsonObject:
        target = Position(request.action.target.x, request.action.target.y) if request.action.target is not None else None
        command = ActionCommand(ActionType(request.action.type), target)
        return {"snapshot": service.apply_action(session_id, command, Actor(request.actor))}

    @app.post("/api/sessions/{session_id}/pause")
    def pause_session(session_id: str) -> JsonObject:
        return {"snapshot": service.pause(session_id)}

    @app.post("/api/sessions/{session_id}/resume")
    def resume_session(session_id: str) -> JsonObject:
        return {"snapshot": service.resume(session_id)}

    @app.post("/api/sessions/{session_id}/save")
    def save_session(session_id: str, request: SaveRequest) -> JsonObject:
        return service.save(
            session_id,
            request.file_name,
            request.run_mode,
            request.algorithm_id,
            checkpoint_path=request.checkpoint_path,
        )

    @app.post("/api/sessions/{session_id}/attach-policy")
    def attach_policy(session_id: str, request: AttachPolicyRequest) -> JsonObject:
        return service.attach_policy(session_id, request.algorithm_id, request.checkpoint_path)

    @app.post("/api/sessions/{session_id}/detach-policy")
    def detach_policy(session_id: str) -> JsonObject:
        return service.detach_policy(session_id)

    @app.post("/api/sessions/{session_id}/agent-step")
    def agent_step(session_id: str) -> JsonObject:
        return service.agent_step(session_id)

    @app.post("/api/replays")
    def load_replay(request: ReplayLoadRequest) -> JsonObject:
        return _replay_response(service.load_replay(request.file_name, request.content), include_gameplay=True)

    @app.post("/api/replays/{replay_id}/step")
    def step_replay(replay_id: str) -> JsonObject:
        return _replay_response(service.step_replay(replay_id))

    @app.post("/api/replays/{replay_id}/jump")
    def jump_replay(replay_id: str, request: ReplayJumpRequest) -> JsonObject:
        return _replay_response(service.jump_replay(replay_id, request.step))

    @app.post("/api/replays/{replay_id}/reset")
    def reset_replay(replay_id: str) -> JsonObject:
        return _replay_response(service.reset_replay(replay_id))

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("aresim.api:app", host="127.0.0.1", port=8000, reload=False)
