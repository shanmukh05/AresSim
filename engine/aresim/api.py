"""FastAPI REST surface used by the React UI.

Maps camelCase JSON to :class:`~aresim.service.AresService` calls and returns UI
snapshots. Errors use ``{ error: { code, message } }`` so the client switches on
``code``, not prose.

**Last updated:** September 12, 2026

**Contains:** route handlers, request models, validation error mapping.

**Run:** ``python -m aresim.api`` (127.0.0.1:8000). Vite proxies ``/api`` there.

**See also:** :mod:`aresim.integrations.ui` (snapshot projection).
"""

from __future__ import annotations

import base64
import binascii
import gzip
import io
import os
from typing import Literal

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .algorithms.catalog import AlgorithmId
from .config import EngineConfig
from .defaults import DEFAULT_ENGINE_CONFIG
from .gameplay import CHECKPOINT_REASON_PRIORITY, TRAJECTORY_EPISODE_SCHEMA, TRAJECTORY_REPLAY_SCHEMA, JsonObject, ReplayCursor
from .integrations.policy import is_deployed_mode, llm_dm_available, rllib_available, muzero_available, world_models_available
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
    compact: bool = False


class ActionsBatchRequest(StrictModel):
    """`POST /api/sessions/{id}/actions-batch`. Batches rapid manual inputs in one round-trip."""
    actions: list[ActionPayload] = Field(min_length=1, max_length=16)
    actor: Literal["Player", "Agent"] = "Player"
    compact: bool = True


class SaveRequest(StrictModel):
    """`POST /api/sessions/{id}/save`. Downloads one unified trajectory JSON."""
    file_name: str = Field(alias="fileName", max_length=180)
    run_mode: Literal["manual", "algorithm"] = Field(alias="runMode")
    algorithm_id: AlgorithmId | None = Field(default=None, alias="algorithmId")
    checkpoint_path: str | None = Field(default=None, alias="checkpointPath", max_length=500)


class LlmDmAttachConfig(StrictModel):
    """Live LLM + Decision Model attachment parameters; keys stay strictly in memory."""
    actor_provider: str = Field(default="decision_models", alias="actorProvider", max_length=64)
    actor_model: str = Field(default="jev-1.14.0", alias="actorModel", max_length=160)
    actor_route: str | None = Field(default=None, alias="actorRoute", max_length=120)
    actor_api_key: str | None = Field(default=None, alias="actorApiKey", max_length=512)
    actor_context_window: int = Field(default=131072, alias="actorContextWindow", ge=8192, le=2000000)
    actor_max_output_tokens: int = Field(default=2048, alias="actorMaxOutputTokens", ge=256, le=65536)
    learner_enabled: bool = Field(default=True, alias="learnerEnabled")
    learner_provider: str = Field(default="gemini", alias="learnerProvider", max_length=64)
    learner_model: str = Field(default="gemini-4-neon", alias="learnerModel", max_length=160)
    learner_route: str | None = Field(default=None, alias="learnerRoute", max_length=120)
    learner_api_key: str | None = Field(default=None, alias="learnerApiKey", max_length=512)
    learner_context_window: int = Field(default=262144, alias="learnerContextWindow", ge=16384, le=2000000)
    learner_max_output_tokens: int = Field(default=16384, alias="learnerMaxOutputTokens", ge=1024, le=131072)
    history_mode: Literal["focused", "facts_recent", "recent", "none"] = Field(default="focused", alias="historyMode")
    advice_limit: int = Field(default=3, alias="adviceLimit", ge=1, le=10)
    attempt_timeout_seconds: float = Field(default=60.0, alias="attemptTimeoutSeconds", ge=10.0, le=300.0)
    max_episode_steps: int = Field(default=256, alias="maxEpisodeSteps", ge=16, le=2000)
    carry_learnings: list[dict[str, object]] = Field(default_factory=list, alias="carryLearnings")


class AttachPolicyRequest(StrictModel):
    """`POST /api/sessions/{id}/attach-policy`."""
    algorithm_id: AlgorithmId = Field(alias="algorithmId")
    checkpoint_path: str | None = Field(default=None, alias="checkpointPath", max_length=500)
    action_mode: Literal["stochastic", "deterministic"] | None = Field(default=None, alias="actionMode")
    llm_dm: LlmDmAttachConfig | None = Field(default=None, alias="llmDm")


class ValidateCheckpointRequest(StrictModel):
    """`POST /api/policies/validate-checkpoint`."""
    checkpoint_path: str = Field(alias="checkpointPath", max_length=500)


class ReplayLoadRequest(StrictModel):
    """`POST /api/replays`. `content` (or gzip+base64 `contentGzipBase64`) is trajectory or legacy gameplay JSON."""
    file_name: str = Field(alias="fileName", max_length=180)
    content: str | None = None
    content_gzip_base64: str | None = Field(default=None, alias="contentGzipBase64")


class ReplayJumpRequest(StrictModel):
    step: int = Field(ge=0)


class AgentStepsRequest(StrictModel):
    """`POST /api/sessions/{id}/agent-steps`. Batch non-LLM steps to amortize network RTT."""
    count: int = Field(default=10, ge=1, le=20)
    compact: bool = False


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


_DEFAULT_CORS_ORIGINS = [
    "https://www.tensorwrites.com",
    "https://tensorwrites.com",
    "https://aressim.vercel.app",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:4173",
    "http://localhost:4173",
]


def _cors_origins() -> list[str]:
    raw = os.environ.get("ARESSIM_CORS_ORIGINS", "").strip()
    if not raw:
        return list(_DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def create_app(config: EngineConfig = DEFAULT_ENGINE_CONFIG) -> FastAPI:
    """Build the API with one `AresService`. Tests pass a custom `config`."""
    service = AresService(config)
    app = FastAPI(title="AresSim Local API", version=config.replay.app_version)
    app.state.ares_service = service
    origins = _cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_origin_regex=None if "*" in origins else r"^https://aressim(-[a-z0-9-]+)?\.vercel\.app$",
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)

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
            "muzeroAvailable": muzero_available(),
            "worldModelsAvailable": world_models_available(),
            "llmDmAvailable": llm_dm_available(),
            "deployed": is_deployed_mode(),
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
        return {"snapshot": service.apply_action(session_id, command, Actor(request.actor), compact=request.compact)}

    @app.post("/api/sessions/{session_id}/actions-batch")
    def apply_session_actions_batch(session_id: str, request: ActionsBatchRequest) -> JsonObject:
        commands = [
            ActionCommand(
                ActionType(item.type),
                Position(item.target.x, item.target.y) if item.target is not None else None,
            )
            for item in request.actions
        ]
        return service.apply_actions_batch(session_id, commands, Actor(request.actor), compact=request.compact)

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
        llm_dm_payload = request.llm_dm.model_dump(by_alias=True) if request.llm_dm is not None else None
        return service.attach_policy(
            session_id,
            request.algorithm_id,
            request.checkpoint_path,
            action_mode=request.action_mode,
            llm_dm_config=llm_dm_payload,
        )

    @app.post("/api/sessions/{session_id}/detach-policy")
    def detach_policy(session_id: str) -> JsonObject:
        return service.detach_policy(session_id)

    @app.post("/api/sessions/{session_id}/agent-step")
    def agent_step(session_id: str) -> JsonObject:
        return service.agent_step(session_id)

    @app.post("/api/sessions/{session_id}/agent-steps")
    def agent_steps(session_id: str, request: AgentStepsRequest) -> JsonObject:
        return service.agent_steps(session_id, request.count, compact=request.compact)

    @app.post("/api/sessions/{session_id}/llm-learn")
    def llm_learn(session_id: str) -> JsonObject:
        return service.llm_learn(session_id)

    @app.post("/api/replays")
    def load_replay(request: ReplayLoadRequest) -> JsonObject:
        content = request.content
        if request.content_gzip_base64 is not None:
            content = _decode_gzip_base64(request.content_gzip_base64, config.replay.max_upload_bytes)
        if content is None:
            raise ServiceError("INVALID_GAMEPLAY", "Provide either replay content or gzipped replay content.", 400)
        return _replay_response(service.load_replay(request.file_name, content), include_gameplay=True)

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


def _decode_gzip_base64(encoded: str, max_bytes: int) -> str:
    """Decode base64 gzip replay content with an explicit decompressed size cap."""
    try:
        compressed = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ServiceError("INVALID_GAMEPLAY", "The compressed gameplay payload is not valid base64.", 400) from error
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as stream:
            raw = stream.read(max_bytes + 1)
    except (OSError, EOFError, ValueError) as error:
        raise ServiceError("INVALID_GAMEPLAY", "The compressed gameplay payload is not valid gzip.", 400) from error
    if len(raw) > max_bytes:
        raise ServiceError("GAMEPLAY_TOO_LARGE", "The gameplay file exceeds the configured upload limit.", 413)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ServiceError("INVALID_GAMEPLAY", "The gameplay file is not valid UTF-8.", 400) from error


app = create_app()


if __name__ == "__main__":
    uvicorn.run("aresim.api:app", host="127.0.0.1", port=8000, reload=False)
