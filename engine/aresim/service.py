"""In-memory live session and replay service between the REST API and the engine.

Holds one live session and one loaded replay. HTTP parsing stays in
:mod:`aresim.api`; gameplay rules stay in :mod:`aresim.core.rules`. Returned
snapshots are camelCase UI JSON from :mod:`aresim.integrations.ui`.

**Last updated:** September 1, 2026

**Contains:** ``AresService``, ``ServiceError``, session/replay use cases.

**Threading:** service methods are synchronized for concurrent HTTP requests.

**See also:** :mod:`aresim.gameplay` (trajectory recorder and replay cursor).
"""

from __future__ import annotations

import json
import secrets
import threading
from dataclasses import dataclass
from typing import Literal

from .config import EngineConfig
from .core.engine import AresEngine
from .gameplay import (
    TrajectoryRecorder,
    JsonObject,
    ReplayCursor,
    UnsupportedGameplaySchema,
    normalize_gameplay_payload,
)
from .integrations.policy import (
    PolicyBridge,
    PolicyError,
    policy_meta_from_result,
    resolve_agent,
    validate_checkpoint_path,
)
from .integrations.ui import snapshot_from_state
from .types import ActionCommand, Actor


class ServiceError(Exception):
    """Use-case failure with a stable `code` the UI can switch on, plus HTTP status."""
    def __init__(self, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass
class AttachedPolicy:
    """One policy bound to the active live session."""

    algorithm_id: str
    agent: object
    policy_id: str
    checkpoint_path: str | None


@dataclass
class LiveSession:
    """The active play session: engine plus the recorder used for Save."""
    engine: AresEngine
    recorder: TrajectoryRecorder
    policy: AttachedPolicy | None = None
    policy_bridge: PolicyBridge | None = None


class AresService:
    """Session, action, save, and replay use cases. Mutations are serialized with a lock."""
    def __init__(self, config: EngineConfig) -> None:
        config.validate()
        self.config = config
        self._live: LiveSession | None = None
        self._replay: ReplayCursor | None = None
        self._lock = threading.RLock()

    def start_session(self, seed: int | None = None) -> JsonObject:
        """Replace the live session. Omit `seed` to draw one in the configured range."""
        with self._lock:
            selected_seed = seed if seed is not None else secrets.randbelow(self.config.world.seed_max - self.config.world.seed_min + 1) + self.config.world.seed_min
            if selected_seed < self.config.world.seed_min or selected_seed > self.config.world.seed_max:
                raise ServiceError("INVALID_SEED", f"Seed must be between {self.config.world.seed_min} and {self.config.world.seed_max}.", 400)
            engine = AresEngine(self.config)
            snapshot = snapshot_from_state(engine.reset(selected_seed))
            self._live = LiveSession(
                engine,
                TrajectoryRecorder.create(self.config.replay, snapshot),
                policy=None,
                policy_bridge=PolicyBridge.for_engine(self.config),
            )
            return snapshot

    def _session(self, session_id: str) -> LiveSession:
        if self._live is None or self._live.engine.state.session_id != session_id:
            raise ServiceError("SESSION_NOT_FOUND", "The requested session is not active.", 404)
        return self._live

    def get_snapshot(self, session_id: str) -> JsonObject:
        """Return the current UI snapshot, or raise if this is not the active session."""
        with self._lock:
            return snapshot_from_state(self._session(session_id).engine.state)

    def apply_action(self, session_id: str, command: ActionCommand, actor: Actor) -> JsonObject:
        """Step the engine and record a delta when the world clock actually advanced."""
        with self._lock:
            session = self._session(session_id)
            before = snapshot_from_state(session.engine.state)
            transition = session.engine.step(command, actor)
            after = snapshot_from_state(transition.state)
            if after["step"] != before["step"]:
                session.recorder.record(before, after)
            return after

    def pause(self, session_id: str) -> JsonObject:
        """Pause without recording a step. Updates the recorder's current snapshot."""
        with self._lock:
            session = self._session(session_id)
            state = session.engine.pause()
            snapshot = snapshot_from_state(state)
            session.recorder.current_snapshot = snapshot
            return snapshot

    def resume(self, session_id: str) -> JsonObject:
        """Resume a paused session without recording a step."""
        with self._lock:
            session = self._session(session_id)
            state = session.engine.resume()
            snapshot = snapshot_from_state(state)
            session.recorder.current_snapshot = snapshot
            return snapshot

    def save(
        self,
        session_id: str,
        file_name: str,
        run_mode: Literal["manual", "algorithm"],
        algorithm_id: str | None,
        checkpoint_path: str | None = None,
    ) -> JsonObject:
        """Export one unified trajectory episode containing complete UI replay data."""
        with self._lock:
            session = self._session(session_id)
            resolved_checkpoint = session.policy.checkpoint_path if session.policy is not None else checkpoint_path
            return session.recorder.export_trajectory(
                file_name,
                run_mode,
                algorithm_id,
                checkpoint_path=resolved_checkpoint,
            )

    def list_policies(self) -> JsonObject:
        """Return the built-in policy catalog and optional RLlib capability."""
        from .integrations.policy import POLICY_CATALOG, rllib_available

        return {"policies": POLICY_CATALOG, "capabilities": {"rllib": rllib_available()}}

    def validate_checkpoint(self, checkpoint_path: str) -> JsonObject:
        """Validate one checkpoint sidecar without attaching it."""
        return validate_checkpoint_path(checkpoint_path)

    def attach_policy(self, session_id: str, algorithm_id: str, checkpoint_path: str | None = None) -> JsonObject:
        """Bind one baseline or checkpoint policy to the active session."""
        with self._lock:
            session = self._session(session_id)
            try:
                agent, policy_id, resolved_path = resolve_agent(algorithm_id, checkpoint_path)
            except PolicyError as error:
                raise _policy_service_error(error) from error
            agent.reset(session.engine.state.seed)
            session.policy = AttachedPolicy(
                algorithm_id=algorithm_id,
                agent=agent,
                policy_id=policy_id,
                checkpoint_path=resolved_path,
            )
            if session.policy_bridge is None:
                session.policy_bridge = PolicyBridge.for_engine(self.config)
            return {"algorithmId": algorithm_id, "policyId": policy_id, "checkpointPath": resolved_path}

    def detach_policy(self, session_id: str) -> JsonObject:
        """Remove any policy bound to the active session."""
        with self._lock:
            session = self._session(session_id)
            session.policy = None
            return {"detached": True}

    def agent_step(self, session_id: str) -> JsonObject:
        """Apply one attached policy decision and return snapshot plus policy metadata."""
        with self._lock:
            session = self._session(session_id)
            if session.policy is None:
                raise ServiceError("POLICY_NOT_ATTACHED", "Attach a policy before running agent steps.", 400)
            if session.policy_bridge is None:
                session.policy_bridge = PolicyBridge.for_engine(self.config)
            before = snapshot_from_state(session.engine.state)
            result = session.policy_bridge.step(
                session.engine,
                session.policy.agent,
                algorithm_id=session.policy.algorithm_id,
            )
            after = snapshot_from_state(result.transition.state)
            if after["step"] != before["step"]:
                session.recorder.record(before, after)
            return {"snapshot": after, "policyMeta": policy_meta_from_result(result)}

    def load_replay(self, file_name: str, content: str) -> ReplayCursor:
        """Parse, normalize, and replace the in-memory replay. Rejects oversized uploads."""
        if len(content.encode("utf-8")) > self.config.replay.max_upload_bytes:
            raise ServiceError("GAMEPLAY_TOO_LARGE", "The gameplay file exceeds the configured upload limit.", 413)
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as error:
            raise ServiceError("INVALID_GAMEPLAY", "The gameplay file is not valid JSON.", 400) from error
        try:
            gameplay = normalize_gameplay_payload(payload, file_name, self.config.replay)
        except UnsupportedGameplaySchema as error:
            raise ServiceError("UNSUPPORTED_GAMEPLAY_SCHEMA", str(error), 400) from error
        except ValueError as error:
            raise ServiceError("INVALID_GAMEPLAY", str(error), 400) from error
        with self._lock:
            self._replay = ReplayCursor.create(gameplay, file_name)
            return self._replay

    def _active_replay(self, replay_id: str) -> ReplayCursor:
        if self._replay is None or self._replay.replay_id != replay_id:
            raise ServiceError("REPLAY_NOT_FOUND", "The requested replay is not active.", 404)
        return self._replay

    def step_replay(self, replay_id: str) -> ReplayCursor:
        """Advance the loaded replay one recorded step."""
        with self._lock:
            replay = self._active_replay(replay_id)
            replay.step()
            return replay

    def jump_replay(self, replay_id: str, step: int) -> ReplayCursor:
        """Seek the loaded replay to `step`."""
        with self._lock:
            replay = self._active_replay(replay_id)
            try:
                replay.jump(step)
            except ValueError as error:
                raise ServiceError("INVALID_REPLAY_POSITION", str(error), 400) from error
            return replay

    def reset_replay(self, replay_id: str) -> ReplayCursor:
        """Return the loaded replay to its initial snapshot."""
        with self._lock:
            replay = self._active_replay(replay_id)
            replay.reset()
            return replay


def _policy_service_error(error: PolicyError) -> ServiceError:
    status = {
        "POLICY_NOT_ATTACHED": 400,
        "CHECKPOINT_NOT_FOUND": 400,
        "CHECKPOINT_INCOMPATIBLE": 400,
        "RLLIB_UNAVAILABLE": 503,
        "UNKNOWN_ALGORITHM": 400,
    }.get(error.code, 400)
    return ServiceError(error.code, error.message, status)
