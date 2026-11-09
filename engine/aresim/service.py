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
from dataclasses import dataclass, field
from typing import Literal

from .algorithms.base import EpisodeStop, TransitionAwareAgent
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
    CHECKPOINT_ALGORITHMS,
    resolve_action_mode,
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
    action_mode: str | None = None


@dataclass
class LiveSession:
    """The active play session: engine plus the recorder used for Save."""
    engine: AresEngine
    recorder: TrajectoryRecorder
    policy: AttachedPolicy | None = None
    policy_bridge: PolicyBridge | None = None
    policy_events: dict[int, str] = field(default_factory=dict)
    revision: int = 0
    pending_policy_request: str | None = None
    policy_needs_reset: bool = False


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
            if self._live is not None:
                self._invalidate_policy(self._live, "session_replaced")
            engine = AresEngine(self.config)
            snapshot = snapshot_from_state(engine.reset(selected_seed), self.config)
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

    def _invalidate_policy(self, session: LiveSession, reason: str) -> None:
        """Invalidate inference authority while leaving any already committed tick intact."""
        session.revision += 1
        session.policy_needs_reset = True
        if session.policy is None:
            return
        agent = session.policy.agent
        cancel = getattr(agent, "cancel", None)
        if callable(cancel):
            cancel(reason)
        if not session.pending_policy_request and isinstance(agent, TransitionAwareAgent) and getattr(agent, "episode_open", True):
            agent.finish_episode({"transitions": getattr(agent, "step_index", 0), "terminated": False,
                                  "truncated": True, "ending_reason": reason})

    def get_snapshot(self, session_id: str) -> JsonObject:
        """Return the current UI snapshot, or raise if this is not the active session."""
        with self._lock:
            return snapshot_from_state(self._session(session_id).engine.state, self.config)

    def apply_action(self, session_id: str, command: ActionCommand, actor: Actor) -> JsonObject:
        """Step the engine and record a delta when the world clock actually advanced."""
        with self._lock:
            session = self._session(session_id)
            self._invalidate_policy(session, "manual_interruption")
            before = snapshot_from_state(session.engine.state, self.config)
            transition = session.engine.step(command, actor)
            after = snapshot_from_state(transition.state, self.config)
            if after["step"] != before["step"]:
                session.recorder.record(before, after)
            return after

    def pause(self, session_id: str) -> JsonObject:
        """Pause without recording a step. Updates the recorder's current snapshot."""
        with self._lock:
            session = self._session(session_id)
            self._invalidate_policy(session, "paused")
            state = session.engine.pause()
            snapshot = snapshot_from_state(state, self.config)
            session.recorder.current_snapshot = snapshot
            return snapshot

    def resume(self, session_id: str) -> JsonObject:
        """Resume a paused session without recording a step."""
        with self._lock:
            session = self._session(session_id)
            state = session.engine.resume()
            snapshot = snapshot_from_state(state, self.config)
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
        from .integrations.policy import POLICY_CATALOG, rllib_available, muzero_available, world_models_available, llm_dm_available

        return {"policies": POLICY_CATALOG, "capabilities": {"rllib": rllib_available(), "muzero": muzero_available(), "world_models": world_models_available(), "llm_dm": llm_dm_available()}}

    def validate_checkpoint(self, checkpoint_path: str) -> JsonObject:
        """Validate one checkpoint sidecar without attaching it."""
        return validate_checkpoint_path(checkpoint_path)

    def attach_policy(self, session_id: str, algorithm_id: str, checkpoint_path: str | None = None, *, action_mode: str | None = None) -> JsonObject:
        """Bind one baseline or checkpoint policy to the active session."""
        with self._lock:
            session = self._session(session_id)
            try:
                mode = resolve_action_mode(algorithm_id, action_mode) if algorithm_id in CHECKPOINT_ALGORITHMS else None
                agent, policy_id, resolved_path = resolve_agent(algorithm_id, checkpoint_path, action_mode=mode)
            except PolicyError as error:
                raise _policy_service_error(error) from error
            current = session.policy
            if (current is not None and current.algorithm_id == algorithm_id
                    and current.checkpoint_path == resolved_path and current.action_mode != mode):
                # Changing action selection must preserve discovered-cell memory,
                # MuZero history and the live agent's sampling RNG.
                current.agent.deterministic = mode == "deterministic"
                current.action_mode = mode
                return {"algorithmId": algorithm_id, "policyId": current.policy_id,
                        "checkpointPath": resolved_path, "actionMode": mode}
            self._invalidate_policy(session, "policy_changed")
            agent.reset(session.engine.state.seed)
            session.policy = AttachedPolicy(
                algorithm_id=algorithm_id,
                agent=agent,
                policy_id=policy_id,
                checkpoint_path=resolved_path,
                action_mode=mode,
            )
            session.policy_events.clear()
            session.policy_bridge = PolicyBridge.for_engine(self.config)
            return {"algorithmId": algorithm_id, "policyId": policy_id, "checkpointPath": resolved_path, "actionMode": mode}

    def detach_policy(self, session_id: str) -> JsonObject:
        """Remove any policy bound to the active session."""
        with self._lock:
            session = self._session(session_id)
            self._invalidate_policy(session, "policy_detached")
            session.policy = None
            return {"detached": True}

    def agent_step(self, session_id: str) -> JsonObject:
        """Infer outside the session lock, then commit only if the captured revision is current."""
        with self._lock:
            session = self._session(session_id)
            if session.policy is None:
                raise ServiceError("POLICY_NOT_ATTACHED", "Attach a policy before running agent steps.", 400)
            if session.pending_policy_request is not None:
                raise ServiceError("POLICY_BUSY", "A policy decision is already pending.", 409)
            attached = session.policy
            agent = attached.agent
            if session.policy_needs_reset:
                agent.reset(session.engine.state.seed)
                session.policy_bridge = PolicyBridge.for_engine(self.config)
                session.policy_needs_reset = False
            if session.policy_bridge is None:
                session.policy_bridge = PolicyBridge.for_engine(self.config)
            bridge = session.policy_bridge
            try:
                observation, mask = bridge.prepare(session.engine, agent)
            except PolicyError as error:
                raise _policy_service_error(error) from error
            revision = session.revision
            request_id = secrets.token_hex(16)
            session.pending_policy_request = request_id
        try:
            action = int(agent.act(observation, mask))
            with self._lock:
                if self._live is not session or session.revision != revision or session.policy is not attached:
                    raise EpisodeStop("cancelled")
                before = snapshot_from_state(session.engine.state, self.config)
                result = bridge.commit(session.engine, agent, action, mask, algorithm_id=attached.algorithm_id)
                after = snapshot_from_state(result.transition.state, self.config)
                summary = result.info.get("summary") if result.info else None
                if isinstance(summary, str) and summary:
                    session.policy_events[int(after["step"])] = summary
                _merge_policy_events(after, session.policy_events)
                if after["step"] != before["step"]:
                    session.recorder.record(before, after)
                meta = policy_meta_from_result(result)
                if attached.action_mode is not None:
                    meta["actionMode"] = attached.action_mode
                return {"snapshot": after, "policyMeta": meta}
        except EpisodeStop as error:
            if isinstance(agent, TransitionAwareAgent):
                agent.finish_episode({"transitions": getattr(agent, "step_index", 0), "terminated": False,
                                      "truncated": True, "ending_reason": error.reason})
            raise ServiceError("POLICY_STOPPED", f"Policy episode stopped: {error.reason}", 409) from error
        except PolicyError as error:
            raise _policy_service_error(error) from error
        finally:
            with self._lock:
                if session.pending_policy_request == request_id:
                    session.pending_policy_request = None

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
            if self._live is not None:
                self._invalidate_policy(self._live, "replay_loaded")
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


def _merge_policy_events(snapshot: JsonObject, notes: dict[int, str]) -> None:
    history = snapshot.get("history")
    if not isinstance(history, list) or not notes:
        return
    for entry in history:
        if not isinstance(entry, dict):
            continue
        note = notes.get(int(entry["step"])) if isinstance(entry.get("step"), int) else None
        if not isinstance(note, str) or not note:
            continue
        events = entry.get("events")
        current = list(events) if isinstance(events, list) else []
        if note not in current:
            current.append(note)
            entry["events"] = current


def _policy_service_error(error: PolicyError) -> ServiceError:
    status = {
        "POLICY_NOT_ATTACHED": 400,
        "CHECKPOINT_NOT_FOUND": 400,
        "CHECKPOINT_INCOMPATIBLE": 400,
        "RLLIB_UNAVAILABLE": 503,
        "MUZERO_UNAVAILABLE": 503,
        "WORLD_MODELS_UNAVAILABLE": 503,
        "POLICY_NOT_RUNNING": 400,
        "DECISION_MODELS_UNAVAILABLE": 503,
        "UNKNOWN_ALGORITHM": 400,
    }.get(error.code, 400)
    return ServiceError(error.code, error.message, status)
