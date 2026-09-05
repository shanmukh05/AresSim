"""Coordinates deterministic simulation: reset, step, pause, and checksums.

``AresEngine`` orchestrates world generation and rule application. Returned
states are deep copies; same seed + command sequence yields identical checksums.

**Last updated:** September 1, 2026

**Contains:** :class:`AresEngine`, :func:`state_checksum`.

**Does not own:** observations, rewards, RL adapters, or HTTP — see
:mod:`aresim.envs` and :mod:`aresim.service`.

**See also:** :mod:`aresim.core.generation`, :mod:`aresim.core.rules`.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict

from ..config import EngineConfig
from ..defaults import DEFAULT_ENGINE_CONFIG
from ..types import ActionCommand, Actor, EngineTransition, GameStatus, WorldState
from .generation import create_world
from .rules import apply_action, initialize_state


def state_checksum(state: WorldState) -> str:
    """SHA-256 of the full world, used to prove identical seeds and commands match.

    Serialization is canonical JSON of `dataclasses.asdict(state)`. If you add a
    field to `WorldState`, it is automatically part of the fingerprint.
    """
    encoded = json.dumps(asdict(state), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class AresEngine:
    """Headless gameplay loop used by the REST service and by Python tests.

    Call `reset` before `step`, `pause`, or `resume`. `state` always returns a
    deep copy; the engine keeps the only mutable world internally.
    """

    def __init__(self, config: EngineConfig = DEFAULT_ENGINE_CONFIG) -> None:
        config.validate()
        self.config = config
        self._state: WorldState | None = None

    @property
    def state(self) -> WorldState:
        """Deep copy of the current world. Raises if `reset` has not been called."""
        if self._state is None:
            raise RuntimeError("engine has not been reset")
        return deepcopy(self._state)

    def reset(self, seed: int) -> WorldState:
        """Build a new world from `seed` and return a copy of the initial state."""
        state = create_world(seed, self.config)
        initialize_state(state, self.config)
        self._state = state
        return deepcopy(state)

    def step(self, command: ActionCommand, actor: Actor) -> EngineTransition:
        """Apply one command and return checksums, reward, events, and a state copy.

        Invalid commands still advance time and record history; they do not move
        the rover or spend resources. The world is mutated in place internally.
        """
        if self._state is None:
            raise RuntimeError("engine has not been reset")
        before_checksum = state_checksum(self._state)
        effective_action, reward, reward_terms, events = apply_action(self._state, command, actor, self.config)
        after_checksum = state_checksum(self._state)
        return EngineTransition(
            command=command,
            effective_action=effective_action,
            actor=actor,
            before_checksum=before_checksum,
            after_checksum=after_checksum,
            reward=reward,
            reward_terms=reward_terms,
            events=tuple(events),
            state=deepcopy(self._state),
        )

    def pause(self) -> WorldState:
        """Mark a running session paused without advancing simulation time."""
        if self._state is None:
            raise RuntimeError("engine has not been reset")
        if self._state.game_status == GameStatus.RUNNING:
            self._state.game_status = GameStatus.PAUSED
            self._state.status_reason = "Simulation paused by player"
        return deepcopy(self._state)

    def resume(self) -> WorldState:
        """Resume a paused session. No-op if the run already ended."""
        if self._state is None:
            raise RuntimeError("engine has not been reset")
        if self._state.game_status == GameStatus.PAUSED:
            self._state.game_status = GameStatus.RUNNING
            self._state.status_reason = "Open exploration active"
        return deepcopy(self._state)
