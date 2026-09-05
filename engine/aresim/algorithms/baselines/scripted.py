"""Deterministic survival heuristic baseline policy from local telemetry.

Implements a partial-observation scripted rover policy for reward diagnosis and
masked-PPO comparison. Reads only ``aresim.obs.local.v1`` fields plus the action
mask; never accesses canonical engine state.

**Last updated:** September 5, 2026

**Contains:** ``ScriptedAgent`` and private navigation helpers.

**Registry name:** ``scripted`` → ``policy_id`` ``aresim.agent.scripted.v1``.

**Decision priority (high to low):** unload → service → build → recharge on pad →
return home → extract/scan → navigate to visible resources → explore → Wait.

**Dependencies:** NumPy only.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

from ...components.actions import ACTION_SCHEMA
from ...components.observations import OBSERVATION_SCHEMA
from ..common.masks import require_mask


PolicyObservation: TypeAlias = dict[str, np.ndarray | int]
MOVE_ACTIONS = (1, 2, 3, 4)
INVERSE_MOVE = {1: 3, 2: 4, 3: 1, 4: 2}


class ScriptedAgent:
    """Apply a deterministic survival/resource heuristic from local telemetry.

    Maintains private memory of home pad location, exploration heading, and last
    move to avoid immediate backtracking. The ``seed`` passed to ``reset`` is
    accepted for API compatibility; behavior is deterministic given observations.
    """

    policy_id = "aresim.agent.scripted.v1"
    observation_schema = OBSERVATION_SCHEMA
    action_schema = ACTION_SCHEMA

    def __init__(self) -> None:
        self._reset = False
        self._home: tuple[float, float] | None = None
        self._heading = 1
        self._last_move: int | None = None

    def reset(self, seed: int) -> None:
        """Forget pad location, heading, and last move for a new episode."""
        self._reset = True
        self._home = None
        self._heading = 1
        self._last_move = None

    @staticmethod
    def _policy_arrays(observation: PolicyObservation) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self_vector = observation.get("self")
        colony = observation.get("colony")
        terrain = observation.get("terrain_type")
        flags = observation.get("cell_flags")
        if not isinstance(self_vector, np.ndarray) or self_vector.shape != (10,):
            raise ValueError("scripted agent requires self float[10]")
        if not isinstance(colony, np.ndarray) or colony.shape != (14,):
            raise ValueError("scripted agent requires colony float[14]")
        if not isinstance(terrain, np.ndarray) or terrain.shape != (8, 8):
            raise ValueError("scripted agent requires terrain_type uint8[8,8]")
        if not isinstance(flags, np.ndarray) or flags.shape != (4, 8, 8):
            raise ValueError("scripted agent requires cell_flags uint8[4,8,8]")
        return self_vector, colony, terrain, flags

    def _choose_move(self, candidates: list[int], mask: np.ndarray) -> int | None:
        legal = [action for action in candidates if mask[action] == 1]
        if not legal:
            return None
        if self._last_move is not None and len(legal) > 1:
            reverse = INVERSE_MOVE[self._last_move]
            legal = [action for action in legal if action != reverse] or legal
        action = legal[0]
        self._heading = action
        self._last_move = action
        return action

    def _toward_home(self, current: tuple[float, float], mask: np.ndarray) -> int | None:
        if self._home is None:
            return None
        delta_x = self._home[0] - current[0]
        delta_y = self._home[1] - current[1]
        horizontal = 2 if delta_x > 0 else 4
        vertical = 3 if delta_y > 0 else 1
        axes = [(abs(delta_x), horizontal), (abs(delta_y), vertical)]
        candidates = [action for distance, action in sorted(axes, reverse=True) if distance > 1e-6]
        return self._choose_move(candidates, mask)

    def _visible_resource_targets(self, terrain: np.ndarray, flags: np.ndarray) -> list[tuple[int, int, int, int]]:
        targets: list[tuple[int, int, int, int]] = []
        anchor = 3
        for local_y in range(8):
            for local_x in range(8):
                terrain_id = int(terrain[local_y, local_x])
                is_ice = terrain_id == 3
                is_unscanned_rock = terrain_id == 2 and flags[2, local_y, local_x] == 0
                distance = abs(local_x - anchor) + abs(local_y - anchor)
                if (is_ice or is_unscanned_rock) and distance > 0:
                    targets.append((distance, 0 if is_ice else 1, local_y, local_x))
        return targets

    def _toward_visible_resource(
        self,
        terrain: np.ndarray,
        flags: np.ndarray,
        mask: np.ndarray,
    ) -> int | None:
        targets = self._visible_resource_targets(terrain, flags)
        if not targets:
            return None
        _, _, target_y, target_x = min(targets)
        delta_x = target_x - 3
        delta_y = target_y - 3
        axes = [(abs(delta_x), 2 if delta_x > 0 else 4), (abs(delta_y), 3 if delta_y > 0 else 1)]
        candidates = [action for distance, action in sorted(axes, reverse=True) if distance > 0]
        return self._choose_move(candidates, mask)

    def _explore(self, mask: np.ndarray) -> int:
        heading_index = MOVE_ACTIONS.index(self._heading)
        candidates = [
            MOVE_ACTIONS[heading_index],
            MOVE_ACTIONS[(heading_index + 1) % 4],
            MOVE_ACTIONS[(heading_index - 1) % 4],
            MOVE_ACTIONS[(heading_index + 2) % 4],
        ]
        return self._choose_move(candidates, mask) or 0

    def _immediate_action(
        self,
        mask: np.ndarray,
        payload_used: float,
        service_needed: bool,
        build_progress: float,
        pad_proximity: int,
        battery: float,
    ) -> int | None:
        if mask[9] and payload_used > 1e-6:
            return 9
        if mask[8] and service_needed:
            return 8
        if mask[7] and build_progress < 1:
            return 7
        if pad_proximity == 2 and battery < 0.8:
            return 0
        return None

    def _return_or_collect(
        self,
        current: tuple[float, float],
        terrain: np.ndarray,
        flags: np.ndarray,
        mask: np.ndarray,
        should_return: bool,
        pad_proximity: int,
    ) -> int | None:
        if should_return and pad_proximity != 2:
            home_action = self._toward_home(current, mask)
            if home_action is not None:
                return home_action
        if mask[6]:
            return 6
        if mask[5]:
            return 5
        return self._toward_visible_resource(terrain, flags, mask)

    def act(self, observation: PolicyObservation, action_mask: np.ndarray) -> int:
        """Choose one masked action from local telemetry and private navigation memory."""
        if not self._reset:
            raise RuntimeError("agent has not been reset")
        mask = require_mask(action_mask)
        self_vector, colony, terrain, flags = self._policy_arrays(observation)
        current = (float(self_vector[0]), float(self_vector[1]))
        if self._home is None:
            self._home = current
        pad_proximity = observation.get("pad_proximity")
        if not isinstance(pad_proximity, (int, np.integer)):
            raise ValueError("scripted agent requires integer pad_proximity")
        battery = float(self_vector[2])
        payload_used = 1 - float(self_vector[7])
        service_needed = colony[12] >= 0.5
        build_progress = float(colony[8])
        immediate = self._immediate_action(mask, payload_used, service_needed, build_progress, int(pad_proximity), battery)
        if immediate is not None:
            return immediate
        should_return = battery < 0.35 or payload_used >= 0.75 or service_needed
        chosen = self._return_or_collect(current, terrain, flags, mask, should_return, int(pad_proximity))
        if chosen is not None:
            return chosen
        return self._explore(mask)


__all__ = ["ScriptedAgent"]
