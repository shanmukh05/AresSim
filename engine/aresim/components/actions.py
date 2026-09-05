"""Maps the fixed ten-action RL head to canonical engine commands and legal masks.

Decodes flat ``Discrete(10)`` indices into :class:`~aresim.types.ActionCommand`
with explicit local targets. Legality is delegated to
:func:`aresim.core.rules.validate_action`.

**Last updated:** September 1, 2026

**Contains:** ``DiscreteActions``, ``ACTION_SCHEMA``, ``ACTION_COUNT``.

**Registry name:** ``discrete`` → ``aresim.action.rover.v1``.

**Action IDs:** 0 Wait, 1–4 moves, 5 Scan, 6 Extract, 7 Build, 8 Service, 9 Unload.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from ..config import EngineConfig
from ..core.rules import validate_action
from ..types import ActionCommand, ActionType, Position, WorldState


ACTION_SCHEMA = "aresim.action.rover.v1"
ACTION_COUNT = 10


class DiscreteActions:
    """Decode the stable ten-action rover policy head and build its legal mask."""

    schema = ACTION_SCHEMA

    def __init__(self) -> None:
        self.space = spaces.Discrete(ACTION_COUNT)
        self.mask_space = spaces.Box(0, 1, shape=(ACTION_COUNT,), dtype=np.int8)

    def decode(self, state: WorldState, action_id: int) -> ActionCommand:
        """Return one canonical command with an explicit local target when needed."""
        if not isinstance(action_id, (int, np.integer)) or not self.space.contains(action_id):
            raise ValueError(f"action must be an integer in [0, {ACTION_COUNT - 1}]")
        rover = state.rovers[0]
        current = Position(rover.x, rover.y)
        if action_id == 0:
            return ActionCommand(ActionType.WAIT)
        if action_id == 1:
            return ActionCommand(ActionType.MOVE, Position(rover.x, rover.y - 1))
        if action_id == 2:
            return ActionCommand(ActionType.MOVE, Position(rover.x + 1, rover.y))
        if action_id == 3:
            return ActionCommand(ActionType.MOVE, Position(rover.x, rover.y + 1))
        if action_id == 4:
            return ActionCommand(ActionType.MOVE, Position(rover.x - 1, rover.y))
        action = {
            5: ActionType.SCAN,
            6: ActionType.EXTRACT,
            7: ActionType.BUILD,
            8: ActionType.SERVICE,
            9: ActionType.UNLOAD,
        }[int(action_id)]
        return ActionCommand(action, current)

    def mask(self, state: WorldState, config: EngineConfig) -> np.ndarray:
        """Return `int8[10]`; each bit delegates to canonical validation."""
        mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        for action_id in range(ACTION_COUNT):
            command = self.decode(state, action_id)
            mask[action_id] = int(validate_action(state, command, config).valid)
        mask[0] = 1
        return mask
