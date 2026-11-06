"""Defines RL episode task outcomes without duplicating simulator failure rules.

``open_exploration`` has no victory condition. ``resource_mission`` succeeds when
the rover delivers target ice and ore samples and the pad is not latched for
service. Engine ``game_status`` still owns battery/health/livability failure.

**Last updated:** September 22, 2026

**Contains:** ``TaskOutcome``, ``OpenExplorationTask``, ``ResourceMissionTask``.

**Registry names:** ``open_exploration``, ``resource_mission``.
"""

from __future__ import annotations

from dataclasses import dataclass
from ..config import TaskConfig

from ..types import EngineTransition, GameStatus, RuleStatus, WorldState


TASK_ID = "phase1_open_exploration_v1"
RESOURCE_MISSION_ID = "phase1_resource_mission_v2"
NEW_CELL_DISTANCE_NORMALIZER = 12.0


@dataclass(frozen=True)
class TaskOutcome:
    """Task-facing terminal state for one completed engine transition."""

    terminated: bool
    success: bool
    terminal_reason: str | None
    progress: float = 0.0
    new_cell: float = 0.0
    new_cell_distance: float = 0.0


def _failure_reason(state: WorldState) -> str:
    failed_rule = next((rule.id for rule in state.rules if rule.status == RuleStatus.FAILED), "unknown")
    return {
        "battery": "battery_depleted",
        "health": "rover_health_depleted",
        "livability": "livability_depleted",
    }.get(failed_rule, "environment_failure")


def _mission_progress(state: WorldState, config: TaskConfig = TaskConfig()) -> float:
    fractions = [min(1.0, delivered / target) for delivered, target in (
        (state.objective_stats.ice_delivered, config.ice_target_kg),
        (state.objective_stats.samples_delivered, config.sample_target_kg),
    ) if target > 0]
    return sum(fractions) / len(fractions)


def _mission_complete(state: WorldState, config: TaskConfig = TaskConfig()) -> bool:
    return (
        state.objective_stats.ice_delivered >= config.ice_target_kg
        and state.objective_stats.samples_delivered >= config.sample_target_kg
        and (not config.require_pad_safe or not state.build_pad_state.service_needed)
    )


class OpenExplorationTask:
    """Continue until the authoritative engine reports a terminal failure."""

    task_id = TASK_ID

    def reset(self, state: WorldState) -> None:
        """Clear task state; open exploration is currently stateless."""

    def evaluate(self, before: WorldState, transition: EngineTransition) -> TaskOutcome:
        """Return failure evidence already calculated by the deterministic core."""
        return self.evaluate_transition(transition)

    def evaluate_transition(self, transition: EngineTransition) -> TaskOutcome:
        """Evaluate a built-in task without needing a pre-action world copy."""
        state = transition.state
        if state.game_status != GameStatus.GAME_OVER:
            return TaskOutcome(False, False, None)
        return TaskOutcome(True, False, _failure_reason(state))


class ResourceMissionTask:
    """Deliver target ice and ore samples, and keep the build pad off the service latch."""

    task_id = RESOURCE_MISSION_ID

    def __init__(self, config: TaskConfig = TaskConfig()) -> None:
        config.validate()
        self.config = config
        self._progress = 0.0
        self._visited: set[tuple[int, int]] = set()
        self._pad_cells: tuple[tuple[int, int], ...] = ()

    def reset(self, state: WorldState) -> None:
        """Reset delivery progress, visited cells and pad-distance bookkeeping."""
        rover = state.rovers[0]
        self._visited = {(rover.x, rover.y)}
        self._progress = _mission_progress(state, self.config)
        self._pad_cells = tuple(
            (cell.x, cell.y)
            for row in state.terrain
            for cell in row
            if cell.terrain.value == "build_pad"
        )

    def evaluate(self, before: WorldState, transition: EngineTransition) -> TaskOutcome:
        """Score capped delivery progress and terminate on success or engine failure."""
        return self.evaluate_transition(transition)

    def evaluate_transition(self, transition: EngineTransition) -> TaskOutcome:
        """Score the transition using episode-local progress and the new state."""
        state = transition.state
        rover = state.rovers[0]
        cell = (rover.x, rover.y)
        new_cell = 0.0 if cell in self._visited else 1.0
        self._visited.add(cell)
        new_cell_distance = 0.0
        if new_cell and self._pad_cells:
            distance = min(
                abs(rover.x - pad_x) + abs(rover.y - pad_y)
                for pad_x, pad_y in self._pad_cells
            )
            new_cell_distance = min(1.0, distance / NEW_CELL_DISTANCE_NORMALIZER)
        progress = _mission_progress(state, self.config)
        delta = max(0.0, progress - self._progress)
        self._progress = max(self._progress, progress)
        if state.game_status == GameStatus.GAME_OVER:
            return TaskOutcome(
                True, False, _failure_reason(state), progress=delta,
                new_cell=new_cell, new_cell_distance=new_cell_distance,
            )
        if _mission_complete(state, self.config):
            return TaskOutcome(
                True, True, "mission_complete", progress=delta,
                new_cell=new_cell, new_cell_distance=new_cell_distance,
            )
        return TaskOutcome(
            False, False, None, progress=delta,
            new_cell=new_cell, new_cell_distance=new_cell_distance,
        )
