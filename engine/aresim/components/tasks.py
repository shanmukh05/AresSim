"""Defines RL episode task outcomes without duplicating simulator failure rules.

Phase 1 open exploration has no victory condition; termination trusts engine
``game_status`` and failed rule records.

**Last updated:** September 1, 2026

**Contains:** ``TaskOutcome``, ``OpenExplorationTask``, ``TASK_ID``.

**Registry name:** ``open_exploration`` → ``phase1_open_exploration_v1``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..types import EngineTransition, GameStatus, RuleStatus, WorldState


TASK_ID = "phase1_open_exploration_v1"


@dataclass(frozen=True)
class TaskOutcome:
    """Task-facing terminal state for one completed engine transition."""

    terminated: bool
    success: bool
    terminal_reason: str | None


class OpenExplorationTask:
    """Continue until the authoritative engine reports a terminal failure."""

    task_id = TASK_ID

    def reset(self, state: WorldState) -> None:
        """Clear task state; open exploration is currently stateless."""

    def evaluate(self, before: WorldState, transition: EngineTransition) -> TaskOutcome:
        """Return failure evidence already calculated by the deterministic core."""
        state = transition.state
        if state.game_status != GameStatus.GAME_OVER:
            return TaskOutcome(False, False, None)
        failed_rule = next((rule.id for rule in state.rules if rule.status == RuleStatus.FAILED), "unknown")
        reason = {
            "battery": "battery_depleted",
            "health": "rover_health_depleted",
            "livability": "livability_depleted",
        }.get(failed_rule, "environment_failure")
        return TaskOutcome(True, False, reason)
