"""Structural protocols for swappable RL environment components.

Third-party implementations register through :mod:`aresim.registry` and need not
inherit AresSim concrete classes.

**Last updated:** September 1, 2026

**Contains:** ``ObservationBuilder``, ``ActionAdapter``, ``RewardFunction``,
``TaskEvaluator`` runtime-checkable protocols.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

import numpy as np
from gymnasium import Space

from ..config import EngineConfig
from ..types import ActionCommand, EngineTransition, WorldState
from .rewards import RewardBreakdown
from .tasks import TaskOutcome


ObservationT_co = TypeVar("ObservationT_co", covariant=True)
ActionT = TypeVar("ActionT")


@runtime_checkable
class ObservationBuilder(Protocol[ObservationT_co]):
    """Build one declared observation and clear episode-local state on reset."""

    schema: str
    space: Space[ObservationT_co]

    def reset(self, state: WorldState, engine_config: EngineConfig) -> ObservationT_co:
        """Clear episode-local memory and return the initial observation."""
        ...

    def build(self, state: WorldState, engine_config: EngineConfig) -> ObservationT_co:
        """Return the current observation without mutating canonical state."""
        ...


@runtime_checkable
class ActionAdapter(Protocol[ActionT]):
    """Decode policy actions and expose their authoritative legal-action mask."""

    schema: str
    space: Space[ActionT]
    mask_space: Space[np.ndarray]

    def decode(self, state: WorldState, action: ActionT) -> ActionCommand:
        """Translate a policy action into one explicit canonical command."""
        ...

    def mask(self, state: WorldState, engine_config: EngineConfig) -> np.ndarray:
        """Return a legal-action mask belonging to ``mask_space``."""
        ...


@runtime_checkable
class RewardFunction(Protocol):
    """Purely project an engine transition and task outcome into RL reward."""

    profile: str

    def calculate(
        self,
        before: WorldState,
        transition: EngineTransition,
        outcome: TaskOutcome,
    ) -> RewardBreakdown:
        """Return a stable reward breakdown without mutating inputs."""
        ...


@runtime_checkable
class TaskEvaluator(Protocol):
    """Track task-local episode state and evaluate completed transitions."""

    task_id: str

    def reset(self, state: WorldState) -> None:
        """Clear task-local progress for a newly initialized episode."""
        ...

    def evaluate(self, before: WorldState, transition: EngineTransition) -> TaskOutcome:
        """Return task success or failure for one completed transition."""
        ...


__all__ = [
    "ActionAdapter",
    "ObservationBuilder",
    "RewardFunction",
    "TaskEvaluator",
]
