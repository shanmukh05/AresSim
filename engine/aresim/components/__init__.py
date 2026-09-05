"""Framework-neutral RL component contracts and built-in Phase 1 implementations.

Extension protocols for observations, actions, rewards, and tasks. Built-ins
read canonical :class:`~aresim.types.WorldState` without mutating it.

**Last updated:** September 1, 2026

**Contains:** ``ObservationBuilder``, ``ActionAdapter``, ``RewardFunction``,
``TaskEvaluator`` protocols; ``LocalObservation``, ``DiscreteActions``,
``ShapedTrainReward``, ``SparseEvalReward``, ``OpenExplorationTask``.

**Registry names:** see :func:`aresim.registry.create_default_registry`.

**See also:** submodules ``observations``, ``actions``, ``rewards``, ``tasks``.
"""

from .actions import DiscreteActions
from .base import ActionAdapter, ObservationBuilder, RewardFunction, TaskEvaluator
from .observations import LocalObservation
from .rewards import ShapedTrainReward, SparseEvalReward
from .tasks import OpenExplorationTask

__all__ = [
    "ActionAdapter",
    "DiscreteActions",
    "LocalObservation",
    "ObservationBuilder",
    "OpenExplorationTask",
    "RewardFunction",
    "ShapedTrainReward",
    "SparseEvalReward",
    "TaskEvaluator",
]
