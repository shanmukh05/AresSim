"""Projects engine transitions into named RL reward profiles.

Learning-facing rewards only; engine/UI history retains authoritative engine
reward terms. Values derive from state deltas and task outcomes, not re-simulated rules.
Dense shaping is a potential toward nearest ice or unscanned ore (empty cargo)
or the pad (loaded). Proxy movement bonuses stay in the breakdown at weight 0.

**Last updated:** September 24, 2026

**Contains:** ``RewardBreakdown``, ``RewardTerm``, ``ShapedTrainReward``,
``SparseEvalReward``.

**Registry names:** ``shaped_train``, ``sparse_eval``.

**Schema:** ``aresim.reward.mission.v1``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

from ..config import RewardProfileConfig
from ..types import ActionType, EngineTransition, TerrainCell, TerrainType, WorldState
from .tasks import TaskOutcome


REWARD_SCHEMA = "aresim.reward.mission.v1"
# Matches PPO gamma in configs/masked_ppo/reference.yaml. Φ / 16 keeps a
# one-cell step around +0.06 so Extract/Unload still dominate.
POTENTIAL_GAMMA = 0.995
POTENTIAL_DISTANCE = 16.0


@dataclass(frozen=True)
class RewardTerm:
    """One raw measurement, configured weight, and weighted contribution."""

    raw: float
    weight: float
    value: float


@dataclass(frozen=True)
class RewardBreakdown:
    """Stable RL reward result with both unclipped and returned totals."""

    schema_version: str
    profile: str
    terms: dict[str, RewardTerm]
    total_unclipped: float
    total: float

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible dictionary for framework `info` mappings."""
        return asdict(self)


def _cargo_mass(state: WorldState) -> float:
    rover = state.rovers[0]
    return max(0.0, rover.cargo_ice + rover.cargo_ore + rover.cargo_samples)


def _rover_manhattan(state: WorldState, cells: Iterable[TerrainCell]) -> int | None:
    rover = state.rovers[0]
    distances = (abs(rover.x - cell.x) + abs(rover.y - cell.y) for cell in cells)
    return min(distances, default=None)


def _pad_cells(state: WorldState):
    return (cell for row in state.terrain for cell in row if cell.terrain == TerrainType.BUILD_PAD)


def _ice_cells(state: WorldState):
    return (
        cell
        for row in state.terrain
        for cell in row
        if cell.terrain == TerrainType.ICE and not cell.extracted
    )


def _ore_cells(state: WorldState):
    return (
        cell
        for row in state.terrain
        for cell in row
        if cell.terrain == TerrainType.ROCK and not cell.scanned
    )


def _nearest_resource_manhattan(state: WorldState) -> int | None:
    distances = [
        dist
        for dist in (
            _rover_manhattan(state, _ice_cells(state)),
            _rover_manhattan(state, _ore_cells(state)),
        )
        if dist is not None
    ]
    return min(distances) if distances else None


def _goal_distance(state: WorldState) -> int:
    if _cargo_mass(state) > 0:
        return _rover_manhattan(state, _pad_cells(state)) or 0
    resource = _nearest_resource_manhattan(state)
    if resource is None:
        return _rover_manhattan(state, _pad_cells(state)) or 0
    return resource


def _goal_potential(state: WorldState) -> float:
    return -_goal_distance(state) / POTENTIAL_DISTANCE


def _goal_potential_raw(before: WorldState, after: WorldState) -> float:
    # Extract/Unload change the goal; event terms own those steps.
    if (_cargo_mass(before) > 0) != (_cargo_mass(after) > 0):
        return 0.0
    return POTENTIAL_GAMMA * _goal_potential(after) - _goal_potential(before)


def _raw_values(
    before: WorldState,
    transition: EngineTransition,
    outcome: TaskOutcome,
    *,
    episode_end: bool = False,
) -> dict[str, float]:
    after = transition.state
    before_stats = before.objective_stats
    after_stats = after.objective_stats
    capacity = after.rovers[0].cargo_capacity_kg
    before_health = sum(structure.health for structure in before.structures) / max(1, len(before.structures))
    after_health = sum(structure.health for structure in after.structures) / max(1, len(after.structures))
    health_recovery = max(0, after_health - before_health) / 100
    dust_recovery = max(0, before.dust_intensity - after.dust_intensity)
    rover = after.rovers[0]
    undelivered = max(0, rover.cargo_ice + rover.cargo_samples) / capacity
    at_end = episode_end or outcome.terminated
    return {
        "mission_success": float(outcome.success),
        "terminal_failure": float(outcome.terminated and not outcome.success),
        "objective_progress": max(0.0, outcome.progress),
        "new_cell": max(0.0, outcome.new_cell) * (1.0 + 2.0 * outcome.new_cell_distance),
        "approach_pad": 0.0,
        "goal_potential": _goal_potential_raw(before, after),
        "new_scan": float(after_stats.terrain_scanned > before_stats.terrain_scanned),
        "ice_collected": float(after_stats.ice_collected > before_stats.ice_collected),
        "ice_delivered": max(0, after_stats.ice_delivered - before_stats.ice_delivered) / capacity,
        "samples_delivered": max(0, after_stats.samples_delivered - before_stats.samples_delivered) / capacity,
        "build_progress": max(0, after_stats.habitat_build_progress - before_stats.habitat_build_progress) / 100,
        "service_recovery": (
            min(1, max(health_recovery, dust_recovery))
            if transition.effective_action == ActionType.SERVICE
            else 0
        ),
        "wait_on_pad": 0.0,
        "undelivered_cargo": undelivered if at_end and undelivered > 0 else 0,
        "hazard_damage": max(0, before.rovers[0].health - after.rovers[0].health) / 100,
        "energy_used": max(0, before.resources.battery - after.resources.battery) / 100,
        "invalid_action": float(transition.effective_action == ActionType.INVALID),
        "time_cost": float(not outcome.terminated),
    }


_SPARSE_ZERO_TERMS = frozenset({
    "objective_progress",
    "new_cell",
    "approach_pad",
    "goal_potential",
    "new_scan",
    "ice_collected",
    "ice_delivered",
    "samples_delivered",
    "build_progress",
    "service_recovery",
    "wait_on_pad",
    "undelivered_cargo",
    "hazard_damage",
    "energy_used",
    "time_cost",
})


class _RewardProfile:
    """Shared named-term calculation for shaped and sparse profiles."""

    profile = ""
    sparse = False

    def __init__(self, config: RewardProfileConfig) -> None:
        config.validate()
        self.config = config

    def _term_weight(self, name: str) -> float:
        if self.sparse and name in _SPARSE_ZERO_TERMS:
            return 0
        return float(getattr(self.config, name))

    def calculate(
        self,
        before: WorldState,
        transition: EngineTransition,
        outcome: TaskOutcome,
        *,
        episode_end: bool = False,
    ) -> RewardBreakdown:
        """Calculate a pure reward projection from one immutable transition."""
        raw_values = _raw_values(before, transition, outcome, episode_end=episode_end)
        terms = {}
        for name, raw in raw_values.items():
            weight = self._term_weight(name)
            terms[name] = RewardTerm(raw, weight, raw * weight)
        total_unclipped = sum(term.value for term in terms.values())
        total = total_unclipped
        if not self.sparse and not outcome.terminated:
            total = min(self.config.clip_max, max(self.config.clip_min, total))
        return RewardBreakdown(REWARD_SCHEMA, self.profile, terms, total_unclipped, total)


class ShapedTrainReward(_RewardProfile):
    """Open-exploration shaped training reward with a clipped nonterminal total."""

    profile = "aresim.reward.shaped_train.v1"


class SparseEvalReward(_RewardProfile):
    """Sparse evaluation reward: terminal failure and invalid-action penalties only."""

    profile = "aresim.reward.sparse_eval.v1"
    sparse = True
