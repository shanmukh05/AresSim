"""Projects engine transitions into named RL reward profiles.

Learning-facing rewards only; engine/UI history retains authoritative engine
reward terms. Values derive from state deltas and task outcomes, not re-simulated rules.
Dense shaping is a task-aware potential toward still-needed resources or home.
Cargo mode changes retain the potential difference; only the base reward clips.
Optional sample acquisition credit tracks delivered plus onboard mass and stops
at the mission quota; it never modifies gameplay rules or engine/UI scores.

**Last updated:** October 8, 2026

**Contains:** ``RewardBreakdown``, ``RewardTerm``, ``ShapedTrainReward``,
``SparseEvalReward``.

**Registry names:** ``shaped_train``, ``sparse_eval``.

**Schema:** ``aresim.reward.mission.v3`` when sample acquisition credit is enabled;
otherwise the v2 breakdown and profile are retained.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

from ..config import RewardProfileConfig, TaskConfig
from ..types import ActionType, EngineTransition, TerrainCell, TerrainType, WorldState
from .tasks import TaskOutcome


REWARD_SCHEMA = "aresim.reward.mission.v3"
LEGACY_REWARD_SCHEMA = "aresim.reward.mission.v2"
SAMPLE_COLLECTION_UNIT_KG = 0.5
# Default discount only: training freezes the actual learner gamma in reward config.
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


@dataclass(frozen=True)
class RewardStateSnapshot:
    """Small pre-action projection needed by the built-in reward profiles."""

    cargo_loaded: bool
    goal_potential: float
    terrain_scanned: float
    ice_collected: float
    ice_delivered: float
    samples_delivered: float
    samples_collected: float
    habitat_build_progress: float
    structure_health_mean: float
    dust_intensity: float
    rover_health: float
    colony_battery: float


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


def _goal_distance(state: WorldState, task: TaskConfig = TaskConfig()) -> int:
    rover = state.rovers[0]
    stats = state.objective_stats
    need_ice = stats.ice_delivered + rover.cargo_ice < task.ice_target_kg
    need_samples = stats.samples_delivered + rover.cargo_samples < task.sample_target_kg
    if not (need_ice or need_samples) or _cargo_mass(state) >= rover.cargo_capacity_kg or (rover.battery < 25 and _cargo_mass(state) > 0):
        return _rover_manhattan(state, _pad_cells(state)) or 0
    distances = [distance for distance in (
        _rover_manhattan(state, _ice_cells(state)) if need_ice else None,
        _rover_manhattan(state, _ore_cells(state)) if need_samples else None,
    ) if distance is not None]
    resource = min(distances) if distances else None
    if resource is None:
        return _rover_manhattan(state, _pad_cells(state)) or 0
    return resource


def _goal_potential(state: WorldState, task: TaskConfig = TaskConfig()) -> float:
    return -_goal_distance(state, task) / POTENTIAL_DISTANCE


def _snapshot(state: WorldState, task: TaskConfig = TaskConfig()) -> RewardStateSnapshot:
    stats = state.objective_stats
    return RewardStateSnapshot(
        cargo_loaded=_cargo_mass(state) > 0,
        goal_potential=_goal_potential(state, task),
        terrain_scanned=stats.terrain_scanned,
        ice_collected=stats.ice_collected,
        ice_delivered=stats.ice_delivered,
        samples_delivered=stats.samples_delivered,
        samples_collected=stats.samples_delivered + state.rovers[0].cargo_samples,
        habitat_build_progress=stats.habitat_build_progress,
        structure_health_mean=sum(structure.health for structure in state.structures) / max(1, len(state.structures)),
        dust_intensity=state.dust_intensity,
        rover_health=state.rovers[0].health,
        colony_battery=state.resources.battery,
    )


def _goal_potential_raw(before: RewardStateSnapshot, after: WorldState, gamma: float, terminated: bool, task: TaskConfig = TaskConfig()) -> float:
    # Goal switches are part of Phi(s), including Extract and Unload. True
    # terminal states are absorbing; external time limits retain their potential.
    return gamma * (0.0 if terminated else _goal_potential(after, task)) - before.goal_potential


def _raw_values(
    before: RewardStateSnapshot,
    transition: EngineTransition,
    outcome: TaskOutcome,
    *,
    episode_end: bool = False,
    gamma: float = POTENTIAL_GAMMA,
    task: TaskConfig = TaskConfig(),
) -> dict[str, float]:
    after = transition.state
    after_stats = after.objective_stats
    capacity = after.rovers[0].cargo_capacity_kg
    after_health = sum(structure.health for structure in after.structures) / max(1, len(after.structures))
    health_recovery = max(0, after_health - before.structure_health_mean) / 100
    dust_recovery = max(0, before.dust_intensity - after.dust_intensity)
    rover = after.rovers[0]
    # Delivered + onboard mass survives Unload, so it cannot reset the bonus.
    sample_acquisition = max(0.0,
        min(task.sample_target_kg, after_stats.samples_delivered + rover.cargo_samples)
        - min(task.sample_target_kg, before.samples_collected))
    undelivered = max(0, rover.cargo_ice + rover.cargo_samples) / capacity
    at_end = episode_end or outcome.terminated
    return {
        "mission_success": float(outcome.success),
        "terminal_failure": float(outcome.terminated and not outcome.success),
        "objective_progress": max(0.0, outcome.progress),
        "new_cell": max(0.0, outcome.new_cell) * (1.0 + 2.0 * outcome.new_cell_distance),
        "approach_pad": 0.0,
        "goal_potential": _goal_potential_raw(before, after, gamma, outcome.terminated, task),
        "new_scan": float(after_stats.terrain_scanned > before.terrain_scanned),
        "ice_collected": float(min(task.ice_target_kg, after_stats.ice_collected) > min(task.ice_target_kg, before.ice_collected)),
        "samples_collected": sample_acquisition / SAMPLE_COLLECTION_UNIT_KG,
        "ice_delivered": max(0, min(task.ice_target_kg, after_stats.ice_delivered) - min(task.ice_target_kg, before.ice_delivered)) / capacity,
        "samples_delivered": max(0, min(task.sample_target_kg, after_stats.samples_delivered) - min(task.sample_target_kg, before.samples_delivered)) / capacity,
        "build_progress": max(0, after_stats.habitat_build_progress - before.habitat_build_progress) / 100,
        "service_recovery": (
            min(1, max(health_recovery, dust_recovery))
            if transition.effective_action == ActionType.SERVICE
            else 0
        ),
        "wait_on_pad": 0.0,
        "undelivered_cargo": undelivered if at_end and undelivered > 0 else 0,
        "hazard_damage": max(0, before.rover_health - after.rovers[0].health) / 100,
        "energy_used": max(0, before.colony_battery - after.resources.battery) / 100,
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
    "samples_collected",
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

    def __init__(self, config: RewardProfileConfig, task: TaskConfig = TaskConfig()) -> None:
        config.validate()
        self.config = config
        self.task_config = task
        if not self.sparse and config.samples_collected != 0:
            self.profile = "aresim.reward.shaped_train.v3"

    def _term_weight(self, name: str) -> float:
        if self.sparse and name in _SPARSE_ZERO_TERMS:
            return 0
        return float(getattr(self.config, name))

    def prepare_step(self, state: WorldState) -> RewardStateSnapshot:
        """Capture reward inputs before the engine mutates its world."""
        return _snapshot(state, self.task_config)

    def calculate(
        self,
        before: WorldState | RewardStateSnapshot,
        transition: EngineTransition,
        outcome: TaskOutcome,
        *,
        episode_end: bool = False,
    ) -> RewardBreakdown:
        """Calculate a pure reward projection from one immutable transition."""
        snapshot = before if isinstance(before, RewardStateSnapshot) else self.prepare_step(before)
        raw_values = _raw_values(snapshot, transition, outcome, episode_end=episode_end, gamma=self.config.gamma, task=self.task_config)
        terms = {}
        for name, raw in raw_values.items():
            weight = self._term_weight(name)
            # Zero-default rewards and sparse evaluation retain their v2 contract.
            if name == "samples_collected" and weight == 0:
                continue
            terms[name] = RewardTerm(raw, weight, raw * weight)
        total_unclipped = sum(term.value for term in terms.values())
        total = total_unclipped
        if not self.sparse and not outcome.terminated:
            potential = terms["goal_potential"].value
            total = min(self.config.clip_max, max(self.config.clip_min, total - potential)) + potential
        schema = REWARD_SCHEMA if "samples_collected" in terms else LEGACY_REWARD_SCHEMA
        return RewardBreakdown(schema, self.profile, terms, total_unclipped, total)


class ShapedTrainReward(_RewardProfile):
    """Mission reward with clipped base terms and an unmodified potential difference."""

    profile = "aresim.reward.shaped_train.v2"


class SparseEvalReward(_RewardProfile):
    """Sparse evaluation reward: mission success, failure and invalid actions."""

    profile = "aresim.reward.sparse_eval.v2"
    sparse = True
