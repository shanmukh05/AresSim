"""Shared enums, commands, and world-state dataclasses for every engine layer.

Pure data definitions only — no gameplay logic. Calculations belong in
:mod:`aresim.core.rules`; HTTP models in :mod:`aresim.api`; camelCase UI
snapshots in :mod:`aresim.integrations.ui`.

**Last updated:** September 1, 2026

**Contains:** terrain/weather/action enums, ``ActionCommand``, ``WorldState``,
``EngineTransition``, rover/structure/resource dataclasses.

**Migration rule:** new ``WorldState`` fields must update the UI snapshot mapper
and any gameplay-save hydration defaults.

**Constraints:** the deterministic core must not import NumPy, Gymnasium, or RL libs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class TerrainType(StrEnum):
    """Cell kinds on the 32×32 map. Crater blocks movement; ridge and dune cost extra battery."""
    REGOLITH = "regolith"
    ROCK = "rock"
    ICE = "ice"
    CRATER = "crater"
    DUNE = "dune"
    BUILD_PAD = "build_pad"
    RIDGE = "ridge"


class WeatherState(StrEnum):
    """Cyclic weather that scales solar output, dust, and rover stress."""
    CLEAR = "Clear"
    DUSTY = "Dusty"
    DUST_FRONT = "Dust Front"
    SEVERE_STORM = "Severe Storm"
    COLD_NIGHT = "Cold Night"


class GameStatus(StrEnum):
    """Session status. Phase 1 ends on rover/habitat failure; `victory` is reserved."""
    RUNNING = "running"
    PAUSED = "paused"
    VICTORY = "victory"
    GAME_OVER = "game_over"


class ActionType(StrEnum):
    """Public verbs plus history-only tags. Players never send `invalid` or `event`."""
    MOVE = "move"
    SCAN = "scan"
    EXTRACT = "extract"
    BUILD = "build"
    SERVICE = "service"
    UNLOAD = "unload"
    WAIT = "wait"
    INVALID = "invalid"
    EVENT = "event"


class Actor(StrEnum):
    """Who issued a command. `System` is only used for generated history events."""
    PLAYER = "Player"
    AGENT = "Agent"
    SYSTEM = "System"


class StructureType(StrEnum):
    """Landing-pad buildings. They are scenery plus power/health sources, not separate agents."""
    HABITAT = "habitat"
    SOLAR = "solar"
    BATTERY = "battery"
    STORAGE = "storage"
    EXTRACTOR = "extractor"


class RuleStatus(StrEnum):
    """HUD rule pill state. `failed` on battery, rover health, or livability ends the run."""
    STABLE = "stable"
    WARNING = "warning"
    FAILED = "failed"
    COMPLETE = "complete"


class BuildPadStatus(StrEnum):
    """Combined habitat-progress and service-needed flag shown on the pad in the UI."""
    NORMAL = "normal"
    NEEDS_SERVICE = "needs_service"
    HABITAT_BUILT = "habitat_built"
    HABITAT_BUILT_NEEDS_SERVICE = "habitat_built_needs_service"


@dataclass(frozen=True)
class Position:
    """Integer grid coordinate. Origin is the north-west corner; +y is south."""
    x: int
    y: int


@dataclass(frozen=True)
class ActionCommand:
    """One requested action. `target` may be omitted; `rules.resolve_command` fills it in."""
    type: ActionType
    target: Position | None = None


@dataclass
class TerrainCell:
    """One map cell. `ice`/`ore` are resource signals; `scanned`/`extracted` are one-shot flags."""
    x: int
    y: int
    terrain: TerrainType
    height: float
    roughness: float
    ice: float
    ore: float
    dust: float
    scanned: bool = False
    extracted: bool = False


@dataclass
class RoverState:
    """The single Phase 1 rover. Cargo masses share `cargo_capacity_kg`; Unload empties them."""
    id: str
    name: str
    x: int
    y: int
    battery: float
    health: float
    cargo_ice: float
    cargo_ore: float
    cargo_samples: float
    cargo_capacity_kg: float
    current_task: str


@dataclass
class StructureState:
    id: str
    type: StructureType
    name: str
    x: int
    y: int
    health: float
    powered: bool
    status: str


@dataclass
class ResourceState:
    """Colony power/life-support meters. Rover battery is mirrored onto `RoverState.battery`."""
    power_generated: float
    power_consumed: float
    battery: float
    water: float
    oxygen: float
    livability: float


@dataclass
class MissionState:
    title: str
    objective: str
    reward_objectives: list[str]
    alerts: list[str]


@dataclass
class RewardTotals:
    ice_collected: float = 0
    terrain_scanned: float = 0
    habitat_built: float = 0
    serviced: float = 0
    delivered: float = 0
    traversal: float = 0
    blocked_penalty: float = 0
    total: float = 0


@dataclass
class ObjectiveStats:
    ice_collected: float = 0
    ice_delivered: float = 0
    samples_collected: float = 0
    samples_delivered: float = 0
    unload_count: int = 0
    ice_sites_extracted: int = 0
    ice_sites_total: int = 0
    terrain_scanned: int = 0
    rock_sites_total: int = 0
    habitat_build_progress: float = 0
    habitat_build_count: int = 0
    service_count: int = 0
    reward_totals: RewardTotals = field(default_factory=RewardTotals)


@dataclass
class BuildPadState:
    service_needed: bool
    status: BuildPadStatus


@dataclass
class GameRule:
    id: str
    label: str
    description: str
    status: RuleStatus
    value: str


@dataclass
class ResourceDelta:
    """Sparse per-step resource change. `None` means the field did not change."""
    power: float | None = None
    battery: float | None = None
    water: float | None = None
    oxygen: float | None = None
    ice: float | None = None
    ore: float | None = None
    samples: float | None = None


@dataclass
class HistoryEntry:
    """Newest-first log of one step. `action` is the effective verb, including `invalid`."""
    id: str
    step: int
    actor: Actor
    action: ActionType
    target: Position | None
    result: str
    reward: float
    reward_terms: dict[str, float]
    resource_delta: ResourceDelta
    events: list[str]


@dataclass
class WorldState:
    """Authoritative world. Mutate only inside `rules.apply_action` / `initialize_state`."""
    session_id: str
    seed: int
    step: int
    sol: int
    local_time: str
    mode: str
    game_status: GameStatus
    status_reason: str
    terrain_width: int
    terrain_height: int
    weather: WeatherState
    dust_intensity: float
    resources: ResourceState
    mission: MissionState
    objective_stats: ObjectiveStats
    build_pad_state: BuildPadState
    rules: list[GameRule]
    terrain: list[list[TerrainCell]]
    rovers: list[RoverState]
    structures: list[StructureState]
    history: list[HistoryEntry]


@dataclass(frozen=True)
class ActionValidation:
    """Result of legality checks before a command mutates the world. `warning` is non-fatal."""
    valid: bool
    reason: str = ""
    warning: str | None = None


@dataclass(frozen=True)
class EngineTransition:
    """One `step` result. `effective_action` is `invalid` when the command was blocked."""
    command: ActionCommand
    effective_action: ActionType
    actor: Actor
    before_checksum: str
    after_checksum: str
    reward: float
    reward_terms: dict[str, float]
    events: tuple[str, ...]
    state: WorldState

