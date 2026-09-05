/**
 * TypeScript shapes for UI snapshots, actions, and `aresim.gameplay.v1` files.
 * These names must stay aligned with `engine/aresim/integrations/ui.py`. The
 * engine owns gameplay meaning; this file only types what the UI already receives.
 */

export type TerrainType = "regolith" | "rock" | "ice" | "crater" | "dune" | "build_pad" | "ridge";
export type WeatherState = "Clear" | "Dusty" | "Dust Front" | "Severe Storm" | "Cold Night";
export type GameStatus = "running" | "paused" | "victory" | "game_over";
export type ActionType =
  | "move"
  | "scan"
  | "extract"
  | "build"
  | "service"
  | "unload"
  | "wait"
  | "invalid"
  | "event";
export type OverlayMode = "none" | "ice" | "ore" | "dust" | "roughness";
export type ViewportZoomMode = "fit" | "manual";
export type CameraView = "survey" | "top" | "rover";
export type RunMode = "manual" | "algorithm" | "load";
export type SavedRunMode = RunMode | "llm";
export type AlgorithmId = "random" | "random_valid" | "wait" | "scripted" | "masked_ppo";
export interface PolicyMeta {
  algorithmId: AlgorithmId;
  policyId: string;
  actionIndex: number;
  action: SimAction;
  actionMask: number[];
}

export interface AgentStepResponse {
  snapshot: SimSnapshot;
  policyMeta: PolicyMeta;
}

export interface PolicyCatalogEntry {
  id: AlgorithmId;
  label: string;
  kind: "baseline" | "checkpoint";
  requiresPath: boolean;
}

export interface PoliciesResponse {
  policies: PolicyCatalogEntry[];
  capabilities: { rllib: boolean };
}

export interface AttachPolicyResponse {
  algorithmId: AlgorithmId;
  policyId: string;
  checkpointPath?: string | null;
}
export type LegacyLlmAgentId = "mock_llm";

/** One cell of the authoritative map. Resource signals are 0–1; flags are one-shot. */
export interface TerrainCell {
  x: number;
  y: number;
  terrain: TerrainType;
  height: number;
  roughness: number;
  ice: number;
  ore: number;
  dust: number;
  scanned: boolean;
  extracted: boolean;
}

/** The acting rover. Cargo masses share `cargoCapacityKg`; Unload empties them on the pad. */
export interface RoverEntity {
  id: string;
  name: string;
  x: number;
  y: number;
  battery: number;
  health: number;
  cargoIce: number;
  cargoOre: number;
  cargoSamples: number;
  cargoCapacityKg: number;
  currentTask: string;
}

export interface StructureEntity {
  id: string;
  type: "habitat" | "solar" | "battery" | "storage" | "extractor";
  name: string;
  x: number;
  y: number;
  health: number;
  powered: boolean;
  status: string;
}

export interface GameRule {
  id: string;
  label: string;
  description: string;
  status: "stable" | "warning" | "failed" | "complete";
  value: string;
}

export type WarningKind = "blocked" | "terrain" | "system" | "terminal" | "progress";
export type WarningSeverity = "info" | "success" | "warning" | "danger";

export interface SimWarning {
  id: number;
  kind: WarningKind;
  title: string;
  message: string;
  severity: WarningSeverity;
  relatedAction?: ActionType;
  target?: { x: number; y: number };
}

export interface AgentHistoryEntry {
  id: string;
  step: number;
  actor: "Player" | "Agent" | "System";
  action: ActionType;
  target?: { x: number; y: number };
  result: string;
  reward: number;
  rewardTerms: Record<string, number>;
  resourceDelta: {
    power?: number;
    battery?: number;
    water?: number;
    oxygen?: number;
    ice?: number;
    ore?: number;
    samples?: number;
  };
  events: string[];
}

export type BuildPadStatus = "normal" | "needs_service" | "habitat_built" | "habitat_built_needs_service";

export interface ObjectiveStats {
  iceCollected: number;
  iceDelivered: number;
  samplesCollected: number;
  samplesDelivered: number;
  unloadCount: number;
  iceSitesExtracted: number;
  iceSitesTotal: number;
  terrainScanned: number;
  rockSitesTotal: number;
  habitatBuildProgress: number;
  habitatBuildCount: number;
  serviceCount: number;
  rewardTotals: {
    iceCollected: number;
    terrainScanned: number;
    habitatBuilt: number;
    serviced: number;
    delivered: number;
    traversal: number;
    blockedPenalty: number;
    total: number;
  };
}

export interface BuildPadState {
  serviceNeeded: boolean;
  status: BuildPadStatus;
}

/**
 * CamelCase world snapshot from the Python API.
 * Treat this as read-only truth. Do not invent legality, rewards, or terminals here.
 */
export interface SimSnapshot {
  sessionId: string;
  seed: number;
  step: number;
  sol: number;
  localTime: string;
  mode: "Play" | "Replay" | "AI Watch" | "Debug";
  gameStatus: GameStatus;
  statusReason: string;
  terrainSize: { width: number; height: number };
  weather: WeatherState;
  dustIntensity: number;
  resources: {
    powerGenerated: number;
    powerConsumed: number;
    battery: number;
    water: number;
    oxygen: number;
    livability: number;
  };
  mission: {
    title: string;
    objective: string;
    rewardObjectives: string[];
    alerts: string[];
  };
  objectiveStats: ObjectiveStats;
  buildPadState: BuildPadState;
  rules: GameRule[];
  terrain: TerrainCell[][];
  rovers: RoverEntity[];
  structures: StructureEntity[];
  history: AgentHistoryEntry[];
}

export type SelectionTarget =
  | { kind: "cell"; x: number; y: number }
  | { kind: "rover"; id: string }
  | { kind: "structure"; id: string }
  | { kind: "status"; id: string }
  | { kind: "history"; id: string };

/** Command sent to the backend. Omit `target` when Wait or when the engine should infer one. */
export interface SimAction {
  type: ActionType;
  target?: { x: number; y: number };
}

export interface LoadedReplay {
  replayId: string;
  fileName: string;
  seed: number;
  totalSteps: number;
  cursor: number;
  loadedAt: string;
  schemaVersion?: string;
  checkpoints?: GameplayCheckpoint[];
  activeCheckpointId?: string;
  timelineStepCount?: number;
}

export type GameplayCheckpointReason = "initial" | "interval" | "event" | "final";

/** One recorded step in a save file. Replay applies `changes` without re-running the engine. */
export interface GameplayStepDelta {
  step: number;
  sol: number;
  localTime: string;
  actor: AgentHistoryEntry["actor"];
  action: ActionType;
  target?: { x: number; y: number };
  result: string;
  events: string[];
  reward: number;
  rewardTerms: Record<string, number>;
  resourceDelta: AgentHistoryEntry["resourceDelta"];
  changes: {
    terrainCells: TerrainCell[];
    rovers: RoverEntity[];
    structures: StructureEntity[];
    resources?: SimSnapshot["resources"];
    objectiveStats?: ObjectiveStats;
    buildPadState?: BuildPadState;
    mission?: SimSnapshot["mission"];
    rules?: GameRule[];
    status?: Pick<SimSnapshot, "gameStatus" | "statusReason" | "weather" | "dustIntensity" | "sol" | "localTime" | "step">;
    appendedHistoryEntry: AgentHistoryEntry;
  };
  after: {
    rover: { x: number; y: number; battery: number; health: number };
    gameStatus: GameStatus;
    totalReward: number;
  };
}

export interface GameplayCheckpoint {
  id: string;
  step: number;
  label: string;
  reason: GameplayCheckpointReason;
  summary: string;
  target?: { x: number; y: number };
  reward?: number;
  snapshot: SimSnapshot;
}

/** Replay projection nested inside every newly exported trajectory. */
export interface AresReplayProjectionV1 {
  schemaVersion: "aresim.trajectory.replay.v1";
  savedAt: string;
  fileName: string;
  appVersion: string;
  metadata: {
    sessionId: string;
    seed: number;
    runMode: SavedRunMode;
    /** UI-selected IDs or a versioned policy ID from a recorded RL rollout. */
    algorithmId?: string;
    /** Read-only compatibility metadata for gameplay files created before LLM runtime removal. */
    llmAgentId?: LegacyLlmAgentId;
    totalSteps: number;
    finalStatus: GameStatus;
  };
  initialSnapshot: SimSnapshot;
  steps: GameplayStepDelta[];
  checkpoints: GameplayCheckpoint[];
  finalSnapshot: SimSnapshot;
  integrity: {
    finalStep: number;
    stepCount: number;
    checkpointCount: number;
  };
}

/** Read-only legacy standalone schema accepted by Replay import. */
export type AresGameplaySaveV1 = Omit<AresReplayProjectionV1, "schemaVersion"> & {
  schemaVersion: "aresim.gameplay.v1";
};

/** Public standalone trajectory: policy data plus a complete UI replay projection. */
export interface AresTrajectoryEpisodeV1 {
  schemaVersion: "aresim.trajectory.episode.v1";
  savedAt: string;
  metadata: {
    episodeId: string;
    source: "ui" | "rollout";
    environmentSeed: number;
    policyId: string | null;
    agentSeed: number | null;
  };
  /** Null for UI-authored sessions, which have no framework-neutral policy trace. */
  policy: Record<string, unknown> | null;
  replay: AresReplayProjectionV1;
}

export interface AnalyticsSeriesPoint {
  step: number;
  sol: number;
  localTime: string;
  actor: AgentHistoryEntry["actor"];
  action: ActionType;
  valid: boolean;
  reward: number;
  rewardTerms: Record<string, number>;
  cumulativeReward: number;
  cumulativeByCategory: Record<string, number>;
  battery: number;
  health: number;
  roverX: number;
  roverY: number;
  water: number;
  oxygen: number;
  livability: number;
  powerGenerated: number;
  powerConsumed: number;
  dustIntensity: number;
  weather: WeatherState;
  gameStatus: GameStatus;
  cargoIce: number;
  cargoSamples: number;
  payloadUsedKg: number;
  payloadCapacityKg: number;
  buildProgress: number;
  serviceCount: number;
}
