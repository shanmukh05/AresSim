/**
 * Zustand store: live snapshot, replay cursor, camera, and HUD interaction.
 * All mutations go through `AresApiClient`. Components should read from here
 * rather than calling the API directly.
 */

import { create } from "zustand";
import { AresApiClient, AresApiError, type ReplayResponse } from "../api/aresClient";
import { BUILD_DEPLOYED } from "../lib/deployConfig";
import { downloadTrajectoryFile } from "../lib/trajectoryFile";
import { payloadUsedKg } from "../lib/payload";
import type { ActionType, AgentStepResponse, AresReplayProjectionV1, AlgorithmId, AnalyticsSeriesPoint, CameraView, LlmDmAttachPayload, LlmDmCatalog, LlmDmStepMeta, LlmFeedMessage, LlmGoalSummary, LlmLearningStatement, LoadedReplay, OverlayMode, PolicyActionMode, RunMode, SelectionTarget, SimAction, SimSnapshot, SimWarning, ViewportZoomMode } from "../types/sim";
import { supportsActionSampling, usesCheckpoint } from "../types/sim";

const client = new AresApiClient();

const DEFAULT_LLM_DM_CONFIG: LlmDmAttachPayload = {
  actorProvider: "decision_models",
  actorModel: "jev-1.14.0",
  actorRoute: null,
  actorApiKey: "",
  actorContextWindow: 131072,
  actorMaxOutputTokens: 2048,
  learnerEnabled: true,
  learnerProvider: "gemini",
  learnerModel: "gemini-4-neon",
  learnerRoute: null,
  learnerApiKey: "",
  learnerContextWindow: 262144,
  learnerMaxOutputTokens: 16384,
  historyMode: "focused",
  adviceLimit: 3,
  attemptTimeoutSeconds: 60,
  maxEpisodeSteps: 256,
};

function nowStamp(): string {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function feedId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
}

function buildInitialSeries(snapshot: SimSnapshot): AnalyticsSeriesPoint[] {
  return [seriesPointFromSnapshot(snapshot)];
}

function seriesPointFromSnapshot(snapshot: SimSnapshot): AnalyticsSeriesPoint {
  const totals = snapshot.objectiveStats.rewardTotals;
  const cumulativeByCategory: Record<string, number> = {
    iceCollected: totals.iceCollected,
    terrainScanned: totals.terrainScanned,
    habitatBuilt: totals.habitatBuilt,
    serviced: totals.serviced,
    delivered: totals.delivered,
    traversal: totals.traversal,
    blockedPenalty: totals.blockedPenalty,
    total: totals.total,
  };
  const rover = snapshot.rovers[0];
  return {
    step: snapshot.step,
    sol: snapshot.sol,
    localTime: snapshot.localTime,
    actor: "System",
    action: "event",
    valid: true,
    reward: 0,
    rewardTerms: {},
    cumulativeReward: totals.total,
    cumulativeByCategory,
    battery: snapshot.resources.battery,
    health: rover.health,
    roverX: rover.x,
    roverY: rover.y,
    water: snapshot.resources.water,
    oxygen: snapshot.resources.oxygen,
    livability: snapshot.resources.livability,
    powerGenerated: snapshot.resources.powerGenerated,
    powerConsumed: snapshot.resources.powerConsumed,
    dustIntensity: snapshot.dustIntensity,
    weather: snapshot.weather,
    gameStatus: snapshot.gameStatus,
    cargoIce: rover.cargoIce,
    cargoSamples: rover.cargoSamples,
    payloadUsedKg: payloadUsedKg(rover),
    payloadCapacityKg: rover.cargoCapacityKg,
    buildProgress: snapshot.objectiveStats.habitatBuildProgress,
    serviceCount: snapshot.objectiveStats.serviceCount,
  };
}

function buildSeriesFromGameplay(gameplay: AresReplayProjectionV1): AnalyticsSeriesPoint[] {
  const cumulativeByCategory: Record<string, number> = {};
  const points: AnalyticsSeriesPoint[] = [];
  let currentRover = gameplay.initialSnapshot.rovers[0];
  for (const delta of gameplay.steps) {
    const totals = delta.changes.objectiveStats?.rewardTotals;
    if (totals) {
      cumulativeByCategory.iceCollected = totals.iceCollected;
      cumulativeByCategory.terrainScanned = totals.terrainScanned;
      cumulativeByCategory.habitatBuilt = totals.habitatBuilt;
      cumulativeByCategory.serviced = totals.serviced;
      cumulativeByCategory.delivered = totals.delivered;
      cumulativeByCategory.traversal = totals.traversal;
      cumulativeByCategory.blockedPenalty = totals.blockedPenalty;
      cumulativeByCategory.total = totals.total;
    }
    const status = delta.changes.status;
    const rover = delta.after.rover;
    currentRover = delta.changes.rovers.find((candidate) => candidate.id === currentRover?.id) ?? currentRover;
    const resources = delta.changes.resources;
    points.push({
      step: delta.step,
      sol: delta.sol,
      localTime: delta.localTime,
      actor: delta.actor,
      action: delta.action,
      valid: delta.action !== "invalid",
      reward: delta.reward,
      rewardTerms: delta.rewardTerms,
      cumulativeReward: delta.after.totalReward,
      cumulativeByCategory: { ...cumulativeByCategory },
      battery: resources?.battery ?? rover.battery,
      health: rover.health,
      roverX: rover.x,
      roverY: rover.y,
      water: resources?.water ?? 0,
      oxygen: resources?.oxygen ?? 0,
      livability: resources?.livability ?? 0,
      powerGenerated: resources?.powerGenerated ?? 0,
      powerConsumed: resources?.powerConsumed ?? 0,
      dustIntensity: status?.dustIntensity ?? 0,
      weather: status?.weather ?? "Clear",
      gameStatus: delta.after.gameStatus,
      cargoIce: currentRover?.cargoIce ?? 0,
      cargoSamples: currentRover?.cargoSamples ?? 0,
      payloadUsedKg: currentRover ? payloadUsedKg(currentRover) : 0,
      payloadCapacityKg: currentRover?.cargoCapacityKg ?? gameplay.initialSnapshot.rovers[0]?.cargoCapacityKg ?? 0,
      buildProgress: delta.changes.objectiveStats?.habitatBuildProgress ?? 0,
      serviceCount: delta.changes.objectiveStats?.serviceCount ?? 0,
    });
  }
  return points;
}

/** Client plus presentation state. `snapshot` is null until `start` succeeds. */
interface AresStore {
  snapshot: SimSnapshot | null;
  selectedTool: ActionType;
  overlayMode: OverlayMode;
  selectedTarget: SelectionTarget | null;
  hoveredTarget: SelectionTarget | null;
  highlightedCell: { x: number; y: number } | null;
  paused: boolean;
  speed: number;
  runMode: RunMode;
  algorithmId: AlgorithmId;
  checkpointPath: string;
  policyActionMode: PolicyActionMode;
  deployedMode: boolean;
  defaultCheckpoints: Record<string, string | boolean>;
  llmDmCatalog: LlmDmCatalog | null;
  llmDmConfig: LlmDmAttachPayload;
  llmLearnings: LlmLearningStatement[];
  llmFeed: LlmFeedMessage[];
  llmCurrentGoal: LlmGoalSummary | null;
  llmLastAppliedIds: string[];
  llmLearnerConfigured: boolean;
  llmLearningBusy: boolean;
  llmAutoLearn: boolean;
  llmFeedOpen: boolean;
  llmConfigOpen: boolean;
  manualControlsOpen: boolean;
  manualQueueCount: number;
  rllibAvailable: boolean;
  muzeroAvailable: boolean;
  worldModelsAvailable: boolean;
  llmDmAvailable: boolean;
  policyEpisodeDone: boolean;
  policyCapabilitiesLoaded: boolean;
  policyAttachKey: string | null;
  loadedReplay: LoadedReplay | null;
  loadedGameplay: AresReplayProjectionV1 | null;
  analyticsSeries: AnalyticsSeriesPoint[];
  analyticsOpen: boolean;
  audioMuted: boolean;
  backendBusy: boolean;
  backendError: string | null;
  setAnalyticsOpen: (open: boolean) => void;
  toggleAudioMuted: () => void;
  historyNewestFirst: boolean;
  historyFilter: ActionType | "all";
  viewportZoomMode: ViewportZoomMode;
  viewportZoomScale: number;
  viewportCenter: { x: number; y: number } | null;
  cameraView: CameraView;
  roverCameraYaw: number;
  showGrid: boolean;
  showRoverVisibility: boolean;
  actionWarning: SimWarning | null;
  lastSavedAt: string | null;
  start: (seed: number) => Promise<void>;
  retryBackend: () => Promise<void>;
  randomize: () => Promise<void>;
  saveRun: (filename: string) => Promise<void>;
  loadGameplay: (content: string, fileName: string) => Promise<void>;
  jumpToReplayCheckpoint: (checkpointId: string) => Promise<void>;
  jumpToReplayStep: (step: number) => Promise<void>;
  resetCurrentRun: () => Promise<void>;
  resetReplay: () => Promise<void>;
  step: () => Promise<void>;
  stepAutoplay: () => Promise<void>;
  dispatchAction: (action: SimAction) => Promise<void>;
  moveRover: (dx: number, dy: number) => Promise<void>;
  clearActionWarning: () => void;
  pauseResume: () => Promise<void>;
  setRunMode: (mode: RunMode) => void;
  setAlgorithmId: (id: AlgorithmId) => void;
  setCheckpointPath: (path: string) => void;
  setPolicyActionMode: (mode: PolicyActionMode) => void;
  updateLlmDmConfig: (patch: Partial<LlmDmAttachPayload>) => void;
  setLlmAutoLearn: (enabled: boolean) => void;
  setLlmFeedOpen: (open: boolean) => void;
  setLlmConfigOpen: (open: boolean) => void;
  setManualControlsOpen: (open: boolean) => void;
  clearLlmFeed: () => void;
  runLlmLearn: () => Promise<void>;
  refreshPolicyCapabilities: () => Promise<void>;
  attachCurrentPolicy: () => Promise<void>;
  setSpeed: (speed: number) => void;
  setSelectedTool: (tool: ActionType) => void;
  setOverlayMode: (mode: OverlayMode) => void;
  selectTarget: (target: SelectionTarget | null) => void;
  hoverTarget: (target: SelectionTarget | null) => void;
  highlightCell: (cell: { x: number; y: number } | null) => void;
  toggleHistoryOrder: () => void;
  setHistoryFilter: (action: ActionType | "all") => void;
  zoomIn: () => void;
  zoomOut: () => void;
  fitViewport: () => void;
  setViewportCenter: (center: { x: number; y: number }) => void;
  setCameraView: (view: CameraView) => void;
  setRoverCameraYaw: (yaw: number | ((current: number) => number)) => void;
  toggleGrid: () => void;
  toggleRoverVisibility: () => void;
}

/** Hook into session/replay/UI state. Call `start` once from `App`. */
export const useAresStore = create<AresStore>((set, get) => {
  let backendRequests = 0;
  let policyRevision = 0;
  let policyDetach = Promise.resolve();
  let stepQueue: AgentStepResponse[] = [];
  let prefetchPromise: Promise<void> | null = null;
  type PendingManualInput =
    | { kind: "move"; dx: number; dy: number }
    | { kind: "action"; action: SimAction };
  let manualQueue: PendingManualInput[] = [];
  let manualFlushActive = false;
  let inFlightPlannedPos: { x: number; y: number } | null = null;

  const clearManualQueue = () => {
    manualQueue = [];
    inFlightPlannedPos = null;
    if (get().manualQueueCount !== 0) set({ manualQueueCount: 0 });
  };

  const computePlannedRoverPos = (): { x: number; y: number } | null => {
    const snap = get().snapshot;
    if (!snap || !snap.rovers[0]) return null;
    let x = inFlightPlannedPos ? inFlightPlannedPos.x : snap.rovers[0].x;
    let y = inFlightPlannedPos ? inFlightPlannedPos.y : snap.rovers[0].y;
    const maxX = snap.terrainSize.width - 1;
    const maxY = snap.terrainSize.height - 1;
    for (const item of manualQueue) {
      if (item.kind === "move") {
        x = Math.max(0, Math.min(maxX, x + item.dx));
        y = Math.max(0, Math.min(maxY, y + item.dy));
      } else if (item.action.type === "move" && item.action.target) {
        x = Math.max(0, Math.min(maxX, item.action.target.x));
        y = Math.max(0, Math.min(maxY, item.action.target.y));
      }
    }
    return { x, y };
  };

  const applyManualSnapshot = (nextSnapshot: SimSnapshot, action: SimAction) => {
    const target = action.type === "move" ? action.target : undefined;
    set((current) => ({
      selectedTool: action.type,
      ...(target ? { selectedTarget: { kind: "cell" as const, ...target } } : {}),
      snapshot: nextSnapshot,
      ...(target && current.viewportZoomMode === "manual"
        ? { viewportCenter: { x: target.x + 0.5, y: target.y + 0.5 } }
        : target
          ? { viewportCenter: null }
          : {}),
      actionWarning: warningFromSnapshot(nextSnapshot),
      analyticsSeries: appendSeriesPoint(current.analyticsSeries, nextSnapshot),
    }));
  };

  const flushManualQueue = async () => {
    if (manualFlushActive) return;
    manualFlushActive = true;
    try {
      while (manualQueue.length > 0) {
        const state = get();
        const snap = state.snapshot;
        if (!snap || state.runMode !== "manual" || snap.gameStatus === "game_over") {
          manualQueue = [];
          set({ manualQueueCount: 0 });
          break;
        }
        const batch = manualQueue.splice(0, 16);
        set({ manualQueueCount: manualQueue.length });

        let rx = snap.rovers[0].x;
        let ry = snap.rovers[0].y;
        const maxX = snap.terrainSize.width - 1;
        const maxY = snap.terrainSize.height - 1;
        const selected = state.selectedTarget;
        const concreteActions: SimAction[] = batch.map((item) => {
          if (item.kind === "move") {
            rx = Math.max(0, Math.min(maxX, rx + item.dx));
            ry = Math.max(0, Math.min(maxY, ry + item.dy));
            return { type: "move", target: { x: rx, y: ry } };
          }
          const actionTarget =
            item.action.target ?? (selected?.kind === "cell" ? { x: selected.x, y: selected.y } : undefined);
          if (item.action.type === "move" && actionTarget) {
            rx = Math.max(0, Math.min(maxX, actionTarget.x));
            ry = Math.max(0, Math.min(maxY, actionTarget.y));
          }
          return { ...item.action, target: actionTarget };
        });
        inFlightPlannedPos = { x: rx, y: ry };

        let returned: SimSnapshot[] = [];
        const failureTitle = concreteActions.length === 1 && concreteActions[0].type === "move" ? "Movement failed" : "Action failed";
        if (concreteActions.length === 1) {
          await runBackend(
            () => client.dispatchAction(concreteActions[0]),
            (nextSnapshot) => {
              returned = [nextSnapshot];
            },
            failureTitle,
            true,
          );
        } else {
          await runBackend(
            () => client.dispatchActionBatch(concreteActions),
            (snapshots) => {
              returned = snapshots;
            },
            failureTitle,
            true,
          );
        }

        if (returned.length === 0) {
          manualQueue = [];
          set({ manualQueueCount: 0 });
          break;
        }

        for (let i = 0; i < returned.length; i += 1) {
          const stepSnap = returned[i];
          const action = concreteActions[Math.min(i, concreteActions.length - 1)];
          applyManualSnapshot(stepSnap, action);
          const latest = stepSnap.history[0];
          if (stepSnap.gameStatus !== "running" || latest?.action === "invalid") {
            manualQueue = [];
            set({ manualQueueCount: 0 });
            break;
          }
          if (i < returned.length - 1) {
            await new Promise((resolve) => window.setTimeout(resolve, 90));
          }
        }
        inFlightPlannedPos = null;
      }
    } finally {
      inFlightPlannedPos = null;
      manualFlushActive = false;
      if (get().manualQueueCount !== manualQueue.length) {
        set({ manualQueueCount: manualQueue.length });
      }
    }
  };

  const clearStepQueue = () => {
    stepQueue = [];
    prefetchPromise = null;
  };

  const applyAgentStepResponse = (response: AgentStepResponse): boolean => {
    const llmMeta = response.policyMeta.llmDm;
    const appendedFeed = llmMeta ? messagesFromLlmStep(response.snapshot.step, llmMeta) : [];
    const shouldAutoLearn = Boolean(
      response.policyMeta.episodeDone &&
      get().algorithmId === "llm_dm" &&
      get().llmAutoLearn &&
      llmMeta?.learnerConfigured,
    );
    set((current) => ({
      snapshot: response.snapshot,
      policyEpisodeDone: response.policyMeta.episodeDone === true,
      llmCurrentGoal: llmMeta?.goal ?? current.llmCurrentGoal,
      llmLastAppliedIds: (llmMeta?.applied_advice ?? []).map((item) => item.id),
      llmLearnerConfigured: llmMeta?.learnerConfigured ?? current.llmLearnerConfigured,
      llmLearnings: llmMeta?.learnings ?? current.llmLearnings,
      llmFeed: appendedFeed.length ? [...current.llmFeed, ...appendedFeed].slice(-160) : current.llmFeed,
      actionWarning: response.policyMeta.episodeDone ? {
        id: response.snapshot.step, kind: "progress", title: "Policy episode complete",
        message: `${response.policyMeta.endingReason ?? "Episode ended"}. Reset the game to start another episode.`, severity: "success",
      } : warningFromSnapshot(response.snapshot),
      analyticsSeries: appendSeriesPoint(current.analyticsSeries, response.snapshot),
    }));
    return shouldAutoLearn;
  };

  /** Fast-forward UI state through any buffered steps so it matches the backend before saving or switching modes. */
  const flushStepQueue = () => {
    if (stepQueue.length === 0) {
      prefetchPromise = null;
      return;
    }
    const buffered = stepQueue;
    stepQueue = [];
    prefetchPromise = null;
    const last = buffered[buffered.length - 1];
    set((current) => {
      let series = current.analyticsSeries;
      for (const item of buffered) {
        series = appendSeriesPoint(series, item.snapshot);
      }
      return {
        snapshot: last.snapshot,
        policyEpisodeDone: last.policyMeta.episodeDone === true,
        actionWarning: last.policyMeta.episodeDone ? {
          id: last.snapshot.step, kind: "progress", title: "Policy episode complete",
          message: `${last.policyMeta.endingReason ?? "Episode ended"}. Reset the game to start another episode.`, severity: "success",
        } : warningFromSnapshot(last.snapshot),
        analyticsSeries: series,
      };
    });
  };

  const batchSizeFor = (algorithmId: AlgorithmId): number => (algorithmId === "muzero" ? 8 : 12);

  const canPrefetchMore = (tail?: AgentStepResponse): boolean => {
    const snap = tail?.snapshot ?? get().snapshot;
    const done = tail?.policyMeta.episodeDone === true || get().policyEpisodeDone;
    return Boolean(snap && snap.gameStatus === "running" && !done);
  };

  const startPrefetch = (expectedRevision: number, count: number) => {
    if (prefetchPromise) return;
    const promise: Promise<void> = client.agentSteps(count).then(
      (steps) => {
        if (
          expectedRevision === policyRevision &&
          get().runMode === "algorithm" &&
          get().algorithmId !== "llm_dm"
        ) {
          stepQueue.push(...steps);
        }
      },
      () => {
        // Silent in background; foreground fetch will recover or surface any persistent error.
      },
    ).finally(() => {
      if (prefetchPromise === promise) prefetchPromise = null;
    });
    prefetchPromise = promise;
  };

  const invalidateLlmPolicy = () => {
    policyRevision += 1;
    flushStepQueue();
    const sessionId = get().snapshot?.sessionId;
    if (!sessionId || get().algorithmId !== "llm_dm") return;
    policyDetach = policyDetach.then(async () => {
      if (get().snapshot?.sessionId === sessionId) await client.detachPolicy();
    }).catch(() => {
      set({ backendError: "Could not detach the previous LLM policy. Reset before continuing." });
    });
  };

  const runBackend = async <T>(operation: () => Promise<T>, apply: (value: T) => void, failureTitle = "Backend request failed", allowWhileBusy = false) => {
    if (get().backendBusy && !allowWhileBusy) return;
    const revision = policyRevision;
    backendRequests += 1;
    set({ backendBusy: true, backendError: null });
    try {
      let value: T;
      try {
        value = await operation();
      } catch (error) {
        // Recover transparently if Modal scaled to zero and dropped in-memory session state.
        const currentSnapshot = get().snapshot;
        if (error instanceof AresApiError && error.code === "SESSION_NOT_FOUND" && currentSnapshot) {
          clearStepQueue();
          clearManualQueue();
          const fresh = await client.startSession({ seed: currentSnapshot.seed });
          set({
            snapshot: fresh,
            policyAttachKey: null,
            policyEpisodeDone: false,
            analyticsSeries: buildInitialSeries(fresh),
          });
          value = await operation();
        } else {
          throw error;
        }
      }
      if (revision === policyRevision) apply(value);
    } catch (error) {
      if (revision !== policyRevision) return;
      const message = error instanceof AresApiError ? error.message : "The backend request failed.";
      set((state) => ({
        backendError: message,
        actionWarning: state.snapshot
          ? { id: state.snapshot.step, kind: "blocked", title: failureTitle, message, severity: "warning" }
          : state.actionWarning,
      }));
    } finally {
      backendRequests -= 1;
      set({ backendBusy: backendRequests > 0 });
    }
  };

  const updateReplay = (response: ReplayResponse) => {
    const entry = response.snapshot.history.find((item) => item.step === response.cursor);
    set((state) => ({
      snapshot: response.snapshot,
      loadedReplay: state.loadedReplay
        ? { ...state.loadedReplay, cursor: response.cursor, activeCheckpointId: response.activeCheckpointId ?? state.loadedReplay.activeCheckpointId }
        : null,
      highlightedCell: entry?.target ?? null,
      selectedTarget: entry ? { kind: "history", id: entry.id } : state.selectedTarget,
    }));
  };

  return {
  snapshot: null,
  selectedTool: "move",
  overlayMode: "none",
  selectedTarget: null,
  hoveredTarget: null,
  highlightedCell: null,
  paused: false,
  speed: 1,
  runMode: "manual",
  algorithmId: "random_valid",
  checkpointPath: "",
  policyActionMode: BUILD_DEPLOYED ? "deterministic" : "stochastic",
  deployedMode: BUILD_DEPLOYED,
  defaultCheckpoints: {},
  llmDmCatalog: null,
  llmDmConfig: { ...DEFAULT_LLM_DM_CONFIG },
  llmLearnings: [],
  llmFeed: [],
  llmCurrentGoal: null,
  llmLastAppliedIds: [],
  llmLearnerConfigured: false,
  llmLearningBusy: false,
  llmAutoLearn: true,
  llmFeedOpen: true,
  llmConfigOpen: false,
  manualControlsOpen: true,
  manualQueueCount: 0,
  rllibAvailable: false,
  muzeroAvailable: false,
  worldModelsAvailable: false,
  llmDmAvailable: false,
  policyEpisodeDone: false,
  policyCapabilitiesLoaded: false,
  policyAttachKey: null,
  loadedReplay: null,
  loadedGameplay: null,
  analyticsSeries: [],
  analyticsOpen: false,
  audioMuted: false,
  backendBusy: false,
  backendError: null,
  setAnalyticsOpen: (open) => set({ analyticsOpen: open }),
  toggleAudioMuted: () => set((state) => ({ audioMuted: !state.audioMuted })),
  historyNewestFirst: true,
  historyFilter: "all",
  viewportZoomMode: "fit",
  viewportZoomScale: 1,
  viewportCenter: null,
  cameraView: "survey",
  roverCameraYaw: 0,
  showGrid: false,
  showRoverVisibility: false,
  actionWarning: null,
  lastSavedAt: null,
  start: async (seed) => {
    policyRevision += 1;
    clearStepQueue();
    clearManualQueue();
    await runBackend(
    async () => {
      const snapshot = await client.startSession({ seed });
      await get().refreshPolicyCapabilities();
      return snapshot;
    },
    (snapshot) => set({
      snapshot,
      policyAttachKey: null,
      policyEpisodeDone: false,
      llmFeed: [],
      llmCurrentGoal: null,
      llmLastAppliedIds: [],
      selectedTarget: null,
      highlightedCell: null,
      paused: false,
      actionWarning: null,
      backendError: null,
      viewportCenter: null,
      cameraView: "survey",
      roverCameraYaw: 0,
      loadedReplay: null,
      loadedGameplay: null,
      analyticsSeries: buildInitialSeries(snapshot),
    }),
    "Session start failed",
    true,
    );
  },
  retryBackend: async () => get().start(1447),
  randomize: async () => {
    policyRevision += 1;
    clearStepQueue();
    clearManualQueue();
    await runBackend(
    async () => {
      const snapshot = await client.randomizeSession();
      await get().refreshPolicyCapabilities();
      return snapshot;
    },
    (snapshot) => set({
      snapshot,
      policyAttachKey: null,
      policyEpisodeDone: false,
      llmFeed: [],
      llmCurrentGoal: null,
      llmLastAppliedIds: [],
      selectedTarget: null,
      highlightedCell: null,
      paused: false,
      historyFilter: "all",
      viewportZoomMode: "fit",
      viewportZoomScale: 1,
      viewportCenter: null,
      cameraView: "survey",
      roverCameraYaw: 0,
      actionWarning: null,
      loadedReplay: null,
      loadedGameplay: null,
      analyticsSeries: buildInitialSeries(snapshot),
    }),
    "Randomization failed",
    true,
    );
  },
  saveRun: async (filename) => {
    flushStepQueue();
    const snapshot = get().snapshot;
    const runMode = get().runMode;
    if (!snapshot || runMode === "load") return;
    const safeName = filename.trim() || `aresim-seed-${snapshot.seed}-step-${snapshot.step}.json`;
    await runBackend(
      () => client.saveSession({
        fileName: safeName,
        runMode,
        algorithmId: runMode === "algorithm" ? get().algorithmId : undefined,
        checkpointPath: runMode === "algorithm" && usesCheckpoint(get().algorithmId) ? get().checkpointPath : undefined,
      }),
      (trajectory) => {
        downloadTrajectoryFile(safeName, trajectory);
        set({
          lastSavedAt: trajectory.savedAt,
          actionWarning: { id: snapshot.step, kind: "progress", title: "Trajectory exported", message: `Exported "${trajectory.replay.fileName}" (seed ${snapshot.seed}, step ${snapshot.step}) as replayable trajectory JSON.`, severity: "success" },
        });
      },
      "Save failed",
    );
  },
  loadGameplay: async (content, fileName) => {
    policyRevision += 1;
    clearStepQueue();
    clearManualQueue();
    await runBackend(
    () => client.loadReplay(content, fileName),
    (response) => {
      const gameplay = response.gameplay!;
      const replay: LoadedReplay = {
        replayId: response.replayId,
        fileName: response.fileName,
        seed: response.seed,
        totalSteps: response.totalSteps,
        cursor: response.cursor,
        loadedAt: new Date().toISOString(),
        schemaVersion: response.schemaVersion,
        checkpoints: gameplay.checkpoints,
        activeCheckpointId: response.activeCheckpointId,
        timelineStepCount: gameplay.steps.length,
      };
      set({
        snapshot: response.snapshot,
        runMode: "load",
        paused: false,
        llmFeed: [],
        llmCurrentGoal: null,
        llmLastAppliedIds: [],
        loadedReplay: replay,
        loadedGameplay: gameplay,
        selectedTarget: null,
        highlightedCell: null,
        viewportZoomMode: "fit",
        viewportZoomScale: 1,
        viewportCenter: null,
        cameraView: "survey",
        roverCameraYaw: 0,
        analyticsSeries: buildSeriesFromGameplay(gameplay),
        actionWarning: { id: response.snapshot.step, kind: "progress", title: "Trajectory loaded", message: `Loaded ${response.fileName} with seed ${response.seed} and ${gameplay.checkpoints.length} checkpoints`, severity: "success" },
      });
    },
    "Load failed",
    true,
    );
  },
  resetCurrentRun: async () => {
    const snapshot = get().snapshot;
    if (!snapshot) return;
    await get().start(snapshot.seed);
    if (get().runMode === "algorithm") {
      await get().attachCurrentPolicy();
    }
  },
  resetReplay: async () => runBackend(() => client.resetReplay(), updateReplay, "Replay reset failed"),
  jumpToReplayCheckpoint: async (checkpointId) => {
    const checkpoint = get().loadedGameplay?.checkpoints.find((item) => item.id === checkpointId);
    if (!checkpoint) return;
    await runBackend(
      () => client.jumpReplay(checkpoint.step),
      (response) => {
        updateReplay(response);
        set((state) => ({ loadedReplay: state.loadedReplay ? { ...state.loadedReplay, activeCheckpointId: checkpointId } : null }));
      },
      "Replay jump failed",
    );
  },
  jumpToReplayStep: async (targetStep) => runBackend(() => client.jumpReplay(Math.round(targetStep)), updateReplay, "Replay jump failed"),
  step: async () => {
    const state = get();
    if (state.runMode === "load") {
      if (!state.loadedReplay) return;
      await runBackend(() => client.stepReplay(), updateReplay, "Replay step failed");
      return;
    }
    if (state.runMode === "algorithm") {
      if (state.algorithmId !== "llm_dm" && stepQueue.length > 0) {
        const next = stepQueue.shift()!;
        applyAgentStepResponse(next);
        const tail = stepQueue[stepQueue.length - 1] ?? next;
        if (stepQueue.length <= 4 && !prefetchPromise && canPrefetchMore(tail)) {
          startPrefetch(policyRevision, batchSizeFor(get().algorithmId));
        }
        return;
      }
      await get().attachCurrentPolicy();
      if (!get().policyAttachKey) return;
      let shouldAutoLearn = false;
      await runBackend(
        () => client.agentStep(),
        (response) => {
          shouldAutoLearn = applyAgentStepResponse(response);
        },
        "Action failed",
      );
      if (shouldAutoLearn) {
        await get().runLlmLearn();
      }
      return;
    }
    const action = { type: state.selectedTool } satisfies SimAction;
    await runBackend(
      () => client.dispatchAction(action),
      (snapshot) => set((current) => ({ snapshot, actionWarning: warningFromSnapshot(snapshot), analyticsSeries: appendSeriesPoint(current.analyticsSeries, snapshot) })),
      "Action failed",
    );
  },
  stepAutoplay: async () => {
    const state = get();
    if (state.runMode !== "algorithm" || state.algorithmId === "llm_dm") {
      await get().step();
      return;
    }
    await get().attachCurrentPolicy();
    if (!get().policyAttachKey) return;
    if (stepQueue.length === 0) {
      if (prefetchPromise) {
        await prefetchPromise;
      }
      if (stepQueue.length === 0) {
        const count = batchSizeFor(get().algorithmId);
        await runBackend(
          () => client.agentSteps(count),
          (steps) => {
            stepQueue.push(...steps);
          },
          "Action failed",
        );
      }
    }
    const next = stepQueue.shift();
    if (!next) return;
    applyAgentStepResponse(next);
    const tail = stepQueue[stepQueue.length - 1] ?? next;
    if (stepQueue.length <= 4 && !prefetchPromise && canPrefetchMore(tail)) {
      startPrefetch(policyRevision, batchSizeFor(get().algorithmId));
    }
  },
  dispatchAction: async (action) => {
    const state = get();
    if (!state.snapshot || state.snapshot.gameStatus === "game_over") return;
    if (state.runMode === "manual") {
      if (manualQueue.length >= 16) return;
      manualQueue.push({ kind: "action", action });
      set({ manualQueueCount: manualQueue.length });
      await flushManualQueue();
      return;
    }
    const selected = state.selectedTarget;
    const target = action.target ?? (selected?.kind === "cell" ? { x: selected.x, y: selected.y } : undefined);
    await runBackend(
      () => client.dispatchAction({ ...action, target }),
      (snapshot) => set((current) => ({ selectedTool: action.type, snapshot, actionWarning: warningFromSnapshot(snapshot), analyticsSeries: appendSeriesPoint(current.analyticsSeries, snapshot) })),
      "Action failed",
    );
  },
  moveRover: async (dx, dy) => {
    const state = get();
    const snapshot = state.snapshot;
    if (!snapshot || snapshot.gameStatus === "game_over") return;
    if (state.runMode === "manual") {
      if (manualQueue.length >= 16) return;
      manualQueue.push({ kind: "move", dx, dy });
      const planned = computePlannedRoverPos();
      set({
        manualQueueCount: manualQueue.length,
        selectedTool: "move",
        ...(planned ? { selectedTarget: { kind: "cell", ...planned } } : {}),
      });
      await flushManualQueue();
      return;
    }
    const rover = snapshot.rovers[0];
    const target = { x: Math.max(0, Math.min(snapshot.terrainSize.width - 1, rover.x + dx)), y: Math.max(0, Math.min(snapshot.terrainSize.height - 1, rover.y + dy)) };
    await runBackend(
      () => client.dispatchAction({ type: "move", target }),
      (nextSnapshot) => applyManualSnapshot(nextSnapshot, { type: "move", target }),
      "Movement failed",
    );
  },
  clearActionWarning: () => set({ actionWarning: null }),
  pauseResume: async () => {
    const state = get();
    if (state.runMode === "load" || (state.runMode === "algorithm" && state.algorithmId !== "llm_dm")) {
      set((current) => ({ paused: !current.paused }));
      return;
    }
    const nextPaused = !state.paused;
    policyRevision += 1;
    await runBackend(
      () => nextPaused ? client.pause() : client.resume(),
      (snapshot) => set({ paused: nextPaused, snapshot }),
      "Playback control failed",
      true,
    );
  },
  setSpeed: (speed) => set({ speed }),
  setRunMode: (mode) => {
    if (get().runMode !== mode) {
      clearManualQueue();
      invalidateLlmPolicy();
    }
    set({ runMode: mode, policyAttachKey: null });
    if (mode === "algorithm") {
      void get().refreshPolicyCapabilities();
    }
  },
  setAlgorithmId: (id) => {
    if (get().algorithmId !== id) invalidateLlmPolicy();
    const { deployedMode, defaultCheckpoints } = get();
    const defaultPath = !deployedMode && typeof defaultCheckpoints[id] === "string"
      ? String(defaultCheckpoints[id])
      : "";
    set({
      algorithmId: id,
      checkpointPath: usesCheckpoint(id) ? defaultPath : "",
      policyActionMode: deployedMode ? "deterministic" : "stochastic",
      policyAttachKey: null,
      policyEpisodeDone: false,
      ...(id === "llm_dm" ? { llmFeedOpen: true } : {}),
    });
  },
  setCheckpointPath: (path) => {
    policyRevision += 1;
    flushStepQueue();
    set({ checkpointPath: path, policyAttachKey: null });
  },
  setPolicyActionMode: (mode) => {
    policyRevision += 1;
    flushStepQueue();
    set({ policyActionMode: mode, policyAttachKey: null });
  },
  updateLlmDmConfig: (patch) => {
    invalidateLlmPolicy();
    set((state) => ({
      llmDmConfig: { ...state.llmDmConfig, ...patch },
      policyAttachKey: null,
    }));
  },
  setLlmAutoLearn: (enabled) => set({ llmAutoLearn: enabled }),
  setLlmFeedOpen: (open) => set({ llmFeedOpen: open }),
  setLlmConfigOpen: (open) => set({ llmConfigOpen: open }),
  setManualControlsOpen: (open) => set({ manualControlsOpen: open }),
  clearLlmFeed: () => set({ llmFeed: [] }),
  runLlmLearn: async () => {
    const state = get();
    if (!state.snapshot || state.algorithmId !== "llm_dm" || state.llmLearningBusy) return;
    set({ llmLearningBusy: true });
    try {
      const result = await client.llmLearn();
      const author = result.learnerModel || state.llmDmConfig.learnerModel || "Learner";
      const message: LlmFeedMessage = {
        id: feedId("learn"),
        kind: "learner",
        step: state.snapshot.step,
        author,
        badge: result.addedLearning ? `${result.addedLearning.code} LEARNED` : "NO CHANGE",
        text: result.addedLearning
          ? result.addedLearning.text
          : result.reason || "Learner reviewed the episode and proposed no new rule change.",
        learning: result.addedLearning ?? null,
        reason: result.reason,
        timestamp: nowStamp(),
      };
      set((current) => ({
        llmLearnings: result.learnings,
        llmFeed: [...current.llmFeed, message].slice(-160),
        actionWarning: result.addedLearning ? {
          id: current.snapshot?.step ?? 0,
          kind: "progress",
          title: `New learning ${result.addedLearning.code}`,
          message: "Learner synthesized a new rule and added it to live memory.",
          severity: "success",
        } : current.actionWarning,
      }));
    } catch (error) {
      const msg = error instanceof AresApiError ? error.message : "Failed to generate learning.";
      set((current) => ({
        actionWarning: current.snapshot
          ? { id: current.snapshot.step, kind: "blocked", title: "Learner reflection failed", message: msg, severity: "warning" }
          : current.actionWarning,
      }));
    } finally {
      set({ llmLearningBusy: false });
    }
  },
  refreshPolicyCapabilities: async () => {
    try {
      const policies = await client.listPolicies();
      const deployed = BUILD_DEPLOYED || policies.deployed === true;
      const defaults = policies.defaultCheckpoints ?? {};
      const currentAlgo = get().algorithmId;
      const autoPath =
        !deployed && usesCheckpoint(currentAlgo) && !get().checkpointPath.trim() && typeof defaults[currentAlgo] === "string"
          ? String(defaults[currentAlgo])
          : get().checkpointPath;
      set({
        deployedMode: deployed,
        defaultCheckpoints: defaults,
        checkpointPath: autoPath,
        policyActionMode: deployed ? "deterministic" : get().policyActionMode,
        llmDmCatalog: policies.llmDmCatalog ?? null,
        rllibAvailable: policies.capabilities.rllib,
        muzeroAvailable: policies.capabilities.muzero === true,
        worldModelsAvailable: policies.capabilities.world_models === true,
        llmDmAvailable: policies.capabilities.llm_dm === true,
        policyCapabilitiesLoaded: true,
      });
    } catch {
      try {
        const health = await client.fetchHealth();
        const deployed = BUILD_DEPLOYED || health.deployed === true;
        set({
          deployedMode: deployed,
          policyActionMode: deployed ? "deterministic" : get().policyActionMode,
          rllibAvailable: health.rllibAvailable,
          muzeroAvailable: health.muzeroAvailable,
          worldModelsAvailable: health.worldModelsAvailable,
          llmDmAvailable: health.llmDmAvailable,
          policyCapabilitiesLoaded: true,
        });
      } catch {
        set({ policyCapabilitiesLoaded: false });
      }
    }
  },
  attachCurrentPolicy: async () => {
    await policyDetach;
    const { algorithmId, checkpointPath, policyActionMode, snapshot, runMode, deployedMode, defaultCheckpoints, llmDmConfig, llmLearnings } = get();
    if (!snapshot || runMode !== "algorithm") return;
    const actionMode = supportsActionSampling(algorithmId)
      ? (deployedMode ? "deterministic" : policyActionMode)
      : undefined;
    const effectiveCheckpoint = usesCheckpoint(algorithmId)
      ? (deployedMode ? undefined : (checkpointPath.trim() || (typeof defaultCheckpoints[algorithmId] === "string" ? String(defaultCheckpoints[algorithmId]) : "")))
      : undefined;
    const llmPayload: LlmDmAttachPayload | undefined = algorithmId === "llm_dm"
      ? {
          ...llmDmConfig,
          actorApiKey: llmDmConfig.actorApiKey?.trim() || null,
          learnerApiKey: llmDmConfig.learnerApiKey?.trim() || null,
          carryLearnings: llmLearnings,
        }
      : undefined;
    const llmKeyPart = llmPayload
      ? `${llmPayload.actorProvider}:${llmPayload.actorModel}:${llmPayload.actorContextWindow}:${llmPayload.actorMaxOutputTokens}:${llmPayload.learnerEnabled}:${llmPayload.learnerProvider}:${llmPayload.learnerModel}:${llmPayload.learnerContextWindow}:${llmPayload.learnerMaxOutputTokens}:${llmPayload.historyMode}:${llmPayload.adviceLimit}:${llmPayload.maxEpisodeSteps}:${Boolean(llmPayload.actorApiKey)}:${Boolean(llmPayload.learnerApiKey)}`
      : "";
    const key = `${algorithmId}:${effectiveCheckpoint ?? ""}:${actionMode ?? ""}:${llmKeyPart}`;
    if (get().policyAttachKey === key) return;
    if (usesCheckpoint(algorithmId) && !deployedMode && !effectiveCheckpoint) {
      set({
        actionWarning: {
          id: snapshot.step,
          kind: "blocked",
          title: "Checkpoint required",
          message: "Enter an absolute path to checkpoint.json for this learned policy.",
          severity: "warning",
        },
      });
      return;
    }
    if (algorithmId === "llm_dm" && get().policyCapabilitiesLoaded && !get().llmDmAvailable) {
      set({ actionWarning: { id: snapshot.step, kind: "blocked", title: "LLM/DM unavailable",
        message: "Install the llm_dm optional dependency (jsonschema) on the backend.", severity: "warning" } });
      return;
    }
    await runBackend(
      () => client.attachPolicy({
        algorithmId,
        checkpointPath: effectiveCheckpoint || undefined,
        actionMode,
        llmDm: llmPayload,
      }),
      (attached) => {
        if (algorithmId === "llm_dm" && attached.llmDm) {
          const info = attached.llmDm;
          const sysMsg: LlmFeedMessage = {
            id: feedId("sys"),
            kind: "system",
            step: snapshot.step,
            author: "System",
            badge: "ONLINE",
            text: `Connected ${info.actorModel}${info.learnerConfigured && info.learnerModel ? ` + Learner ${info.learnerModel}` : " (Actor only)"}${info.learnings.length ? ` · ${info.learnings.length} learning${info.learnings.length === 1 ? "" : "s"} active` : ""}.`,
            timestamp: nowStamp(),
          };
          set((current) => ({
            policyAttachKey: key,
            actionWarning: null,
            llmLearnerConfigured: info.learnerConfigured,
            llmLearnings: info.learnings,
            llmFeed: [...current.llmFeed, sysMsg].slice(-160),
          }));
          return;
        }
        set({ policyAttachKey: key, actionWarning: null });
      },
      "Policy attach failed",
    );
  },
  setSelectedTool: (tool) => set({ selectedTool: tool }),
  setOverlayMode: (mode) => set({ overlayMode: mode }),
  selectTarget: (target) => set({ selectedTarget: target }),
  hoverTarget: (target) => set({ hoveredTarget: target }),
  highlightCell: (cell) => set({ highlightedCell: cell }),
  toggleHistoryOrder: () => set((state) => ({ historyNewestFirst: !state.historyNewestFirst })),
  setHistoryFilter: (action) => set({ historyFilter: action }),
  zoomIn: () => set((state) => ({ viewportZoomMode: "manual", viewportZoomScale: Math.min(3.25, Number((state.viewportZoomScale + 0.35).toFixed(2))) })),
  zoomOut: () => set((state) => ({ viewportZoomMode: "manual", viewportZoomScale: Math.max(0.75, Number((state.viewportZoomScale - 0.25).toFixed(2))) })),
  fitViewport: () => set({ viewportZoomMode: "fit", viewportZoomScale: 1, viewportCenter: null }),
  setViewportCenter: (center) => {
    const snapshot = get().snapshot;
    const maxX = snapshot?.terrainSize.width ?? 32;
    const maxY = snapshot?.terrainSize.height ?? 32;
    set({
      viewportZoomMode: "manual",
      viewportCenter: {
        x: Math.max(0.5, Math.min(maxX - 0.5, center.x)),
        y: Math.max(0.5, Math.min(maxY - 0.5, center.y)),
      },
    });
  },
  setCameraView: (cameraView) => set({ cameraView }),
  setRoverCameraYaw: (nextYaw) => set((state) => ({ roverCameraYaw: typeof nextYaw === "function" ? nextYaw(state.roverCameraYaw) : nextYaw })),
  toggleGrid: () => set((state) => ({ showGrid: !state.showGrid })),
  toggleRoverVisibility: () => set((state) => ({ showRoverVisibility: !state.showRoverVisibility })),
  };
});

function messagesFromLlmStep(step: number, meta: LlmDmStepMeta): LlmFeedMessage[] {
  const messages: LlmFeedMessage[] = [];
  const ts = nowStamp();
  if (meta.goal_event?.goal) {
    const g = meta.goal_event.goal;
    const coords = Array.isArray(g.position) ? ` (${g.position[0]}, ${g.position[1]})` : "";
    messages.push({
      id: feedId("goal"),
      kind: "goal",
      step,
      author: "Goal Planner",
      badge: g.kind.replace(/_/g, " ").toUpperCase(),
      text: `Targeted ${g.kind.replace(/_/g, " ")}${coords} · ${meta.goal_event.reason.replace(/_/g, " ")}`,
      timestamp: ts,
    });
  }
  const actorName = meta.model || meta.provider || "Actor";
  const actionLabel = (meta.action || "wait").replace(/_/g, " ");
  const confidence = typeof meta.actor_details?.confidence === "number" ? meta.actor_details.confidence : undefined;
  const probabilities = meta.actor_details?.probabilities;
  messages.push({
    id: feedId("act"),
    kind: "actor",
    step,
    author: actorName,
    badge: meta.fallback ? "FALLBACK" : "ACTOR",
    action: meta.action,
    text: meta.fallback ? `Fallback → ${actionLabel} (${meta.fallback})` : `Chose ${actionLabel}`,
    confidence,
    probabilities,
    latencySec: typeof meta.latency_seconds === "number" ? meta.latency_seconds : undefined,
    appliedAdvice: meta.applied_advice && meta.applied_advice.length ? meta.applied_advice : undefined,
    fallback: meta.fallback ?? null,
    timestamp: ts,
  });
  return messages;
}

function appendSeriesPoint(series: AnalyticsSeriesPoint[], snapshot: SimSnapshot): AnalyticsSeriesPoint[] {
  const point = seriesPointFromSnapshot(snapshot);
  const entry = snapshot.history[0];
  if (entry) {
    point.step = entry.step;
    point.actor = entry.actor;
    point.action = entry.action;
    point.valid = entry.action !== "invalid";
    point.reward = entry.reward;
    point.rewardTerms = entry.rewardTerms;
  }
  if (series.length && series[series.length - 1].step === point.step) {
    return [...series.slice(0, -1), point];
  }
  return [...series, point];
}

function warningFromSnapshot(snapshot: SimSnapshot) {
  if (snapshot.gameStatus === "game_over") {
    return { id: snapshot.step, kind: "terminal", title: "Exploration ended", message: snapshot.statusReason, severity: "danger" } satisfies SimWarning;
  }

  const entry = snapshot.history[0];
  if (!entry) return null;
  const warning = entry.events.find((event) => event.startsWith("Warning: "));
  if (warning) {
    const message = warning.replace("Warning: ", "");
    const isProgress = message.toLowerCase().includes("mission");
    return {
      id: entry.step,
      kind: isProgress ? "progress" : "terrain",
      title: isProgress ? "Mission progress" : "Terrain hazard",
      message,
      severity: isProgress ? "success" : "warning",
      relatedAction: entry.action,
      target: entry.target,
    } satisfies SimWarning;
  }
  const system = entry.events.find((event) => event.startsWith("System: ") && event !== "System: Build pad service needed");
  if (system) {
    return {
      id: entry.step,
      kind: "system",
      title: "System warning",
      message: system.replace("System: ", ""),
      severity: "warning",
      relatedAction: entry.action,
      target: entry.target,
    } satisfies SimWarning;
  }
  if (entry.action === "invalid") {
    return {
      id: entry.step,
      kind: "blocked",
      title: "Blocked action",
      message: entry.result,
      severity: "warning",
      relatedAction: entry.action,
      target: entry.target,
    } satisfies SimWarning;
  }
  return null;
}
