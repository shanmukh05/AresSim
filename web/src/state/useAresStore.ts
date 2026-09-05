/**
 * Zustand store: live snapshot, replay cursor, camera, and HUD interaction.
 * All mutations go through `AresApiClient`. Components should read from here
 * rather than calling the API directly.
 */

import { create } from "zustand";
import { AresApiClient, AresApiError, type ReplayResponse } from "../api/aresClient";
import { downloadTrajectoryFile } from "../lib/trajectoryFile";
import { payloadUsedKg } from "../lib/payload";
import type { ActionType, AresReplayProjectionV1, AlgorithmId, AnalyticsSeriesPoint, CameraView, LoadedReplay, OverlayMode, RunMode, SelectionTarget, SimAction, SimSnapshot, SimWarning, ViewportZoomMode } from "../types/sim";

const client = new AresApiClient();

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
      payloadCapacityKg: currentRover?.cargoCapacityKg ?? gameplay.initialSnapshot.rovers[0]?.cargoCapacityKg ?? 12,
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
  rllibAvailable: boolean;
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
  dispatchAction: (action: SimAction) => Promise<void>;
  moveRover: (dx: number, dy: number) => Promise<void>;
  clearActionWarning: () => void;
  pauseResume: () => Promise<void>;
  setRunMode: (mode: RunMode) => void;
  setAlgorithmId: (id: AlgorithmId) => void;
  setCheckpointPath: (path: string) => void;
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
  const runBackend = async <T>(operation: () => Promise<T>, apply: (value: T) => void, failureTitle = "Backend request failed") => {
    if (get().backendBusy) return;
    set({ backendBusy: true, backendError: null });
    try {
      const value = await operation();
      apply(value);
    } catch (error) {
      const message = error instanceof AresApiError ? error.message : "The backend request failed.";
      set((state) => ({
        backendError: message,
        actionWarning: state.snapshot
          ? { id: state.snapshot.step, kind: "blocked", title: failureTitle, message, severity: "warning" }
          : state.actionWarning,
      }));
    } finally {
      set({ backendBusy: false });
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
  algorithmId: "random",
  checkpointPath: "",
  rllibAvailable: false,
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
  start: async (seed) => runBackend(
    async () => {
      const snapshot = await client.startSession({ seed });
      await get().refreshPolicyCapabilities();
      return snapshot;
    },
    (snapshot) => set({
      snapshot,
      policyAttachKey: null,
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
  ),
  retryBackend: async () => get().start(1447),
  randomize: async () => runBackend(
    async () => {
      const snapshot = await client.randomizeSession();
      await get().refreshPolicyCapabilities();
      return snapshot;
    },
    (snapshot) => set({
      snapshot,
      policyAttachKey: null,
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
  ),
  saveRun: async (filename) => {
    const snapshot = get().snapshot;
    const runMode = get().runMode;
    if (!snapshot || runMode === "load") return;
    const safeName = filename.trim() || `aresim-seed-${snapshot.seed}-step-${snapshot.step}.json`;
    await runBackend(
      () => client.saveSession({
        fileName: safeName,
        runMode,
        algorithmId: runMode === "algorithm" ? get().algorithmId : undefined,
        checkpointPath: runMode === "algorithm" && get().algorithmId === "masked_ppo" ? get().checkpointPath : undefined,
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
  loadGameplay: async (content, fileName) => runBackend(
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
  ),
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
      await get().attachCurrentPolicy();
      await runBackend(
        () => client.agentStep(),
        (response) => set((current) => ({
          snapshot: response.snapshot,
          actionWarning: warningFromSnapshot(response.snapshot),
          analyticsSeries: appendSeriesPoint(current.analyticsSeries, response.snapshot),
        })),
        "Action failed",
      );
      return;
    }
    const action = { type: state.selectedTool } satisfies SimAction;
    await runBackend(
      () => client.dispatchAction(action),
      (snapshot) => set((current) => ({ snapshot, actionWarning: warningFromSnapshot(snapshot), analyticsSeries: appendSeriesPoint(current.analyticsSeries, snapshot) })),
      "Action failed",
    );
  },
  dispatchAction: async (action) => {
    const selected = get().selectedTarget;
    const target = action.target ?? (selected?.kind === "cell" ? { x: selected.x, y: selected.y } : undefined);
    await runBackend(
      () => client.dispatchAction({ ...action, target }),
      (snapshot) => set((current) => ({ selectedTool: action.type, snapshot, actionWarning: warningFromSnapshot(snapshot), analyticsSeries: appendSeriesPoint(current.analyticsSeries, snapshot) })),
      "Action failed",
    );
  },
  moveRover: async (dx, dy) => {
    const snapshot = get().snapshot;
    if (!snapshot) return;
    const rover = snapshot.rovers[0];
    const target = { x: Math.max(0, Math.min(snapshot.terrainSize.width - 1, rover.x + dx)), y: Math.max(0, Math.min(snapshot.terrainSize.height - 1, rover.y + dy)) };
    await runBackend(
      () => client.dispatchAction({ type: "move", target }),
      (nextSnapshot) => set((current) => ({
        selectedTool: "move",
        selectedTarget: { kind: "cell", ...target },
        snapshot: nextSnapshot,
        viewportCenter: current.viewportZoomMode === "manual" ? { x: target.x + 0.5, y: target.y + 0.5 } : null,
        actionWarning: warningFromSnapshot(nextSnapshot),
        analyticsSeries: appendSeriesPoint(current.analyticsSeries, nextSnapshot),
      })),
      "Movement failed",
    );
  },
  clearActionWarning: () => set({ actionWarning: null }),
  pauseResume: async () => {
    if (get().runMode === "load") {
      set((state) => ({ paused: !state.paused }));
      return;
    }
    const nextPaused = !get().paused;
    await runBackend(
      () => nextPaused ? client.pause() : client.resume(),
      (snapshot) => set({ paused: nextPaused, snapshot }),
      "Playback control failed",
    );
  },
  setSpeed: (speed) => set({ speed }),
  setRunMode: (mode) => {
    set({ runMode: mode });
    if (mode === "algorithm") {
      void get().refreshPolicyCapabilities();
    }
  },
  setAlgorithmId: (id) => set({ algorithmId: id, policyAttachKey: null }),
  setCheckpointPath: (path) => set({ checkpointPath: path, policyAttachKey: null }),
  refreshPolicyCapabilities: async () => {
    try {
      const policies = await client.listPolicies();
      set({ rllibAvailable: policies.capabilities.rllib, policyCapabilitiesLoaded: true });
    } catch {
      try {
        const health = await client.fetchHealth();
        set({ rllibAvailable: health.rllibAvailable, policyCapabilitiesLoaded: true });
      } catch {
        set({ policyCapabilitiesLoaded: false });
      }
    }
  },
  attachCurrentPolicy: async () => {
    const { algorithmId, checkpointPath, snapshot, runMode } = get();
    if (!snapshot || runMode !== "algorithm") return;
    const key = `${algorithmId}:${algorithmId === "masked_ppo" ? checkpointPath.trim() : ""}`;
    if (get().policyAttachKey === key) return;
    if (algorithmId === "masked_ppo" && !checkpointPath.trim()) {
      set({
        actionWarning: {
          id: snapshot.step,
          kind: "blocked",
          title: "Checkpoint required",
          message: "Enter an absolute path to checkpoint.json for Masked PPO.",
          severity: "warning",
        },
      });
      return;
    }
    await runBackend(
      () => client.attachPolicy({
        algorithmId,
        checkpointPath: algorithmId === "masked_ppo" ? checkpointPath.trim() : undefined,
      }),
      () => set({ policyAttachKey: key, actionWarning: null }),
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
