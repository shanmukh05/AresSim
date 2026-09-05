/** Unit tests for the AresSim Zustand store. */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AresTrajectoryEpisodeV1 } from "../types/sim";
import * as trajectoryFile from "../lib/trajectoryFile";
import { useAresStore } from "./useAresStore";

describe("useAresStore", () => {
  beforeEach(async () => {
    useAresStore.getState().setRunMode("manual");
    await useAresStore.getState().start(1447);
  });

  afterEach(() => vi.restoreAllMocks());

  it("updates selected tool, hover, pinned inspector, overlay, and speed", () => {
    useAresStore.getState().setSelectedTool("scan");
    useAresStore.getState().hoverTarget({ kind: "cell", x: 2, y: 3 });
    useAresStore.getState().selectTarget({ kind: "cell", x: 4, y: 5 });
    useAresStore.getState().setOverlayMode("ice");
    useAresStore.getState().setSpeed(4);

    const state = useAresStore.getState();
    expect(state.selectedTool).toBe("scan");
    expect(state.hoveredTarget).toEqual({ kind: "cell", x: 2, y: 3 });
    expect(state.selectedTarget).toEqual({ kind: "cell", x: 4, y: 5 });
    expect(state.overlayMode).toBe("ice");
    expect(state.speed).toBe(4);
  });

  it("moves the rover with directional controls and updates zoom state", async () => {
    const start = useAresStore.getState().snapshot!.rovers[0];

    await useAresStore.getState().moveRover(1, 0);
    useAresStore.getState().zoomIn();
    useAresStore.getState().zoomOut();
    useAresStore.getState().setViewportCenter({ x: 10, y: 12 });
    useAresStore.getState().toggleGrid();
    useAresStore.getState().toggleRoverVisibility();
    useAresStore.getState().fitViewport();
    useAresStore.getState().setHistoryFilter("move");

    const state = useAresStore.getState();
    expect(state.snapshot?.rovers[0].x).toBe(start.x + 1);
    expect(state.snapshot?.history[0].action).toBe("move");
    expect(state.viewportZoomMode).toBe("fit");
    expect(state.viewportZoomScale).toBe(1);
    expect(state.viewportCenter).toBeNull();
    expect(state.showGrid).toBe(true);
    expect(state.showRoverVisibility).toBe(true);
    expect(state.historyFilter).toBe("move");
  });

  it("exports the current run with a filename", async () => {
    const download = vi.spyOn(trajectoryFile, "downloadTrajectoryFile").mockImplementation(() => {});
    await useAresStore.getState().dispatchAction({ type: "wait" });

    await useAresStore.getState().saveRun("test-run");

    const state = useAresStore.getState();
    const saved = download.mock.calls.at(-1)?.[1] as AresTrajectoryEpisodeV1;
    expect(state.lastSavedAt).toBeTruthy();
    expect(state.actionWarning?.title).toBe("Trajectory exported");
    expect(state.actionWarning?.message).toContain("test-run");
    expect(download).toHaveBeenCalled();
    expect(saved.schemaVersion).toBe("aresim.trajectory.episode.v1");
    expect(saved.replay.initialSnapshot.seed).toBe(state.snapshot?.seed);
    expect(saved.replay.finalSnapshot.step).toBe(state.snapshot?.step);
    expect(saved.replay.steps.length).toBeGreaterThan(0);
    expect(saved.replay.checkpoints[0].reason).toBe("initial");
  });

  it("falls back to a default save name when none is provided", async () => {
    const download = vi.spyOn(trajectoryFile, "downloadTrajectoryFile").mockImplementation(() => {});

    await useAresStore.getState().saveRun("");

    const state = useAresStore.getState();
    expect(state.actionWarning?.title).toBe("Trajectory exported");
    expect(download.mock.calls.at(-1)?.[0]).toContain(String(state.snapshot?.seed));
  });

  it("persists manual camera center until fit view is restored", () => {
    useAresStore.getState().zoomIn();
    useAresStore.getState().setViewportCenter({ x: 40, y: -4 });

    let state = useAresStore.getState();
    expect(state.viewportZoomMode).toBe("manual");
    expect(state.viewportZoomScale).toBeGreaterThan(1);
    expect(state.viewportCenter).toEqual({ x: 31.5, y: 0.5 });

    useAresStore.getState().fitViewport();

    state = useAresStore.getState();
    expect(state.viewportZoomMode).toBe("fit");
    expect(state.viewportCenter).toBeNull();
  });

  it("keeps manual zoom across actions, policy steps, and speed changes", async () => {
    useAresStore.getState().zoomIn();
    const zoomScale = useAresStore.getState().viewportZoomScale;

    await useAresStore.getState().dispatchAction({ type: "wait" });
    useAresStore.getState().setRunMode("algorithm");
    useAresStore.getState().setSpeed(3);
    await useAresStore.getState().step();

    expect(useAresStore.getState().viewportZoomMode).toBe("manual");
    expect(useAresStore.getState().viewportZoomScale).toBe(zoomScale);
  });

  it("randomize updates snapshot and restarts history", async () => {
    const previousSeed = useAresStore.getState().snapshot?.seed;
    await useAresStore.getState().dispatchAction({ type: "scan" });
    expect(useAresStore.getState().snapshot?.history.length).toBeGreaterThan(1);

    await useAresStore.getState().randomize();

    expect(useAresStore.getState().snapshot?.seed).not.toBe(previousSeed);
    expect(useAresStore.getState().snapshot?.history).toHaveLength(1);
  });

  it("steps algorithm mode with the selected baseline policy", async () => {
    useAresStore.getState().setRunMode("algorithm");
    useAresStore.getState().setAlgorithmId("wait");
    await useAresStore.getState().attachCurrentPolicy();
    await useAresStore.getState().step();

    const entry = useAresStore.getState().snapshot?.history[0];
    expect(entry?.actor).toBe("Agent");
    expect(entry?.action).toBe("wait");
  });

  it("loads legacy LLM gameplay metadata as a replay", async () => {
    const download = vi.spyOn(trajectoryFile, "downloadTrajectoryFile").mockImplementation(() => {});
    await useAresStore.getState().dispatchAction({ type: "wait" });
    await useAresStore.getState().saveRun("legacy-llm");
    const legacy = structuredClone(download.mock.calls.at(-1)?.[1] as AresTrajectoryEpisodeV1);
    legacy.replay.metadata.runMode = "llm";
    legacy.replay.metadata.llmAgentId = "mock_llm";

    await useAresStore.getState().loadGameplay(JSON.stringify(legacy), "legacy-llm.json");

    expect(useAresStore.getState().runMode).toBe("load");
    expect(useAresStore.getState().loadedGameplay?.metadata.runMode).toBe("llm");
    expect(useAresStore.getState().loadedGameplay?.metadata.llmAgentId).toBe("mock_llm");
  });

  it("loads and resets gameplay replay state", async () => {
    await useAresStore.getState().dispatchAction({ type: "wait" });
    const snapshot = useAresStore.getState().snapshot!;

    await useAresStore.getState().loadGameplay(JSON.stringify({ savedAt: "now", snapshot }), "run.json");

    expect(useAresStore.getState().runMode).toBe("load");
    expect(useAresStore.getState().loadedReplay?.fileName).toBe("run.json");
    expect(useAresStore.getState().snapshot?.mode).toBe("Replay");

    await useAresStore.getState().step();
    expect(useAresStore.getState().loadedReplay?.cursor).toBe(snapshot.step);

    await useAresStore.getState().resetReplay();
    expect(useAresStore.getState().loadedReplay?.cursor).toBe(snapshot.step);
  });

  it("jumps to loaded gameplay checkpoints", async () => {
    const download = vi.spyOn(trajectoryFile, "downloadTrajectoryFile").mockImplementation(() => {});
    for (let index = 0; index < 10; index += 1) {
      await useAresStore.getState().dispatchAction({ type: "wait" });
    }
    await useAresStore.getState().saveRun("checkpoint-run");
    const saveContent = JSON.stringify(download.mock.calls.at(-1)?.[1]);
    expect(saveContent).toBeTruthy();

    await useAresStore.getState().loadGameplay(saveContent, "checkpoint-run.json");
    const finalCheckpoint = useAresStore.getState().loadedGameplay?.checkpoints.find((checkpoint) => checkpoint.reason === "final");
    expect(finalCheckpoint).toBeTruthy();

    await useAresStore.getState().jumpToReplayCheckpoint(finalCheckpoint!.id);

    expect(useAresStore.getState().loadedReplay?.activeCheckpointId).toBe(finalCheckpoint!.id);
    expect(useAresStore.getState().loadedReplay?.cursor).toBe(finalCheckpoint!.step);
    expect(useAresStore.getState().snapshot?.step).toBe(finalCheckpoint!.step);
  });

  it("rejects invalid gameplay JSON without replacing the current snapshot", async () => {
    const seed = useAresStore.getState().snapshot?.seed;

    await useAresStore.getState().loadGameplay("{ nope", "bad.json");

    expect(useAresStore.getState().snapshot?.seed).toBe(seed);
    expect(useAresStore.getState().actionWarning?.title).toBe("Load failed");
  });

  it("surfaces and clears invalid action warnings", async () => {
    await useAresStore.getState().dispatchAction({ type: "build", target: { x: 0, y: 0 } });

    expect(useAresStore.getState().actionWarning?.message).toMatch(/blocked/i);

    useAresStore.getState().clearActionWarning();

    expect(useAresStore.getState().actionWarning).toBeNull();
  });

  it("records analytics series points as the live session steps", async () => {
    await useAresStore.getState().start(512);
    const initial = useAresStore.getState().analyticsSeries;

    expect(initial).toHaveLength(1);
    expect(initial[0].step).toBe(0);

    await useAresStore.getState().dispatchAction({ type: "scan" });
    await useAresStore.getState().dispatchAction({ type: "wait" });

    const stepped = useAresStore.getState().analyticsSeries;
    expect(stepped).toHaveLength(3);
    expect(stepped[1].action).toBe("scan");
    expect(stepped[2].action).toBe("wait");
    expect(stepped[2].cumulativeReward).toBe(useAresStore.getState().snapshot?.objectiveStats.rewardTotals.total);
  });

  it("rebuilds analytics series from loaded gameplay steps", async () => {
    const baseline = useAresStore.getState().snapshot!;
    const file = JSON.stringify({
      savedAt: "now",
      snapshot: baseline,
    });

    await useAresStore.getState().loadGameplay(file, "an-run.json");

    const series = useAresStore.getState().analyticsSeries;
    expect(useAresStore.getState().runMode).toBe("load");
    expect(series.length).toBe(useAresStore.getState().loadedGameplay?.steps.length);
    if (series.length > 0) {
      expect(series[0].step).toBe(1);
    }
  });

  it("serializes rapid actions and retains the last snapshot on network failure", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const first = useAresStore.getState().dispatchAction({ type: "wait" });
    const duplicate = useAresStore.getState().dispatchAction({ type: "wait" });
    await Promise.all([first, duplicate]);
    const actionCalls = fetchSpy.mock.calls.filter(([input]) => String(input).includes("/actions"));
    expect(actionCalls).toHaveLength(1);

    const snapshot = useAresStore.getState().snapshot;
    fetchSpy.mockRejectedValueOnce(new TypeError("offline"));
    await useAresStore.getState().dispatchAction({ type: "wait" });
    expect(useAresStore.getState().snapshot).toEqual(snapshot);
    expect(useAresStore.getState().backendError).toMatch(/unavailable/i);
    expect(useAresStore.getState().actionWarning?.title).toBe("Action failed");
  });
});
