# AresSim Trajectory Episode and Replay Projection

Last updated: 2026-08-02

Status: Canonical implemented trajectory/replay contract. New UI and rollout exports use `aresim.trajectory.episode.v1` with an `aresim.trajectory.replay.v1` projection. Python creates, validates, normalizes, and reconstructs these files; the browser downloads them and presents backend-returned replay snapshots. Standalone `aresim.gameplay.v1` is import-only compatibility.

## 1. Purpose

A trajectory episode must contain enough information to analyze the policy trace and replay, inspect, and reconstruct the complete AresSim run without relying on hidden browser state.

The file is used for:

- Loading a run in the `web/` UI.
- Jumping directly to important replay checkpoints.
- Inspecting every agent/player decision.
- Debugging rewards, resource changes, terrain mutation, and rover path.
- Backend-owned save and replay reconstruction.
- Future RL dataset export.

`TrajectoryWriter` creates one `aresim.trajectory.episode.v1` file per recorded RL episode under `<dataset>/episodes/`. The top-level `policy` object retains local observations, masks, selected action IDs, selected and engine rewards, flags, events, schemas, spaces, seeds, and state checksums. The nested `replay` object retains full initial/final snapshots and every authoritative step delta. It uses only endpoint checkpoints for rollout exports to avoid duplicating large world snapshots; any intermediate step is still reconstructed exactly from the deltas. UI exports use the same envelope with `policy: null` because UI sessions do not currently expose a framework-neutral policy trace.

## 2. Schemas

Public standalone schema:

```ts
interface AresTrajectoryEpisodeV1 {
  schemaVersion: "aresim.trajectory.episode.v1";
  savedAt: string;
  metadata: {
    episodeId: string;
    source: "ui" | "rollout";
    environmentSeed: number;
    policyId: string | null;
    agentSeed: number | null;
  };
  policy: Record<string, unknown> | null;
  replay: AresTrajectoryReplayV1;
}
```

The replay projection is:

```ts
interface AresTrajectoryReplayV1 {
  schemaVersion: "aresim.trajectory.replay.v1";
  savedAt: string;
  fileName: string;
  appVersion: string;
  metadata: {
    sessionId: string;
    seed: number;
    runMode: "manual" | "algorithm" | "load" | "llm";
    algorithmId?: string; // UI selector ID or versioned RL policy ID
    // Deprecated read-only compatibility field for older saves.
    llmAgentId?: "mock_llm";
    totalSteps: number;
    finalStatus: "running" | "paused" | "victory" | "game_over";
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
```

The active runtime modes are Manual, Algorithm, and Replay (`load`). The `"llm"` value and `llmAgentId` remain accepted only so older `aresim.gameplay.v1` files can be replayed; there is no active LLM UI/store mode, and new exports never emit either legacy value. Normal UI exports originate only from live Manual or Algorithm runs.

The standalone schema is intentionally verbose. Dataset JSONL shards remain the compact and optionally gzip-compressed storage path; standalone episodes prioritize correctness, debuggability, and direct UI replay.

## 3. Snapshot Strategy

The save file stores:

- `initialSnapshot`: full environment state at step 0.
- `steps`: chronological per-timestep deltas from step 1 onward.
- `checkpoints`: full environment snapshots at automatic replay anchors.
- `finalSnapshot`: full latest environment state.

This gives two replay paths:

- Full reconstruction: start from `initialSnapshot` and apply `steps[]` in order.
- Direct jump: load a checkpoint snapshot immediately without replaying earlier deltas.

Checkpoint snapshots are deliberately stored because Load mode needs fast random access for analysis.

## 4. Step Delta Format

Each `GameplayStepDelta` stores one timestep:

```ts
interface GameplayStepDelta {
  step: number;
  sol: number;
  localTime: string;
  actor: "Player" | "Agent" | "System";
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
```

The active selectable action union is `move | scan | extract | build | service | unload | wait`. `invalid` and `event` remain recorded outcome/history categories. Scan and Extract are separate collection transitions; Unload is the only delivery transition.

Delta rules:

- Every consumed environment decision should produce a step record.
- Terrain deltas include changed cells only.
- Rover and structure deltas include changed entities only.
- Resource/objective/build-pad/rule snapshots are included only when changed.
- `appendedHistoryEntry` is always included because history is the audit trail.
- Reward terms remain structured so future RL debugging can separate progress, energy, safety, invalid-action penalties, and total reward.
- A successful extraction delta contains the changed cell with `terrain: "regolith"` and `extracted: true`; replay must replace the previous ice cell rather than layering or revealing a foundation tile.
- `ActionType` includes the player-selectable `unload` action. Its rover delta clears `cargoIce`, `cargoOre`, and `cargoSamples`; its resource delta records negative delivered cargo plus any resulting water and oxygen; and its objective delta updates delivered counters and `unloadCount`.
- A successful Scan includes `resourceDelta.samples: 0.5` and a rover delta with the updated sample payload. A successful Extract similarly records `resourceDelta.ice: 2`.
- Rover snapshots persist `cargoSamples` and `cargoCapacityKg` in addition to ice and ore cargo.

## 5. Checkpoint Generation

Checkpoints are automatic, not user-authored.

Required checkpoint types:

- Step 0: `Initial`.
- Every 10 steps: interval checkpoint labeled `Step <n>`.
- Significant event steps: event checkpoint.
- Final step: `Final`.

Manual/UI trajectories use all checkpoint types above. Rollout trajectories use only Initial and Final checkpoints; their complete `steps[]` timeline remains authoritative and supports exact reconstruction of every intermediate state.

Significant event checkpoints include:

- Invalid or blocked action.
- Terrain hazard.
- Extract success.
- Unload success.
- Habitat build progress change.
- Service success.
- Low battery or system warning.
- Game over.

Checkpoint shape:

```ts
interface GameplayCheckpoint {
  id: string;
  step: number;
  label: string;
  reason: "initial" | "interval" | "event" | "final";
  summary: string;
  target?: { x: number; y: number };
  reward?: number;
  snapshot: SimSnapshot;
}
```

If multiple checkpoint reasons occur at the same step, each reason may create a separate checkpoint item. For example, step 10 can be both an interval checkpoint and an event checkpoint.

## 6. Save Behavior

`Export trajectory` creates an `aresim.trajectory.episode.v1` JSON file and downloads it.

Default filename:

```text
aresim-seed-<seed>-step-<step>.json
```

The Save dialog also offers a short `aresim-<session-prefix>` placeholder. If the store is invoked without a user-supplied name, the seed/step filename above is the programmatic fallback. The download helper appends `.json` when the chosen name does not already include it.

The UI also keeps a local browser copy under:

```text
aresim-save-<name>
```

This browser copy is a convenience for Phase 1 tests and local iteration. The downloaded JSON file is the actual portable replay artifact.

Trajectory replay projections intentionally exclude transient presentation state:

- camera viewpoint (3D Survey/Top/Rover), camera zoom, rover-follow target, fit/manual view, 3D rotation angle, Rover POV look yaw, and field of view;
- active analytical layer, cell-boundary visibility, and rover-visibility flashlight preview;
- audio mute;
- open drawer/modal, hover, selection, and Inspector state;
- temporary playback UI state.

These settings do not affect environment reconstruction. Loading a trajectory starts in the default fitted 3D Survey view with no pinned target and keeps its replay projection as the state authority.

The `8 x 8` rover observation is derived deterministically from each restored snapshot and rover position, so a UI-authored trajectory does not invent policy tensors. The active observation uses `self[10]`, categorical `pad_proximity`, and no occupancy/entity tensors; its policy action is the flat masked `aresim.action.rover.v1`. Rollout trajectories record those schema identifiers plus the actual transition-aligned observation/mask arrays. An external `max_episode_steps` cutoff belongs in trajectory experiment metadata and produces truncation, not replayed mission failure. A replay projection remains privileged reconstruction data and must not be passed directly to a policy.

Gymnasium, PettingZoo, and RLlib adapters alter call shape, not simulator or replay semantics. A future multi-rover replay requires a schema revision recording one joint world tick with commands, rewards, events, and stable agent IDs. Do not encode simultaneous actions as an arbitrary sequence of Phase 1 deltas. Training and evaluation workflows follow [RL Algorithms, Training, and Evaluation](../rl/rl_quickstart.md); trajectory schemas follow the [RL Usage Guide](../rl/usage.md).

## 7. Load Behavior

Replay mode accepts:

- New `aresim.trajectory.episode.v1` files.
- Legacy standalone `aresim.gameplay.v1` replay projections.
- Legacy `{ savedAt, snapshot }` payloads.
- Raw `SimSnapshot` JSON payloads.

New gameplay files load with:

- Embedded seed.
- Initial snapshot.
- Per-step deltas.
- Checkpoints.
- Final snapshot.
- Replay cursor metadata.

Legacy loads are upgraded in backend memory:

- A gameplay wrapper is created.
- Initial and final checkpoints are generated where possible.
- A single legacy snapshot is not treated as enough evidence to synthesize transition-aligned deltas.
- If legacy data has only one snapshot, direct checkpoint inspection is available at that snapshot's original step, but no invented intermediate replay is exposed.

## 8. Load Mode Timeline

The reserved footer keeps the same three-zone structure in Replay:

- Left, Mode & Source: Replay mode, upload/replace icon, and loaded filename.
- Center, Replay Controls: play/pause, step, repeat, speed, cursor, and timeline scrubber.
- Right, World Setup: the labeled zone remains visible for spatial consistency but contains no seed, apply, randomize, or other environment mutation controls.

Before a file is loaded, replay transport remains visible but disabled. After a gameplay file is loaded, the center controls provide:

- Replay Pause/Resume.
- Step replay.
- Repeat replay.
- Speed slider.
- Step scrubber (continuous slider from step 0 to final step).

Step scrubber:

- A continuous slider from step 0 to `integrity.finalStep` that lets the user choose any saved step.
- The current cursor position and total steps are shown as a numeric readout.
- When its value changes, the store loads the nearest checkpoint at or before the target, then applies saved timestep deltas forward to reach the exact step.
- Checkpoints are reconstruction anchors in the file; the current compact footer does not draw checkpoint markers or a separate checkpoint picker on the slider.

Scrubber behavior:

- Set the visible snapshot to the target step's reconstructed state.
- Set replay cursor to the target step.
- Highlight the target cell when present.
- Pin/select the matching history row when available.
- Keep the run in Load mode.

Replay uses the same synthesized action-feedback path as live play. Advancing to a saved action can therefore trigger its action sound/animation, subject to the UI-only mute setting; these effects do not modify replay state.

Stepping after a scrub continues from the cursor by applying the next saved timestep delta.

Repeat replay returns to the Initial checkpoint. Canonical runs normally begin at step 0; a raw legacy snapshot keeps its original step as its initial cursor.

## 9. Reconstruction Rules

Direct checkpoint reconstruction:

```text
currentSnapshot = checkpoint.snapshot
cursor = checkpoint.step
```

Step replay reconstruction:

```text
currentSnapshot = applyStepDelta(currentSnapshot, steps[cursor + 1])
cursor = cursor + 1
```

The grid, action rules, rewards, and terrain validity are not re-simulated during Load mode. The uploaded gameplay is treated as the replay authority.

View behavior during replay follows the normal renderer contract: action/step changes do not reset an active zoom, while loading a new file initializes Fit view. Manual commands and environment randomization remain unavailable.

## 10. Engine API

The Python backend actively supports these concepts:

- Save/export gameplay by session id.
- Return full initial/final snapshots.
- Return ordered step deltas.
- Generate automatic checkpoints.
- Load gameplay for replay.
- Jump directly to checkpoints.
- Continue replay stepping from any checkpoint.

Active endpoints:

- `POST /api/sessions/{id}/save`
- `POST /api/replays`
- `POST /api/replays/{id}/jump`
- `POST /api/replays/{id}/step`
- `POST /api/replays/{id}/reset`

The backend is the authority for generated save files and replay reconstruction. It keeps only one replay in memory; the downloaded JSON is the durable portable artifact.

## 10.1 Analytics Consumers

The nested replay `steps[]` array in a trajectory episode is the source for the Run Analytics modal when a run is loaded:

- `reward`, `rewardTerms`, and `after.totalReward` feed all Rewards-tab plots.
- `resourceDelta`, rover payload deltas, and `changes.resources` feed Resources-tab plots, including the ice/sample/capacity chart.
- `actor`, `action`, `target`, and `after.rover` feed Behavior-tab plots and the rover path scatter.
- `changes.status.weather` and `changes.status.dustIntensity` feed Environment-tab plots.
- `changes.objectiveStats` drives Progress-tab cumulative charts (habitat build, ice collected, payload delivered, scan count, service count).

On load, the store rebuilds an unbounded `analyticsSeries: AnalyticsSeriesPoint[]` once from `loadedGameplay.steps[]`. The same series shape is recorded live during Manual and Algorithm sessions, so live and loaded paths render identical charts via the `useAnalyticsData()` hook.

Legacy single-snapshot uploads cannot fully reconstruct the series, so the Analytics modal shows a "Limited replay data" banner and only renders plots supported by the available fields.

## 11. Testing Expectations

Tests should verify:

- Saved files include initial snapshot, steps, checkpoints, final snapshot, and integrity metadata.
- Checkpoints are generated for initial, interval, event, and final reasons.
- Load mode accepts new and legacy payloads.
- Jumping to a checkpoint restores the exact checkpoint snapshot.
- Step replay after a checkpoint advances from the checkpoint cursor.
- Final checkpoint restores final environment state.
- Timeline controls remain inside the reserved 60 px footer without horizontal zone overflow at desktop and tablet widths.
- Replay exposes no seed/randomization or manual-action controls, and unloaded transport is disabled.
- New exports contain no LLM mode/agent metadata, while legacy LLM metadata still loads into Replay.
- New exports preserve the explicit Scan/Extract/Unload transitions. Merely arriving at a build-pad cell must never be reconstructed as an implicit delivery.
- Replays created before payload fields existed hydrate `cargoSamples` to `0`, `cargoCapacityKg` to `12`, delivered counters and `unloadCount` to `0`, and the delivery reward total to `0`.
- Camera, layers, cell boundaries, audio mute, drawers, and selections are not required for deterministic reconstruction.
