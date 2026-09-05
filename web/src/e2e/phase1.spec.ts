/** Playwright checks that the Phase 1 shell renders and the canvas is not blank. */

import { expect, test } from "@playwright/test";
import initialSnapshot from "../test/fixtures/snapshot-246.json";
import type { SimSnapshot } from "../types/sim";

test("loads the simulator shell and renders a nonblank canvas", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("ARESIM")).toBeVisible();
  await expect(page.getByTestId("header-power")).toBeVisible();
  await expect(page.getByTestId("header-battery")).toBeVisible();
  await expect(page.getByTestId("header-health")).toBeVisible();
  await expect(page.getByTestId("header-habitat")).toBeVisible();
  await expect(page.getByTestId("header-livability")).toBeVisible();
  await expect(page.getByTestId("ambient-time-sky")).toHaveAttribute("data-local-time", "08:00");
  await expect(page.getByTestId("ambient-time-sky")).toHaveAttribute("data-time-phase", "day");
  await expect(page.getByTestId("ambient-time-sky")).toHaveAttribute("data-night-objects", "0.000");
  await expect(page.getByTestId("ambient-time-sky")).toHaveAttribute("aria-hidden", "true");
  await expect(page.locator("[data-celestial-object]")).toHaveCount(6);
  await expect(page.getByTestId("mission-hud")).toBeVisible();
  await expect(page.getByTestId("alert-hud")).toBeVisible();
  await expect(page.getByText("Colony Overview")).toHaveCount(0);
  await expect(page.getByTestId("mission-drawer")).toHaveCount(0);
  await expect(page.getByText("Mission Tasks")).toHaveCount(0);
  await expect(page.locator("header").getByText("Dust", { exact: true })).toHaveCount(0);
  await expect(page.getByLabel("Open agent history")).toBeVisible();
  await expect(page.getByText("Agent Step History")).toHaveCount(0);
  const canvas = page.locator("canvas").first();
  await expect(canvas).toBeVisible();
  await expect.poll(async () => (await canvas.boundingBox())?.height ?? 0).toBeGreaterThanOrEqual(300);
  const box = await canvas.boundingBox();
  expect(box?.width).toBeGreaterThanOrEqual(300);
  expect(box?.height).toBeGreaterThanOrEqual(300);
  const viewportBox = await page.getByTestId("game-viewport").boundingBox();
  const footerBox = await page.getByTestId("action-bar").boundingBox();
  const controlsBox = await page.getByTestId("action-bar-controls").boundingBox();
  expect(footerBox?.height).toBeLessThanOrEqual(80);
  expect(controlsBox?.width).toBeLessThanOrEqual(1180);
  expect(footerBox!.y).toBeGreaterThanOrEqual(viewportBox!.y + viewportBox!.height - 1);
});

test("history row can highlight its target", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Scan").click();
  await page.getByLabel("Open agent history").click();
  await page.getByTestId("history-row-1").click();
  await page.getByLabel("Close drawer").click();
  await page.getByTestId("selection-hud").click();
  await expect(page.getByText("Step 1: scan")).toBeVisible();
});

test("randomize visibly changes the session seed", async ({ page }) => {
  await page.goto("/");
  const seedInput = page.getByLabel("World seed");
  const seedBefore = await seedInput.inputValue();
  await page.getByLabel("Randomize world").click();
  await expect.poll(async () => seedInput.inputValue()).not.toBe(seedBefore);
  await expect(page.getByTestId("game-viewport")).toHaveAttribute("data-camera-center", "16.00,16.00");
  await expect(page.getByTestId("game-viewport")).toHaveAttribute("data-visible-rect", "0.00,0.00,32.00,32.00");

  await seedInput.fill("24680");
  await page.getByLabel("Apply seed").click();
  await expect.poll(async () => seedInput.inputValue()).toBe("24680");
});

test("keyboard movement and zoom controls are wired", async ({ page }, testInfo) => {
  await page.goto("/");
  await expect(page.getByTestId("mission-hud")).toBeVisible();
  await expect(page.getByTestId("game-viewport")).toHaveAttribute("data-camera-view", "survey");
  await expect(page.getByTestId("game-viewport")).toHaveAttribute("data-survey-projection", "angled-orthographic-3d");
  await page.getByLabel("Top view").click();
  await expect(page.getByTestId("game-viewport")).toHaveAttribute("data-top-projection", "window-aligned-square");
  await expect(page.getByLabel("World angle")).toHaveCount(0);
  await page.getByLabel("3D survey view").click();
  await expect(page.getByLabel("World angle")).toBeVisible();
  const fitRectBeforeMove = await page.getByTestId("game-viewport").getAttribute("data-visible-rect");
  const fitCenterBeforeMove = await page.getByTestId("game-viewport").getAttribute("data-camera-center");
  await page.keyboard.press("ArrowRight");
  await page.getByLabel("Open agent history").click();
  await expect(page.getByTestId("history-row-1")).toContainText("move");
  await page.getByLabel("Close drawer").click();
  await expect.poll(async () => page.getByTestId("game-viewport").getAttribute("data-visible-rect")).toBe(fitRectBeforeMove);
  await expect(page.getByTestId("game-viewport")).toHaveAttribute("data-camera-center", fitCenterBeforeMove ?? "16.00,16.00");

  await page.getByLabel("Zoom in").click();
  await expect(page.getByTestId("zoom-readout")).not.toContainText("Fit");
  await expect(page.getByTestId("mini-view")).toBeVisible();
  const centerBeforeMapPan = await page.getByTestId("game-viewport").getAttribute("data-camera-center");
  await page.getByTestId("mini-view").click({ position: { x: 20, y: 20 } });
  await expect.poll(async () => page.getByTestId("game-viewport").getAttribute("data-camera-center")).not.toBe(centerBeforeMapPan);
  const zoomBeforeAction = await page.getByTestId("zoom-readout").textContent();
  await page.getByLabel("Scan").click();
  await expect(page.getByTestId("zoom-readout")).toHaveText(zoomBeforeAction ?? "135% zoom");
  const centerBeforeMove = await page.getByTestId("game-viewport").getAttribute("data-camera-center");
  await page.keyboard.press("ArrowRight");
  await expect(page.getByTestId("zoom-readout")).toHaveText(zoomBeforeAction ?? "135% zoom");
  await expect.poll(async () => page.getByTestId("game-viewport").getAttribute("data-camera-center")).not.toBe(centerBeforeMove);
  await expect(page.getByLabel("Rover navigation overview")).toBeVisible();
  await page.getByLabel("World angle").fill("137");
  await expect(page.getByTestId("game-viewport")).toHaveAttribute("data-camera-rotation", "137");
  await page.getByLabel("Fit environment").click();
  await expect(page.getByTestId("zoom-readout")).toHaveCount(0);
  if (testInfo.project.name === "desktop") {
    await page.getByTestId("game-viewport").hover();
    await page.mouse.wheel(0, -260);
    await expect(page.getByTestId("zoom-readout")).toContainText("135% zoom");
  }
  await expect(page.getByLabel("Show cell boundaries")).toBeVisible();
});

test("Rover POV keeps movement level, faces travel direction, and supports wheel FOV zoom", async ({ page }, testInfo) => {
  await page.goto("/");
  const viewport = page.getByTestId("game-viewport");
  await expect(viewport).toHaveAttribute("data-camera-view", "survey");

  await page.getByLabel("Rover point of view").click();
  await expect(viewport).toHaveAttribute("data-camera-view", "rover");
  await expect(viewport).toHaveAttribute("data-rover-motion", "level-smoothed");
  await expect(page.getByTestId("rover-pov-readout")).toContainText("62° field of view");
  await expect(page.getByLabel("World angle")).toHaveCount(0);

  if (testInfo.project.name === "desktop") {
    await viewport.hover();
    await page.mouse.wheel(0, -260);
    await expect(viewport).toHaveAttribute("data-rover-fov", "58");
    await expect(page.getByTestId("rover-pov-readout")).toContainText("58° field of view");
  }

  const headingBefore = await viewport.getAttribute("data-rover-heading");
  await page.keyboard.press("ArrowRight");
  await expect.poll(async () => viewport.getAttribute("data-rover-heading")).not.toBe(headingBefore);
  const movementHeading = await viewport.getAttribute("data-rover-heading");
  await page.getByLabel("Scan").click();
  await expect(viewport).toHaveAttribute("data-rover-heading", movementHeading ?? "1.00,-0.00");
  if (testInfo.project.name === "desktop") await expect(viewport).toHaveAttribute("data-rover-fov", "58");

  const centerAfterRightTurnValue = (await viewport.getAttribute("data-camera-center"))!;
  const centerAfterRightTurn = centerAfterRightTurnValue.split(",").map(Number);
  await page.keyboard.press("ArrowUp");
  await expect.poll(async () => viewport.getAttribute("data-camera-center")).not.toBe(centerAfterRightTurnValue);
  const centerAfterForward = (await viewport.getAttribute("data-camera-center"))!.split(",").map(Number);
  expect(centerAfterForward[0]).toBe(centerAfterRightTurn[0] + 1);
  expect(centerAfterForward[1]).toBe(centerAfterRightTurn[1]);
  await expect(viewport).toHaveAttribute("data-rover-heading", movementHeading ?? "1.00,-0.00");

  if (testInfo.project.name === "desktop") {
    const box = await viewport.boundingBox();
    await page.mouse.move(box!.x + box!.width / 2, box!.y + box!.height / 2);
    await page.mouse.down();
    await page.mouse.move(box!.x + box!.width / 2 + 90, box!.y + box!.height / 2, { steps: 6 });
    await page.mouse.up();
    await expect.poll(async () => viewport.getAttribute("data-rover-heading")).not.toBe(movementHeading);
  }

  await page.getByLabel("Return to fitted 3D view").click();
  await expect(viewport).toHaveAttribute("data-camera-view", "survey");
  await expect(page.getByLabel("World angle")).toBeVisible();
});

test("environment display controls toggle cell boundaries and analytical layers", async ({ page }) => {
  await page.goto("/");
  const viewport = page.getByTestId("game-viewport");
  await expect(viewport).toHaveAttribute("data-grid-visible", "false");
  await page.getByLabel("Show cell boundaries").click();
  await expect(viewport).toHaveAttribute("data-grid-visible", "true");
  await expect(page.getByLabel("Hide cell boundaries")).toBeVisible();

  await expect(viewport).toHaveAttribute("data-observation-preview", "off");
  await expect(viewport).toHaveAttribute("data-observation-window", /-?\d+,-?\d+,8,8/);
  await page.getByLabel("Show rover visibility").click();
  await expect(viewport).toHaveAttribute("data-observation-preview", "local-8");
  await expect(page.getByLabel("Hide rover visibility")).toBeVisible();
  await page.getByLabel("Top view").click();
  await expect(viewport).toHaveAttribute("data-camera-view", "top");
  await expect(viewport).toHaveAttribute("data-observation-preview", "local-8");
  await page.getByLabel("Rover point of view").click();
  await expect(viewport).toHaveAttribute("data-camera-view", "rover");
  await expect(viewport).toHaveAttribute("data-observation-preview", "local-8");

  await page.getByLabel("Ice layer").click();
  await expect(page.getByLabel("Ice layer")).toHaveClass(/bg-cyan/);
  await page.getByLabel("Ice layer").click();
  await expect(page.getByLabel("Ice layer")).not.toHaveClass(/bg-cyan/);
});

test("game guide modal explains rules", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Open game guide").click();
  await expect(page.getByText("AresSim Game Guide")).toBeVisible();
  await expect(page.getByRole("heading", { name: "How To Play" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Controls" })).toBeVisible();
  await page.getByRole("tab", { name: "terrain" }).click();
  await expect(page.getByRole("heading", { name: "Terrain Legend" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Build Pad Visuals" })).toBeVisible();
  await page.getByRole("tab", { name: "actions" }).click();
  await expect(page.getByRole("heading", { name: "Validity Rules" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Action Space" })).toBeVisible();
  await page.getByRole("tab", { name: "rewards" }).click();
  await expect(page.getByRole("heading", { name: "How Rewards Are Computed" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Per-Action Reward Terms" })).toBeVisible();
  await page.getByRole("tab", { name: "rules" }).click();
  await expect(page.getByRole("heading", { name: "Terminal Failure Conditions" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Battery Drain Formula" })).toBeVisible();
});

test("rover movement has no deadline rule and action panel is simplified", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("mission-hud")).toBeVisible();

  await expect(page.getByLabel("Extract")).toBeVisible();
  await expect(page.getByLabel("Service")).toBeVisible();
  await expect(page.getByLabel("Simulation mode")).toBeVisible();
  await expect(page.getByLabel("Mine ice")).toHaveCount(0);
  await expect(page.getByLabel("Invalid action test")).toHaveCount(0);

  const sky = page.getByTestId("ambient-time-sky");
  let previousTime = await sky.getAttribute("data-local-time");
  for (let index = 0; index < 8; index += 1) {
    await page.keyboard.press("ArrowRight");
    await expect.poll(async () => sky.getAttribute("data-local-time")).not.toBe(previousTime);
    previousTime = await sky.getAttribute("data-local-time");
  }

  await page.getByLabel("Open agent history").click();
  await expect(page.getByTestId("history-row-8")).toBeVisible();
  await expect.poll(async () => Number(await page.getByTestId("game-viewport").getAttribute("data-path-arrows"))).toBeGreaterThan(0);
  await expect(page.getByText(/deadline/i)).toHaveCount(0);
});

test("compact footer swaps manual, algorithm, and replay controls", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByLabel("Simulation mode")).toBeVisible();
  await expect(page.getByLabel("Scan")).toBeVisible();

  await page.getByLabel("Algorithm mode").click();
  await expect(page.getByLabel("Algorithm policy")).toBeVisible();
  await expect(page.getByLabel("Scan")).toHaveCount(0);
  await expect(page.getByLabel("World seed")).toBeVisible();
  await expect(page.getByLabel("Randomize world")).toBeVisible();
  await expect(page.getByLabel("Reset run")).toBeVisible();
  await page.getByLabel("Step simulation").click();
  await page.getByLabel("Open agent history").click();
  await expect(page.getByTestId("history-row-1")).toBeVisible();
  await page.getByLabel("Close drawer").click();

  await page.getByLabel("Replay mode").click();
  await expect(page.getByLabel("Upload trajectory JSON")).toBeVisible();
  await expect(page.getByLabel("World seed")).toBeVisible();
  await expect(page.getByLabel("World seed")).toBeEditable();
  await expect(page.getByLabel("Randomize world")).toBeVisible();
  await expect(page.getByLabel("Replay controls")).toBeVisible();
  await expect(page.getByLabel("Play replay")).toBeDisabled();
  await expect(page.getByLabel("LLM mode")).toHaveCount(0);
});

test("UI export downloads the unified replayable trajectory schema", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Wait").click();
  await page.getByLabel("Export trajectory").click();
  await page.getByLabel("Trajectory filename").fill("ui-trajectory-test");

  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export JSON" }).click();
  const download = await downloadPromise;
  const stream = await download.createReadStream();
  const chunks: Buffer[] = [];
  for await (const chunk of stream) chunks.push(Buffer.from(chunk));
  const trajectory = JSON.parse(Buffer.concat(chunks).toString("utf-8"));

  expect(download.suggestedFilename()).toBe("ui-trajectory-test.json");
  expect(trajectory.schemaVersion).toBe("aresim.trajectory.episode.v1");
  expect(trajectory.policy).toBeNull();
  expect(trajectory.replay.schemaVersion).toBe("aresim.trajectory.replay.v1");
  expect(trajectory.replay.steps).toHaveLength(1);
  expect(trajectory.replay.finalSnapshot.step).toBe(1);
});

test("uploaded legacy gameplay restores seed and history in replay mode", async ({ page }) => {
  await page.goto("/");
  const initial = structuredClone(initialSnapshot) as SimSnapshot;
  const final = JSON.parse(JSON.stringify(initial));
  final.step = 10;
  final.statusReason = "Checkpoint test final state";
  const replay = JSON.stringify({
    schemaVersion: "aresim.gameplay.v1",
    savedAt: "now",
    fileName: "saved-run.json",
    appVersion: "0.1.0",
    metadata: {
      sessionId: initial.sessionId,
      seed: initial.seed,
      runMode: "manual",
      totalSteps: final.step,
      finalStatus: final.gameStatus,
    },
    initialSnapshot: initial,
    steps: [],
    checkpoints: [
      { id: "checkpoint-0-initial", step: 0, label: "Initial", reason: "initial", summary: "Seed 24680", snapshot: initial },
      { id: "checkpoint-10-final", step: 10, label: "Final", reason: "final", summary: "Final state", target: { x: 3, y: 3 }, reward: 1.25, snapshot: final },
    ],
    finalSnapshot: final,
    integrity: { finalStep: 10, stepCount: 0, checkpointCount: 2 },
  });

  await page.getByLabel("Replay mode").click();
  await page.getByLabel("Upload trajectory JSON").setInputFiles({
    name: "saved-run.json",
    mimeType: "application/json",
    buffer: Buffer.from(replay),
  });

  await expect(page.getByRole("alert")).toContainText("Trajectory loaded");
  await expect(page.getByLabel("Replay controls")).toBeVisible();
  await expect(page.getByLabel("Repeat replay")).toBeVisible();
  await expect(page.getByTestId("step-scrubber")).toBeVisible();
  await expect(page.getByTestId("replay-cursor")).toBeVisible();
  await expect(page.getByLabel("World seed")).toBeDisabled();
  await expect(page.getByLabel("World seed")).toHaveValue("24680");
  await expect(page.getByLabel("Randomize world")).toBeDisabled();
});

test("extract records a valid resource action", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("mission-hud")).toBeVisible();
  await page.getByLabel("Extract").click();

  await page.getByLabel("Open agent history").click();
  await expect(page.getByTestId("history-row-1")).toContainText("extract");
  await expect.poll(async () => Number(await page.getByTestId("game-viewport").getAttribute("data-extracted-markers"))).toBeGreaterThan(0);
});

test("payload remains aboard until explicit build-pad unload", async ({ page }) => {
  await page.goto("/");
  const payload = page.getByTestId("header-storage");
  await expect(payload).toHaveAttribute("aria-valuenow", "0");
  await expect(payload).toHaveAttribute("aria-valuemax", "12");
  await page.getByLabel("Zoom in").click();

  await page.getByLabel("Scan").click();
  await expect(payload).toHaveAttribute("aria-valuenow", "0.5");

  await page.getByLabel("Extract").click();
  await expect(payload).toHaveAttribute("aria-valuenow", "2.5");

  await page.getByLabel("Wait").click();
  await expect(payload).toHaveAttribute("aria-valuenow", "2.5");

  await page.getByLabel("Unload payload").click();
  await expect(payload).toHaveAttribute("aria-valuenow", "0");
  await expect(page.getByText("ICE TANK · 2.0 KG")).toBeVisible();
  await expect(page.getByText("SAMPLE VAULT · 0.5 KG")).toBeVisible();
  await expect(page.getByText("UNLOADING · 2.0 KG ICE · 0.5 KG SAMPLES")).toBeVisible();
  await page.getByLabel("Open mission").click();
  await expect(page.getByText("Payload Delivered")).toBeVisible();
  await expect(page.getByTestId("mission-drawer")).toContainText("2 kg ice");
  await expect(page.getByTestId("mission-drawer")).toContainText("1 unload");
});

test("objective markers, decimals, and habitat build state update", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("header").getByText(/\d+\.\d{2}%/).first()).toBeVisible();
  await expect(page.locator("header").getByText(/[+-]?\d+\.\d{2} kW/).first()).toBeVisible();

  await page.getByLabel("Scan").click();
  await expect.poll(async () => Number(await page.getByTestId("game-viewport").getAttribute("data-scanned-markers"))).toBeGreaterThan(0);

  for (let index = 0; index < 10; index += 1) {
    await page.getByRole("button", { name: "Build", exact: true }).click();
  }

  await page.getByLabel("Open mission").click();
  await expect(page.getByText("100.00% (10/10)")).toBeVisible();
  await expect.poll(async () => page.getByTestId("game-viewport").getAttribute("data-build-pad-status")).toContain("habitat_built");
});

test("analytics modal opens from ribbon and renders tabs", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("analytics-button").click();
  await expect(page.getByText("Run Analytics")).toBeVisible();
  await expect(page.getByRole("tab", { name: /Rewards/ })).toBeVisible();
  await expect(page.getByRole("tab", { name: /Environment/ })).toBeVisible();
});
