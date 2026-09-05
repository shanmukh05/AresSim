/** Unit tests for camera-relative movement deltas. */

import { describe, expect, it } from "vitest";
import { movementDeltaForArrow } from "./cameraControls";

describe("movementDeltaForArrow", () => {
  it("keeps world-cardinal arrow controls outside Rover POV", () => {
    expect(movementDeltaForArrow("ArrowUp", "survey", Math.PI / 2)).toEqual([0, -1]);
    expect(movementDeltaForArrow("ArrowLeft", "top", Math.PI)).toEqual([-1, 0]);
  });

  it("rotates all arrow operations with an east-facing rover", () => {
    expect(movementDeltaForArrow("ArrowUp", "rover", Math.PI / 2)).toEqual([1, 0]);
    expect(movementDeltaForArrow("ArrowLeft", "rover", Math.PI / 2)).toEqual([0, -1]);
    expect(movementDeltaForArrow("ArrowRight", "rover", Math.PI / 2)).toEqual([0, 1]);
    expect(movementDeltaForArrow("ArrowDown", "rover", Math.PI / 2)).toEqual([-1, 0]);
  });

  it("makes Up continue forward after a left turn", () => {
    const west = -Math.PI / 2;
    expect(movementDeltaForArrow("ArrowUp", "rover", west)).toEqual([-1, 0]);
    expect(movementDeltaForArrow("ArrowLeft", "rover", west)).toEqual([0, 1]);
  });

  it("ignores non-arrow keys", () => {
    expect(movementDeltaForArrow("KeyW", "rover", 0)).toBeNull();
  });
});
