/** Unit tests for the local rover observation crop. */

import { describe, expect, it } from "vitest";
import { fixtureSnapshot } from "../test/fixtures/snapshots";
import { createRoverObservation, isInsideRoverObservation, ROVER_OBSERVATION_ANCHOR, ROVER_OBSERVATION_SIZE } from "./roverObservation";

describe("rover local observation", () => {
  it("builds a fixed 8x8 crop with the rover at local cell 3,3", () => {
    const snapshot = fixtureSnapshot(145);
    snapshot.rovers[0].x = 12;
    snapshot.rovers[0].y = 14;
    const observation = createRoverObservation(snapshot);

    expect(observation.schema).toBe("aresim.obs.local.v1");
    expect(observation.size).toBe(ROVER_OBSERVATION_SIZE);
    expect(observation.anchor).toEqual({ x: ROVER_OBSERVATION_ANCHOR, y: ROVER_OBSERVATION_ANCHOR });
    expect(observation.origin).toEqual({ x: 9, y: 11 });
    expect(observation.cells).toHaveLength(64);
    expect(observation.cells.find((cell) => cell.localX === 3 && cell.localY === 3)).toMatchObject({ worldX: 12, worldY: 14, visible: true });
    expect(isInsideRoverObservation(16, 18, snapshot.rovers[0])).toBe(true);
    expect(isInsideRoverObservation(17, 18, snapshot.rovers[0])).toBe(false);
  });

  it("pads out-of-bounds cells as unknown instead of shifting the crop", () => {
    const snapshot = fixtureSnapshot(246);
    snapshot.rovers[0].x = 0;
    snapshot.rovers[0].y = 0;
    const observation = createRoverObservation(snapshot);

    expect(observation.origin).toEqual({ x: -3, y: -3 });
    expect(observation.cells.filter((cell) => !cell.visible)).toHaveLength(39);
    expect(observation.cells.filter((cell) => !cell.visible).every((cell) => cell.terrain === null)).toBe(true);
  });
});
