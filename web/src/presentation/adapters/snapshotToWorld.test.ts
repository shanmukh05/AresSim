/** Unit tests for snapshot-to-world presentation adapters. */

import { describe, expect, it } from "vitest";
import { fixtureSnapshot } from "../../test/fixtures/snapshots";
import { snapshotToWorld } from "./snapshotToWorld";

describe("snapshotToWorld", () => {
  it("preserves grid coordinates, stable ids, entities, and deterministic path data", () => {
    const snapshot = fixtureSnapshot(1447);
    const world = snapshotToWorld(snapshot);

    expect(world.dimensions).toEqual(snapshot.terrainSize);
    expect(world.terrain).toHaveLength(snapshot.terrainSize.width * snapshot.terrainSize.height);
    expect(world.terrain[0]).toMatchObject({ id: "cell-0-0", x: 0, y: 0 });
    expect(world.rovers.map((rover) => rover.id)).toEqual(snapshot.rovers.map((rover) => rover.id));
    expect(world.structures.map((structure) => structure.id)).toEqual(snapshot.structures.map((structure) => structure.id));
    expect(world.buildPadStatus).toBe(snapshot.buildPadState.status);
  });

  it("maps ridges above regolith and craters below the surface", () => {
    const snapshot = fixtureSnapshot(1447);
    const world = snapshotToWorld(snapshot);
    const ridge = world.terrain.find((cell) => cell.terrain === "ridge");
    const crater = world.terrain.find((cell) => cell.terrain === "crater");
    const regolith = world.terrain.find((cell) => cell.terrain === "regolith");

    expect(ridge).toBeDefined();
    expect(crater).toBeDefined();
    expect(regolith).toBeDefined();
    expect(ridge!.elevation).toBeGreaterThan(regolith!.elevation);
    expect(crater!.elevation).toBeLessThan(0);
  });

  it("builds the rover trail from movement entries only and preserves movement order", () => {
    const snapshot = fixtureSnapshot(1447);
    const entry = (id: string, step: number, action: "move" | "scan", x: number, y: number) => ({
      id,
      step,
      actor: "Player" as const,
      action,
      target: { x, y },
      result: action,
      reward: 0,
      rewardTerms: {},
      resourceDelta: {},
      events: [],
    });
    snapshot.history = [
      entry("move-2", 3, "move", 12, 11),
      entry("scan", 2, "scan", 3, 27),
      entry("move-1", 1, "move", 11, 11),
    ];

    expect(snapshotToWorld(snapshot).path).toEqual([{ x: 11, y: 11 }, { x: 12, y: 11 }]);
  });
});
