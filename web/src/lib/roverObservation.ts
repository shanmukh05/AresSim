/**
 * Fixed 8×8 rover-centered crop used by the flashlight overlay.
 * This is a preview of the planned RL observation (`aresim.obs.local.v1`);
 * it is not a second source of world truth.
 */

import type { SimSnapshot, TerrainCell } from "../types/sim";

export const ROVER_OBSERVATION_SIZE = 8;
export const ROVER_OBSERVATION_ANCHOR = Math.floor((ROVER_OBSERVATION_SIZE - 1) / 2);

export interface RoverObservationCell {
  localX: number;
  localY: number;
  worldX: number;
  worldY: number;
  visible: boolean;
  terrain: TerrainCell | null;
}

export interface RoverObservationV1 {
  schema: "aresim.obs.local.v1";
  size: 8;
  anchor: { x: number; y: number };
  origin: { x: number; y: number };
  cells: RoverObservationCell[];
}

/** World-space bounds of the 8×8 crop with the rover at local cell (3, 3). */
export function roverObservationBounds(rover: { x: number; y: number }) {
  const minX = rover.x - ROVER_OBSERVATION_ANCHOR;
  const minY = rover.y - ROVER_OBSERVATION_ANCHOR;
  return {
    minX,
    minY,
    maxXExclusive: minX + ROVER_OBSERVATION_SIZE,
    maxYExclusive: minY + ROVER_OBSERVATION_SIZE,
  };
}

/** True if `(x, y)` sits inside the acting rover's local observation. */
export function isInsideRoverObservation(x: number, y: number, rover: { x: number; y: number }) {
  const bounds = roverObservationBounds(rover);
  return x >= bounds.minX && x < bounds.maxXExclusive && y >= bounds.minY && y < bounds.maxYExclusive;
}

/** Build `aresim.obs.local.v1`. Out-of-map cells are present but `visible: false`. */
export function createRoverObservation(snapshot: SimSnapshot): RoverObservationV1 {
  const rover = snapshot.rovers[0];
  if (!rover) throw new Error("A rover is required to build a local observation");
  const bounds = roverObservationBounds(rover);
  const cells: RoverObservationCell[] = [];

  for (let localY = 0; localY < ROVER_OBSERVATION_SIZE; localY += 1) {
    for (let localX = 0; localX < ROVER_OBSERVATION_SIZE; localX += 1) {
      const worldX = bounds.minX + localX;
      const worldY = bounds.minY + localY;
      const visible = worldX >= 0 && worldX < snapshot.terrainSize.width && worldY >= 0 && worldY < snapshot.terrainSize.height;
      cells.push({
        localX,
        localY,
        worldX,
        worldY,
        visible,
        terrain: visible ? structuredClone(snapshot.terrain[worldY][worldX]) : null,
      });
    }
  }

  return {
    schema: "aresim.obs.local.v1",
    size: ROVER_OBSERVATION_SIZE,
    anchor: { x: ROVER_OBSERVATION_ANCHOR, y: ROVER_OBSERVATION_ANCHOR },
    origin: { x: bounds.minX, y: bounds.minY },
    cells,
  };
}
