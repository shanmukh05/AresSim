/**
 * Fixed 8×8 rover-centered crop for the flashlight overlay.
 * Visual preview only; the policy observation is built in `engine/aresim/components/observations.py`.
 */

export const ROVER_OBSERVATION_SIZE = 8;
export const ROVER_OBSERVATION_ANCHOR = Math.floor((ROVER_OBSERVATION_SIZE - 1) / 2);

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
