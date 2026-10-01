/**
 * Rover-centered crop bounds for the flashlight overlay.
 * Size comes from the snapshot (`observationWindowSize` ← `defaults.py`).
 * Visual preview only; the policy observation is built in `engine/aresim/components/observations.py`.
 */

/** World-space bounds of the rover-centered crop (even windows bias toward +x/+y). */
export function roverObservationBounds(rover: { x: number; y: number }, windowSize: number) {
  const anchor = Math.floor((windowSize - 1) / 2);
  const minX = rover.x - anchor;
  const minY = rover.y - anchor;
  return {
    minX,
    minY,
    maxXExclusive: minX + windowSize,
    maxYExclusive: minY + windowSize,
    windowSize,
  };
}

/** True if `(x, y)` sits inside the acting rover's local observation. */
export function isInsideRoverObservation(
  x: number,
  y: number,
  rover: { x: number; y: number },
  windowSize: number,
) {
  const bounds = roverObservationBounds(rover, windowSize);
  return x >= bounds.minX && x < bounds.maxXExclusive && y >= bounds.minY && y < bounds.maxYExclusive;
}
