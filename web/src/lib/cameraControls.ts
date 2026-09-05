/**
 * Map arrow keys to grid deltas.
 * Survey/Top stay world-cardinal; Rover POV rotates with the rover's facing.
 */

import type { CameraView } from "../types/sim";

export type MovementArrow = "ArrowUp" | "ArrowDown" | "ArrowLeft" | "ArrowRight";

const WORLD_DELTAS: Record<MovementArrow, [number, number]> = {
  ArrowUp: [0, -1],
  ArrowDown: [0, 1],
  ArrowLeft: [-1, 0],
  ArrowRight: [1, 0],
};

/** `[dx, dy]` for a movement key, or `null` if the key is not an arrow. */
export function movementDeltaForArrow(key: string, view: CameraView, roverYaw: number): [number, number] | null {
  if (!(key in WORLD_DELTAS)) return null;
  const arrow = key as MovementArrow;
  if (view !== "rover") return WORLD_DELTAS[arrow];

  const rawX = Math.sin(roverYaw);
  const rawY = -Math.cos(roverYaw);
  const forward: [number, number] = Math.abs(rawX) > Math.abs(rawY)
    ? [rawX >= 0 ? 1 : -1, 0]
    : [0, rawY >= 0 ? 1 : -1];

  if (arrow === "ArrowUp") return forward;
  if (arrow === "ArrowDown") return cleanDelta(-forward[0], -forward[1]);
  if (arrow === "ArrowLeft") return cleanDelta(forward[1], -forward[0]);
  return cleanDelta(-forward[1], forward[0]);
}

function cleanDelta(x: number, y: number): [number, number] {
  return [x === 0 ? 0 : x, y === 0 ? 0 : y];
}
