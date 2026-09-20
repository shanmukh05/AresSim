/**
 * Display-only rover cargo totals for HUD and analytics.
 * Capacity checks and cargo mutation live in `engine/aresim/core/rules.py`.
 */

import type { RoverEntity } from "../types/sim";

type PayloadCargo = Pick<RoverEntity, "cargoIce" | "cargoOre" | "cargoSamples" | "cargoCapacityKg">;

/** Combined ice + ore + sample mass currently on the rover. */
export function payloadUsedKg(rover: Pick<RoverEntity, "cargoIce" | "cargoOre" | "cargoSamples">) {
  return Number(((rover.cargoIce ?? 0) + (rover.cargoOre ?? 0) + (rover.cargoSamples ?? 0)).toFixed(2));
}

export function payloadRemainingKg(rover: PayloadCargo) {
  return Number((Math.max(0, rover.cargoCapacityKg - payloadUsedKg(rover))).toFixed(2));
}
