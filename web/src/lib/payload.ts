/**
 * Rover cargo helpers for HUD math and the retired mock backend.
 * Authoritative capacity checks live in `engine/aresim/core/rules.py`.
 */

import type { RoverEntity } from "../types/sim";

export const ROVER_PAYLOAD_CAPACITY_KG = 12;
export const SCAN_SAMPLE_MASS_KG = 0.5;
export const ICE_EXTRACTION_MASS_KG = 2;

type PayloadCargo = Pick<RoverEntity, "cargoIce" | "cargoOre" | "cargoSamples" | "cargoCapacityKg">;
type PayloadAddition = Partial<Pick<RoverEntity, "cargoIce" | "cargoOre" | "cargoSamples">>;

/** Combined ice + ore + sample mass currently on the rover. */
export function payloadUsedKg(rover: Pick<RoverEntity, "cargoIce" | "cargoOre" | "cargoSamples">) {
  return Number(((rover.cargoIce ?? 0) + (rover.cargoOre ?? 0) + (rover.cargoSamples ?? 0)).toFixed(2));
}

export function payloadRemainingKg(rover: Pick<RoverEntity, "cargoIce" | "cargoOre" | "cargoSamples" | "cargoCapacityKg">) {
  return Number((Math.max(0, (rover.cargoCapacityKg ?? ROVER_PAYLOAD_CAPACITY_KG) - payloadUsedKg(rover))).toFixed(2));
}

export function payloadAdditionKg(addition: PayloadAddition) {
  return Number(((addition.cargoIce ?? 0) + (addition.cargoOre ?? 0) + (addition.cargoSamples ?? 0)).toFixed(2));
}

/** Collection is atomic: the entire addition fits or no cargo is changed. */
export function canFitPayload(rover: PayloadCargo, addition: PayloadAddition) {
  const additions = [addition.cargoIce ?? 0, addition.cargoOre ?? 0, addition.cargoSamples ?? 0];
  if (!additions.every((value) => Number.isFinite(value) && value >= 0)) return false;
  const additionKg = payloadAdditionKg(addition);
  const capacityKg = rover.cargoCapacityKg ?? ROVER_PAYLOAD_CAPACITY_KG;
  return Number.isFinite(capacityKg) && capacityKg >= 0 && payloadUsedKg(rover) + additionKg <= capacityKg + 1e-9;
}

/** Returns false without mutation when the complete addition would exceed capacity. */
export function tryAddPayload(rover: RoverEntity, addition: PayloadAddition) {
  if (!canFitPayload(rover, addition)) return false;
  rover.cargoIce = Number(((rover.cargoIce ?? 0) + (addition.cargoIce ?? 0)).toFixed(2));
  rover.cargoOre = Number(((rover.cargoOre ?? 0) + (addition.cargoOre ?? 0)).toFixed(2));
  rover.cargoSamples = Number(((rover.cargoSamples ?? 0) + (addition.cargoSamples ?? 0)).toFixed(2));
  return true;
}

/** Adds payload fields introduced after aresim.gameplay.v1 without breaking legacy replays. */
export function hydrateRoverPayload(rover: RoverEntity): RoverEntity {
  return {
    ...rover,
    cargoIce: rover.cargoIce ?? 0,
    cargoOre: rover.cargoOre ?? 0,
    cargoSamples: rover.cargoSamples ?? 0,
    cargoCapacityKg: rover.cargoCapacityKg ?? ROVER_PAYLOAD_CAPACITY_KG,
  };
}
