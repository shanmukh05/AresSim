/**
 * Display helpers for power margin and Wait/pad recharge.
 * Mirrors engine coefficients for HUD text; the engine still applies the real charge.
 */

import type { SimSnapshot } from "../types/sim";

export const WAIT_CHARGE_PER_KW = 0.85;
export const WAIT_MAX_CHARGE = 18;
export const PAD_TRICKLE_CHARGE_PER_KW = 0.25;
export const PAD_TRICKLE_MAX_CHARGE = 4;

/** Generated minus consumed. Negative means the rover battery will drain from deficit. */
export function getPowerMargin(snapshot: SimSnapshot) {
  return Number((snapshot.resources.powerGenerated - snapshot.resources.powerConsumed).toFixed(2));
}

export function waitRechargeForMargin(powerMargin: number) {
  return Number(Math.max(0, Math.min(WAIT_MAX_CHARGE, powerMargin * WAIT_CHARGE_PER_KW)).toFixed(2));
}

export function padTrickleRechargeForMargin(powerMargin: number) {
  return Number(Math.max(0, Math.min(PAD_TRICKLE_MAX_CHARGE, powerMargin * PAD_TRICKLE_CHARGE_PER_KW)).toFixed(2));
}

export function estimateWaitRecharge(snapshot: SimSnapshot) {
  return waitRechargeForMargin(getPowerMargin(snapshot));
}
