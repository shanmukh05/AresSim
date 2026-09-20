/**
 * Display helpers for power margin and a Wait-charge HUD preview.
 * The engine still applies the real charge in `core/rules.py`.
 */

import type { SimSnapshot } from "../types/sim";

// ponytail: HUD preview copies PowerConfig wait_charge_per_kw/wait_max_charge.
// Upgrade: snapshot.waitChargeEstimate from rules.wait_recharge so replay deltas stay honest.
const WAIT_CHARGE_PER_KW = 0.85;
const WAIT_MAX_CHARGE = 18;

/** Generated minus consumed. Negative means the rover battery will drain from deficit. */
export function getPowerMargin(snapshot: SimSnapshot) {
  return Number((snapshot.resources.powerGenerated - snapshot.resources.powerConsumed).toFixed(2));
}

export function estimateWaitRecharge(snapshot: SimSnapshot) {
  const margin = getPowerMargin(snapshot);
  return Number(Math.max(0, Math.min(WAIT_MAX_CHARGE, margin * WAIT_CHARGE_PER_KW)).toFixed(2));
}
