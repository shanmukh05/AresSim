/** Display projections of backend-owned power values. */

import type { SimSnapshot } from "../types/sim";

export function getPowerMargin(snapshot: SimSnapshot) {
  return Number((snapshot.resources.powerGenerated - snapshot.resources.powerConsumed).toFixed(2));
}

export function estimateWaitRecharge(snapshot: SimSnapshot): number | null {
  // Historical replays do not carry the originating engine configuration.
  return snapshot.waitChargeEstimate ?? null;
}
