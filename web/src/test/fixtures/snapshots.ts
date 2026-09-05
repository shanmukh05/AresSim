/** Frozen engine snapshots for presentation and observation unit tests. */

import type { SimSnapshot } from "../../types/sim";
import snapshot145 from "./snapshot-145.json";
import snapshot1447 from "./snapshot-1447.json";
import snapshot246 from "./snapshot-246.json";

const FIXTURES = {
  1447: snapshot1447,
  145: snapshot145,
  246: snapshot246,
} as const satisfies Record<number, SimSnapshot>;

export function fixtureSnapshot(seed: keyof typeof FIXTURES): SimSnapshot {
  return structuredClone(FIXTURES[seed]) as SimSnapshot;
}
