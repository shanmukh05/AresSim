/** Unit tests for rover payload helpers. */

import { describe, expect, it } from "vitest";
import type { RoverEntity } from "../types/sim";
import { canFitPayload, payloadUsedKg, tryAddPayload } from "./payload";

function rover(overrides: Partial<RoverEntity> = {}): RoverEntity {
  return {
    id: "rover-test",
    name: "Test rover",
    x: 0,
    y: 0,
    battery: 100,
    health: 100,
    cargoIce: 0,
    cargoOre: 0,
    cargoSamples: 0,
    cargoCapacityKg: 12,
    currentTask: "Testing",
    ...overrides,
  };
}

describe("payload capacity", () => {
  it("allows an exact fill and atomically rejects the next sample", () => {
    const subject = rover({ cargoIce: 10, cargoSamples: 1.5 });

    expect(tryAddPayload(subject, { cargoSamples: 0.5 })).toBe(true);
    expect(payloadUsedKg(subject)).toBe(12);
    expect(subject.cargoSamples).toBe(2);

    expect(tryAddPayload(subject, { cargoSamples: 0.5 })).toBe(false);
    expect(payloadUsedKg(subject)).toBe(12);
    expect(subject.cargoSamples).toBe(2);
  });

  it("counts all cargo types against the shared capacity", () => {
    const subject = rover({ cargoIce: 9.5, cargoOre: 1.5, cargoSamples: 0.75 });

    expect(canFitPayload(subject, { cargoSamples: 0.5 })).toBe(false);
    expect(tryAddPayload(subject, { cargoSamples: 0.5 })).toBe(false);
    expect(payloadUsedKg(subject)).toBe(11.75);
    expect(subject.cargoSamples).toBe(0.75);
  });

  it("blocks every further addition when a loaded rover is already over capacity", () => {
    const subject = rover({ cargoIce: 12, cargoSamples: 0.5 });

    expect(tryAddPayload(subject, { cargoSamples: 0.5 })).toBe(false);
    expect(payloadUsedKg(subject)).toBe(12.5);
    expect(subject.cargoSamples).toBe(0.5);
  });

  it("rejects invalid or negative additions without changing cargo", () => {
    const subject = rover({ cargoSamples: 1 });

    expect(tryAddPayload(subject, { cargoSamples: -0.5 })).toBe(false);
    expect(tryAddPayload(subject, { cargoIce: Number.NaN })).toBe(false);
    expect(payloadUsedKg(subject)).toBe(1);
  });
});
