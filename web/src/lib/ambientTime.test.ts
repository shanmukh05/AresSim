/** Unit tests for ambient day/night phase derivation. */

import { describe, expect, it } from "vitest";
import { ambientTimeFor } from "./ambientTime";

describe("ambientTimeFor", () => {
  it("derives the four restrained time phases", () => {
    expect(ambientTimeFor("02:00").phase).toBe("night");
    expect(ambientTimeFor("06:15").phase).toBe("dawn");
    expect(ambientTimeFor("12:00").phase).toBe("day");
    expect(ambientTimeFor("18:45").phase).toBe("dusk");
  });

  it("places the sun highest at noon and hides it at night", () => {
    const sunrise = ambientTimeFor("06:00");
    const noon = ambientTimeFor("12:00");
    const night = ambientTimeFor("23:00");
    expect(noon.sunY).toBeLessThan(sunrise.sunY);
    expect(noon.sunX).toBeGreaterThan(sunrise.sunX);
    expect(noon.sunOpacity).toBeGreaterThan(0);
    expect(night.sunOpacity).toBe(0);
    expect(night.stars).toBeGreaterThan(0.9);
    expect(night.nightObjects).toBeGreaterThan(0.9);
  });

  it("moves Phobos faster than Deimos across successive sols", () => {
    const first = ambientTimeFor("22:00", 1);
    const next = ambientTimeFor("22:00", 2);
    expect(Math.abs(next.phobosX - first.phobosX)).toBeGreaterThan(Math.abs(next.deimosX - first.deimosX));
  });

  it("keeps invalid input safe", () => {
    const state = ambientTimeFor("invalid");
    expect(state.phase).toBe("night");
    expect(Number.isFinite(state.sunX)).toBe(true);
  });
});
