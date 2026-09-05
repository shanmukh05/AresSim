/**
 * Sky phase and celestial positions from `snapshot.localTime`.
 * Pure presentation. Sol is unused for lighting but kept for future day-cycle work.
 */

export type AmbientPhase = "night" | "dawn" | "day" | "dusk";

export interface AmbientTimeState {
  phase: AmbientPhase;
  hour: number;
  daylight: number;
  stars: number;
  sunX: number;
  sunY: number;
  sunOpacity: number;
  nightObjects: number;
  phobosX: number;
  phobosY: number;
  deimosX: number;
  deimosY: number;
  planetDrift: number;
  skyTop: string;
  skyHorizon: string;
}

export function ambientTimeFor(localTime: string, sol = 1): AmbientTimeState {
  const [rawHour = 0, rawMinute = 0] = localTime.split(":").map(Number);
  const hour = ((Number.isFinite(rawHour) ? rawHour : 0) + (Number.isFinite(rawMinute) ? rawMinute : 0) / 60 + 24) % 24;
  const solarProgress = clamp((hour - 6) / 12);
  const daylight = clamp(Math.sin(solarProgress * Math.PI));
  const twilight = clamp(Math.max(bell(hour, 6, 1.65), bell(hour, 18, 1.65)));
  const phase: AmbientPhase = hour < 5.5 || hour >= 20 ? "night" : hour < 7 ? "dawn" : hour < 18 ? "day" : "dusk";
  const skyTop = mixHex("#03050a", "#32140d", daylight * 0.88);
  const daylightHorizon = mixHex("#11070a", "#6a2413", daylight * 0.82);
  const skyHorizon = mixHex(daylightHorizon, "#8b3219", twilight * 0.42);
  const absoluteHours = Math.max(0, sol - 1) * 24.6597 + hour;
  const phobosOrbit = fraction(absoluteHours / 7.653);
  const deimosOrbit = fraction(absoluteHours / 30.312);

  return {
    phase,
    hour,
    daylight,
    stars: clamp(1 - daylight * 2.4 - twilight * 0.48),
    sunX: 7 + solarProgress * 86,
    sunY: 76 - Math.sin(solarProgress * Math.PI) * 62,
    sunOpacity: daylight < 0.001 ? 0 : 0.26 + daylight * 0.54,
    nightObjects: clamp(1 - daylight * 2.6 - twilight * 0.42),
    phobosX: 6 + phobosOrbit * 88,
    phobosY: 68 - Math.sin(phobosOrbit * Math.PI) * 46,
    deimosX: 94 - deimosOrbit * 88,
    deimosY: 56 - Math.sin(deimosOrbit * Math.PI) * 34,
    planetDrift: Math.sin(Math.max(1, sol) * 0.31) * 3.5,
    skyTop,
    skyHorizon,
  };
}

function bell(value: number, center: number, width: number) {
  const distance = (value - center) / width;
  return Math.exp(-(distance * distance));
}

function clamp(value: number) {
  return Math.max(0, Math.min(1, value));
}

function fraction(value: number) {
  return value - Math.floor(value);
}

function mixHex(from: string, to: string, amount: number) {
  const start = hexChannels(from);
  const end = hexChannels(to);
  const channel = (index: number) => Math.round(start[index] + (end[index] - start[index]) * clamp(amount)).toString(16).padStart(2, "0");
  return `#${channel(0)}${channel(1)}${channel(2)}`;
}

function hexChannels(color: string) {
  const value = color.slice(1);
  return [0, 2, 4].map((offset) => Number.parseInt(value.slice(offset, offset + 2), 16));
}
