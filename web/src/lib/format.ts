/** Small number-formatting helpers for HUD and analytics. */

export function formatFixed(value: number, unit = "") {
  return `${value.toFixed(2)}${unit}`;
}

export function formatSignedFixed(value: number, unit = "") {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}${unit}`;
}
