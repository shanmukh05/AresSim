/**
 * Deployment mode flag and base-path URL helpers for standalone vs TensorWrites `/aressim`.
 * Toggle deployment mode via `VITE_DEPLOYED=true`, backend `ARESSIM_DEPLOYED=1`, or `?deployed=1`.
 */

const deployedEnv = String(import.meta.env.VITE_DEPLOYED ?? "").trim().toLowerCase();
const urlDeployed =
  typeof window !== "undefined" &&
  new URLSearchParams(window.location.search).get("deployed") === "1";

/**
 * Compile-time / URL deployment toggle.
 * Flip this (or build with `VITE_DEPLOYED=true`) to run AresSim in public blog deployment mode:
 * - hides dev Agent Settings (RL models auto-run their best default checkpoints)
 * - requires user-supplied API keys for LLM + Decision Model runs instead of server `.env` keys
 */
export const BUILD_DEPLOYED =
  deployedEnv === "true" || deployedEnv === "1" || urlDeployed;

/** Normalized base URL (e.g. `/` in local dev or `/aressim/` when mounted on TensorWrites). */
export const APP_BASE_URL = (import.meta.env.BASE_URL || "/").replace(/\/?$/, "/");

/** Resolve a public asset path (e.g. `logo.png`) against the configured mount base path. */
export function assetUrl(relativePath: string): string {
  return `${APP_BASE_URL}${relativePath.replace(/^\/+/, "")}`;
}

/** Resolve an API path (e.g. `/api/sessions`) against `VITE_API_BASE` or the mount base path. */
export function apiUrl(path: string): string {
  const explicit = String(import.meta.env.VITE_API_BASE ?? "").trim().replace(/\/+$/, "");
  const normalized = path.startsWith("/") ? path : `/${path}`;
  if (explicit) return `${explicit}${normalized}`;
  const prefix = APP_BASE_URL === "/" ? "" : APP_BASE_URL.replace(/\/$/, "");
  return `${prefix}${normalized}`;
}
