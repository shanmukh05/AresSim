/** Proxy Vitest `fetch` calls to the Python API started in `vitest.global-setup.ts`. */

import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const PORT_FILE = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../.vitest-api-port");

function apiBase() {
  const port = readFileSync(PORT_FILE, "utf8").trim();
  return `http://127.0.0.1:${port}`;
}

export function resetMockApi() {
  // Each test starts a fresh session through the store; no global simulator state remains.
}

export function installMockApi() {
  const base = apiBase();
  const originalFetch = globalThis.fetch.bind(globalThis);
  globalThis.fetch = async (input, init) => {
    const href = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const url = new URL(href, "http://aresim.test");
    if (url.pathname.startsWith("/api")) {
      return originalFetch(`${base}${url.pathname}${url.search}`, init);
    }
    return originalFetch(input, init);
  };
}
