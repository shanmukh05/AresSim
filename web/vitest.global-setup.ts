/** Start the Python API once for Vitest so browser tests hit authoritative engine rules. */

import { spawn, type ChildProcess } from "node:child_process";
import { writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const WEB_DIR = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(WEB_DIR, "..");
const ENGINE_DIR = path.join(ROOT, "engine");
const PORT_FILE = path.join(WEB_DIR, ".vitest-api-port");
const PORT = 18_765;

let server: ChildProcess | null = null;

async function waitForHealth(port: number) {
  for (let attempt = 0; attempt < 60; attempt += 1) {
    try {
      const response = await fetch(`http://127.0.0.1:${port}/api/health`);
      if (response.ok) return;
    } catch {
      // Server still starting.
    }
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  throw new Error("AresSim test API did not become healthy in time");
}

export async function setup() {
  writeFileSync(PORT_FILE, String(PORT), "utf8");
  server = spawn(
    path.join(ENGINE_DIR, ".venv", "bin", "python"),
    ["-m", "uvicorn", "aresim.api:create_app", "--factory", "--host", "127.0.0.1", "--port", String(PORT)],
    { cwd: ENGINE_DIR, stdio: "pipe", env: { ...process.env, PYTHONPATH: ENGINE_DIR } },
  );
  await waitForHealth(PORT);
}

export async function teardown() {
  server?.kill("SIGTERM");
  server = null;
}
