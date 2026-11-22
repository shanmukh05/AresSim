/**
 * HTTP client for the Python engine REST API.
 * Vite proxies `/api` to `127.0.0.1:8000`. Keep this file transport-only; store
 * logic belongs in `useAresStore`.
 */

import { apiUrl } from "../lib/deployConfig";
import type { AgentStepResponse, AlgorithmId, AttachPolicyResponse, AresReplayProjectionV1, AresTrajectoryEpisodeV1, LlmDmAttachPayload, LlmLearnResponse, PoliciesResponse, PolicyActionMode, RunMode, SimAction, SimSnapshot, TerrainCell } from "../types/sim";

type CompactSimSnapshot = Omit<SimSnapshot, "terrain"> & {
  terrain?: TerrainCell[][];
  terrainDelta?: TerrainCell[];
};

interface SnapshotResponse {
  snapshot: CompactSimSnapshot;
}

interface BatchSnapshotResponse {
  snapshots: CompactSimSnapshot[];
}

interface CompactAgentStepResponse {
  snapshot: CompactSimSnapshot;
  policyMeta: AgentStepResponse["policyMeta"];
}

export interface ReplayResponse {
  replayId: string;
  fileName: string;
  seed: number;
  totalSteps: number;
  cursor: number;
  schemaVersion: string;
  activeCheckpointId?: string;
  snapshot: SimSnapshot;
  gameplay?: AresReplayProjectionV1;
}

interface ApiErrorPayload {
  error?: {
    code?: string;
    message?: string;
  };
}

/** Backend error with a machine-readable `code` the store can switch on. */
export class AresApiError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

/** Tracks the live `sessionId` and loaded `replayId` for subsequent calls. */
export class AresApiClient {
  private sessionId: string | null = null;
  private replayId: string | null = null;
  private terrainCache: TerrainCell[][] | null = null;

  /** Start or replace the live session. Pass no seed to let the backend choose. */
  async startSession({ seed }: { seed?: number } = {}): Promise<SimSnapshot> {
    const response = await request<SnapshotResponse>("/api/sessions", {
      method: "POST",
      body: JSON.stringify({ seed: seed ?? null }),
    });
    const snapshot = this.hydrateSnapshot(response.snapshot, true);
    this.sessionId = snapshot.sessionId;
    this.replayId = null;
    return snapshot;
  }

  async randomizeSession(): Promise<SimSnapshot> {
    return this.startSession();
  }

  /** Manual play: send `actor: Player` with compact terrain deltas when cached. */
  async dispatchAction(action: SimAction): Promise<SimSnapshot> {
    return this.applyAction(action, "Player");
  }

  /** Batched manual play: executes queued rapid inputs in one network round-trip. */
  async dispatchActionBatch(actions: SimAction[]): Promise<SimSnapshot[]> {
    const sessionId = this.requireSession();
    const compact = this.terrainCache !== null;
    const response = await request<BatchSnapshotResponse>(
      `/api/sessions/${encodeURIComponent(sessionId)}/actions-batch`,
      {
        method: "POST",
        body: JSON.stringify({ actions, actor: "Player", compact }),
      },
    );
    const items = Array.isArray(response.snapshots) ? response.snapshots : [];
    return items.map((item) => this.hydrateSnapshot(item, true));
  }

  /** Algorithm autoplay: server runs the attached policy and steps as Agent. */
  async agentStep(): Promise<AgentStepResponse> {
    const sessionId = this.requireSession();
    const response = await request<CompactAgentStepResponse>(`/api/sessions/${encodeURIComponent(sessionId)}/agent-step`, {
      method: "POST",
    });
    return {
      snapshot: this.hydrateSnapshot(response.snapshot, true),
      policyMeta: response.policyMeta,
    };
  }

  /** Batched algorithm steps: amortizes remote round-trip latency and shrinks payloads via terrain deltas. */
  async agentSteps(count = 10): Promise<AgentStepResponse[]> {
    const sessionId = this.requireSession();
    const compact = this.terrainCache !== null;
    const response = await request<{ steps: CompactAgentStepResponse[] }>(
      `/api/sessions/${encodeURIComponent(sessionId)}/agent-steps`,
      {
        method: "POST",
        body: JSON.stringify({ count, compact }),
      },
    );
    const steps = Array.isArray(response.steps) ? response.steps : [];
    return steps.map((item) => ({
      snapshot: this.hydrateSnapshot(item.snapshot, true),
      policyMeta: item.policyMeta,
    }));
  }

  async listPolicies(): Promise<PoliciesResponse> {
    return request<PoliciesResponse>("/api/policies");
  }

  async attachPolicy({ algorithmId, checkpointPath, actionMode, llmDm }: { algorithmId: AlgorithmId; checkpointPath?: string; actionMode?: PolicyActionMode; llmDm?: LlmDmAttachPayload }): Promise<AttachPolicyResponse> {
    const sessionId = this.requireSession();
    return request<AttachPolicyResponse>(`/api/sessions/${encodeURIComponent(sessionId)}/attach-policy`, {
      method: "POST",
      body: JSON.stringify({
        algorithmId,
        checkpointPath: checkpointPath ?? null,
        actionMode,
        ...(llmDm ? { llmDm } : {}),
      }),
    });
  }

  async detachPolicy(): Promise<void> {
    const sessionId = this.requireSession();
    await request<{ detached: boolean }>(`/api/sessions/${encodeURIComponent(sessionId)}/detach-policy`, {
      method: "POST",
    });
  }

  async llmLearn(): Promise<LlmLearnResponse> {
    const sessionId = this.requireSession();
    return request<LlmLearnResponse>(`/api/sessions/${encodeURIComponent(sessionId)}/llm-learn`, {
      method: "POST",
    });
  }

  async validateCheckpoint(checkpointPath: string): Promise<{ valid: boolean; policyId?: string; loaderId?: string; error?: string; code?: string }> {
    return request("/api/policies/validate-checkpoint", {
      method: "POST",
      body: JSON.stringify({ checkpointPath }),
    });
  }

  async fetchHealth(): Promise<{ rllibAvailable: boolean; muzeroAvailable: boolean; worldModelsAvailable: boolean; llmDmAvailable: boolean; deployed: boolean }> {
    const response = await request<{ rllibAvailable?: boolean; muzeroAvailable?: boolean; worldModelsAvailable?: boolean; llmDmAvailable?: boolean; deployed?: boolean }>("/api/health");
    return {
      rllibAvailable: response.rllibAvailable === true,
      muzeroAvailable: response.muzeroAvailable === true,
      worldModelsAvailable: response.worldModelsAvailable === true,
      llmDmAvailable: response.llmDmAvailable === true,
      deployed: response.deployed === true,
    };
  }

  /** @deprecated Use agentStep after attachPolicy. Kept for manual pre-decoded agent actions if needed. */
  async step(action: SimAction): Promise<SimSnapshot> {
    return this.applyAction(action, "Agent");
  }

  async pause(): Promise<SimSnapshot> {
    return this.sessionRequest("pause");
  }

  async resume(): Promise<SimSnapshot> {
    return this.sessionRequest("resume");
  }

  async saveSession({ fileName, runMode, algorithmId, checkpointPath }: { fileName: string; runMode: Exclude<RunMode, "load">; algorithmId?: AlgorithmId; checkpointPath?: string }): Promise<AresTrajectoryEpisodeV1> {
    const sessionId = this.requireSession();
    return request<AresTrajectoryEpisodeV1>(`/api/sessions/${encodeURIComponent(sessionId)}/save`, {
      method: "POST",
      body: JSON.stringify({ fileName, runMode, algorithmId, checkpointPath: checkpointPath ?? null }),
    });
  }

  /** Upload a trajectory (or legacy gameplay file) and switch into replay mode. */
  async loadReplay(content: string, fileName: string): Promise<ReplayResponse> {
    const compact = compactJsonIfValid(content);
    const compressed = await gzipToBase64(compact);
    const body = compressed !== null
      ? JSON.stringify({ fileName, contentGzipBase64: compressed })
      : JSON.stringify({ fileName, content: compact });
    const response = await request<ReplayResponse>("/api/replays", {
      method: "POST",
      body,
    });
    assertSnapshot(response.snapshot);
    if (!response.gameplay) throw new AresApiError("INVALID_RESPONSE", "Backend replay response did not include gameplay data.", 500);
    this.replayId = response.replayId;
    return response;
  }

  async stepReplay(): Promise<ReplayResponse> {
    return this.replayRequest("step");
  }

  async jumpReplay(step: number): Promise<ReplayResponse> {
    return this.replayRequest("jump", { step });
  }

  async resetReplay(): Promise<ReplayResponse> {
    return this.replayRequest("reset");
  }

  private async applyAction(action: SimAction, actor: "Player" | "Agent"): Promise<SimSnapshot> {
    const sessionId = this.requireSession();
    const compact = this.terrainCache !== null;
    const response = await request<SnapshotResponse>(`/api/sessions/${encodeURIComponent(sessionId)}/actions`, {
      method: "POST",
      body: JSON.stringify({ action, actor, compact }),
    });
    return this.hydrateSnapshot(response.snapshot, true);
  }

  private async sessionRequest(operation: "pause" | "resume"): Promise<SimSnapshot> {
    const sessionId = this.requireSession();
    const response = await request<SnapshotResponse>(`/api/sessions/${encodeURIComponent(sessionId)}/${operation}`, { method: "POST" });
    return this.hydrateSnapshot(response.snapshot, true);
  }

  private hydrateSnapshot(raw: CompactSimSnapshot, updateCache: boolean): SimSnapshot {
    let terrain = raw.terrain;
    if (!Array.isArray(terrain) && Array.isArray(raw.terrainDelta) && this.terrainCache) {
      if (raw.terrainDelta.length === 0) {
        terrain = this.terrainCache;
      } else {
        const nextTerrain = this.terrainCache.slice();
        const clonedRows = new Set<number>();
        for (const cell of raw.terrainDelta) {
          if (cell.y >= 0 && cell.y < nextTerrain.length && cell.x >= 0 && cell.x < nextTerrain[cell.y].length) {
            if (!clonedRows.has(cell.y)) {
              nextTerrain[cell.y] = nextTerrain[cell.y].slice();
              clonedRows.add(cell.y);
            }
            nextTerrain[cell.y][cell.x] = cell;
          }
        }
        terrain = nextTerrain;
      }
    }
    const { terrainDelta: _ignored, ...rest } = raw;
    const hydrated = { ...rest, terrain } as SimSnapshot;
    assertSnapshot(hydrated);
    if (updateCache) {
      this.terrainCache = hydrated.terrain;
    }
    return hydrated;
  }

  private async replayRequest(operation: "step" | "jump" | "reset", body?: object): Promise<ReplayResponse> {
    if (!this.replayId) throw new AresApiError("REPLAY_NOT_LOADED", "Load a gameplay replay first.", 400);
    const response = await request<ReplayResponse>(`/api/replays/${encodeURIComponent(this.replayId)}/${operation}`, {
      method: "POST",
      body: body ? JSON.stringify(body) : undefined,
    });
    assertSnapshot(response.snapshot);
    return response;
  }

  private requireSession(): string {
    if (!this.sessionId) throw new AresApiError("SESSION_NOT_STARTED", "Start an AresSim session first.", 400);
    return this.sessionId;
  }
}

async function request<T>(url: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(apiUrl(url), {
      ...init,
      headers: { "Content-Type": "application/json", ...init.headers },
    });
  } catch {
    throw new AresApiError("BACKEND_UNAVAILABLE", "The AresSim backend is unavailable. Start the Python API and retry.", 0);
  }
  const rawText = await response.text().catch(() => "");
  let payload: ApiErrorPayload | T | null = null;
  if (rawText) {
    try {
      payload = JSON.parse(rawText) as ApiErrorPayload | T;
    } catch {
      payload = null;
    }
  }
  if (!response.ok) {
    const error = payload && typeof payload === "object" && "error" in payload ? payload.error : undefined;
    const fallbackMessage =
      response.status === 413
        ? "The uploaded file is too large for the hosted proxy (4.5 MB compressed limit)."
        : response.status === 504
          ? "The inference server timed out while processing the request."
          : rawText && rawText.length < 180
            ? `Backend error (${response.status}): ${rawText.trim()}`
            : `The backend request failed (HTTP ${response.status}).`;
    throw new AresApiError(error?.code ?? `HTTP_${response.status}`, error?.message ?? fallbackMessage, response.status);
  }
  if (payload === null) throw new AresApiError("INVALID_RESPONSE", "The backend returned an invalid JSON response.", response.status);
  return payload as T;
}

function compactJsonIfValid(raw: string): string {
  try {
    return JSON.stringify(JSON.parse(raw));
  } catch {
    return raw;
  }
}

async function gzipToBase64(text: string): Promise<string | null> {
  if (typeof CompressionStream !== "function") return null;
  try {
    const stream = new Blob([text]).stream().pipeThrough(new CompressionStream("gzip"));
    const bytes = new Uint8Array(await new Response(stream).arrayBuffer());
    const chunks: string[] = [];
    const chunkSize = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkSize) {
      chunks.push(String.fromCharCode(...bytes.subarray(i, i + chunkSize)));
    }
    return btoa(chunks.join(""));
  } catch {
    return null;
  }
}

function assertSnapshot(value: unknown): asserts value is SimSnapshot {
  if (!value || typeof value !== "object") throw new AresApiError("INVALID_RESPONSE", "The backend returned an invalid snapshot.", 500);
  const snapshot = value as Partial<SimSnapshot>;
  if (typeof snapshot.sessionId !== "string" || typeof snapshot.seed !== "number" || !Array.isArray(snapshot.terrain) || !Array.isArray(snapshot.rovers) || !Array.isArray(snapshot.history)) {
    throw new AresApiError("INVALID_RESPONSE", "The backend returned an invalid snapshot.", 500);
  }
}
