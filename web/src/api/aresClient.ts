/**
 * HTTP client for the Python engine REST API.
 * Vite proxies `/api` to `127.0.0.1:8000`. Keep this file transport-only; store
 * logic belongs in `useAresStore`.
 */

import type { AgentStepResponse, AlgorithmId, AttachPolicyResponse, AresReplayProjectionV1, AresTrajectoryEpisodeV1, PoliciesResponse, RunMode, SimAction, SimSnapshot } from "../types/sim";

interface SnapshotResponse {
  snapshot: SimSnapshot;
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

  /** Start or replace the live session. Pass no seed to let the backend choose. */
  async startSession({ seed }: { seed?: number } = {}): Promise<SimSnapshot> {
    const response = await request<SnapshotResponse>("/api/sessions", {
      method: "POST",
      body: JSON.stringify({ seed: seed ?? null }),
    });
    assertSnapshot(response.snapshot);
    this.sessionId = response.snapshot.sessionId;
    this.replayId = null;
    return response.snapshot;
  }

  async randomizeSession(): Promise<SimSnapshot> {
    return this.startSession();
  }

  /** Manual play: send `actor: Player`. */
  async dispatchAction(action: SimAction): Promise<SimSnapshot> {
    return this.applyAction(action, "Player");
  }

  /** Algorithm autoplay: server runs the attached policy and steps as Agent. */
  async agentStep(): Promise<AgentStepResponse> {
    const sessionId = this.requireSession();
    const response = await request<AgentStepResponse>(`/api/sessions/${encodeURIComponent(sessionId)}/agent-step`, {
      method: "POST",
    });
    assertSnapshot(response.snapshot);
    return response;
  }

  async listPolicies(): Promise<PoliciesResponse> {
    return request<PoliciesResponse>("/api/policies");
  }

  async attachPolicy({ algorithmId, checkpointPath }: { algorithmId: AlgorithmId; checkpointPath?: string }): Promise<AttachPolicyResponse> {
    const sessionId = this.requireSession();
    return request<AttachPolicyResponse>(`/api/sessions/${encodeURIComponent(sessionId)}/attach-policy`, {
      method: "POST",
      body: JSON.stringify({
        algorithmId,
        checkpointPath: checkpointPath ?? null,
      }),
    });
  }

  async detachPolicy(): Promise<void> {
    const sessionId = this.requireSession();
    await request<{ detached: boolean }>(`/api/sessions/${encodeURIComponent(sessionId)}/detach-policy`, {
      method: "POST",
    });
  }

  async validateCheckpoint(checkpointPath: string): Promise<{ valid: boolean; policyId?: string; loaderId?: string; error?: string; code?: string }> {
    return request("/api/policies/validate-checkpoint", {
      method: "POST",
      body: JSON.stringify({ checkpointPath }),
    });
  }

  async fetchHealth(): Promise<{ rllibAvailable: boolean }> {
    const response = await request<{ rllibAvailable?: boolean }>("/api/health");
    return { rllibAvailable: response.rllibAvailable === true };
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
    const response = await request<ReplayResponse>("/api/replays", {
      method: "POST",
      body: JSON.stringify({ fileName, content }),
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
    const response = await request<SnapshotResponse>(`/api/sessions/${encodeURIComponent(sessionId)}/actions`, {
      method: "POST",
      body: JSON.stringify({ action, actor }),
    });
    assertSnapshot(response.snapshot);
    return response.snapshot;
  }

  private async sessionRequest(operation: "pause" | "resume"): Promise<SimSnapshot> {
    const sessionId = this.requireSession();
    const response = await request<SnapshotResponse>(`/api/sessions/${encodeURIComponent(sessionId)}/${operation}`, { method: "POST" });
    assertSnapshot(response.snapshot);
    return response.snapshot;
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

async function request<T>(url: string, init: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, {
      ...init,
      headers: { "Content-Type": "application/json", ...init.headers },
    });
  } catch {
    throw new AresApiError("BACKEND_UNAVAILABLE", "The AresSim backend is unavailable. Start the Python API and retry.", 0);
  }
  const payload = await response.json().catch(() => null) as ApiErrorPayload | T | null;
  if (!response.ok) {
    const error = payload && typeof payload === "object" && "error" in payload ? payload.error : undefined;
    throw new AresApiError(error?.code ?? "BACKEND_ERROR", error?.message ?? "The backend request failed.", response.status);
  }
  if (payload === null) throw new AresApiError("INVALID_RESPONSE", "The backend returned an invalid JSON response.", response.status);
  return payload as T;
}

function assertSnapshot(value: unknown): asserts value is SimSnapshot {
  if (!value || typeof value !== "object") throw new AresApiError("INVALID_RESPONSE", "The backend returned an invalid snapshot.", 500);
  const snapshot = value as Partial<SimSnapshot>;
  if (typeof snapshot.sessionId !== "string" || typeof snapshot.seed !== "number" || !Array.isArray(snapshot.terrain) || !Array.isArray(snapshot.rovers) || !Array.isArray(snapshot.history)) {
    throw new AresApiError("INVALID_RESPONSE", "The backend returned an invalid snapshot.", 500);
  }
}
