/**
 * Vercel serverless proxy between the AresSim UI (`/aressim/api/*`) and the
 * proxy-auth-protected Modal FastAPI backend (`requires_proxy_auth=True`).
 * Keeps `Modal-Key` and `Modal-Secret` strictly server-side in Vercel env vars.
 */

import { gzipSync } from "node:zlib";

const ALLOWED_PATH = /^\/api\/(health|policies(\/validate-checkpoint)?|sessions(\/[A-Za-z0-9_-]+(\/(actions|actions-batch|pause|resume|save|attach-policy|detach-policy|agent-step|agent-steps|llm-learn))?)?|replays(\/[A-Za-z0-9_-]+\/(step|jump|reset))?)$/;

function resolveApiPath(req) {
  const raw = req.query?.path;
  if (Array.isArray(raw) && raw.length > 0) {
    return `/api/${raw.join("/").replace(/^\/+/, "")}`;
  }
  if (typeof raw === "string" && raw.trim()) {
    return `/api/${raw.trim().replace(/^\/+/, "")}`;
  }
  const url = new URL(req.url || "/", "http://localhost");
  const fromQuery = url.searchParams.get("path");
  if (fromQuery) {
    return `/api/${fromQuery.replace(/^\/+/, "")}`;
  }
  return url.pathname.replace(/^\/aressim(?=\/api\/)/, "");
}

export default async function handler(req, res) {
  res.setHeader("Cache-Control", "no-store");

  if (req.method !== "GET" && req.method !== "POST") {
    res.setHeader("Allow", "GET, POST");
    return res.status(405).json({ error: { code: "METHOD_NOT_ALLOWED", message: "Method not allowed." } });
  }

  const base = (process.env.ARESSIM_MODAL_URL || "").trim().replace(/\/+$/, "");
  const key = (process.env.ARESSIM_MODAL_KEY || "").trim();
  const secret = (process.env.ARESSIM_MODAL_SECRET || "").trim();
  if (!base || !key || !secret) {
    return res.status(503).json({
      error: { code: "BACKEND_NOT_CONFIGURED", message: "AresSim inference backend credentials are not configured." },
    });
  }

  const apiPath = resolveApiPath(req);
  if (!ALLOWED_PATH.test(apiPath)) {
    return res.status(404).json({ error: { code: "NOT_FOUND", message: `Unknown AresSim API endpoint: ${apiPath}` } });
  }

  let body;
  if (req.method === "POST") {
    if (req.body === undefined || req.body === null || req.body === "") {
      body = undefined;
    } else if (typeof req.body === "string") {
      body = req.body;
    } else {
      body = JSON.stringify(req.body);
    }
  }

  try {
    const upstream = await fetch(`${base}${apiPath}`, {
      method: req.method,
      headers: {
        "Content-Type": "application/json",
        "Modal-Key": key,
        "Modal-Secret": secret,
      },
      body,
      redirect: "error",
      signal: AbortSignal.timeout(55000),
    });

    const text = await upstream.text();
    const contentType = upstream.headers.get("content-type") || "application/json";
    res.status(upstream.status).setHeader("Content-Type", contentType);
    const acceptEncoding = String(req.headers["accept-encoding"] || "");
    if (acceptEncoding.includes("gzip") && text.length > 1024) {
      const compressed = gzipSync(Buffer.from(text, "utf-8"));
      res.setHeader("Content-Encoding", "gzip");
      res.setHeader("Vary", "Accept-Encoding");
      res.send(compressed);
      return;
    }
    res.send(text);
  } catch (error) {
    const timedOut = error?.name === "TimeoutError" || error?.name === "AbortError";
    res.status(timedOut ? 504 : 502).json({
      error: {
        code: timedOut ? "UPSTREAM_TIMEOUT" : "UPSTREAM_UNAVAILABLE",
        message: timedOut
          ? "The inference server took too long to respond."
          : "Could not reach the AresSim inference server.",
      },
    });
  }
}
