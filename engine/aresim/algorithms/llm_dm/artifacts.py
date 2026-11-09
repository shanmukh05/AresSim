"""Save essential run results, compact game/Learner records and one replay per game.

Normal output omits duplicate packets, prompt copies and per-request files. Exact
model inputs and lifecycle events are available in one optional compressed trace.
The shared trajectory owner continues to serialize lossless UI replays.
"""

from __future__ import annotations

import gzip
import json
import os
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ...training.trajectories import EpisodeTrajectory, ReplayDirectoryWriter
from ...gameplay import TrajectoryRecorder
from .contracts import Json, canonical, digest, json_value
from .reporting import compact_turn, render_summary


def atomic_json(path: Path, value: object) -> None:
    """Publish readable JSON atomically so interruption cannot leave half a manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


class RunArtifacts:
    """Five primary files plus one portable replay per completed game."""

    def __init__(
        self, directory: Path, run_id: str, *, debug_trace: bool = False
    ) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        self.directory, self.run_id, self.debug_trace = directory, run_id, debug_trace
        self.manifest: Json = {}
        self.games: dict[str, Json] = {}
        self.progress: dict[str, list[Json]] = {}
        self.learning: list[Json] = []
        self.replays: list[Json] = []
        self.versions: dict[str, Json] = {}
        self._roles: dict[str, Json] = {}
        self._latencies: dict[str, list[float]] = {}
        self._content_ids: dict[str, str] = {}

    def write(self, relative: str, value: object) -> str:
        atomic_json(self.directory / relative, value)
        return relative

    def record(self, relative: str, record: Json) -> str:
        """Append an essential record and return its stable selector."""
        identity = digest(record)
        value = {
            "recorded_at": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            **record,
            "record_id": identity,
        }
        path = self.directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(canonical(value) + "\n")
        return f"{relative}#record_id={identity}"

    def trace(self, category: str, value: object) -> None:
        """Opt in to exact input/output debugging in a single compressed stream."""
        if self.debug_trace:
            with gzip.open(
                self.directory / "debug.jsonl.gz", "at", encoding="utf-8"
            ) as stream:
                stream.write(canonical({"category": category, "value": value}) + "\n")

    def content(self, category: str, value: object) -> str:
        """Keep Learner evidence once; other exact content is debug-only and digest-addressed."""
        identity = digest(value)
        if identity not in self._content_ids:
            reference = f"sha256:{identity}"
            if category == "evidence":
                reference = self.record(
                    "learning.jsonl",
                    {"event": "evidence", "content_id": identity, "value": value},
                )
            else:
                self.trace(category, {"content_id": identity, "value": value})
            self._content_ids[identity] = reference
        return self._content_ids[identity]

    def remember_bundle(self, bundle: Json) -> None:
        """Keep versioned knowledge without repeating shared prompts and configuration."""
        self.versions[bundle["bundle_id"]] = {
            key: deepcopy(bundle[key])
            for key in ("bundle_id", "parent_bundle_id", "knowledge")
        }

    def select_bundle(self, bundle: Json) -> None:
        self.remember_bundle(bundle)
        self.write("selected-bundle.json", bundle)

    def set_manifest(self, manifest: Json) -> None:
        self.manifest = deepcopy(manifest)
        self.flush()

    def flush(self, *, summary: bool = True) -> None:
        """Refresh the two entry points without multiplying files or copying full traces."""
        calls = deepcopy(self._roles)
        for role, values in self._latencies.items():
            if values:
                calls[role]["latency_seconds"] = {
                    "mean": float(np.mean(values)),
                    "p50": float(np.quantile(values, 0.5)),
                    "p95": float(np.quantile(values, 0.95)),
                }
        self.manifest.update(
            output_format="aresim.llm.essential.v1",
            debug_trace=self.debug_trace,
            games=list(self.games.values()),
            replays=self.replays,
            model_calls=calls,
            policy_versions=list(self.versions.values()),
        )
        self.write("manifest.json", self.manifest)
        if summary:
            partial = self.directory / "summary.log.partial"
            partial.write_text(
                render_summary(self.manifest, self.learning, self.progress),
                encoding="utf-8",
            )
            os.replace(partial, self.directory / "summary.log")

    def budget_updated(self, budget: Json) -> None:
        self.manifest["budget"] = deepcopy(budget)
        self.flush()

    def call_record(self, event: Json) -> None:
        """Aggregate successful calls; retain actionable errors and optional exact trace."""
        self.trace("model_call", event)
        if event["event"] != "attempt_completed":
            return
        role, result = event["role"], event["result"]
        stats = self._roles.setdefault(
            role,
            {
                "attempts": 0,
                "ok": 0,
                "errors": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_input_tokens": 0,
                "cache_write_input_tokens": 0,
                "reasoning_output_tokens": 0,
                "unknown_usage_attempts": 0,
            },
        )
        stats["attempts"] += 1
        stage = stats.setdefault("stages", {}).setdefault(
            event.get("request_kind", role), {"attempts": 0, "input_bytes": 0}
        )
        stage["attempts"] += 1
        stage["input_bytes"] += event.get("input_bytes", 0)
        stats["ok" if result["status"] == "ok" else "errors"] += 1
        models = stats.setdefault("reported_models", {})
        model = result.get("model") or "unknown"
        models[model] = models.get(model, 0) + 1
        usage = result.get("usage")
        transport = (result.get("details") or {}).get("transport", {})
        if "request_bytes" in transport:
            stage["wire_bytes"] = (
                stage.get("wire_bytes", 0) + transport["request_bytes"]
            )
        if usage:
            for key in ("input_tokens", "output_tokens"):
                stats[key] += usage[key]
            details = usage.get("provider_details", {})
            incoming, outgoing = (
                details.get("input_tokens_details", {}),
                details.get("output_tokens_details", {}),
            )
            stats["cached_input_tokens"] += incoming.get("cached_tokens", 0)
            stats["cache_write_input_tokens"] += incoming.get("cache_write_tokens", 0)
            stats["reasoning_output_tokens"] += outgoing.get("reasoning_tokens", 0)
        else:
            stats["unknown_usage_attempts"] += 1
        self._latencies.setdefault(role, []).append(event["elapsed_seconds"])
        self.manifest["budget"] = deepcopy(event["budget"])
        if result["status"] != "ok":
            self.record(
                "learning.jsonl" if role == "learner" else "games.jsonl",
                {
                    "event": "model_error",
                    "role": role,
                    "context": event.get("context"),
                    "attempt_id": event["attempt_id"],
                    "status": result["status"],
                    "error_code": result.get("error_code"),
                    "retryable": result.get("retryable"),
                    "provider_request_id": result.get("request_id"),
                },
            )
        self.flush(summary=False)

    def decision_record(self, event: Json) -> None:
        self.trace("decision", event)
        if event["event"] == "decision_unexecuted":
            self.record("games.jsonl", event)

    def episode_record(self, episode_id: str, event: Json) -> None:
        """Save compact telemetry per real turn; full observations remain in the replay."""
        self.trace("game", {"episode_id": episode_id, **event})
        kind = event["event"]
        if kind == "reset":
            self.games[episode_id] = {
                "episode_id": episode_id,
                "source_episode_id": event["source_episode_id"],
                "phase": event["phase"],
                "bundle_id": event["bundle_id"],
                "status": "running",
            }
            self.progress[episode_id] = []
        game = self.games[episode_id]
        row = {
            "event": kind,
            "episode_id": episode_id,
            "phase": game["phase"],
            "source_episode_id": game["source_episode_id"],
            "step_index": event.get("step_index", 0),
        }
        if kind == "transition":
            row.update(compact_turn(event))
            self.progress[episode_id].append(row)
        elif kind == "reset":
            state = event.get("state")
            row["initial_rover"] = deepcopy(state["rover"]) if state else None
        else:
            row.update(
                {
                    k: event[k]
                    for k in ("ending_reason", "terminated", "truncated")
                    if k in event
                }
            )
        self.record("games.jsonl", row)
        if kind == "reset":
            self.flush()

    def game_summary(self, summary: Json) -> None:
        """Keep one structured result and render meaningful progression for humans."""
        value = deepcopy(summary)
        metrics = value["metrics"]
        latencies = metrics.pop("latencies", [])
        if latencies:
            metrics["latency_seconds"] = {
                "p50": float(np.quantile(latencies, 0.5)),
                "p95": float(np.quantile(latencies, 0.95)),
            }
        value.pop("timeline", None)
        value.pop("events", None)
        value["status"] = "closed"
        self.games[value["episode_id"]] = value
        self.flush()

    def learning_record(self, record: Json) -> str:
        reference = self.record("learning.jsonl", record)
        self.learning.append(deepcopy(record))
        self.flush()
        return reference


class ReplayWriter(ReplayDirectoryWriter):
    """Continuously publish native game JSON and index it in the LLM run manifest."""

    def __init__(self, artifacts: RunArtifacts, phase: str) -> None:
        self.artifacts, self.phase = artifacts, phase
        super().__init__(artifacts.directory, prefix=f"{phase}--")

    def _sync(self, *, summary: bool = True) -> None:
        for entry in self.entries.values():
            value = {"phase": self.phase, **entry}
            existing = next(
                (r for r in self.artifacts.replays if r["path"] == entry["path"]), None
            )
            if existing is None:
                self.artifacts.replays.append(value)
            else:
                existing.update(value)
            for game in self.artifacts.games.values():
                if (
                    game["phase"] == self.phase
                    and game["source_episode_id"] == entry["source_episode_id"]
                ):
                    game["environment_seed"] = entry["environment_seed"]
        self.artifacts.flush(summary=summary)

    def write_progress(self, recorder: TrajectoryRecorder) -> None:
        super().write_progress(recorder)
        self._sync(summary=False)

    def write_episode(self, episode: EpisodeTrajectory) -> None:
        super().write_episode(episode)
        self._sync()

    def finalize(self, *, max_episode_steps: int) -> Path:
        self.artifacts.flush()
        return self.artifacts.directory / "manifest.json"
