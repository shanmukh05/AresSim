#!/usr/bin/env python3
"""Compare LLM Actor+Learner pairs through AresSim's existing five-game learning pipeline.

Materialize an independent config per model, preflight the whole panel, and retain
normal run artifacts. Read native replay statistics for the comparison CSV; never
send evaluator-only replay state to either model. No paid calls occur in dry-run/fake mode.
The same report directory receives the metric bars and the shared action-path figure.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from itertools import count
import json
import logging
from pathlib import Path
import re
from statistics import fmean, pstdev
import sys

import yaml

from aresim.algorithms.common.config_decode import mapping
from aresim.algorithms.common.learnings import resolve_learnings_file
from aresim.algorithms.llm_dm.budget import BudgetLedger
from aresim.algorithms.llm_dm.config import (
    RunConfig,
    load_llm_dm_run,
    load_role_credentials,
    validate_experiment_name,
)
from aresim.algorithms.llm_dm.contracts import digest, json_value
from aresim.algorithms.llm_dm.providers import make_provider
from aresim.algorithms.llm_dm.rollout import run_llm_dm

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "aresim.llm_dm.model_panel.v1"
CREDENTIALS = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "fake": "",
}
METRICS = (
    "shaped_return",
    "engine_return",
    "sparse_evaluation_return",
    "length",
    "ice_collected",
    "ice_delivered",
    "samples_collected",
    "samples_delivered",
    "success",
)
CSV_FIELDS = (
    "model_name",
    "provider",
    "model",
    "episode_id",
    "environment_seed",
    "agent_seed",
    "recording_complete",
    "length",
    "shaped_return",
    "engine_return",
    "sparse_evaluation_return",
    "ice_collected",
    "ice_delivered",
    "samples_collected",
    "samples_delivered",
    "success",
    "terminated",
    "truncated",
    "ending_reason",
    "extract_opportunities",
    "fallbacks",
    "learning_statement_count",
    "learning_context_decisions",
    "history_budget_reductions",
    "action_counts",
    "effective_action_counts",
    "task_id",
    "reward_profile",
    "policy_bundle_id",
    "run_manifest",
    "replay_path",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelRun:
    name: str
    config_path: Path
    config: RunConfig

    @property
    def manifest(self) -> Path:
        return self.config.output_directory / self.name / "manifest.json"


def existing_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    for candidate in (base / path, ROOT / path, path):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(value)


def allocate_directory(base: Path, name: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    for index in count(1):
        path = base / (name if index == 1 else f"{name}-{index:03d}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue


def model_name(model: dict) -> str:
    if "name" in model:
        return validate_experiment_name(model["name"])
    value = re.sub(
        r"[^A-Za-z0-9_-]+", "-", f"{model['provider']}-{model['model']}"
    ).strip("-_")
    if len(value) > 80:
        value = value[:65] + "-" + digest(model)[:12]
    return validate_experiment_name(value)


def prepare(
    source: Path, args: argparse.Namespace
) -> tuple[Path, dict, list[ModelRun], Path | None]:
    panel = mapping(yaml.safe_load(source.read_text()), "LLM model panel")
    allowed = {
        "schema_version",
        "experiment_name",
        "output_directory",
        "env_file",
        "models",
        "run",
    }
    if set(panel) - allowed or panel.get("schema_version") != SCHEMA:
        raise ValueError("unsupported LLM model panel schema or unknown fields")
    run = mapping(panel.get("run"), "run")
    episodes = run.get("episodes", [])
    if not isinstance(episodes, list) or len(episodes) != 5:
        raise ValueError("run.episodes must declare exactly five shared games")
    if len({row["environment_seed"] for row in episodes}) != 5:
        raise ValueError("the five environment seeds must be distinct")
    if run.get("policy_bundle"):
        raise ValueError(
            "model comparison starts fresh; policy_bundle is not supported"
        )
    models = panel.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a nonempty provider/model list")
    selected = set(args.model or [])
    chosen, matched, seen = [], set(), set()
    for raw in models:
        item = mapping(raw, "model")
        extra = set(item) - {
            "name",
            "provider",
            "model",
            "credential_env",
            "route",
            "generation",
            "actor",
            "learner",
            "pricing",
        }
        if (
            extra
            or item.get("provider") not in CREDENTIALS
            or not isinstance(item.get("model"), str)
            or not item["model"].strip()
        ):
            raise ValueError(
                "each model requires an LLM provider and nonempty model ID; Decision Models are excluded"
            )
        name = model_name(item)
        keys = {name, f"{item['provider']}/{item['model']}"}
        if selected and not keys & selected:
            continue
        matched.update(keys & selected)
        if name.casefold() in seen:
            raise ValueError(
                f"duplicate model name: {name}; use distinct explicit names for variants"
            )
        seen.add(name.casefold())
        chosen.append((name, item))
    if selected - matched:
        raise ValueError(
            f"unknown --model selection: {', '.join(sorted(selected - matched))}"
        )
    output = (args.output or ROOT / panel["output_directory"]).expanduser().resolve()
    directory = allocate_directory(
        output, validate_experiment_name(panel["experiment_name"])
    )
    (directory / "configs").mkdir()
    price_source = existing_path(run["budgets"]["price_sheet"], source.parent)
    prices = json.loads(price_source.read_text())
    for _, item in chosen:
        if "pricing" in item:
            key = f"{item['provider']}/{item['model']}" + (
                f"@{item['route']}" if item.get("route") else ""
            )
            prices["models"][key] = mapping(item["pricing"], "pricing")
    prices_path = directory / "prices.json"
    prices_path.write_text(json.dumps(prices, indent=2) + "\n")
    jobs = []
    for name, item in chosen:
        payload = deepcopy(run)
        payload.update(
            schema_version="aresim.llm_dm.run.v1",
            agent="llm_dm",
            experiment_name=name,
            output_directory=str(directory / "runs"),
            execution_mode="sequential_diagnostic",
            split="diagnostic",
        )
        if args.fake:
            payload["transport"] = "fake"
        if args.max_episode_steps is not None:
            payload["max_episode_steps"] = args.max_episode_steps
        payload.pop("policy_bundle", None)
        if payload.get("learnings_file") is not None:
            payload["learnings_file"] = str(
                resolve_learnings_file(payload["learnings_file"], source.parent)
            )
        payload["budgets"]["price_sheet"] = str(prices_path)
        for role in ("actor", "learner"):
            values = mapping(payload.get(role), role)
            values.pop("evaluation_manifest", None)
            values.pop("evaluation_episodes", None)
            values.update(
                generation=deepcopy(item.get("generation", {})), route=item.get("route")
            )
            overrides = mapping(item.get(role, {}), role + " overrides")
            if set(overrides) & {
                "provider",
                "model",
                "credential_env",
                "output_mode",
                "batch_episodes",
                "max_rounds",
                "enabled",
                "evaluation_episodes",
                "evaluation_manifest",
            }:
                raise ValueError(
                    f"{name}: role overrides cannot change model identity or learning schedule"
                )
            values.update(overrides)
            values.update(
                provider=item["provider"],
                model=item["model"],
                output_mode="native_schema",
                credential_env=item.get(
                    "credential_env", CREDENTIALS[item["provider"]]
                ),
            )
            if values.get("prompt_path"):
                values["prompt_path"] = str(
                    existing_path(values["prompt_path"], source.parent)
                )
            if role == "learner":
                values.update(enabled=True, batch_episodes=1, max_rounds=4)
            payload[role] = values
        path = directory / "configs" / f"{name}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False))
        jobs.append(ModelRun(name, path, load_llm_dm_run(path)))
    configured_env = args.env_file or panel.get("env_file")
    env_file = (
        existing_path(str(configured_env), source.parent)
        if configured_env
        and not args.dry_run
        and any(job.config.transport == "live" for job in jobs)
        else None
    )
    # Preflight every selected model before the first paid call.
    for job in jobs:
        if env_file:
            load_role_credentials(job.config, env_file)
        ledger = BudgetLedger(job.config.budgets, fake=job.config.transport == "fake")
        for role in (job.config.actor, job.config.learner.role):
            ledger.price(role)
            if not args.dry_run:
                provider = make_provider(role, fake=job.config.transport == "fake")
                try:
                    provider.preflight()
                finally:
                    provider.close()
    return directory, panel, jobs, env_file


def episode_rows(job: ModelRun, manifest: dict, directory: Path) -> list[dict]:
    rows = []
    for entry in manifest.get("replays", []):
        planned = next(
            episode
            for episode in job.config.episodes
            if episode.episode_id == entry["source_episode_id"]
        )
        path = job.manifest.parent / entry["path"]
        saved = json.loads(path.read_text())
        policy, replay = saved.get("policy"), saved["replay"]
        game = next(
            (
                g
                for g in manifest.get("games", [])
                if g["phase"] == entry["phase"]
                and g["source_episode_id"] == entry["source_episode_id"]
            ),
            {},
        )
        metrics, outcome = game.get("metrics", {}), game.get("outcome", {})
        final = replay.get("finalSnapshot", replay["initialSnapshot"])
        stats = final["objectiveStats"]
        row = {
            "model_name": job.name,
            "provider": job.config.actor.provider,
            "model": job.config.actor.model,
            "episode_id": entry["source_episode_id"],
            "environment_seed": saved.get("metadata", {}).get(
                "environmentSeed", final.get("seed")
            ),
            "agent_seed": planned.agent_seed,
            "recording_complete": bool(policy),
            "length": policy["length"] if policy else len(replay["steps"]),
            "shaped_return": policy["episode_return"] if policy else None,
            "engine_return": policy["engine_return"] if policy else None,
            "sparse_evaluation_return": sum(
                float(b.get("terms", {}).get(k, {}).get("value", 0))
                for b in policy["reward_breakdowns"]
                for k in ("mission_success", "terminal_failure", "invalid_action")
            )
            if policy
            else None,
            "ice_collected": stats["iceCollected"],
            "ice_delivered": stats["iceDelivered"],
            "samples_collected": stats["samplesCollected"],
            "samples_delivered": stats["samplesDelivered"],
            "success": outcome.get("ending_reason") == "mission_complete",
            "terminated": outcome.get("terminated"),
            "truncated": outcome.get("truncated"),
            "ending_reason": outcome.get(
                "ending_reason", manifest.get("ending_reason")
            ),
            "extract_opportunities": sum(
                bool(mask[6]) for mask in policy["action_masks"][:-1]
            )
            if policy
            else None,
            "action_counts": metrics.get("action_counts"),
            "effective_action_counts": dict(Counter(policy["effective_actions"]))
            if policy
            else None,
            "task_id": policy["task_id"] if policy else None,
            "reward_profile": policy["reward_profile"] if policy else None,
            "policy_bundle_id": game.get("bundle_id"),
            "run_manifest": str(job.manifest.relative_to(directory)),
            "replay_path": str(path.relative_to(directory)),
        }
        row.update(
            {
                key: metrics.get(key)
                for key in (
                    "fallbacks",
                    "learning_statement_count",
                    "learning_context_decisions",
                    "history_budget_reductions",
                )
            }
        )
        rows.append(row)
    return rows


def model_result(
    job: ModelRun, manifest: dict, rows: list[dict], directory: Path
) -> dict:
    aggregate = {}
    for key in METRICS:
        values = [
            float(row[key])
            for row in rows
            if row["recording_complete"] and row.get(key) is not None
        ]
        aggregate[key] = {
            "n": len(values),
            "mean": fmean(values) if values else None,
            "std": pstdev(values) if values else None,
            "total": sum(values) if values else None,
        }
    return {
        "name": job.name,
        "provider": job.config.actor.provider,
        "model": job.config.actor.model,
        "status": manifest.get("status", "failed"),
        "ending_reason": manifest.get("ending_reason"),
        "scheduled_games": 5,
        "recorded_games": len(rows),
        "recording_complete_games": sum(row["recording_complete"] for row in rows),
        "resolved_config": str(job.config_path.relative_to(directory)),
        "run_manifest": str(job.manifest.relative_to(directory))
        if job.manifest.exists()
        else None,
        "budget": manifest.get("budget"),
        "model_calls": manifest.get("model_calls"),
        "learning_cycles": manifest.get("learning_cycles", []),
        "metrics": aggregate,
    }


def save_reports(directory: Path, summary: dict, rows: list[dict]) -> None:
    partial = directory / "summary.json.partial"
    partial.write_text(
        json.dumps(json_value(summary), indent=2, allow_nan=False) + "\n"
    )
    partial.replace(directory / "summary.json")
    partial = directory / "episodes.csv.partial"
    with partial.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, dict)
                    else value
                    for key, value in row.items()
                }
            )
    partial.replace(directory / "episodes.csv")
    write_figures(directory, summary, rows)


def write_figures(directory: Path, summary: dict, rows: list[dict]) -> None:
    """Write metric bars and one shared action-path figure beside the comparison CSV."""
    recorded = [row for row in rows if row.get("recording_complete") in (True, "True") and row.get("replay_path")]
    algorithms = []
    labels = {}
    for model in summary.get("models", []):
        metrics = model.get("metrics") or {}
        means = {key: (metrics.get(key) or {}).get("mean") for key in METRICS}
        if means.get("length") is None:
            continue
        recorded_games = model.get("recorded_games")
        scheduled = model.get("scheduled_games")
        label = model.get("model") or model["name"]
        if recorded_games is not None and scheduled is not None:
            label = f"{label} ({recorded_games}/{scheduled})"
        labels[model["name"]] = label
        algorithms.append({
            "name": label,
            "means": {**means, "sparse_return": means["sparse_evaluation_return"]},
        })
    write_learning_figures(directory)
    if not algorithms:
        return
    from aresim.training.panel import _write_action_paths, _write_figures

    figure_dir = directory / "figures"
    _write_figures(figure_dir, {
        "algorithms": algorithms,
        "episode_count": summary.get("games_per_model", len(recorded)),
        "split": "diagnostic",
        "action_mode": "sequential",
    })
    action_rows = []
    for row in recorded:
        replay = json.loads((directory / row["replay_path"]).read_text(encoding="utf-8"))
        actions = replay.get("policy", {}).get("actions")
        if isinstance(actions, list) and actions:
            action_rows.append({"name": labels.get(row["model_name"], row["model_name"]), "actions": actions})
    if action_rows:
        _write_action_paths(figure_dir / "action_paths.png", action_rows)


_ENDING = {
    "mission_complete": "mission complete",
    "battery_depleted": "battery empty",
    "max_episode_steps": "time limit",
    "goal_selection_failed": "stopped",
}
_ENDING_COLOR = {
    "mission_complete": "#1b7f6b",
    "battery_depleted": "#c44536",
    "max_episode_steps": "#c48a1a",
    "goal_selection_failed": "#5c6770",
}
_MODEL_COLOR = ("#3b6ea5", "#e07a3d", "#2a9d8f", "#6d597a", "#b56576")


def _clip(text: str, limit: int) -> str:
    words = text.split()
    if not words:
        return ""
    kept = []
    for word in words:
        trial = " ".join([*kept, word])
        if len(trial) > limit:
            break
        kept.append(word)
    clipped = " ".join(kept) or words[0][:limit]
    return clipped if clipped == " ".join(words) else clipped + "…"


def _plain_sentence(text: str) -> str:
    cleaned = re.sub(r"\b(?:lesson\s+)?[0-9a-f]{8,}\b", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;.")
    sentence = re.split(r"(?<=[.!])\s", cleaned, maxsplit=1)[0]
    return _clip(sentence, 220)


def _no_change_text(proposal: dict) -> str:
    benefit = _plain_sentence(str(proposal.get("expected_benefit") or ""))
    if benefit and "transition" not in benefit.lower():
        return benefit
    return "No new lesson. The review kept the advice already written."


def _learning_rows(directory: Path) -> list[dict]:
    summary_path = directory / "summary.json"
    episodes_path = directory / "episodes.csv"
    if not summary_path.is_file() or not episodes_path.is_file():
        return []
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with episodes_path.open(encoding="utf-8") as stream:
        grouped: dict[str, list[dict]] = {}
        for row in csv.DictReader(stream):
            grouped.setdefault(row["model_name"], []).append(row)
    models = []
    for index, model in enumerate(summary.get("models", [])):
        reviews = {}
        path = directory / "runs" / model["name"] / "learning.jsonl"
        if path.is_file():
            for line in path.open(encoding="utf-8"):
                event = json.loads(line)
                if event.get("event") == "learner_proposed":
                    item = event["candidate_item"]
                    reviews[int(event["batch"])] = ("lesson", item["text"], item.get("conditions") or [])
                elif event.get("event") == "learner_no_change":
                    reviews[int(event["batch"])] = ("no_change", _no_change_text(event.get("proposal") or {}))
        games = []
        for row in grouped.get(model["name"], []):
            games.append({
                "game": int(str(row["episode_id"]).rsplit("-", 1)[-1]),
                "ice": float(row["ice_delivered"]),
                "samples": float(row["samples_delivered"]),
                "shaped": float(row["shaped_return"]),
                "success": str(row["success"]) in {"True", "true", "1"},
                "ending": row["ending_reason"],
                "statements": int(float(row["learning_statement_count"] or 0)),
                "advice": int(float(row["learning_context_decisions"] or 0)),
            })
        models.append({
            "name": model.get("model") or model["name"],
            "color": _MODEL_COLOR[index % len(_MODEL_COLOR)],
            "games": games,
            "reviews": reviews,
        })
    return models


def write_learning_figures(directory: Path) -> None:
    """Plot per-game outcomes and the lesson each model wrote between those games."""
    models = _learning_rows(directory)
    if not models or not any(model["games"] for model in models):
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = directory / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    _write_learning_outcomes(figure_dir / "learning_by_game.png", models)
    _write_lesson_board(figure_dir / "what_they_learned.png", models)
    plt.close("all")


def _write_learning_outcomes(path: Path, models: list[dict]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    games = range(1, 6)
    metrics = (
        ("ice", "Ice delivered (kg)"),
        ("samples", "Samples delivered (kg)"),
        ("shaped", "Shaped return"),
        ("advice", "Decisions shown advice"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 7.4))
    width = 0.24
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * width
    for axis, (key, title) in zip(axes.flat, metrics, strict=True):
        for offset, model in zip(offsets, models, strict=True):
            by_game = {game["game"]: game for game in model["games"]}
            values = [by_game[game][key] if game in by_game else np.nan for game in games]
            bars = axis.bar(np.arange(1, 6) + offset, values, width=width, color=model["color"], label=model["name"])
            if key == "ice":
                for bar, game_number in zip(bars, games, strict=True):
                    game = by_game.get(game_number)
                    if game and game["success"]:
                        axis.plot(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.45, marker="*", color="#111", markersize=9, linestyle="None")
        axis.set_title(title)
        axis.set_xticks(list(games))
        axis.set_xlabel("game")
        axis.grid(axis="y", linestyle=":", alpha=0.6)
        axis.set_axisbelow(True)
        axis.margins(y=0.18)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(models), frameon=False, bbox_to_anchor=(0.5, 1.02))
    figure.suptitle("Same five worlds, played in order. A star marks a completed mission.", y=1.08)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _write_lesson_board(path: Path, models: list[dict]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import textwrap

    columns = [_model_lessons(model) for model in models]
    row_count = max(len(column) for column in columns)
    figure, axes = plt.subplots(row_count, len(models), figsize=(4.8 * len(models), 2.55 * row_count + 0.55))
    if row_count == 1:
        axes = [axes]
    for column, (model, lessons) in enumerate(zip(models, columns, strict=True)):
        for row in range(row_count):
            axis = axes[row][column]
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.axis("off")
            if row == 0:
                axis.set_title(model["name"], color=model["color"], fontsize=14, fontweight="bold", pad=10)
            if row >= len(lessons):
                continue
            lesson = lessons[row]
            axis.add_patch(FancyBboxPatch(
                (0.04, 0.08), 0.92, 0.84, boxstyle="round,pad=0.012,rounding_size=0.04",
                facecolor=model["color"], edgecolor="none", alpha=0.12,
            ))
            body = textwrap.fill(lesson["summary"], 38)
            axis.text(0.5, 0.58, body, ha="center", va="center", fontsize=12, color="#1f2933")
            axis.text(0.5, 0.16, f"after game {lesson['after']}", ha="center", va="center", fontsize=9, color="#667085")
    for row in range(row_count):
        axes[row][0].text(
            -0.06, 0.5, str(row + 1), transform=axes[row][0].transAxes,
            ha="right", va="center", fontsize=16, fontweight="bold", color="#98a2b3", clip_on=False,
        )
    figure.suptitle("What each model learned", fontsize=16)
    figure.tight_layout(rect=(0.03, 0, 1, 0.94))
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _model_lessons(model: dict) -> list[dict]:
    lessons = []
    for batch in sorted(model["reviews"]):
        review = model["reviews"][batch]
        if review[0] != "lesson":
            continue
        _kind, text, conditions = review
        lessons.append({"after": batch + 1, "summary": _short_lesson(text, conditions)})
    return lessons


def _condition(conditions: list, field: str) -> tuple[str, object] | None:
    for item in conditions:
        if item.get("field") == field:
            return str(item.get("operator")), item.get("value")
    return None


def _mission_state(conditions: list, name: str) -> str:
    found = _condition(conditions, f"mission.{name}_remaining_fraction")
    if found is None:
        return ""
    operator, value = found
    if operator in {"eq", "=="} and float(value) == 0:
        return "done"
    if operator in {"gt", "gte", ">", ">="}:
        return "open"
    return ""


def _short_lesson(text: str, conditions: list) -> str:
    """Restate one lesson the way a person would say it."""
    ice = _mission_state(conditions, "ice")
    samples = _mission_state(conditions, "samples")
    place = _condition(conditions, "rover.pad_proximity")
    on_pad = place is not None and place[1] == "on_build_pad"
    off_pad = place is not None and place[1] == "outside_service_range"
    lowered = text.lower()
    if samples == "done" and ice == "open" and "build" in lowered:
        return "Samples are done, ice isn't. Go get the ice and bring it back. Don't bother building unless you have to."
    if ice == "open" and samples == "open" and "waiting away" in lowered:
        return "Battery's under 18% and you still need both. Head back to the pad. Sitting still off the pad won't charge you."
    if off_pad and samples == "done" and ice == "open":
        return "Samples are done, ice isn't, you're off the pad, and the battery is down to 18%. Stop digging and take the short way back. Don't start another ice run if you can't get home."
    if on_pad and ice == "done" and samples == "open":
        return "Ice is in, samples aren't, and you're on the pad at 50% or less with an empty hold. Charge up before you leave. If you can't charge, grab samples nearby. Don't head out on a long trip."
    if on_pad and samples == "open":
        return "You're on the pad, samples still to do, battery at 60% or less. Don't roll out yet. Charge first. Leaving around 54% wasn't enough for a long sample run."
    return _clip(text, 140)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/evaluation/llm_dm_models.yaml"
    )
    parser.add_argument(
        "--model",
        action="append",
        help="select a configured name or provider/model; repeat to select several",
    )
    parser.add_argument("--env-file", type=Path)
    parser.add_argument(
        "--output", type=Path, help="override the output base directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write and validate per-model configs without API calls",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="run real simulator games with offline model responses",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        help="override the per-game limit, useful for bounded offline verification",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    source = args.config.expanduser().resolve()
    directory, panel, jobs, _ = prepare(source, args)
    summary = {
        "schema_version": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "source_config": str(source),
        "source_sha256": sha256(source.read_bytes()).hexdigest(),
        "experiment_name": panel["experiment_name"],
        "transport": jobs[0].config.transport,
        "status": "planned" if args.dry_run else "running",
        "dry_run": args.dry_run,
        "comparison_kind": "sequential_adaptation",
        "games_per_model": 5,
        "learner_reviews_per_model": 4,
        "episodes": json_value(jobs[0].config.episodes),
        "max_episode_steps": jobs[0].config.max_episode_steps,
        "environment": json_value(jobs[0].config.environment),
        "environment_hash": digest(jobs[0].config.environment),
        "max_usd_per_model": jobs[0].config.budgets.max_usd,
        "max_usd_panel": sum(job.config.budgets.max_usd for job in jobs),
        "note": "Same worlds and settings per model; knowledge is isolated between models. Five sequential learning games are descriptive adaptation results, not held-out frozen-policy evaluation. Means include complete recordings, including external-cutoff/failure recordings; check model status and per-game ending reasons before comparing incomplete runs.",
        "models": [
            {
                "name": job.name,
                "provider": job.config.actor.provider,
                "model": job.config.actor.model,
                "status": "pending",
                "resolved_config": str(job.config_path.relative_to(directory)),
            }
            for job in jobs
        ],
    }
    rows = []
    save_reports(directory, summary, rows)
    logger.info("Panel output: %s", directory)
    logger.info(
        "%d models; five identical worlds each; budget ceiling $%.2f per model / $%.2f panel",
        len(jobs),
        summary["max_usd_per_model"],
        summary["max_usd_panel"],
    )
    if args.dry_run:
        logger.info(
            "Dry run: all resolved configs and prices validated; no model calls."
        )
        print(directory / "summary.json")
        return 0
    interrupted = False
    for index, job in enumerate(jobs):
        logger.info(
            "\nModel %d/%d: %s as both Actor and Learner",
            index + 1,
            len(jobs),
            job.name,
        )
        summary["models"][index]["status"] = "running"
        save_reports(directory, summary, rows)
        error = None
        try:
            path = run_llm_dm(job.config)
            if path != job.manifest:
                raise RuntimeError(
                    "unexpected run directory; refusing ambiguous panel results"
                )
        except KeyboardInterrupt:
            interrupted = True
        except Exception as exception:
            error = type(exception).__name__
            logger.error("%s failed: %s", job.name, exception)
        manifest = (
            json.loads(job.manifest.read_text())
            if job.manifest.exists()
            else {"status": "failed", "ending_reason": error or "interrupted"}
        )
        recorded = episode_rows(job, manifest, directory)
        rows.extend(recorded)
        summary["models"][index] = model_result(job, manifest, recorded, directory)
        save_reports(directory, summary, rows)
        if interrupted:
            break
    summary["status"] = (
        "interrupted"
        if interrupted
        else "completed"
        if all(m["status"] == "completed" for m in summary["models"])
        else "incomplete"
    )
    save_reports(directory, summary, rows)
    for model in summary["models"]:
        metrics = model.get("metrics", {})
        logger.info(
            "%s: %s; games=%s; mean reward=%s; samples collected=%s, delivered=%s",
            model["name"],
            model["status"],
            model.get("recorded_games", 0),
            metrics.get("shaped_return", {}).get("mean"),
            metrics.get("samples_collected", {}).get("total"),
            metrics.get("samples_delivered", {}).get("total"),
        )
    print(directory / "summary.json")
    return 130 if interrupted else 0 if summary["status"] == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())
