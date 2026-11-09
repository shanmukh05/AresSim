"""Compare frozen checkpoints on one shared list of worlds.

The panel generates episode identities from a seed manifest, scores each checkpoint
with :func:`aresim.training.evaluation.evaluate_episodes`, and writes the raw rows
before the aggregate bars and the shared action-path figure. Each policy keeps its
frozen environment. Shaped return therefore follows that checkpoint's reward profile.

**Contains:** :func:`load_panel`, :func:`generate_episodes`, :func:`run_panel`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev

from .evaluation import evaluate_episodes
from .experiments import parse_experiment
from .provenance import environment_config, verify_provenance
from .runner import EpisodeSpec
from .seeds import load_seed_manifest, resolve_checkout_file

PANEL_SCHEMA = "aresim.evaluation.panel.v1"
_METRICS = (
    ("shaped_return", "Shaped return"),
    ("engine_return", "Engine return"),
    ("sparse_return", "Sparse return"),
    ("length", "Timesteps"),
    ("ice_collected", "Ice collected (kg)"),
    ("ice_delivered", "Ice delivered (kg)"),
    ("samples_collected", "Samples collected (kg)"),
    ("samples_delivered", "Samples delivered (kg)"),
    ("success", "Success rate"),
)
_PANEL_FIELDS = {
    "schema_version", "output", "seed_manifest", "split", "episode_count", "action_mode", "workers",
    "allow_environment_mismatch", "checkpoints",
}
_CHECKPOINT_FIELDS = {"name", "path"}
_REWARD_FIELDS = ("samples_collected", "time_cost", "ice_collected", "ice_delivered", "samples_delivered", "mission_success")


@dataclass(frozen=True)
class PanelCheckpoint:
    """One labeled frozen policy in a comparison panel."""

    name: str
    path: Path


@dataclass(frozen=True)
class PanelSpec:
    """Validated comparison: shared worlds, then one run per checkpoint."""

    seed_manifest: str
    split: str
    episode_count: int
    action_mode: str
    workers: int
    allow_environment_mismatch: bool
    output: Path
    checkpoints: tuple[PanelCheckpoint, ...]


def load_panel(path: str | Path) -> PanelSpec:
    """Load a panel YAML and reject unknown fields, missing checkpoints, and unsafe output paths."""
    import yaml

    source = resolve_checkout_file(path, kind="evaluation panel")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != PANEL_SCHEMA:
        raise ValueError("unsupported evaluation panel")
    unknown = sorted(set(payload) - _PANEL_FIELDS)
    if unknown:
        raise ValueError(f"unknown evaluation panel fields: {', '.join(unknown)}")
    split = payload["split"]
    if split not in {"validation", "test"}:
        raise ValueError("panel split must be validation or test")
    episode_count = payload["episode_count"]
    workers = payload["workers"]
    if _invalid_count(episode_count) or _invalid_count(workers):
        raise ValueError("episode_count and workers must be positive integers")
    action_mode = payload["action_mode"]
    if action_mode not in {"deterministic", "stochastic"}:
        raise ValueError("panel action_mode must be deterministic or stochastic")
    allow_mismatch = payload.get("allow_environment_mismatch", False)
    if not isinstance(allow_mismatch, bool):
        raise ValueError("allow_environment_mismatch must be a boolean")
    output = _output_directory(payload["output"])
    checkpoints = tuple(_checkpoint(item) for item in payload["checkpoints"])
    if len({item.name for item in checkpoints}) != len(checkpoints):
        raise ValueError("panel checkpoint names must be unique")
    manifest = load_seed_manifest(payload["seed_manifest"])
    if episode_count > len(getattr(manifest, split)):
        raise ValueError("episode_count exceeds the selected seed split")
    return PanelSpec(
        str(payload["seed_manifest"]), split, episode_count, action_mode, workers, allow_mismatch, output, checkpoints)


def generate_episodes(spec: PanelSpec) -> tuple[EpisodeSpec, ...]:
    """Take the first ``episode_count`` worlds from the panel split, in manifest order."""
    manifest = load_seed_manifest(spec.seed_manifest)
    return manifest.episodes(spec.split)[:spec.episode_count]


def run_panel(config: str | Path | PanelSpec) -> Path:
    """Generate worlds, score every checkpoint, then write raw rows, summary, and bar charts."""
    spec = config if isinstance(config, PanelSpec) else load_panel(config)
    if spec.output.exists():
        raise FileExistsError(f"panel output already exists: {spec.output}")
    episodes = generate_episodes(spec)
    spec.output.mkdir(parents=True)
    _write_json(spec.output / "environments.json", _environment_payload(spec, episodes))
    rows: list[dict[str, object]] = []
    for item in spec.checkpoints:
        print(f"Panel {item.name}: {len(episodes)} {spec.action_mode} episodes", flush=True)
        evaluated = evaluate_episodes(
            item.path, episodes, deterministic=spec.action_mode == "deterministic", workers=spec.workers,
            allow_environment_mismatch=spec.allow_environment_mismatch)
        item_rows = [_episode_record(item, _checkpoint_context(item.path), row) for row in evaluated]
        _append_jsonl(spec.output / "episodes.jsonl", item_rows)
        rows.extend(item_rows)
    summary = _summarize(spec, episodes, rows)
    _write_json(spec.output / "summary.json", summary)
    _write_summary_csv(spec.output / "summary.csv", summary)
    figure_dir = spec.output / "figures"
    _write_figures(figure_dir, summary)
    _write_action_paths(figure_dir / "action_paths.png", rows)
    return spec.output


def _invalid_count(value: object) -> bool:
    return isinstance(value, bool) or not isinstance(value, int) or value < 1


def _output_directory(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("panel output must be a relative directory")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("panel output must stay inside the repository")
    return Path(__file__).resolve().parents[3] / path


def _checkpoint(value: object) -> PanelCheckpoint:
    if not isinstance(value, dict):
        raise ValueError("panel checkpoint must be an object")
    unknown = sorted(set(value) - _CHECKPOINT_FIELDS)
    if unknown:
        raise ValueError(f"unknown panel checkpoint fields: {', '.join(unknown)}")
    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("panel checkpoint name must be text")
    return PanelCheckpoint(name, resolve_checkout_file(value["path"], kind="checkpoint"))


def _environment_payload(spec: PanelSpec, episodes: tuple[EpisodeSpec, ...]) -> dict[str, object]:
    manifest = load_seed_manifest(spec.seed_manifest)
    return {
        "schema_version": PANEL_SCHEMA,
        "seed_manifest": spec.seed_manifest,
        "manifest_id": manifest.manifest_id,
        "scenario_id": manifest.scenario_id,
        "split": spec.split,
        "action_mode": spec.action_mode,
        "episodes": [
            {"episode_id": episode.episode_id, "environment_seed": episode.environment_seed, "agent_seed": episode.agent_seed}
            for episode in episodes
        ],
    }


def _environment_comparable(path: Path, allow_mismatch: bool) -> bool:
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    return bool(verify_provenance(parse_experiment(sidecar["experiment"]), allow_environment_mismatch=allow_mismatch))


def _checkpoint_context(path: Path) -> dict[str, object]:
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    reward = environment_config(parse_experiment(sidecar["experiment"])).reward_config
    return {
        "algorithm_id": sidecar.get("algorithm_id"),
        "environment_steps": sidecar.get("environment_steps"),
        "phase": sidecar.get("phase"),
        "reward": {name: getattr(reward, name) for name in _REWARD_FIELDS},
    }


def _episode_record(item: PanelCheckpoint, context: dict[str, object], row) -> dict[str, object]:
    return {
        "name": item.name,
        "algorithm_id": context["algorithm_id"],
        "checkpoint": str(item.path),
        "environment_steps": context["environment_steps"],
        "phase": context["phase"],
        "episode_id": row.episode_id,
        "environment_seed": row.environment_seed,
        "agent_seed": row.agent_seed,
        "action_mode": row.action_mode,
        "length": row.length,
        "shaped_return": row.shaped_return,
        "engine_return": row.engine_return,
        "sparse_return": row.sparse_return,
        "success": row.success,
        "terminated": row.terminated,
        "truncated": row.truncated,
        "ending_reason": row.ending_reason,
        "ice_collected": row.ice_collected,
        "ice_delivered": row.ice_delivered,
        "samples_collected": row.samples_collected,
        "samples_delivered": row.samples_delivered,
        "extract_opportunities": row.extract_opportunities,
        "action_counts": row.action_counts,
        "actions": list(row.actions),
    }


def _summarize(spec: PanelSpec, episodes: tuple[EpisodeSpec, ...], rows: list[dict[str, object]]) -> dict[str, object]:
    algorithms = []
    for item in spec.checkpoints:
        selected = [row for row in rows if row["name"] == item.name]
        context = _checkpoint_context(item.path)
        algorithms.append({
            "name": item.name,
            "checkpoint": str(item.path),
            "environment_comparable": _environment_comparable(item.path, spec.allow_environment_mismatch),
            **{key: context[key] for key in ("algorithm_id", "environment_steps", "phase", "reward")},
            "episode_count": len(selected),
            "means": {key: fmean(float(row[key]) for row in selected) for key, _title in _METRICS},
            "std": {key: pstdev(float(row[key]) for row in selected) for key, _title in _METRICS},
            "ending_reasons": _counts(selected, "ending_reason"),
        })
    profiles = [item["reward"] for item in algorithms]
    return {
        "schema_version": PANEL_SCHEMA,
        "seed_manifest": spec.seed_manifest,
        "split": spec.split,
        "action_mode": spec.action_mode,
        "episode_count": len(episodes),
        "environment_seeds": [episode.environment_seed for episode in episodes],
        "shaped_returns_share_reward_profile": all(profile == profiles[0] for profile in profiles),
        "algorithms": algorithms,
    }


def _counts(rows: list[dict[str, object]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row[key] or "unknown")
        counts[label] = counts.get(label, 0) + 1
    return counts


def _write_json(path: Path, payload: dict[str, object]) -> None:
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    partial.replace(path)


def _append_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _write_summary_csv(path: Path, summary: dict[str, object]) -> None:
    import csv

    fieldnames = ["name", "algorithm_id", "environment_steps", "phase", *[key for key, _title in _METRICS]]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for item in summary["algorithms"]:
            means = item["means"]
            writer.writerow({
                "name": item["name"],
                "algorithm_id": item["algorithm_id"],
                "environment_steps": item["environment_steps"],
                "phase": item["phase"],
                **{key: means[key] for key, _title in _METRICS},
            })


def _write_figures(directory: Path, summary: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    names = [item["name"] for item in summary["algorithms"]]
    colors = [plt.get_cmap("tab10")(index) for index in range(len(names))]
    for key, title in _METRICS:
        _write_bar(directory / f"{key}.png", names, [item["means"][key] for item in summary["algorithms"]], title, colors)
    figure, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, _METRICS, strict=True):
        values = [item["means"][key] for item in summary["algorithms"]]
        axis.bar(names, values, color=colors)
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=30)
        axis.grid(axis="y", linestyle=":", alpha=0.6)
    figure.suptitle(f"{summary['episode_count']} {summary['split']} games, {summary['action_mode']}")
    figure.savefig(directory / "all_metrics.png", dpi=150)
    plt.close(figure)


_ACTION_LABELS = ("wait", "move_n", "move_e", "move_s", "move_w", "scan", "extract", "build", "service", "unload")


def _write_action_paths(path: Path, rows: list[dict[str, object]]) -> None:
    """One stacked-timeline figure: each panel is an algorithm, columns share the legend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    names = list(dict.fromkeys(str(row["name"]) for row in rows))
    columns = 3 if len(names) > 3 else max(1, len(names))
    grid_rows = (len(names) + columns - 1) // columns
    figure, axes = plt.subplots(grid_rows, columns, figsize=(4.4 * columns, 3.2 * grid_rows), squeeze=False)
    bins = 100
    progress = np.linspace(0, 100, bins)
    colors = plt.cm.tab10(np.linspace(0, 1, len(_ACTION_LABELS)))
    for index, name in enumerate(names):
        axis = axes.flat[index]
        mix = _action_mix([row["actions"] for row in rows if row["name"] == name], bins)
        bottom = np.zeros(bins)
        for action_id, _label in enumerate(_ACTION_LABELS):
            values = mix[:, action_id]
            axis.fill_between(progress, bottom, bottom + values, color=colors[action_id], linewidth=0)
            bottom += values
        axis.set_ylim(0, 1)
        axis.set_title(name)
        axis.set_xlabel("episode progress (%)")
        axis.set_ylabel("action fraction")
    for axis in axes.flat[len(names):]:
        axis.axis("off")
    figure.legend(
        [Patch(facecolor=colors[index], label=label) for index, label in enumerate(_ACTION_LABELS)],
        _ACTION_LABELS, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=False,
    )
    figure.suptitle("Action path over the episode", y=1.08)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _action_mix(sequences: list[object], bins: int):
    import numpy as np

    counts = np.zeros((bins, len(_ACTION_LABELS)), dtype=np.float64)
    for sequence in sequences:
        actions = np.asarray(sequence, dtype=np.int16)
        if actions.size == 0:
            continue
        for bin_index, step_index in enumerate(np.linspace(0, actions.size - 1, bins).astype(int)):
            counts[bin_index, int(actions[step_index])] += 1
    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    return counts / totals


def _write_bar(path: Path, names: list[str], values: list[float], title: str, colors) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    axis.bar(names, values, color=colors)
    axis.set_title(title)
    axis.tick_params(axis="x", labelrotation=20)
    axis.grid(axis="y", linestyle=":", alpha=0.6)
    figure.savefig(path, dpi=150)
    plt.close(figure)
