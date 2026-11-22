"""Compare frozen checkpoints on one shared list of worlds.

The panel generates episode identities from a seed manifest, scores each checkpoint
with :func:`aresim.training.evaluation.evaluate_episodes`, and writes the raw rows
before the aggregate bars, the shared action-path figure, the action-share /
step-reward figure, and one visit heatmap per policy. Each policy keeps its
frozen environment. Shaped return therefore follows that checkpoint's reward profile.
``baselines: true`` also scores Wait, Random-valid, and Scripted in the first
checkpoint's frozen environment.

**Contains:** :func:`load_panel`, :func:`generate_episodes`, :func:`run_panel`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev

from .evaluation import evaluate_agent, evaluate_episodes
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
    "allow_environment_mismatch", "checkpoints", "baselines",
}
_BASELINE_LABELS = {
    "wait": "Wait",
    "random": "Random",
    "random_valid": "Random-valid",
    "scripted": "Scripted",
}
_DEFAULT_BASELINES = ("wait", "random_valid", "scripted")
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
    baselines: tuple[str, ...] = ()


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
    baselines = _baselines(payload.get("baselines", False))
    if baselines and not checkpoints:
        raise ValueError("baselines need a checkpoint whose frozen environment they can play")
    return PanelSpec(
        str(payload["seed_manifest"]), split, episode_count, action_mode, workers, allow_mismatch, output,
        checkpoints, baselines)


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
    if spec.baselines:
        rows.extend(_score_baselines(spec, episodes))
    summary = _summarize(spec, episodes, rows)
    _write_json(spec.output / "summary.json", summary)
    _write_summary_csv(spec.output / "summary.csv", summary)
    figure_dir = spec.output / "figures"
    _write_figures(figure_dir, summary)
    _write_action_paths(figure_dir / "action_paths.png", rows)
    _write_action_rewards(figure_dir / "action_rewards.png", rows)
    _write_path_heatmaps(figure_dir / "path_heatmaps.png", rows)
    return spec.output


def _baselines(value: object) -> tuple[str, ...]:
    if value is None or value is False:
        return ()
    if value is True:
        names = _DEFAULT_BASELINES
    elif isinstance(value, list) and value and all(isinstance(name, str) and name.strip() for name in value):
        names = tuple(value)
    else:
        raise ValueError("baselines must be true or a non-empty list of baseline names")
    unknown = [name for name in names if name not in _BASELINE_LABELS]
    if unknown or len(set(names)) != len(names):
        raise ValueError(f"unknown or duplicate baselines: {', '.join(names)}")
    return names


def _score_baselines(spec: PanelSpec, episodes: tuple[EpisodeSpec, ...]) -> list[dict[str, object]]:
    """Play the built-in agents in the first checkpoint's frozen environment."""
    reference = spec.checkpoints[0]
    sidecar = json.loads(reference.path.read_text(encoding="utf-8"))
    experiment = parse_experiment(sidecar["experiment"])
    config = environment_config(experiment)
    context = _checkpoint_context(reference.path)
    rows = []
    for agent in spec.baselines:
        label = _BASELINE_LABELS[agent]
        print(f"Panel {label}: {len(episodes)} {spec.action_mode} episodes", flush=True)
        evaluated = evaluate_agent(agent, episodes, config, experiment.environment.max_episode_steps)
        item_rows = []
        for row in evaluated:
            record = _episode_record(reference, context, row)
            record.update(name=label, algorithm_id=agent, phase="baseline", checkpoint=agent, environment_steps=None)
            item_rows.append(record)
        _append_jsonl(spec.output / "episodes.jsonl", item_rows)
        rows.extend(item_rows)
    return rows


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
        "action_shaped_sums": list(row.action_shaped_sums),
        "positions": [list(point) for point in row.positions],
        "map_size": row.map_size,
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
    if spec.baselines:
        reference = algorithms[0]
        for agent in spec.baselines:
            label = _BASELINE_LABELS[agent]
            selected = [row for row in rows if row["name"] == label]
            algorithms.append({
                "name": label,
                "checkpoint": agent,
                "environment_comparable": reference["environment_comparable"],
                "algorithm_id": agent,
                "environment_steps": None,
                "phase": "baseline",
                "reward": reference["reward"],
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


def _write_action_rewards(path: Path, rows: list[dict[str, object]]) -> None:
    """One figure: each action is a subplot of step share and mean shaped reward."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(dict.fromkeys(str(row["name"]) for row in rows))
    shares = np.zeros((len(_ACTION_LABELS), len(names)))
    rewards = np.full((len(_ACTION_LABELS), len(names)), np.nan)
    for name_index, name in enumerate(names):
        selected = [row for row in rows if row["name"] == name]
        totals = np.zeros(len(_ACTION_LABELS))
        summed = np.zeros(len(_ACTION_LABELS))
        steps = 0
        for row in selected:
            actions = np.asarray(row["actions"], dtype=np.int16)
            sums = np.asarray(row.get("action_shaped_sums") or [], dtype=np.float64)
            if actions.size == 0 or sums.shape != (len(_ACTION_LABELS),):
                continue
            counts = np.bincount(actions, minlength=len(_ACTION_LABELS))[:len(_ACTION_LABELS)]
            totals += counts
            summed += sums
            steps += int(actions.size)
        if steps:
            shares[:, name_index] = 100 * totals / steps
            present = totals > 0
            rewards[present, name_index] = summed[present] / totals[present]
    figure, axes = plt.subplots(2, 5, figsize=(16, 7.4), constrained_layout=True)
    positions = np.arange(len(names))
    ticks = [_short_name(name) for name in names]
    for action_id, label in enumerate(_ACTION_LABELS):
        axis = axes.flat[action_id]
        axis.barh(positions, shares[action_id], color="#4C78A8", height=0.62)
        reward_axis = axis.twiny()
        reward_axis.plot(rewards[action_id], positions, color="#E45756", marker="o", linewidth=1.4)
        axis.set_title(label)
        axis.set_yticks(positions, ticks if action_id % 5 == 0 else [])
        axis.invert_yaxis()
        axis.set_xlabel("step share (%)" if action_id >= 5 else "")
        reward_axis.set_xlabel("mean shaped reward" if action_id < 5 else "")
        axis.grid(axis="x", linestyle=":", alpha=0.5)
    figure.suptitle("Action share (bars) and mean shaped reward (line) on steps that issued that action")
    figure.legend(
        [plt.Rectangle((0, 0), 1, 1, color="#4C78A8"), plt.Line2D([0], [0], color="#E45756", marker="o")],
        ["step share", "mean shaped reward"],
        loc="outside lower center", ncol=2, frameon=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_path_heatmaps(path: Path, rows: list[dict[str, object]]) -> None:
    """One figure: where each policy spent time, one heatmap per policy.

    The color scale is shared and logarithmic. Subplots line up only when every
    policy was scored on the same seeds and the same frozen world settings, so a
    cell is the same place on a given seed. Policies with no recorded positions
    are skipped.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(dict.fromkeys(str(row["name"]) for row in rows if row.get("positions")))
    if not names:
        return
    size = max(int(row.get("map_size") or 0) for row in rows)
    for row in rows:
        for point in row.get("positions") or []:
            size = max(size, int(point[0]) + 1, int(point[1]) + 1)
    counts = {name: np.zeros((size, size), dtype=np.float64) for name in names}
    for row in rows:
        grid = counts.get(str(row["name"]))
        if grid is None:
            continue
        for point in row.get("positions") or []:
            x, y = int(point[0]), int(point[1])
            grid[y, x] += 1
    columns = 3 if len(names) > 3 else len(names)
    grid_rows = (len(names) + columns - 1) // columns
    figure, axes = plt.subplots(
        grid_rows, columns, figsize=(4.2 * columns, 4.0 * grid_rows), squeeze=False, constrained_layout=True,
    )
    image = None
    vmax = max(float(np.log1p(grid).max()) for grid in counts.values())
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#f4f4f4")
    for index, name in enumerate(names):
        axis = axes.flat[index]
        shown = np.ma.masked_where(counts[name] == 0, np.log1p(counts[name]))
        image = axis.imshow(shown, cmap=cmap, vmin=0, vmax=vmax, origin="upper", interpolation="nearest")
        axis.set_title(_short_name(name))
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.flat[len(names):]:
        axis.axis("off")
    figure.colorbar(image, ax=axes, fraction=0.03, pad=0.02, label="log(1 + steps in cell)")
    figure.suptitle("Visit time on the same validation maps. North is up.")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _short_name(name: str) -> str:
    return {
        "Double Duelling DQN": "DQN",
        "DreamerV3": "Dreamer",
        "Plan2Explore": "P2E",
        "Recurrent DQN": "RDQN",
    }.get(name, name)


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
