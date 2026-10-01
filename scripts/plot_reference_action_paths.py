#!/usr/bin/env python3
"""Roll out reference checkpoints on validation seeds and plot action paths.

Default writes combined PPO + DQN figures under ``results/reports/``. Caches
per-trial action sequences as JSON so plots can be regenerated without RLlib.

Usage::

    engine/.venv/bin/python scripts/plot_reference_action_paths.py
    engine/.venv/bin/python scripts/plot_reference_action_paths.py --plot-only
    engine/.venv/bin/python scripts/plot_reference_action_paths.py --trials seed_27 seed_29 dqn_seed_1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from aresim.algorithms.baselines.scripted import ScriptedAgent
from aresim.factory import make_gym_env
from aresim.integrations.policy import load_checkpoint_agent
from aresim.training.evaluation import _environment_config
from aresim.training.experiments import load_experiment

ACTION_LABELS = (
    "wait",
    "move_n",
    "move_e",
    "move_s",
    "move_w",
    "scan",
    "extract",
    "build",
    "service",
    "unload",
)
ROOT = Path(__file__).resolve().parents[1]
PPO_EXPERIMENT = ROOT / "results" / "rllib_masked_ppo_reference"
DQN_EXPERIMENT = ROOT / "results" / "rllib_masked_double_dueling_dqn_reference"
DEFAULT_OUTPUT = ROOT / "results" / "reports"
PPO_TRIALS = ("seed_7", "seed_11", "seed_13", "seed_14", "seed_17", "seed_19", "seed_21", "seed_23", "seed_25", "seed_27", "seed_29")
DEFAULT_TRIALS = (*PPO_TRIALS, "dqn_seed_1")
VAL_SEEDS = tuple(range(2000, 2032))
NORMALIZED_STEPS = 100
PAD_VALUE = -1
GRID_COLUMNS = 4


def _experiment_for(label: str) -> Path:
    return DQN_EXPERIMENT if label.startswith("dqn_") else PPO_EXPERIMENT


def _checkpoint_trial_id(label: str) -> str:
    return label.removeprefix("dqn_")


def _legacy_cache_path(label: str) -> Path:
    experiment = _experiment_for(label)
    return experiment / "reports" / "action_path_cache" / f"{_checkpoint_trial_id(label)}.json"


def _rollout_episode(env, agent, seed: int) -> dict[str, object]:
    obs, info = env.reset(seed=seed)
    if hasattr(agent, "reset"):
        agent.reset(seed + 1_000_000)
    actions: list[int] = []
    effective: list[str] = []
    done = False
    while not done:
        action = int(agent.act(obs["observation"], obs["action_mask"]))
        obs, _reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        effective.append(str(info.get("effective_action", "unknown")))
        done = terminated or truncated
    state = env.unwrapped.environment.world_state
    stats = state.objective_stats
    rover = state.rovers[0]
    return {
        "environment_seed": seed,
        "length": len(actions),
        "actions": actions,
        "effective_actions": effective,
        "ending_reason": info.get("terminal_reason") or info.get("truncation_reason"),
        "ice_collected": stats.ice_collected,
        "ice_delivered": stats.ice_delivered,
        "samples_delivered": stats.samples_delivered,
        "cargo_ice_end": rover.cargo_ice,
        "cargo_samples_end": rover.cargo_samples,
    }


def _collect_trial(label: str, seeds: tuple[int, ...], cache: Path) -> list[dict[str, object]]:
    cache_path = cache / f"{label}.json"
    for candidate in (cache_path, _legacy_cache_path(label)):
        if candidate.is_file():
            episodes = json.loads(candidate.read_text(encoding="utf-8"))
            if candidate != cache_path:
                cache.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(episodes), encoding="utf-8")
            return episodes

    trial_id = _checkpoint_trial_id(label)
    experiment = _experiment_for(label)
    checkpoint = experiment / trial_id / "checkpoints" / "final" / "checkpoint.json"
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing final checkpoint for {label}")
    spec = load_experiment(experiment / trial_id / "resolved_config.yaml")
    env = make_gym_env(_environment_config(spec))
    agent = load_checkpoint_agent(str(checkpoint.resolve()))
    episodes = [_rollout_episode(env, agent, seed) for seed in seeds]
    cache.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(episodes, indent=2), encoding="utf-8")
    return episodes


def _collect_scripted(seeds: tuple[int, ...], cache: Path) -> list[dict[str, object]]:
    cache_path = cache / "scripted.json"
    legacy = PPO_EXPERIMENT / "reports" / "action_path_cache" / "scripted.json"
    for candidate in (cache_path, legacy, DQN_EXPERIMENT / "reports" / "action_path_cache" / "scripted.json"):
        if candidate.is_file():
            episodes = json.loads(candidate.read_text(encoding="utf-8"))
            if candidate != cache_path:
                cache.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(episodes), encoding="utf-8")
            return episodes

    config = PPO_EXPERIMENT / "seed_29" / "resolved_config.yaml"
    spec = load_experiment(config)
    env = make_gym_env(_environment_config(spec))
    agent = ScriptedAgent()
    episodes = [_rollout_episode(env, agent, seed) for seed in seeds]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(episodes, indent=2), encoding="utf-8")
    return episodes


def _pad_matrix(episodes: list[dict[str, object]]) -> np.ndarray:
    max_len = max(int(ep["length"]) for ep in episodes)
    matrix = np.full((len(episodes), max_len), PAD_VALUE, dtype=np.int16)
    for row, episode in enumerate(episodes):
        actions = episode["actions"]
        matrix[row, : len(actions)] = actions
    return matrix


def _normalized_mix(episodes: list[dict[str, object]], bins: int) -> np.ndarray:
    """Return ``[bins, action_count]`` fractions averaged over episodes."""
    counts = np.zeros((bins, len(ACTION_LABELS)), dtype=np.float64)
    for episode in episodes:
        actions = np.asarray(episode["actions"], dtype=np.int16)
        if actions.size == 0:
            continue
        positions = np.linspace(0, actions.size - 1, bins).astype(int)
        for bin_idx, step_idx in enumerate(positions):
            counts[bin_idx, actions[step_idx]] += 1
    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    return counts / totals


def _subplot_grid(count: int, panel_width: float, panel_height: float):
    columns = min(GRID_COLUMNS, count)
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(panel_width * columns, panel_height * rows),
        squeeze=False,
    )
    return fig, axes


def _iter_grid_panels(axes, series: dict[str, list[dict[str, object]]]):
    panels = list(series.items())
    used = len(panels)
    for index, ax in enumerate(axes.flat):
        if index >= used:
            ax.axis("off")
            continue
        yield ax, panels[index]


def _save_grid_figure(fig, output: Path, title: str, handles: list[Patch]) -> None:
    fig.suptitle(title, y=0.995)
    fig.legend(
        handles,
        list(ACTION_LABELS),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=5,
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _action_legend(cmap) -> list[Patch]:
    return [Patch(facecolor=cmap(i), label=ACTION_LABELS[i]) for i in range(len(ACTION_LABELS))]


def _plot_heatmaps(series: dict[str, list[dict[str, object]]], output: Path) -> None:
    cmap = ListedColormap(plt.cm.tab10(np.linspace(0, 1, len(ACTION_LABELS))))
    fig, axes = _subplot_grid(len(series), 4.2, 2.6)
    for ax, (label, episodes) in _iter_grid_panels(axes, series):
        matrix = _pad_matrix(episodes)
        masked = np.ma.masked_where(matrix == PAD_VALUE, matrix)
        ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=9)
        ax.set_title(label)
        ax.set_xlabel("step")
        ax.set_ylabel("episode")
        episode_labels = [str(ep["environment_seed"]) for ep in episodes]
        if len(episode_labels) <= 16:
            ax.set_yticks(range(len(episode_labels)))
            ax.set_yticklabels(episode_labels, fontsize=7)
    _save_grid_figure(fig, output, "Action path per episode (rows sorted by seed)", _action_legend(cmap))


def _plot_mix_over_time(series: dict[str, list[dict[str, object]]], output: Path) -> None:
    fig, axes = _subplot_grid(len(series), 4.2, 3.4)
    x = np.linspace(0, 100, NORMALIZED_STEPS)
    colors = plt.cm.tab10(np.linspace(0, 1, len(ACTION_LABELS)))
    for ax, (label, episodes) in _iter_grid_panels(axes, series):
        mix = _normalized_mix(episodes, NORMALIZED_STEPS)
        bottom = np.zeros(NORMALIZED_STEPS)
        for action_id, _name in enumerate(ACTION_LABELS):
            values = mix[:, action_id]
            ax.fill_between(x, bottom, bottom + values, color=colors[action_id], alpha=0.85)
            bottom += values
        ax.set_ylim(0, 1)
        ax.set_title(label)
        ax.set_xlabel("episode progress (%)")
        ax.set_ylabel("action fraction")
    handles = [Patch(facecolor=colors[i], label=name) for i, name in enumerate(ACTION_LABELS)]
    _save_grid_figure(fig, output, "Action mix over normalized episode timeline", handles)


def _plot_early_window(series: dict[str, list[dict[str, object]]], output: Path, window: int = 120) -> None:
    cmap = ListedColormap(plt.cm.tab10(np.linspace(0, 1, len(ACTION_LABELS))))
    fig, axes = _subplot_grid(len(series), 4.2, 2.4)
    for ax, (label, episodes) in _iter_grid_panels(axes, series):
        matrix = np.full((len(episodes), window), PAD_VALUE, dtype=np.int16)
        for row, episode in enumerate(episodes):
            actions = episode["actions"][:window]
            matrix[row, : len(actions)] = actions
        masked = np.ma.masked_where(matrix == PAD_VALUE, matrix)
        ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=9)
        ax.set_title(label)
        ax.set_xlabel("step")
        ax.set_ylabel("episode")
    _save_grid_figure(fig, output, f"Early-episode action paths (first {window} steps)", _action_legend(cmap))


def _write_summary(series: dict[str, list[dict[str, object]]], output: Path) -> None:
    summary: dict[str, object] = {}
    for label, episodes in series.items():
        lengths = [int(ep["length"]) for ep in episodes]
        all_actions = [a for ep in episodes for a in ep["actions"]]
        total = len(all_actions) or 1
        fractions = {ACTION_LABELS[i]: round(100 * sum(1 for a in all_actions if a == i) / total, 2) for i in range(10)}
        summary[label] = {
            "episode_count": len(episodes),
            "mean_length": round(sum(lengths) / len(lengths), 1),
            "extract_episodes": sum(6 in ep["actions"] for ep in episodes),
            "unload_episodes": sum(9 in ep["actions"] for ep in episodes),
            "wait_fraction_pct": fractions["wait"],
            "move_fraction_pct": round(sum(fractions[k] for k in ACTION_LABELS[1:5]), 2),
            "service_fraction_pct": fractions["service"],
            "mean_ice_collected": round(sum(float(ep["ice_collected"]) for ep in episodes) / len(episodes), 2),
            "mean_ice_delivered": round(sum(float(ep["ice_delivered"]) for ep in episodes) / len(episodes), 2),
            "action_fractions_pct": fractions,
        }
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _load_cached_series(cache: Path, trial_ids: tuple[str, ...], include_scripted: bool) -> dict[str, list[dict[str, object]]]:
    series: dict[str, list[dict[str, object]]] = {}
    for trial_id in trial_ids:
        candidates = (cache / f"{trial_id}.json", _legacy_cache_path(trial_id))
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"missing cache for {trial_id}: run collection first")
        series[trial_id] = json.loads(path.read_text(encoding="utf-8"))
    if include_scripted:
        scripted_candidates = (
            cache / "scripted.json",
            PPO_EXPERIMENT / "reports" / "action_path_cache" / "scripted.json",
            DQN_EXPERIMENT / "reports" / "action_path_cache" / "scripted.json",
        )
        scripted = next((candidate for candidate in scripted_candidates if candidate.is_file()), None)
        if scripted is None:
            raise FileNotFoundError("missing scripted cache: run collection first")
        series["scripted"] = json.loads(scripted.read_text(encoding="utf-8"))
    return series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="directory for figures and cache (default: results/reports)")
    parser.add_argument("--trials", nargs="+", default=list(DEFAULT_TRIALS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(VAL_SEEDS))
    parser.add_argument("--include-scripted", action="store_true", default=True)
    parser.add_argument("--no-scripted", action="store_false", dest="include_scripted")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="regenerate figures from cached JSON without loading checkpoints",
    )
    args = parser.parse_args()
    reports = args.output if args.output.is_absolute() else ROOT / args.output
    cache = reports / "action_path_cache"
    seeds = tuple(args.seeds)
    reports.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        series = _load_cached_series(cache, tuple(args.trials), args.include_scripted)
    else:
        series: dict[str, list[dict[str, object]]] = {}
        for trial_id in args.trials:
            print(f"collecting {trial_id}...")
            series[trial_id] = _collect_trial(trial_id, seeds, cache)
        if args.include_scripted:
            print("collecting scripted...")
            series["scripted"] = _collect_scripted(seeds, cache)

    _plot_heatmaps(series, reports / "action_path_heatmaps.png")
    _plot_mix_over_time(series, reports / "action_mix_over_time.png")
    _plot_early_window(series, reports / "action_path_early_120.png")
    _write_summary(series, reports / "action_paths_summary.json")
    print(f"wrote figures and summary under {reports}")


if __name__ == "__main__":
    main()
