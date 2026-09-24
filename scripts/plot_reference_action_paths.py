#!/usr/bin/env python3
"""Roll out reference checkpoints on validation seeds and plot action paths.

Writes heatmaps and action-mix-over-time figures under
``results/rllib_masked_ppo_reference/reports/``. Caches per-trial action
sequences as JSON so plots can be regenerated without reloading RLlib.

Usage::

    engine/.venv/bin/python scripts/plot_reference_action_paths.py
    engine/.venv/bin/python scripts/plot_reference_action_paths.py --trials seed_7 seed_13
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
REFERENCE = ROOT / "results" / "rllib_masked_ppo_reference"
REPORTS = REFERENCE / "reports"
CACHE = REPORTS / "action_path_cache"
DEFAULT_TRIALS = ("seed_7", "seed_11", "seed_13", "seed_14", "seed_17", "seed_19", "seed_21", "seed_23")
VAL_SEEDS = tuple(range(2000, 2032))
NORMALIZED_STEPS = 100
PAD_VALUE = -1


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


def _collect_trial(trial_id: str, seeds: tuple[int, ...]) -> list[dict[str, object]]:
    cache_path = CACHE / f"{trial_id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    checkpoint = REFERENCE / trial_id / "checkpoints" / "final" / "checkpoint.json"
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing final checkpoint for {trial_id}")
    spec = load_experiment(REFERENCE / trial_id / "resolved_config.yaml")
    env = make_gym_env(_environment_config(spec))
    agent = load_checkpoint_agent(str(checkpoint.resolve()))
    episodes = [_rollout_episode(env, agent, seed) for seed in seeds]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(episodes, indent=2), encoding="utf-8")
    return episodes


def _collect_scripted(seeds: tuple[int, ...]) -> list[dict[str, object]]:
    cache_path = CACHE / "scripted.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    spec = load_experiment(REFERENCE / "seed_13" / "resolved_config.yaml")
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


def _plot_heatmaps(series: dict[str, list[dict[str, object]]], output: Path) -> None:
    cmap = ListedColormap(plt.cm.tab10(np.linspace(0, 1, len(ACTION_LABELS))))
    fig, axes = plt.subplots(len(series), 1, figsize=(14, 2.8 * len(series)), squeeze=False)
    for ax, (label, episodes) in zip(axes[:, 0], series.items(), strict=True):
        matrix = _pad_matrix(episodes)
        masked = np.ma.masked_where(matrix == PAD_VALUE, matrix)
        im = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=9)
        ax.set_title(f"{label} — action path per episode (rows=sorted by seed)")
        ax.set_xlabel("step")
        ax.set_ylabel("episode")
        episode_labels = [str(ep["environment_seed"]) for ep in episodes]
        if len(episode_labels) <= 16:
            ax.set_yticks(range(len(episode_labels)))
            ax.set_yticklabels(episode_labels, fontsize=7)
        fig.colorbar(im, ax=ax, ticks=range(len(ACTION_LABELS)), label="action id")
    handles = [Patch(facecolor=cmap(i), label=ACTION_LABELS[i]) for i in range(len(ACTION_LABELS))]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_mix_over_time(series: dict[str, list[dict[str, object]]], output: Path) -> None:
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, len(series), figsize=(4.2 * len(series), 4), squeeze=False)
    x = np.linspace(0, 100, NORMALIZED_STEPS)
    colors = plt.cm.tab10(np.linspace(0, 1, len(ACTION_LABELS)))
    for ax, (label, episodes) in zip(axes[0], series.items(), strict=True):
        mix = _normalized_mix(episodes, NORMALIZED_STEPS)
        bottom = np.zeros(NORMALIZED_STEPS)
        for action_id, name in enumerate(ACTION_LABELS):
            values = mix[:, action_id]
            ax.fill_between(x, bottom, bottom + values, color=colors[action_id], alpha=0.85)
            bottom += values
        ax.set_ylim(0, 1)
        ax.set_title(label)
        ax.set_xlabel("episode progress (%)")
        ax.set_ylabel("action fraction")
    handles = [Patch(facecolor=colors[i], label=name) for i, name in enumerate(ACTION_LABELS)]
    labels = list(ACTION_LABELS)
    fig.suptitle("Action mix over normalized episode timeline", y=0.98)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=5,
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_early_window(series: dict[str, list[dict[str, object]]], output: Path, window: int = 120) -> None:
    cmap = ListedColormap(plt.cm.tab10(np.linspace(0, 1, len(ACTION_LABELS))))
    fig, axes = plt.subplots(len(series), 1, figsize=(14, 2.4 * len(series)), squeeze=False)
    for ax, (label, episodes) in zip(axes[:, 0], series.items(), strict=True):
        matrix = np.full((len(episodes), window), PAD_VALUE, dtype=np.int16)
        for row, episode in enumerate(episodes):
            actions = episode["actions"][:window]
            matrix[row, : len(actions)] = actions
        masked = np.ma.masked_where(matrix == PAD_VALUE, matrix)
        ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=9)
        ax.set_title(f"{label} — first {window} steps")
        ax.set_xlabel("step")
        ax.set_ylabel("episode")
    fig.suptitle("Early-episode action paths (loop detection)", y=1.01)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


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


def _load_cached_series(trial_ids: tuple[str, ...], include_scripted: bool) -> dict[str, list[dict[str, object]]]:
    series: dict[str, list[dict[str, object]]] = {}
    for trial_id in trial_ids:
        cache_path = CACHE / f"{trial_id}.json"
        if not cache_path.exists():
            raise FileNotFoundError(f"missing cache for {trial_id}: run collection first")
        series[trial_id] = json.loads(cache_path.read_text(encoding="utf-8"))
    if include_scripted:
        scripted_path = CACHE / "scripted.json"
        if not scripted_path.exists():
            raise FileNotFoundError("missing scripted cache: run collection first")
        series["scripted"] = json.loads(scripted_path.read_text(encoding="utf-8"))
    return series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    seeds = tuple(args.seeds)
    REPORTS.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        series = _load_cached_series(tuple(args.trials), args.include_scripted)
    else:
        series: dict[str, list[dict[str, object]]] = {}
        for trial_id in args.trials:
            print(f"collecting {trial_id}...")
            series[trial_id] = _collect_trial(trial_id, seeds)
        if args.include_scripted:
            print("collecting scripted...")
            series["scripted"] = _collect_scripted(seeds)

    _plot_heatmaps(series, REPORTS / "action_path_heatmaps.png")
    _plot_mix_over_time(series, REPORTS / "action_mix_over_time.png")
    _plot_early_window(series, REPORTS / "action_path_early_120.png")
    _write_summary(series, REPORTS / "action_paths_summary.json")
    print(f"wrote figures and summary under {REPORTS}")


if __name__ == "__main__":
    main()
