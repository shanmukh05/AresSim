"""Generate training report plots from W&B metrics and local evaluation artifacts.

Pulls canonical training history from W&B and writes matplotlib figures plus exported
metric tables under ``<run>/reports/``. Does not start Ray or the simulator.

**Last updated:** October 13, 2026

**Contains:** :func:`generate_report`, :func:`wandb_run_path`.

**CLI:** ``aresim-rl report <run_dir>``.

**Dependencies:** ``wandb``, ``matplotlib``, ``pandas`` (``rllib`` extra).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Canonical PPO/DQN metrics plus MuZero loss and replay-weight diagnostics.
TRAINING_METRICS = (
    "train/environment_steps",
    "train/episodes",
    "train/episode_length",
    "train/shaped_return",
    "system/environment_steps_per_second",
    "learner/total_loss",
    "learner/policy_loss",
    "learner/value_loss",
    "learner/reward_loss",
    "learner/constraint_loss",
    "learner/consistency_loss",
    "learner/consistency_loss_unweighted",
    "learner/consistency_target_std",
    "learner/policy_target_age_mean",
    "learner/policy_target_age_max",
    "train/reanalysis_start_fraction",
    "system/reanalysis_sampled_starts",
    "system/reanalysis_search_roots",
    "system/reanalysis_value_roots",
    "learner/total_loss_unweighted",
    "learner/policy_loss_unweighted",
    "learner/value_loss_unweighted",
    "learner/reward_loss_unweighted",
    "learner/constraint_loss_unweighted",
    "learner/importance_weight_mean",
    "learner/importance_weight_min",
    "learner/importance_weight_max",
    "learner/effective_sample_size",
    "learner/entropy",
    "learner/approx_kl",
    "learner/clip_fraction",
    "learner/learning_rate",
    "learner/gradient_norm",
    "learner/explained_variance",
    "learner/td_error",
    "learner/q_mean",
    "learner/q_max",
    "learner/q_min",
    "train/search_simulations",
    "train/replay_steps",
    "train/learner_updates",
    "system/elapsed_seconds",
    "learner/reconstruction_loss",
    "learner/dynamics_kl",
    "learner/representation_kl",
    "learner/continuation_loss",
    "learner/legality_loss",
    "learner/actor_loss",
    "learner/disagreement_loss",
    "learner/imagined_reward",
    "learner/return_scale",
    "learner/imagined_mask_density",
    "learner/recurrent_learned_timesteps",
    "learner/recurrent_burn_in_timesteps",
    "learner/recurrent_padding_fraction",
    "train/actual_replay_ratio",
    "train/observed_positions",
    "train/pretraining_environment_steps",
    "train/cumulative_environment_steps",
    "system/prefix_reconstruction_seconds",
    "system/collector_reconstruction_seconds",

) + tuple(f"learner/legality_{metric}/{action}" for metric in ("recall", "false_positive_rate", "positive_count", "negative_count") for action in range(10))

PLOT_GROUPS: dict[str, tuple[tuple[str, ...], str]] = {
    "world_model_losses": (("learner/reconstruction_loss", "learner/dynamics_kl", "learner/representation_kl", "learner/continuation_loss", "learner/legality_loss"), "World-model losses"),
    "imagined_behavior": (("learner/actor_loss", "learner/value_loss", "learner/entropy", "learner/imagined_reward"), "Imagined behavior"),
    "exploration": (("learner/disagreement_loss", "learner/imagined_reward", "train/observed_positions"), "Plan2Explore exploration"),
    "recurrent_replay": (("learner/recurrent_learned_timesteps", "learner/recurrent_burn_in_timesteps", "learner/recurrent_padding_fraction"), "Recurrent DQN replay"),
    "world_model_replay": (("train/actual_replay_ratio", "system/prefix_reconstruction_seconds", "system/collector_reconstruction_seconds"), "World-model replay and history cost"),
    "phase_budget": (("train/environment_steps", "train/pretraining_environment_steps", "train/cumulative_environment_steps"), "Phase interaction budgets"),
    "training_returns": (("train/shaped_return", "train/episode_length"), "Training returns"),
    "learner_losses": (("learner/total_loss", "learner/policy_loss", "learner/value_loss", "learner/reward_loss", "learner/constraint_loss", "learner/consistency_loss"), "Learner losses"),
    "learner_losses_unweighted": (
        ("learner/total_loss_unweighted", "learner/policy_loss_unweighted", "learner/value_loss_unweighted",
         "learner/reward_loss_unweighted", "learner/constraint_loss_unweighted", "learner/consistency_loss_unweighted"),
        "MuZero losses without replay importance weights",
    ),
    "replay_importance_weights": (
        ("learner/importance_weight_mean", "learner/importance_weight_min", "learner/importance_weight_max"),
        "MuZero replay importance weights",
    ),
    "replay_effective_sample_size": (("learner/effective_sample_size",), "MuZero effective replay batch size"),
    "muzero_legality_recall": (tuple(f"learner/legality_recall/{a}" for a in (5, 6, 9)), "MuZero rare-action legality recall"),
    "muzero_legality_false_positives": (tuple(f"learner/legality_false_positive_rate/{a}" for a in (5, 6, 9)), "MuZero rare-action legality false positives"),
    "muzero_target_age": (("learner/policy_target_age_mean", "learner/policy_target_age_max"), "MuZero policy target age (learner updates)"),
    "muzero_reanalysis_coverage": (("train/reanalysis_start_fraction",), "MuZero reanalysed fraction of learned starts"),
    "muzero_consistency": (("learner/consistency_loss_unweighted", "learner/consistency_target_std"), "MuZero temporal consistency supervision"),
    "learner_diagnostics": (
        ("learner/entropy", "learner/explained_variance", "learner/approx_kl"),
        "Learner diagnostics",
    ),
    "learner_q": (("learner/td_error", "learner/q_mean"), "DQN Q-values"),
}


def wandb_run_path(run_directory: str | Path) -> str:
    """Resolve the W&B path recorded by one completed AresSim run."""
    manifest = json.loads((Path(run_directory) / "manifest.json").read_text(encoding="utf-8"))
    experiment = manifest.get("experiment", {})
    tracking = experiment.get("tracking", {}) if isinstance(experiment, dict) else {}
    mode = tracking.get("mode")
    if mode not in {"online", "offline"}:
        raise ValueError("reports require a tracked W&B run (tracking.mode online or offline)")
    run_id = manifest.get("wandb_run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("manifest is missing wandb_run_id")
    entity = tracking.get("entity")
    if not entity:
        import wandb

        entity = wandb.Api().default_entity
    return f"{entity}/{tracking['project']}/{run_id}"


def _load_manifest(run_path: Path) -> dict[str, Any]:
    return json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))


def _fetch_wandb_history(run_path: str) -> Any:
    import pandas as pd
    import wandb

    history = wandb.Api().run(run_path).history(samples=10_000)
    if history.empty:
        return history
    columns = [name for name in TRAINING_METRICS if name in history.columns]
    evaluation_columns = sorted(column for column in history.columns if str(column).startswith("evaluation/"))
    selected = columns + evaluation_columns
    if not selected:
        return history
    return history[selected].dropna(how="all")


def _step_series(history) -> tuple[Any, str]:
    if "train/environment_steps" in history.columns and history["train/environment_steps"].notna().any():
        return history["train/environment_steps"], "environment steps"
    if "_step" in history.columns:
        return history["_step"], "W&B step"
    return range(len(history)), "sample"


def _plot_metric_group(history, metrics: tuple[str, ...], title: str, output: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = {}
    for name in metrics:
        if name not in history.columns:
            continue
        values = history[name]
        for metric, denominator in (("recall", "positive"), ("false_positive_rate", "negative")):
            prefix = f"learner/legality_{metric}/"
            if name.startswith(prefix):
                count = f"learner/legality_{denominator}_count/{name.removeprefix(prefix)}"
                if count in history.columns:
                    values = values.where(history[count] > 0)
        if values.notna().any():
            series[name] = values
    present = list(series)
    if not present:
        return False
    steps, xlabel = _step_series(history)
    figure, axis = plt.subplots(figsize=(8, 4))
    for name in present:
        axis.plot(steps, series[name], label=name.split("/", 1)[-1])
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("value")
    axis.legend(loc="best")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return True


def _latest_evaluation_summary(run_path: Path) -> tuple[Path | None, dict[str, Any] | None]:
    summaries = sorted(run_path.glob("evaluation/*/summary.json"))
    if not summaries:
        return None, None
    latest = summaries[-1]
    return latest, json.loads(latest.read_text(encoding="utf-8"))


def _plot_evaluation_comparison(summary: dict[str, Any], output: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baselines = summary.get("baseline_comparisons")
    if not isinstance(baselines, list) or not baselines:
        return False
    labels = ["checkpoint"]
    values = [float(summary.get("mean_shaped_return", 0.0))]
    for item in baselines:
        if not isinstance(item, dict):
            continue
        policy_id = str(item.get("policy_id", "baseline"))
        labels.append(policy_id.rsplit(".", 1)[-1])
        values.append(float(item.get("mean_shaped_return", 0.0)))
    if len(labels) < 2:
        return False
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(labels, values)
    axis.set_title("Mean shaped return (evaluation)")
    axis.set_ylabel("mean_shaped_return")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return True


def _export_history(history, output: Path) -> list[str]:
    exported: list[str] = []
    if history.empty:
        return exported
    metrics_path = output / "metrics_history.json"
    records = history.where(history.notna(), None).to_dict(orient="records")
    metrics_path.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    exported.append(metrics_path.name)
    csv_path = output / "metrics_history.csv"
    history.to_csv(csv_path, index=False)
    exported.append(csv_path.name)
    return exported


def generate_report(run_directory: str | Path) -> Path:
    """Fetch W&B history, write plots and exported metrics under ``<run>/reports/``."""
    run_path = Path(run_directory).resolve()
    wandb_path = wandb_run_path(run_path)
    reports_dir = run_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    history = _fetch_wandb_history(wandb_path)
    plots: list[str] = []
    for filename, (metrics, title) in PLOT_GROUPS.items():
        if _plot_metric_group(history, metrics, title, reports_dir / f"{filename}.png"):
            plots.append(f"{filename}.png")

    evaluation_path, evaluation_summary = _latest_evaluation_summary(run_path)
    if evaluation_summary is not None:
        copied = reports_dir / "evaluation_summary.json"
        copied.write_text(json.dumps(evaluation_summary, indent=2, sort_keys=True), encoding="utf-8")
        if _plot_evaluation_comparison(evaluation_summary, reports_dir / "evaluation_baseline_comparison.png"):
            plots.append("evaluation_baseline_comparison.png")

    exported = _export_history(history, reports_dir)
    manifest = {
        "schema_version": "aresim.report.v1",
        "wandb_run": wandb_path,
        "plots": plots,
        "exported_files": exported,
        "evaluation_summary": str(evaluation_path.relative_to(run_path)) if evaluation_path else None,
        "metric_columns": [column for column in history.columns] if not history.empty else [],
    }
    (reports_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return reports_dir


__all__ = ["generate_report", "wandb_run_path"]
