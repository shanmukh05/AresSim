"""Periodic held-out validation and preservation of the best complete checkpoint."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .provenance import digest, environment_config


def validation_score(summary, spec):
    """Never rank by shaped rewards or uncapped resource farming."""
    task = environment_config(spec).task_config
    delivery = summary.get("mean_capped_delivery_progress")
    if delivery is None:
        delivery = sum(min(1., float(summary.get(key, 0.)) / target) for key, target in (
            ("mean_ice_delivered", task.ice_target_kg), ("mean_samples_delivered", task.sample_target_kg)) if target > 0)
    return (float(summary.get("success_rate", 0.)), delivery,
            float(summary["mean_sparse_evaluation_return"]))


def validate_and_select(checkpoint: Path, spec, run_directory: Path, *, registry=None, component_registry=None):
    """Use validation seeds only; best survives rotating checkpoint pruning."""
    from .evaluation import evaluate_checkpoint

    payload = json.loads(checkpoint.read_text())
    fingerprint = digest(payload)
    destination = run_directory / "evaluation" / f"{checkpoint.parent.name}-{fingerprint[:12]}"
    if not (destination / "summary.json").is_file():
        if destination.exists():
            import time
            destination.rename(destination.with_name(f"{destination.name}.incomplete-{time.time_ns()}"))
        evaluate_checkpoint(checkpoint, split="validation", output_directory=destination,
                            registry=registry, component_registry=component_registry, record_trajectories=False)
    summary = json.loads((destination / "summary.json").read_text())
    summary = summary["action_modes"][spec.evaluation.selection_action_mode]
    score = validation_score(summary, spec)
    selection_path = run_directory / "checkpoint_selection.json"
    previous = json.loads(selection_path.read_text()) if selection_path.is_file() else None
    if previous is not None and score <= tuple(previous["score"]):
        return destination
    best = run_directory / "checkpoints" / "best"
    temporary = best.with_name("best.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(checkpoint.parent, temporary)
    payload["native_path"] = "native"
    (temporary / "checkpoint.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if best.exists():
        shutil.rmtree(best)
    temporary.replace(best)
    selection = {"schema_version": "aresim.checkpoint_selection.v1", "split": "validation",
                 "action_mode": summary["action_mode"], "score": score,
                 "metric_order": ["success_rate", "capped_delivery_progress", "sparse_return"],
                 "source_checkpoint": str(checkpoint), "evaluation": str(destination),
                 "checkpoint": "checkpoints/best/checkpoint.json"}
    temporary_selection = selection_path.with_suffix(".json.partial")
    temporary_selection.write_text(json.dumps(selection, indent=2) + "\n")
    temporary_selection.replace(selection_path)
    return destination


def require_success_gate(spec, run_directory):
    """Keep artifacts and stop scaling when a configured validation gate fails."""
    threshold = spec.evaluation.minimum_success_rate
    if threshold is None:
        return
    selection = json.loads((run_directory / "checkpoint_selection.json").read_text())
    actual = float(selection["score"][0])
    if actual < threshold:
        raise RuntimeError(f"validation success gate failed: {actual:.1%} < {threshold:.1%}; best checkpoint and evaluation retained")
