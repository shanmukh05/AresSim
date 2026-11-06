"""Resolve native RLlib training continuation from run directories and sidecars.

Experiment compatibility is shared with native trainers via training.resume.
This module classifies paths; it does not start Ray or restore weights.
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ...training.resume import require_resume_compatible
from ...training.checkpoints import RLLIB_CHECKPOINT_SCHEMA
from ...training.experiments import EXPERIMENT_SCHEMA, parse_experiment

if TYPE_CHECKING:
    from ...training.experiments import ExperimentSpec

CHECKPOINT_SCHEMA = RLLIB_CHECKPOINT_SCHEMA
_STEP_LABEL = re.compile(r"^step_(\d+)$")


@dataclass(frozen=True)
class ResolvedResume:
    """Validated run ownership, native path and completed-step information."""

    run_directory: Path
    ray_directory: Path
    native_checkpoint: Path | None
    completed_environment_steps: int | None
    explicit_checkpoint: bool


def _native_from_sidecar(sidecar: Path) -> Path:
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    native = Path(str(payload["native_path"]))
    return native if native.is_absolute() else (sidecar.parent / native).resolve()


def _validate_resume_sidecar(sidecar: Path, spec: ExperimentSpec) -> dict[str, object]:
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported checkpoint sidecar: {sidecar}")
    experiment = payload.get("experiment")
    if not isinstance(experiment, dict) or experiment.get("schema_version") != EXPERIMENT_SCHEMA:
        raise ValueError(f"checkpoint experiment envelope is invalid: {sidecar}")
    stored = parse_experiment(experiment)
    require_resume_compatible(stored, spec)
    return payload


def _environment_steps_from_label(label: str, rollout_batch_size: int, native: Path | None) -> int | None:
    match = _STEP_LABEL.match(label)
    if match:
        return int(match.group(1))
    if label != "final" or native is None:
        return None
    state_path = native / "algorithm_state.pkl"
    if not state_path.is_file():
        return None
    with state_path.open("rb") as handle:
        state = pickle.load(handle)
    iteration = state.get("training_iteration")
    if isinstance(iteration, int) and not isinstance(iteration, bool):
        return iteration * rollout_batch_size
    return None


def _locate_resume_files(path: Path) -> tuple[Path | None, Path | None]:
    """Return ``(sidecar, native_dir_hint)`` for a user-supplied resume path."""
    if path.is_file():
        return (path, None) if path.name == "checkpoint.json" else (None, None)
    if not path.is_dir():
        return None, None
    sidecar = path / "checkpoint.json"
    if sidecar.is_file():
        return sidecar, None
    if (path / "rllib_checkpoint.json").is_file():
        candidate = path.parent / "checkpoint.json"
        return (candidate if candidate.is_file() else None), path
    if (path / "native" / "rllib_checkpoint.json").is_file():
        return None, (path / "native").resolve()
    return None, None


def _resume_from_sidecar(sidecar: Path, native: Path | None, spec: ExperimentSpec) -> ResolvedResume:
    payload = _validate_resume_sidecar(sidecar, spec)
    native = native or _native_from_sidecar(sidecar)
    if not native.is_dir() or not (native / "rllib_checkpoint.json").is_file():
        raise FileNotFoundError(f"native RLlib checkpoint does not exist: {native}")
    resolved = _resolved(
        sidecar.parents[2],
        native,
        spec.algorithm_config.rollout_batch_size,
        sidecar.parent.name,
        explicit_checkpoint=True,
    )
    actual_steps = payload.get("environment_steps")
    if isinstance(actual_steps, int) and not isinstance(actual_steps, bool):
        from dataclasses import replace
        resolved = replace(resolved, completed_environment_steps=actual_steps)
    return resolved


def _resume_from_run(run_directory: Path, spec: ExperimentSpec) -> ResolvedResume:
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    experiment = manifest.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("run manifest is missing experiment provenance")
    require_resume_compatible(parse_experiment(experiment), spec)
    return _resolved(run_directory, None, spec.algorithm_config.rollout_batch_size, None, explicit_checkpoint=False)


def _resolved(
    run_directory: Path,
    native: Path | None,
    rollout_batch_size: int,
    label: str | None,
    *,
    explicit_checkpoint: bool,
) -> ResolvedResume:
    completed = (
        _environment_steps_from_label(label, rollout_batch_size, native) if label else None
    )
    return ResolvedResume(
        run_directory=run_directory,
        ray_directory=run_directory / "ray",
        native_checkpoint=native,
        completed_environment_steps=completed,
        explicit_checkpoint=explicit_checkpoint,
    )


def resolve_resume_target(value: str | Path, spec: ExperimentSpec) -> ResolvedResume:
    """Resolve a run directory or checkpoint path into a validated resume target."""
    requested = Path(value).expanduser()
    default_run = Path(spec.artifacts.root).resolve() / spec.experiment_id / spec.trial_id
    path = (default_run / requested).resolve() if not requested.is_absolute() else requested.resolve()
    sidecar, native = _locate_resume_files(path)
    if sidecar is not None:
        return _resume_from_sidecar(sidecar, native, spec)
    if path.is_dir() and (path / "manifest.json").is_file():
        return _resume_from_run(path, spec)
    raise FileNotFoundError(f"resume target not found: {path}")
