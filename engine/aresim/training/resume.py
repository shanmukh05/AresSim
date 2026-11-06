"""Framework-neutral experiment compatibility for training continuation."""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .experiments import ExperimentSpec


def require_resume_compatible(stored: ExperimentSpec, current: ExperimentSpec) -> None:
    """Reject resume when anything besides the training budget changed."""
    from .provenance import verify_provenance
    verify_provenance(stored, training=True)
    if stored.provenance != current.provenance:
        raise ValueError("resume provenance differs")
    stored_dict = stored.as_dict()
    current_dict = current.as_dict()
    for key in (
        "schema_version",
        "experiment_id",
        "trial_id",
        "algorithm",
        "model",
        "learner_seed",
        "environment",
        "model_config",
        "resources",
        "evaluation",
        "checkpoint",
        "tracking",
    ):
        if stored_dict[key] != current_dict[key]:
            raise ValueError(f"resume experiment mismatch: {key}")
    stored_algo = dict(stored_dict["algorithm_config"])
    current_algo = dict(current_dict["algorithm_config"])
    stored_steps = stored_algo.pop("total_environment_steps")
    current_steps = current_algo.pop("total_environment_steps")
    if stored_algo != current_algo:
        raise ValueError("resume algorithm_config mismatch")
    if current_steps < stored_steps:
        raise ValueError("total_environment_steps cannot be reduced on resume")


