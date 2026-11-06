"""Plan2Explore checkpoint identity over the shared recurrent world-model loader.

Sidecar phase selects exploration versus task behavior without a separate UI control.
"""
from pathlib import Path

from ..dreamerv3.checkpoint import BuiltinCheckpointLoader as WorldModelLoader, validate_sidecar
from ...training.provenance import environment_config, experiment_seeds, serialize, verify_provenance


def validate_transfer(sidecar: Path, spec):
    """Require a complete pretraining export compatible with the adaptation task."""
    payload, parent, native = validate_sidecar(sidecar, "plan2explore")
    if spec.algorithm != "plan2explore" or spec.algorithm_config.phase != "adapt":
        raise ValueError("pretraining initialization is only available for Plan2Explore adaptation")
    if payload["phase"] != "pretrain":
        raise ValueError("adaptation requires a pretraining parent")
    verify_provenance(parent)
    before = serialize(environment_config(parent))
    after = serialize(environment_config(spec))
    for key in ("task", "reward", "reward_config"):
        before.pop(key)
        after.pop(key)
    if (before != after or parent.model_config != spec.model_config or parent.algorithm_config.gamma != spec.algorithm_config.gamma
            or experiment_seeds(parent) != experiment_seeds(spec)
            or parent.environment.max_episode_steps != spec.environment.max_episode_steps
            or parent.algorithm_config.ensemble_size != spec.algorithm_config.ensemble_size):
        raise ValueError("Plan2Explore transfer physics, model, episode limit, or seed family differs")
    return payload, parent, native



class BuiltinCheckpointLoader(WorldModelLoader):
    """Reject Dreamer sidecars while retaining the shared portable native format."""

    loader_id = "torch_plan2explore"
    algorithm_id = "plan2explore"
