"""Framework-neutral rollouts, evaluation and lazy registered training entry points.

Algorithms own networks/objectives; training.rllib and training.world_models own their
respective collection/lifecycle integrations. Importing this surface does not load Torch or Ray.
"""

from .experiments import ExperimentSpec, load_experiment, load_experiment_with_overrides
from .checkpoints import make_checkpoint_agent
from .runner import DEFAULT_MAX_EPISODE_STEPS, EpisodeSpec, EpisodeSummary, RolloutConfig, RolloutResult, RolloutRunner
from .seeds import SeedSplitManifest, load_seed_manifest
from .trajectories import (
    EpisodeTrajectory,
    TrajectoryManifest,
    TrajectoryValidationError,
    TrajectoryWriter,
    iter_trajectory_episodes,
    validate_trajectory_episode,
    validate_trajectory_dataset,
)

def create_training_registry():
    from ..algorithms.registry import create_training_registry as create

    return create()


def run_experiment(spec, registry=None, **kwargs):
    """Resolve once and dispatch through the explicitly registered training runner."""
    from pathlib import Path
    selected = create_training_registry() if registry is None else registry
    resolved = load_experiment(spec, registry=selected) if isinstance(spec, (str, Path)) else spec
    return selected.run(resolved, **kwargs)


def evaluate_checkpoint(*args, **kwargs):
    from .evaluation import evaluate_checkpoint as evaluate

    return evaluate(*args, **kwargs)


def run_panel(*args, **kwargs):
    """Generate shared worlds, score the configured checkpoints, and write raw rows plus bars."""
    from .panel import run_panel as run

    return run(*args, **kwargs)


def generate_report(*args, **kwargs):
    from .reports import generate_report as generate

    return generate(*args, **kwargs)


def __getattr__(name: str):
    if name in {"AlgorithmFactory", "CheckpointLoader", "ModelFactory", "TrainingContext", "TrainingRunner", "TrainingRegistry"}:
        from ..algorithms.registry import (
            AlgorithmFactory,
            CheckpointLoader,
            ModelFactory,
            TrainingContext,
            TrainingRunner,
            TrainingRegistry,
        )

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EpisodeTrajectory",
    "AlgorithmFactory",
    "CheckpointLoader",
    "ExperimentSpec",
    "ModelFactory",
    "DEFAULT_MAX_EPISODE_STEPS",
    "EpisodeSpec",
    "EpisodeSummary",
    "RolloutConfig",
    "RolloutResult",
    "RolloutRunner",
    "SeedSplitManifest",
    "TrajectoryManifest",
    "TrajectoryValidationError",
    "TrajectoryWriter",
    "TrainingContext",
    "TrainingRunner",
    "TrainingRegistry",
    "create_training_registry",
    "evaluate_checkpoint",
    "run_panel",
    "generate_report",
    "iter_trajectory_episodes",
    "load_experiment",
    "load_experiment_with_overrides",
    "load_seed_manifest",
    "make_checkpoint_agent",
    "run_experiment",
    "validate_trajectory_episode",
    "validate_trajectory_dataset",
]
