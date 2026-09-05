"""Rollout, trajectory, and optional learned-policy training public API.

Orchestration layer: does not implement simulator rules or algorithm training
internals. Learned PPO lives in :mod:`aresim.algorithms.ppo`; baselines in
:mod:`aresim.algorithms`.

**Last updated:** September 1, 2026

**Contains:** lazy exports for ``run_experiment``, ``make_checkpoint_agent``,
rollout/trajectory types, and training registry symbols.

**CLI entry:** ``aresim-rl`` → :mod:`aresim.training.cli`.

**Dependencies:** core rollouts need ``env`` extra; PPO paths need ``rllib`` extra.
"""

from .experiments import ExperimentSpec, load_experiment_with_overrides
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


def load_experiment(path, registry=None):
    """Load YAML using the selected algorithm factory's typed decoder."""
    import yaml

    from .experiments import parse_experiment
    from .seeds import resolve_checkout_file

    payload = yaml.safe_load(resolve_checkout_file(path, kind="experiment config").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("algorithm"), str):
        raise ValueError("experiment must select a registered algorithm")
    selected = create_training_registry() if registry is None else registry
    decoder = lambda raw: selected.decode_algorithm_config(payload["algorithm"], raw)
    return parse_experiment(payload, algorithm_config_decoder=decoder)


def run_experiment(*args, **kwargs):
    from ..algorithms.ppo.train import run_experiment as run

    return run(*args, **kwargs)


def make_checkpoint_agent(*args, **kwargs):
    from ..algorithms.ppo.checkpoint import make_checkpoint_agent as make

    return make(*args, **kwargs)


def evaluate_checkpoint(*args, **kwargs):
    from .evaluation import evaluate_checkpoint as evaluate

    return evaluate(*args, **kwargs)


def generate_report(*args, **kwargs):
    from .reports import generate_report as generate

    return generate(*args, **kwargs)


def __getattr__(name: str):
    if name in {"AlgorithmFactory", "CheckpointLoader", "ModelFactory", "TrainingContext", "TrainingRegistry"}:
        from ..algorithms.registry import (
            AlgorithmFactory,
            CheckpointLoader,
            ModelFactory,
            TrainingContext,
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
    "TrainingRegistry",
    "create_training_registry",
    "evaluate_checkpoint",
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
