"""Lazy public exports for the masked PPO algorithm package.

Import lightweight symbols from :mod:`aresim.algorithms.ppo.config` directly when
avoiding Ray/Torch. Training and checkpoint symbols load on first attribute access.

**Last updated:** September 1, 2026

**Contains:** lazy exports for config, train, and checkpoint modules.

**Entry points:** :func:`run_experiment`, :func:`make_checkpoint_agent` (also
re-exported from :mod:`aresim.training`).
"""

from .config import MaskedPPOConfig, ModelConfig

__all__ = [
    "AresMaskedPPORLModule",
    "LocalMaskedActorCritic",
    "MaskedPPOConfig",
    "MaskedPPOFactory",
    "ModelConfig",
    "canonicalize_rllib_metrics",
    "make_checkpoint_agent",
    "run_experiment",
]


def __getattr__(name: str):
    if name in {
        "AresMaskedPPORLModule",
        "LocalMaskedActorCritic",
        "MaskedPPOFactory",
        "canonicalize_rllib_metrics",
        "run_experiment",
    }:
        from .train import (
            AresMaskedPPORLModule,
            LocalMaskedActorCritic,
            MaskedPPOFactory,
            canonicalize_rllib_metrics,
            run_experiment,
        )

        return locals()[name]
    if name == "make_checkpoint_agent":
        from .checkpoint import make_checkpoint_agent

        return make_checkpoint_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
