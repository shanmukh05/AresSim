"""Lazy public exports for the mask-aware DQN algorithm package.

Import :class:`MaskedDQNConfig` directly when avoiding Ray/Torch. Training and
checkpoint symbols load on first attribute access.

**Last updated:** September 12, 2026
"""

from .config import MaskedDQNConfig

__all__ = [
    "AresMaskedDQNRLModule",
    "LocalMaskedQNetwork",
    "MaskedDQNConfig",
    "MaskedDQNFactory",
]


def __getattr__(name: str):
    if name in {"AresMaskedDQNRLModule", "LocalMaskedQNetwork", "MaskedDQNFactory"}:
        from .train import AresMaskedDQNRLModule, LocalMaskedQNetwork, MaskedDQNFactory

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
