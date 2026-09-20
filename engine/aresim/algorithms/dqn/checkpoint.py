"""Checkpoint sidecar loading for mask-aware DQN.

Reuses the shared ``aresim.checkpoint.rllib.v1`` sidecar writer and
:class:`~aresim.algorithms.ppo.checkpoint.RLlibCheckpointAgent`. Provenance
requires ``masked_dqn`` / ``local_cnn_q``.

**Last updated:** September 12, 2026

**Loader id:** ``rllib_masked_dqn``.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..ppo.checkpoint import (
    CHECKPOINT_SCHEMA,
    RLlibCheckpointAgent,
    _from_checkpoint,
    _module_checkpoint,
    _native_path,
    _nonempty_text,
    _validate_inventory,
)
from ...training.experiments import parse_experiment


class BuiltinCheckpointLoader:
    """Load masked-DQN RLlib checkpoints registered as ``rllib_masked_dqn``."""

    loader_id = "rllib_masked_dqn"

    def load(self, path: str, *, deterministic: bool = True) -> RLlibCheckpointAgent:
        """Restore ``path`` (sidecar JSON) into an :class:`RLlibCheckpointAgent`."""
        sidecar = Path(path)
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        _validate_sidecar(payload, self.loader_id)
        native = _native_path(sidecar, payload)
        _validate_inventory(native, payload.get("native_inventory"))
        module = _from_checkpoint(_module_checkpoint(native))
        return RLlibCheckpointAgent(module, str(payload["policy_id"]), deterministic=deterministic)


def _validate_sidecar(payload: dict[str, object], loader_id: str) -> None:
    if payload.get("schema_version") != CHECKPOINT_SCHEMA or payload.get("loader_id") != loader_id:
        raise ValueError("unsupported RLlib checkpoint sidecar")
    expected = {
        "framework_id": "rllib",
        "algorithm_id": "masked_dqn",
        "model_id": "local_cnn_q",
        "observation_schema": "aresim.obs.local.v1",
        "action_schema": "aresim.action.rover.v1",
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("RLlib checkpoint provenance is incompatible with this loader")
    if not all(_nonempty_text(payload.get(key)) for key in ("task_id", "reward_profile")):
        raise ValueError("RLlib checkpoint task/reward provenance is invalid")
    experiment = payload.get("experiment")
    if not isinstance(experiment, dict) or parse_experiment(experiment).config_hash != payload.get("config_hash"):
        raise ValueError("RLlib checkpoint configuration hash is invalid")


__all__ = ["BuiltinCheckpointLoader"]
