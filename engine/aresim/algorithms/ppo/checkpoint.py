"""Frozen masked-PPO restoration and historical checkpoint exports.

Shared RLlib transport and validation live in training.rllib.checkpoint.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import torch
from ray.rllib.core.columns import Columns

from ...training.checkpoints import make_checkpoint_agent
from ...training.rllib.checkpoint import (
    CHECKPOINT_SCHEMA, RLlibCheckpointAgent, from_checkpoint, module_checkpoint,
    native_path, validate_inventory, validate_sidecar, write_checkpoint_sidecar,
)


class _FrozenActorCritic:
    """``forward_inference`` adapter over :class:`LocalMaskedActorCritic`.

    Avoids constructing Ray's ``RLModule``, which still instantiates deprecated
    ``RLModuleConfig`` and warns once per process.
    """

    def __init__(self, core: torch.nn.Module) -> None:
        self.core = core.eval()

    def forward_inference(self, batch: dict[str, Any]) -> dict[str, Any]:
        policy_input = batch[Columns.OBS]
        logits, _ = self.core(policy_input["observation"], policy_input["action_mask"])
        return {Columns.ACTION_DIST_INPUTS: logits}


class BuiltinCheckpointLoader:
    """Load masked-PPO RLlib checkpoints registered as ``rllib_masked_ppo``.

    Validates sidecar schema, provenance fields, config hash, and native file
    inventory before restoring frozen actor-critic weights for inference.
    """

    loader_id = "rllib_masked_ppo"

    def load(self, path: str, *, deterministic: bool = True) -> RLlibCheckpointAgent:
        """Restore ``path`` (sidecar JSON) into an :class:`RLlibCheckpointAgent`."""
        sidecar = Path(path)
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        validate_sidecar(payload, loader_id=self.loader_id, algorithm_id="masked_ppo", model_id="local_cnn_actor_critic")
        native = native_path(sidecar, payload)
        validate_inventory(native, payload.get("native_inventory"))
        module = _load_inference_module(native, payload)
        agent = RLlibCheckpointAgent(module, str(payload["policy_id"]), deterministic=deterministic)
        agent.observation_schema = str(payload["observation_schema"])
        from ...training.experiments import parse_experiment
        from ...training.provenance import environment_config
        agent.environment_config = environment_config(parse_experiment(payload["experiment"]))
        return agent


def _actor_state_dict(state: dict[str, object]) -> dict[str, object]:
    """Map ``core.*`` RLModule keys onto :class:`LocalMaskedActorCritic`."""
    stripped = {str(key).removeprefix("core."): value for key, value in state.items()}
    if any(key.startswith("encoder.") for key in stripped):
        return stripped
    return {
        key if key.startswith(("policy.", "value.")) else f"encoder.{key}": value
        for key, value in stripped.items()
    }


def _load_inference_module(native: Path, payload: dict[str, object]) -> Any:
    """Restore frozen inference without constructing Ray's deprecated module config."""
    module_dir = module_checkpoint(native)
    state_path = module_dir / "module_state.pkl"
    if not state_path.is_file():
        return from_checkpoint(module_dir)
    state = pickle.loads(state_path.read_bytes())
    if not isinstance(state, dict) or not state or not all(str(key).startswith("core.") for key in state):
        return from_checkpoint(module_dir)
    from ...training.experiments import parse_experiment
    from .train import LocalMaskedActorCritic

    spec = parse_experiment(payload["experiment"])
    from ...training.provenance import environment_config
    config = environment_config(spec)
    core = LocalMaskedActorCritic(spec.model_config, config.observation_config.window_size,
                                  memory=spec.environment.observation == "local_memory")
    try:
        core.load_state_dict({key: torch.as_tensor(value) for key, value in _actor_state_dict(state).items()})
    except RuntimeError:
        return from_checkpoint(module_dir)
    return _FrozenActorCritic(core)


__all__ = [
    "CHECKPOINT_SCHEMA",
    "BuiltinCheckpointLoader",
    "RLlibCheckpointAgent",
    "make_checkpoint_agent",
    "write_checkpoint_sidecar",
]
