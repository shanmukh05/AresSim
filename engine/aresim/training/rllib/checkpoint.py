"""RLlib checkpoint transport, validation and framework-neutral inference adapters.

Algorithm packages own their model-specific restoration and action selection.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule

from ...algorithms.base import Agent
from ..artifacts import atomic_json, file_inventory, validate_inventory
from ..checkpoints import RLLIB_CHECKPOINT_SCHEMA


CHECKPOINT_SCHEMA = RLLIB_CHECKPOINT_SCHEMA


def write_checkpoint_sidecar(
    path: str | Path,
    *,
    native_path: str,
    experiment: dict[str, object],
    config_hash: str,
    loader_id: str,
    observation_schema: str,
    action_schema: str,
    task_id: str,
    reward_profile: str,
    environment_steps: int | None = None,
) -> Path:
    """Write auditable JSON metadata beside one immutable native RLlib checkpoint.

    Atomically publishes ``checkpoint.json`` (via a ``.partial`` temp file) with
    experiment provenance, schema ids, and a SHA-256 inventory of native files.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    native = Path(native_path).resolve()
    if not native.exists():
        raise FileNotFoundError(f"native RLlib checkpoint does not exist: {native}")
    inventory = file_inventory(native)
    try:
        recorded_native_path = native.relative_to(destination.parent.resolve()).as_posix()
    except ValueError:
        recorded_native_path = str(native)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA, "loader_id": loader_id, "native_path": recorded_native_path,
        "framework_id": "rllib", "algorithm_id": str(experiment["algorithm"]), "model_id": str(experiment["model"]),
        "config_hash": config_hash, "policy_id": f"{experiment['experiment_id']}:{experiment['trial_id']}:{destination.parent.name}",
        "observation_schema": observation_schema, "action_schema": action_schema,
        "task_id": task_id, "reward_profile": reward_profile,
        "experiment": experiment, "native_inventory": inventory,
        "environment_steps": environment_steps,
    }
    atomic_json(destination, payload)
    return destination


def module_checkpoint(native: Path) -> Path:
    if (native / "metadata.json").exists():
        return native
    candidates = sorted(native.rglob("metadata.json"))
    module_candidates = [path.parent for path in candidates if "rl_module" in path.parts and "default_policy" in path.parts]
    if not module_candidates:
        raise ValueError("RLlib checkpoint does not contain a default-policy RLModule")
    return module_candidates[0]


class RLlibCheckpointAgent(Agent[dict[str, object], int]):
    """Run frozen policy inference through the framework-neutral ``Agent`` contract.

    Wraps a restored actor-critic (or RLModule fallback) and applies the same
    action mask at inference as during training. Supports deterministic argmax
    or stochastic sampling from masked logits.
    """

    observation_schema = "aresim.obs.local.v1"
    action_schema = "aresim.action.rover.v1"

    def __init__(self, module: Any, policy_id: str, deterministic: bool = True) -> None:
        """Bind a restored module; ``policy_id`` comes from the checkpoint sidecar."""
        self.module = module
        self.policy_id = policy_id
        self.deterministic = deterministic
        self.generator = torch.Generator(device="cpu")

    def reset(self, seed: int) -> None:
        """Re-seed stochastic action sampling for one evaluation episode."""
        self.generator.manual_seed(seed)

    @staticmethod
    def _tensor(value: object) -> Any:
        if isinstance(value, dict):
            return {key: RLlibCheckpointAgent._tensor(item) for key, item in value.items()}
        array = np.asarray(value)
        return torch.as_tensor(array).unsqueeze(0)

    def act(self, observation: dict[str, object], action_mask: np.ndarray) -> int:
        """Return one action index from masked policy logits or Q-values."""
        batch = {Columns.OBS: {"observation": self._tensor(observation), "action_mask": self._tensor(action_mask)}}
        with torch.no_grad():
            output = self.module.forward_inference(batch)
        if Columns.ACTIONS in output:
            return int(output[Columns.ACTIONS][0].cpu().item())
        scores = output.get(Columns.ACTION_DIST_INPUTS, output.get("qf_preds"))
        if scores is None:
            raise ValueError("RLModule inference did not return actions, logits, or Q-values")
        logits = scores[0].cpu()
        if self.deterministic:
            return int(torch.argmax(logits).item())
        return int(torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=self.generator).item())


def from_checkpoint(path: Path) -> RLModule:
    """Load an RLModule while hiding Ray's internal ``RLModuleConfig`` warning."""
    logger = logging.getLogger("ray._common.deprecation")

    class _DropConfigWarning(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "RLModule(config=" not in record.getMessage()

    filt = _DropConfigWarning()
    logger.addFilter(filt)
    try:
        return RLModule.from_checkpoint(path)
    finally:
        logger.removeFilter(filt)


def native_path(sidecar: Path, payload: dict[str, object]) -> Path:
    native = Path(str(payload["native_path"]))
    return native if native.is_absolute() else (sidecar.parent / native).resolve()


def validate_sidecar(payload: dict[str, object], *, loader_id: str, algorithm_id: str, model_id: str):
    """Require identity, schemas, configuration hash and model contract to agree."""
    from ..experiments import experiment_payload_hash
    from ..provenance import validate_checkpoint_contract

    expected = {"schema_version": CHECKPOINT_SCHEMA, "loader_id": loader_id,
                "framework_id": "rllib", "algorithm_id": algorithm_id,
                "model_id": model_id, "action_schema": "aresim.action.rover.v1"}
    if (any(payload.get(key) != value for key, value in expected.items())
            or payload.get("observation_schema") not in {"aresim.obs.local.v1", "aresim.obs.local_memory.v2"}):
        raise ValueError("RLlib checkpoint provenance is incompatible with this loader")
    for key in ("policy_id", "task_id", "reward_profile"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ValueError(f"RLlib checkpoint {key} is invalid")
    experiment = payload.get("experiment")
    if not isinstance(experiment, dict) or experiment_payload_hash(experiment) != payload.get("config_hash"):
        raise ValueError("RLlib checkpoint configuration hash is invalid")
    return validate_checkpoint_contract(payload)


__all__ = ["CHECKPOINT_SCHEMA", "RLlibCheckpointAgent", "from_checkpoint", "module_checkpoint",
           "native_path", "validate_inventory", "validate_sidecar", "write_checkpoint_sidecar"]
