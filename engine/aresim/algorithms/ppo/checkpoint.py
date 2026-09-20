"""Checkpoint sidecar I/O and frozen-policy inference for masked PPO and DQN.

Owns the deploy/eval path: writing auditable JSON beside native RLlib checkpoints,
validating provenance and file inventory, and adapting frozen actor-critic weights
to the framework-neutral :class:`~aresim.algorithms.base.Agent` contract. Inference
loads ``module_state.pkl`` into ``LocalMaskedActorCritic`` so Ray's deprecated
``RLModuleConfig`` is not constructed.

**Last updated:** September 12, 2026

**Contains:** ``CHECKPOINT_SCHEMA``, :func:`write_checkpoint_sidecar`,
:class:`RLlibCheckpointAgent`, :class:`BuiltinCheckpointLoader`,
:func:`make_checkpoint_agent`.

**Loader id:** ``rllib_masked_ppo`` (registered in :mod:`aresim.algorithms.registry`).

**Sidecar schema:** ``aresim.checkpoint.rllib.v1`` — stores experiment hash,
schema ids, native path, and SHA-256 inventory of framework files.

**Dependencies:** Ray RLlib, PyTorch, NumPy. Not imported by baseline-only code paths.

**See also:** :mod:`aresim.algorithms.ppo.train` for training-time checkpoint export;
:mod:`aresim.training.evaluation` for fixed-seed rollout comparison.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule

from ..base import Agent


CHECKPOINT_SCHEMA = "aresim.checkpoint.rllib.v1"


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
    inventory = []
    for item in sorted(candidate for candidate in native.rglob("*") if candidate.is_file()):
        inventory.append({
            "path": item.relative_to(native).as_posix(),
            "size": item.stat().st_size,
            "sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
        })
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
    }
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def _module_checkpoint(native: Path) -> Path:
    if (native / "metadata.json").exists():
        return native
    candidates = sorted(native.rglob("metadata.json"))
    module_candidates = [path.parent for path in candidates if "rl_module" in path.parts and "default_policy" in path.parts]
    if not module_candidates:
        raise ValueError("RLlib checkpoint does not contain a default-policy RLModule")
    return module_candidates[0]


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
        _validate_sidecar(payload, self.loader_id)
        native = _native_path(sidecar, payload)
        _validate_inventory(native, payload.get("native_inventory"))
        module = _load_inference_module(native, payload)
        return RLlibCheckpointAgent(module, str(payload["policy_id"]), deterministic=deterministic)


def _actor_state_dict(state: dict[str, object]) -> dict[str, object]:
    """Map ``core.*`` RLModule keys onto :class:`LocalMaskedActorCritic`."""
    stripped = {str(key).removeprefix("core."): value for key, value in state.items()}
    if any(key.startswith("encoder.") for key in stripped):
        return stripped
    return {
        key if key.startswith(("policy.", "value.")) else f"encoder.{key}": value
        for key, value in stripped.items()
    }


def _from_checkpoint(path: Path) -> RLModule:
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


def _load_inference_module(native: Path, payload: dict[str, object]) -> Any:
    """Restore frozen inference without constructing Ray's deprecated module config."""
    module_dir = _module_checkpoint(native)
    state_path = module_dir / "module_state.pkl"
    if not state_path.is_file():
        return _from_checkpoint(module_dir)
    state = pickle.loads(state_path.read_bytes())
    if not isinstance(state, dict) or not state or not all(str(key).startswith("core.") for key in state):
        return _from_checkpoint(module_dir)
    from ...training.experiments import parse_experiment
    from .train import LocalMaskedActorCritic

    spec = parse_experiment(payload["experiment"])
    core = LocalMaskedActorCritic(spec.model_config)
    try:
        core.load_state_dict({key: torch.as_tensor(value) for key, value in _actor_state_dict(state).items()})
    except RuntimeError:
        return _from_checkpoint(module_dir)
    return _FrozenActorCritic(core)


def _validate_sidecar(payload: dict[str, object], loader_id: str) -> None:
    if payload.get("schema_version") != CHECKPOINT_SCHEMA or payload.get("loader_id") != loader_id:
        raise ValueError("unsupported RLlib checkpoint sidecar")
    if any(payload.get(key) != value for key, value in _EXPECTED_PROVENANCE.items()):
        raise ValueError("RLlib checkpoint provenance is incompatible with this loader")
    if not all(_nonempty_text(payload.get(key)) for key in ("task_id", "reward_profile")):
        raise ValueError("RLlib checkpoint task/reward provenance is invalid")
    from ...training.experiments import parse_experiment

    experiment = payload.get("experiment")
    if not isinstance(experiment, dict) or parse_experiment(experiment).config_hash != payload.get("config_hash"):
        raise ValueError("RLlib checkpoint configuration hash is invalid")


_EXPECTED_PROVENANCE = {
    "framework_id": "rllib",
    "algorithm_id": "masked_ppo",
    "model_id": "local_cnn_actor_critic",
    "observation_schema": "aresim.obs.local.v1",
    "action_schema": "aresim.action.rover.v1",
}


def _nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _native_path(sidecar: Path, payload: dict[str, object]) -> Path:
    native = Path(str(payload["native_path"]))
    return native if native.is_absolute() else (sidecar.parent / native).resolve()


def _validate_inventory(native: Path, inventory: object) -> None:
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("RLlib checkpoint has no native file inventory")
    for entry in inventory:
        _validate_inventory_entry(native, entry)


def _validate_inventory_entry(native: Path, entry: object) -> None:
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise ValueError("RLlib checkpoint inventory is malformed")
    relative = Path(entry["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("RLlib checkpoint inventory contains an unsafe path")
    item = native / relative
    valid = (
        item.is_file()
        and item.stat().st_size == entry.get("size")
        and hashlib.sha256(item.read_bytes()).hexdigest() == entry.get("sha256")
    )
    if not valid:
        raise ValueError("RLlib checkpoint native file checksum mismatch")


def make_checkpoint_agent(path: str | Path, registry=None, *, deterministic: bool = True) -> RLlibCheckpointAgent:
    """Load a checkpoint sidecar via the training registry as a public ``Agent``.

    Reads ``loader_id`` from the JSON at ``path`` and dispatches to the matching
    registered loader. This is the supported entry point for evaluation and UI
    diagnostics.
    """
    from ..registry import create_training_registry

    selected = create_training_registry() if registry is None else registry
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return selected.build_checkpoint_loader(str(payload.get("loader_id", ""))).load(str(path), deterministic=deterministic)


__all__ = [
    "CHECKPOINT_SCHEMA",
    "BuiltinCheckpointLoader",
    "RLlibCheckpointAgent",
    "make_checkpoint_agent",
    "write_checkpoint_sidecar",
]
