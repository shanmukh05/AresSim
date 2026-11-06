"""Versioned MuZero checkpoint export and frozen search-policy inference.

The sidecar carries schema provenance and checksums. Frozen agents satisfy the
common Agent protocol used by evaluation and the live policy bridge.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..base import Agent
from ...training.artifacts import atomic_json, file_inventory, validate_inventory
from ..common.masks import require_mask
from .config import MuZeroConfig, MuZeroModelConfig
from .model import MuZeroNetwork
from .search import choose_action, search


CHECKPOINT_SCHEMA = "aresim.checkpoint.muzero.v1"


def write_checkpoint_sidecar(path: Path, experiment: dict[str, object], config_hash: str,
                             schemas: dict[str, str], *, environment_steps: int | None = None) -> Path:
    """Atomically publish metadata after the native files are complete."""
    native = path.parent / "native"
    payload = {
        "schema_version": CHECKPOINT_SCHEMA, "loader_id": "torch_muzero", "framework_id": "torch",
        "algorithm_id": "muzero", "model_id": "local_muzero", "native_path": "native",
        "config_hash": config_hash,
        "policy_id": f"{experiment['experiment_id']}:{experiment['trial_id']}:{path.parent.name}",
        "observation_schema": schemas["observation_schema"], "action_schema": schemas["action_schema"],
        "task_id": schemas["task_id"], "reward_profile": schemas["reward_profile"],
        "experiment": experiment, "native_inventory": file_inventory(native), "environment_steps": environment_steps,
    }
    atomic_json(path, payload)
    return path


def validate_sidecar(path: Path) -> tuple[dict[str, object], Path]:
    """Check sidecar identity, experiment hash, and each native file digest."""
    from ...training.experiments import experiment_payload_hash, parse_experiment

    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {"schema_version": CHECKPOINT_SCHEMA, "loader_id": "torch_muzero", "framework_id": "torch",
                "algorithm_id": "muzero", "model_id": "local_muzero",
                "observation_schema": "aresim.obs.local.v1", "action_schema": "aresim.action.rover.v1"}
    if (payload.get("observation_schema") not in {"aresim.obs.local.v1", "aresim.obs.local_memory.v2"}
            or any(payload.get(key) != value for key, value in expected.items() if key != "observation_schema")):
        raise ValueError("MuZero checkpoint provenance is incompatible")
    experiment = payload.get("experiment")
    if not isinstance(experiment, dict) or experiment_payload_hash(experiment) != payload.get("config_hash"):
        raise ValueError("MuZero checkpoint experiment hash is invalid")
    from ...training.provenance import validate_checkpoint_contract
    validate_checkpoint_contract(payload)
    native_value = Path(str(payload.get("native_path", "")))
    if native_value.is_absolute() or ".." in native_value.parts:
        raise ValueError("MuZero native checkpoint path is unsafe")
    native = path.parent / native_value
    inventory = payload.get("native_inventory")
    validate_inventory(native, inventory)
    if {entry["path"] for entry in inventory} != {"model.pt", "trainer.pt"}:
        raise ValueError("MuZero checkpoint must contain model and trainer state")
    return payload, native


def history_tensors(observations: list[dict[str, object]], actions: list[int],
                    length: int, device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Encode one recent history, padding with the first observation and no-action 10."""
    selected = [observations[max(0, len(observations) - length + offset)] for offset in range(length)]
    previous = [actions[position - 1] if position > 0 else 10
                for position in (len(observations) - length + offset for offset in range(length))]
    tensors = {key: torch.as_tensor(np.stack([np.asarray(item[key]) for item in selected]), device=device).unsqueeze(0)
               for key in selected[0]}
    return tensors, torch.tensor([previous], device=device)


def batch_history_tensors(observations: list[list[dict[str, object]]], actions: list[list[int]],
                          length: int, device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Collate independent histories before one device transfer per input field."""
    if not observations or len(observations) != len(actions):
        raise ValueError("MuZero history batch must contain aligned observations and actions")
    selected = [[items[max(0, len(items) - length + offset)] for offset in range(length)]
                for items in observations]
    previous = [[moves[position - 1] if position > 0 else 10
                 for position in (len(items) - length + offset for offset in range(length))]
                for items, moves in zip(observations, actions, strict=True)]
    tensors = {key: torch.as_tensor(np.stack([
        np.stack([np.asarray(item[key]) for item in history]) for history in selected]), device=device)
        for key in selected[0][0]}
    return tensors, torch.tensor(previous, device=device)


class MuZeroCheckpointAgent(Agent[dict[str, object], int]):
    """Keep only episode observation/action history while choosing legal actions."""

    observation_schema = "aresim.obs.local.v1"
    action_schema = "aresim.action.rover.v1"

    def __init__(self, network: MuZeroNetwork, config: MuZeroConfig, policy_id: str,
                 *, deterministic: bool = True) -> None:
        self.network = network.eval()
        self.config = config
        self.policy_id = policy_id
        self.deterministic = deterministic
        self.device = next(network.parameters()).device
        self.rng = np.random.default_rng(0)
        self.observations: list[dict[str, object]] = []
        self.actions: list[int] = []

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.observations.clear()
        self.actions.clear()

    def act(self, observation: dict[str, object], action_mask: np.ndarray) -> int:
        require_mask(action_mask)
        self.observations.append({key: np.asarray(value).copy() for key, value in observation.items()})
        tensors, previous = history_tensors(self.observations, self.actions,
                                            self.network.config.history_length, self.device)
        result = search(self.network, tensors, previous, action_mask, self.config,
                        training=False)
        action = choose_action(result, training=not self.deterministic,
                               temperature=self.config.temperature, rng=self.rng)
        self.actions.append(action)
        return action


class BuiltinCheckpointLoader:
    loader_id = "torch_muzero"

    def load(self, path: str, *, deterministic: bool = True) -> MuZeroCheckpointAgent:
        sidecar = Path(path)
        payload, native = validate_sidecar(sidecar)
        experiment = payload["experiment"]
        model_config = MuZeroModelConfig(**{**experiment["model_config"],
            "predict_constraints": experiment["model_config"].get("predict_constraints", False)})
        algorithm_config = MuZeroConfig(**experiment["algorithm_config"])
        from ...training.experiments import parse_experiment
        from ...training.provenance import environment_config
        spec = parse_experiment(experiment)
        network = MuZeroNetwork(model_config, environment_config(spec).observation_config.window_size,
                                memory=spec.environment.observation == "local_memory")
        network.load_state_dict(torch.load(native / "model.pt", map_location="cpu", weights_only=True))
        agent = MuZeroCheckpointAgent(network, algorithm_config, str(payload["policy_id"]), deterministic=deterministic)
        agent.observation_schema = str(payload["observation_schema"])
        from ...training.experiments import parse_experiment
        from ...training.provenance import environment_config
        agent.environment_config = environment_config(parse_experiment(payload["experiment"]))
        return agent


__all__ = ["BuiltinCheckpointLoader", "MuZeroCheckpointAgent", "history_tensors", "batch_history_tensors",
           "validate_sidecar", "write_checkpoint_sidecar"]
