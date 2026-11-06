"""Stateful epsilon-greedy inference for recurrent DQN native RLlib checkpoints.

Every agent owns its LSTM state and generator; cached models must never share episode memory.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from ray.rllib.core.columns import Columns
from ..dqn.checkpoint import DQNCheckpointAgent
from ...training.rllib.checkpoint import native_path, validate_inventory, module_checkpoint, from_checkpoint, validate_sidecar
from ...training.provenance import environment_config


class RecurrentDQNAgent(DQNCheckpointAgent):
    """Advance LSTM state once per observation, including random exploration decisions."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.state = None

    def reset_memory(self) -> None:
        self.state = None

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.reset_memory()

    def act(self, observation: dict[str, object], action_mask: np.ndarray) -> int:
        from ..common.masks import require_mask
        mask = require_mask(action_mask)
        if not mask.any():
            raise ValueError("recurrent DQN requires a legal action")
        batch = {Columns.OBS: {"observation": self._tensor(observation), "action_mask": self._tensor(mask)}}
        if self.state is not None:
            batch[Columns.STATE_IN] = self.state
        with torch.no_grad():
            output = self.module.compute_q_values(batch)
        self.state = output[Columns.STATE_OUT]
        values = output["qf_preds"][0].cpu()
        legal = torch.as_tensor(mask, dtype=torch.bool)
        if not torch.isfinite(values[legal]).all():
            raise ValueError("recurrent DQN returned non-finite legal Q-values")
        if not self.deterministic and torch.rand((), generator=self.generator) < self.epsilon:
            return int(torch.multinomial(legal.float(), 1, generator=self.generator).item())
        return int(values.masked_fill(~legal, -torch.inf).argmax().item())


class BuiltinCheckpointLoader:
    """Validate identity/inventory and restore the stateful RLModule."""

    loader_id = "rllib_recurrent_dqn"

    def load(self, path: str, *, deterministic: bool = True) -> RecurrentDQNAgent:
        sidecar = Path(path)
        payload = json.loads(sidecar.read_text())
        spec = validate_sidecar(payload, loader_id=self.loader_id, algorithm_id="recurrent_dqn", model_id="local_cnn_lstm_q")
        native = native_path(sidecar, payload)
        validate_inventory(native, payload.get("native_inventory"))
        module = from_checkpoint(module_checkpoint(native))
        agent = RecurrentDQNAgent(module, payload["policy_id"], deterministic=deterministic, epsilon=spec.algorithm_config.epsilon_end)
        agent.observation_schema = payload["observation_schema"]
        agent.environment_config = environment_config(spec)
        return agent
