"""Checkpoint sidecar loading for mask-aware DQN.

Reuses the shared ``aresim.checkpoint.rllib.v1`` sidecar writer and
:class:`~aresim.training.rllib.checkpoint.RLlibCheckpointAgent`. Provenance
requires ``masked_dqn`` / ``local_cnn_q``.

**Last updated:** October 7, 2026

**Loader id:** ``rllib_masked_dqn``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from ray.rllib.core.columns import Columns

from ...training.rllib.checkpoint import (
    RLlibCheckpointAgent, from_checkpoint, module_checkpoint, native_path,
    validate_inventory, validate_sidecar,
)
from ...training.experiments import parse_experiment



class DQNCheckpointAgent(RLlibCheckpointAgent):
    """Seeded epsilon-greedy inference over legal Q-values, without training."""

    def __init__(self, module, policy_id: str, *, deterministic: bool = False, epsilon: float = .05) -> None:
        super().__init__(module, policy_id, deterministic=deterministic)
        if not np.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError("DQN sampling epsilon must be in [0, 1]")
        self.epsilon = epsilon

    def act(self, observation: dict[str, object], action_mask: np.ndarray) -> int:
        legal = torch.as_tensor(action_mask, dtype=torch.bool, device="cpu")
        if legal.ndim != 1 or not legal.any():
            raise ValueError("DQN requires a nonempty legal action mask")
        batch = {Columns.OBS: {"observation": self._tensor(observation), "action_mask": self._tensor(action_mask)}}
        with torch.no_grad():
            # RLlib's forward_inference returns greedy ACTIONS. Read Q-values
            # directly so that this adapter owns seeded exploration instead.
            output = self.module.compute_q_values(batch) if hasattr(self.module, "compute_q_values") else self.module.forward_inference(batch)
        scores = output.get("qf_preds")
        if scores is None:
            raise ValueError("DQN inference did not return Q-values")
        values = scores[0].cpu()
        if values.shape != legal.shape or not torch.isfinite(values[legal]).all():
            raise ValueError("DQN inference returned incompatible or non-finite legal Q-values")
        if not self.deterministic and self.epsilon > 0 and torch.rand((), generator=self.generator).item() < self.epsilon:
            return int(torch.multinomial(legal.float(), 1, generator=self.generator).item())
        return int(values.masked_fill(~legal, -torch.inf).argmax().item())


class BuiltinCheckpointLoader:
    """Load masked-DQN RLlib checkpoints registered as ``rllib_masked_dqn``."""

    loader_id = "rllib_masked_dqn"

    def load(self, path: str, *, deterministic: bool = True) -> DQNCheckpointAgent:
        """Restore masked Q-values with optional seeded epsilon-greedy selection."""
        sidecar = Path(path)
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        validate_sidecar(payload, loader_id=self.loader_id, algorithm_id="masked_dqn", model_id="local_cnn_q")
        native = native_path(sidecar, payload)
        validate_inventory(native, payload.get("native_inventory"))
        module = from_checkpoint(module_checkpoint(native))
        spec = parse_experiment(payload["experiment"])
        agent = DQNCheckpointAgent(module, str(payload["policy_id"]), deterministic=deterministic,
                                   epsilon=spec.algorithm_config.epsilon_end)
        agent.observation_schema = str(payload["observation_schema"])
        from ...training.provenance import environment_config
        agent.environment_config = environment_config(spec)
        return agent



__all__ = ["BuiltinCheckpointLoader", "DQNCheckpointAgent"]
