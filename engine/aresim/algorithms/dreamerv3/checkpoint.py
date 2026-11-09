"""Portable world-model sidecars and seeded recurrent actor inference.

Frozen loading validates native files without loading trainer replay or importing Ray/RLlib.
Dreamer and Plan2Explore retain distinct registered loader identities and active behaviors.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
from ..common.masks import require_mask
from .model import WorldModel
from .behavior import ActorCritic
from .distributions import sample_latent, policy_probs
from ...training.artifacts import atomic_json, file_inventory, validate_inventory
from ...training.experiments import experiment_payload_hash
from ...training.provenance import validate_checkpoint_contract, environment_config
from ...factory import make_env

CHECKPOINT_SCHEMA = "aresim.checkpoint.world_model.v1"


def validate_sidecar(path: Path, algorithm_id: str | None = None):
    """Validate hashed experiment, phase, model identity and complete native inventory."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("world-model sidecar must be a mapping")
    _validate_identity(payload, algorithm_id)
    spec = validate_checkpoint_contract(payload)
    if payload.get("phase") != getattr(spec.algorithm_config, "phase", "task"):
        raise ValueError("checkpoint phase differs from experiment")
    native = _native_directory(path, payload)
    inventory = payload.get("native_inventory")
    validate_inventory(native, inventory)
    if {entry["path"] for entry in inventory} != {"model.pt", "trainer.pt"}:
        raise ValueError("world-model checkpoint requires model.pt and trainer.pt")
    return payload, spec, native


def _validate_identity(payload: dict, algorithm_id: str | None) -> None:
    if payload.get("schema_version") != CHECKPOINT_SCHEMA or payload.get("framework_id") != "torch":
        raise ValueError("unsupported world-model checkpoint")
    identities = {"dreamerv3": ("local_rssm", "torch_dreamerv3"), "plan2explore": ("local_rssm_disagreement", "torch_plan2explore")}
    name = payload.get("algorithm_id")
    if name not in identities or (algorithm_id is not None and name != algorithm_id):
        raise ValueError("checkpoint algorithm does not match loader")
    model, loader = identities[name]
    _validate_policy_fields(payload)
    expected = {"model_id": model, "loader_id": loader, "action_schema": "aresim.action.rover.v1"}
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("incompatible world-model checkpoint identity")
    if not isinstance(payload.get("experiment"), dict) or experiment_payload_hash(payload["experiment"]) != payload.get("config_hash"):
        raise ValueError("world-model configuration hash differs")


def _validate_policy_fields(payload: dict) -> None:
    if not all(isinstance(payload.get(key), str) and payload[key].strip() for key in ("policy_id", "task_id", "reward_profile")):
        raise ValueError("world-model policy/task/reward metadata is missing")

def _native_directory(path: Path, payload: dict) -> Path:
    relative = Path(str(payload.get("native_path", "")))
    native = (path.parent / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not native.is_relative_to(path.parent.resolve()):
        raise ValueError("unsafe world-model native path")
    return native


def write_sidecar(directory: Path, spec, schemas: dict[str, str], steps: int, parent: dict | None) -> Path:
    """Publish metadata only after immutable model/trainer files have been saved."""
    path = directory / "checkpoint.json"
    payload = {"schema_version": CHECKPOINT_SCHEMA, "framework_id": "torch", "algorithm_id": spec.algorithm,
        "model_id": spec.model, "loader_id": "torch_" + spec.algorithm, "native_path": "native",
        "policy_id": f"{spec.experiment_id}:{spec.trial_id}:{directory.name}", "config_hash": spec.config_hash,
        "experiment": spec.as_dict(), "native_inventory": file_inventory(directory / "native"),
        "environment_steps": steps, "phase": getattr(spec.algorithm_config, "phase", "task"), "parent": parent, **schemas}
    atomic_json(path, payload)
    return path


class WorldModelAgent:
    """Own posterior state and separate action/latent RNGs; models may be shared read-only."""

    action_schema = "aresim.action.rover.v1"

    def __init__(self, model: WorldModel, behavior: ActorCritic, policy_id: str, *, deterministic: bool = False, latent_deterministic: bool = False) -> None:
        self.model = model.eval()
        self.behavior = behavior.eval()
        self.policy_id = policy_id
        self.device = next(model.parameters()).device
        self.observation_schema = "aresim.obs.local_memory.v2" if model.encoder.memory else "aresim.obs.local.v1"
        self.deterministic = deterministic
        self.latent_deterministic = latent_deterministic
        self.generator = torch.Generator(device=self.device)
        self.latent_generator = torch.Generator(device=self.device)
        self.reset(0)

    def reset_memory(self) -> None:
        """Clear model history while preserving RNG streams (for manual live interventions)."""
        self.state = None
        self.previous_action = 10

    def reset(self, seed: int) -> None:
        self.generator.manual_seed(seed)
        self.latent_generator.manual_seed((seed + 104729) % (2**63 - 1))
        self.reset_memory()

    @staticmethod
    @torch.no_grad()
    def observe_many(agents: list[WorldModelAgent], observations: list[dict[str, object]], *, reconstruction_generator=None) -> Tensor:
        """Batch shared neural inference but sample each environment's latent with its own RNG."""
        model = agents[0].model
        device = agents[0].device
        inputs = {key: torch.as_tensor(np.stack([np.asarray(obs[key]) for obs in observations]), device=device) for key in observations[0]}
        embeddings = model.encoder(inputs)
        states = [agent.state if agent.state is not None else model.rssm.initial(1, device) for agent in agents]
        state = tuple(torch.cat([item[index] for item in states], 0) for index in range(2))
        previous = torch.nn.functional.one_hot(torch.tensor([a.previous_action for a in agents], device=device), 11)[..., :10].float()
        hidden, posterior = model.rssm.posterior_step(state, previous, embeddings)
        stochastic = _sample_posteriors(agents, posterior, reconstruction_generator)
        for index, agent in enumerate(agents):
            agent.state = hidden[index:index + 1], stochastic[index:index + 1]
        return model.rssm.features((hidden, stochastic))

    @torch.no_grad()
    def reconstruct(self, observations: list[dict[str, object]], actions: list[int], generator) -> None:
        """Rebuild posterior memory under new weights, batching stateless crop encoding.

        Chunked encoding bounds activation memory; posterior sampling remains in
        original time order using the caller's reconstruction-only RNG.
        """
        self.reset_memory()
        if len(observations) != len(actions):
            raise ValueError("reconstruction requires one observation per previous decision")
        for first in range(0, len(observations), 128):
            chunk = observations[first:first + 128]
            inputs = {key: torch.as_tensor(np.stack([np.asarray(obs[key]) for obs in chunk]), device=self.device) for key in chunk[0]}
            embeddings = self.model.encoder(inputs)
            previous = [10 if index == 0 else actions[index - 1] for index in range(first, first + len(chunk))]
            encoded_actions = torch.nn.functional.one_hot(torch.tensor(previous, device=self.device), 11)[..., :10].float()
            for offset in range(len(chunk)):
                state = self.state if self.state is not None else self.model.rssm.initial(1, self.device)
                hidden, posterior = self.model.rssm.posterior_step(state, encoded_actions[offset:offset + 1], embeddings[offset:offset + 1])
                self.state = hidden, _sample_posteriors([self], posterior, generator)
            self.previous_action = actions[first + len(chunk) - 1]

    @staticmethod
    @torch.no_grad()
    def act_many(agents: list[WorldModelAgent], observations: list[dict[str, object]], masks: list[np.ndarray], *, uniform: bool = False) -> list[int]:
        features = WorldModelAgent.observe_many(agents, observations)
        mask = torch.as_tensor(np.stack([require_mask(m) for m in masks]), device=features.device).bool()
        probabilities = mask.float() / mask.sum(-1, keepdim=True) if uniform else policy_probs(
            agents[0].behavior.actor(features), mask, agents[0].model.config.unimix)
        actions = []
        for index, agent in enumerate(agents):
            if agent.deterministic and not uniform:
                action = int(probabilities[index].argmax())
            else:
                action = int(torch.multinomial(probabilities[index], 1, generator=agent.generator))
            agent.previous_action = action
            actions.append(action)
        return actions

    def act(self, observation: dict[str, object], action_mask: np.ndarray) -> int:
        return self.act_many([self], [observation], [action_mask])[0]


class BuiltinCheckpointLoader:
    """Restore CPU Dreamer inference without starting a distributed runtime."""

    loader_id = "torch_dreamerv3"
    algorithm_id = "dreamerv3"

    def load(self, path: str, *, deterministic: bool = True, latent_deterministic: bool = False) -> WorldModelAgent:
        payload, spec, native = validate_sidecar(Path(path), self.algorithm_id)
        config = environment_config(spec)
        model = WorldModel(spec.model_config, make_env(config, audit=False).observation_space)
        behavior = ActorCritic(model.feature_size, spec.model_config)
        weights = torch.load(native / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(weights["model"])
        key = "exploration" if payload["phase"] == "pretrain" else "task"
        behavior.load_state_dict(weights[key])
        agent = WorldModelAgent(model, behavior, payload["policy_id"], deterministic=deterministic, latent_deterministic=latent_deterministic)
        agent.environment_config = config
        agent.phase = payload["phase"]
        agent.evaluation_metadata = {"latent_mode": "modal" if latent_deterministic else "seeded_sample", "phase": agent.phase}
        return agent


def _sample_posteriors(agents: list[WorldModelAgent], posterior: Tensor, reconstruction_generator) -> Tensor:
    samples = []
    for index, agent in enumerate(agents):
        generator = agent.latent_generator if reconstruction_generator is None else reconstruction_generator
        samples.append(sample_latent(posterior[index:index + 1], agent.model.config.unimix,
                                     generator=generator, deterministic=agent.latent_deterministic))
    return torch.cat(samples, 0)
