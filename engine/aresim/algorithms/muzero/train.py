"""Coordinate MuZero collection, replay, batched reanalysis and a PyTorch learner.

Ray collectors search on CPU while the driver learns from refreshed targets and
recorded future observations. Importance-weighted losses retain a stable scale;
checkpoints sort numerically and save at quiescent chunk boundaries.
Environment transitions come only from AresSim; training never runs evaluation.
"""

from __future__ import annotations

import json
import hashlib
import platform
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ...factory import make_env
from ...training.provenance import environment_config, experiment_seeds, freeze_experiment
from ...training.experiments import ExperimentSpec
from ...training.artifacts import atomic_json, numbered_checkpoints, latest_checkpoint
from ...training.torch_utils import cpu_state
from ..common.config_decode import decode_dataclass
from ..common.tracking import new_wandb_run_id
from ..registry import TrainingContext, TrainingRegistry, create_training_registry
from .checkpoint import batch_history_tensors, validate_sidecar, write_checkpoint_sidecar
from .collection import RayCollectionPool
from .config import MuZeroConfig, MuZeroModelConfig, MuZeroResourceConfig
from .model import MuZeroNetwork, support_loss, support_value
from .replay import Episode, Replay
from .search import search_many


class MuZeroFactory:
    """Registry entry for the dedicated MuZero runner and frozen loader."""

    algorithm_id = "muzero"
    trainable = "muzero"
    checkpoint_loader_id = "torch_muzero"
    observation_schema = "aresim.obs.local.v1"
    action_schema = "aresim.action.rover.v1"

    def decode_config(self, payload: object) -> MuZeroConfig:
        return decode_dataclass(MuZeroConfig, payload, "algorithm_config")

    def build(self, context: TrainingContext) -> MuZeroConfig:
        if not isinstance(context.experiment.algorithm_config, MuZeroConfig):
            raise TypeError("MuZero requires MuZeroConfig")
        return context.experiment.algorithm_config


def _status(run_directory: Path, status: str, error: str | None = None) -> None:
    atomic_json(run_directory / "status.json", {"status": status, "error": error})


def _manifest(run_directory: Path, spec: ExperimentSpec, status: str, steps: int,
              run_id: str, *, elapsed: float = 0.0) -> None:
    artifacts = []
    if status == "completed":
        for item in sorted(path for path in run_directory.rglob("*") if path.is_file()):
            if item.name in {"manifest.json", "status.json"} or "native" in item.parts or "wandb" in item.parts:
                continue
            artifacts.append({"path": item.relative_to(run_directory).as_posix(),
                              "size": item.stat().st_size, "sha256": hashlib.sha256(item.read_bytes()).hexdigest()})
    payload = {"schema_version": "aresim.run.v1", "framework_id": "torch", "status": status,
               "config_hash": spec.config_hash, "experiment": spec.as_dict(), "environment_steps": steps,
               "elapsed_seconds": elapsed, "wandb_run_id": run_id,
               "installed_versions": {"python": platform.python_version(), "torch": torch.__version__},
               "open_exploration_has_success": spec.environment.task != "open_exploration",
               "promoted_checkpoint": "checkpoints/best/checkpoint.json" if (run_directory / "checkpoint_selection.json").is_file() else None,
               "artifacts": artifacts}
    atomic_json(run_directory / "manifest.json", payload)


def _preflight(spec: ExperimentSpec, registry: TrainingRegistry, component_registry) -> tuple[dict[str, str], object]:
    if not isinstance(spec.algorithm_config, MuZeroConfig) or not isinstance(spec.model_config, MuZeroModelConfig) or not isinstance(spec.resources, MuZeroResourceConfig):
        raise TypeError("MuZero experiment has incompatible typed configuration")
    if spec.algorithm_config.consistency_loss_weight and not spec.model_config.consistency_projection_width:
        raise ValueError("positive consistency_loss_weight requires consistency_projection_width")
    if not spec.model_config.predict_constraints and any(weight != 1 for weight in spec.algorithm_config.legality_positive_weights):
        raise ValueError("balanced legality weights require predict_constraints")
    if spec.resources.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("MuZero config requests CUDA but CUDA is unavailable")
    context = TrainingContext(spec, component_registry=component_registry)
    model = registry.build_model(spec.model, context)
    algorithm = registry.build_algorithm(spec.algorithm, TrainingContext(spec, model, component_registry))
    if (model.observation_schema, model.action_schema) != (algorithm.observation_schema, algorithm.action_schema):
        raise ValueError("MuZero model and algorithm schemas differ")
    config = environment_config(spec)
    environment = make_env(config, registry=component_registry,
                           max_episode_steps=spec.environment.max_episode_steps, audit=False)
    reset = environment.reset(seed=spec.learner_seed)
    schemas = {key: str(reset.info[key]) for key in ("observation_schema", "action_schema", "task_id", "reward_profile")}
    if (schemas["observation_schema"], schemas["action_schema"]) != (algorithm.observation_schema, algorithm.action_schema):
        raise ValueError("MuZero environment schemas are incompatible")
    if environment.action_space.n != 10:
        raise ValueError("MuZero requires ten discrete rover actions")
    experiment_seeds(spec)
    return schemas, environment


def _batch(samples: list[tuple[Episode, int, float]], config: MuZeroConfig,
           model: MuZeroNetwork, device: torch.device):
    observations, previous = batch_history_tensors(
        [ep.observations[:index + 1] for ep, index, _ in samples],
        [ep.actions[:index] for ep, index, _ in samples], model.config.history_length, device)
    weights = torch.tensor([weight for _, _, weight in samples], device=device)
    actions = []
    values = []
    policies = []
    rewards = []
    active = []
    policy_active = []
    masks = []
    continuations = []
    legality_active = []
    for episode, start, _ in samples:
        count = len(episode.actions)
        actions.append([episode.actions[start + k] if start + k < count else 0
                        for k in range(config.unroll_steps)])
        values.append([episode.value_target(start + k, config.td_steps, config.gamma)
                       for k in range(config.unroll_steps + 1)])
        policies.append([episode.policies[start + k] if start + k < count else np.zeros(10, dtype=np.float32)
                         for k in range(config.unroll_steps + 1)])
        rewards.append([episode.rewards[start + k] if start + k < count else 0.0
                        for k in range(config.unroll_steps)])
        active.append([episode.terminated or start + k < count
                       for k in range(config.unroll_steps + 1)])
        policy_active.append([start + k < count for k in range(config.unroll_steps + 1)])
        masks.append([episode.masks[min(start + k, count)] for k in range(config.unroll_steps + 1)])
        continuations.append([not (episode.terminated and start + k >= count)
                              for k in range(config.unroll_steps + 1)])
        legality_active.append([start + k < count or (episode.truncated and start + k == count)
                                for k in range(config.unroll_steps + 1)])
    return (observations, previous, weights, torch.tensor(actions, device=device),
            torch.tensor(values, device=device, dtype=torch.float32),
            torch.tensor(np.asarray(policies), device=device),
            torch.tensor(rewards, device=device, dtype=torch.float32),
            torch.tensor(active, device=device, dtype=torch.float32),
            torch.tensor(policy_active, device=device, dtype=torch.float32),
            torch.as_tensor(np.asarray(masks), device=device, dtype=torch.float32),
            torch.tensor(continuations, device=device, dtype=torch.float32),
            torch.tensor(legality_active, device=device, dtype=torch.float32))


@torch.no_grad()
def _consistency_targets(network: MuZeroNetwork, samples: list[tuple[Episode, int, float]],
                         config: MuZeroConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode recorded future histories once per frame, with no target gradient.

    The final real observation is a valid target for either ending. Repeated
    padding past that observation never contributes to consistency supervision.
    """
    length = network.config.history_length
    positions = [[min(max(0, start - length + 2 + offset), len(ep.actions))
                  for offset in range(length + config.unroll_steps - 1)] for ep, start, _ in samples]
    observations = {key: torch.as_tensor(np.stack([
        np.stack([np.asarray(ep.observations[index][key]) for index in indices])
        for (ep, _, _), indices in zip(samples, positions, strict=True)]), device=device)
        for key in samples[0][0].observations[0]}
    previous = torch.tensor([[ep.actions[index - 1] if index > 0 else 10 for index in indices]
        for (ep, _, _), indices in zip(samples, positions, strict=True)], device=device)
    valid = torch.tensor([[start + k <= len(ep.actions) for k in range(1, config.unroll_steps + 1)]
        for ep, start, _ in samples], device=device, dtype=torch.float32)
    targets = network.project_consistency(network.encode_sequence(observations, previous))
    return F.normalize(targets, dim=-1), valid


def _update(network: MuZeroNetwork, optimizer: torch.optim.Optimizer, replay: Replay,
            config: MuZeroConfig, rng: np.random.Generator, device: torch.device, *,
            samples: list[tuple[Episode, int, float]] | None = None,
            policy_version: int | None = None) -> dict[str, float]:
    samples = replay.sample(config.batch_size, config.priority_beta, rng) if samples is None else samples
    (observations, previous, weights, actions, values, policies, rewards, active, policy_active,
     mask_targets, continuation_targets, legality_active) = _batch(samples, config, network, device)
    network.train()
    targets = consistency_active = None
    target_std = torch.zeros((), device=device)
    if config.consistency_loss_weight:
        targets, consistency_active = _consistency_targets(network, samples, config, device)
        count = consistency_active.sum().clamp(min=1)
        mean = (targets * consistency_active[..., None]).sum((0, 1)) / count
        variance = (targets.square() * consistency_active[..., None]).sum((0, 1)) / count - mean.square()
        target_std = variance.clamp(min=0).sqrt().mean()
    latent, policy_logits, value_logits = network.initial(observations, previous)
    policy_loss = torch.zeros(len(samples), device=device)
    value_loss = torch.zeros_like(policy_loss)
    reward_loss = torch.zeros_like(policy_loss)
    constraint_loss = torch.zeros_like(policy_loss)
    consistency_loss = torch.zeros_like(policy_loss)
    legality_counts = torch.zeros((4, 10), device=device)
    positive_weights = torch.tensor(config.legality_positive_weights, device=device)
    root_errors = support_value(value_logits, network.config.support_size).detach() - values[:, 0]
    reward_active = torch.tensor([[episode.terminated or start + k < len(episode.actions)
                                   for k in range(config.unroll_steps)]
                                  for episode, start, _ in samples], device=device, dtype=torch.float32)
    for k in range(config.unroll_steps + 1):
        if network.config.predict_constraints:
            mask_logits, continuation_logits = network.constraints(latent)
            mask_loss = F.binary_cross_entropy_with_logits(mask_logits, mask_targets[:, k],
                pos_weight=positive_weights, reduction="none").mean(-1) * legality_active[:, k]
            continuation_loss = F.binary_cross_entropy_with_logits(continuation_logits,
                continuation_targets[:, k], reduction="none") * active[:, k]
            constraint_loss += mask_loss + continuation_loss
            with torch.no_grad():
                positive = mask_targets[:, k] * legality_active[:, k, None]
                negative = (1 - mask_targets[:, k]) * legality_active[:, k, None]
                predicted = (mask_logits >= 0).float()
                legality_counts += torch.stack(((predicted * positive).sum(0), positive.sum(0),
                                                (predicted * negative).sum(0), negative.sum(0)))
        value_loss += support_loss(value_logits, values[:, k], network.config.support_size) * active[:, k]
        policy_loss += -(policies[:, k] * F.log_softmax(policy_logits, dim=-1)).sum(-1) * policy_active[:, k]
        if k == config.unroll_steps:
            break
        latent, reward_logits, policy_logits, value_logits = network.recurrent(latent, actions[:, k])
        latent.register_hook(lambda gradient: gradient * 0.5)
        if targets is not None:
            predicted = network.project_consistency(latent, prediction=True)
            consistency_loss += (1 - F.cosine_similarity(predicted, targets[:, k], dim=-1)) * consistency_active[:, k]
        reward_loss += support_loss(reward_logits, rewards[:, k], network.config.support_size) * reward_active[:, k]
    divisor = config.unroll_steps + 1
    per_sample_losses = torch.stack((policy_loss, value_loss, reward_loss, constraint_loss,
                                    consistency_loss * config.consistency_loss_weight)) / divisor
    weighted_losses = (per_sample_losses * weights).sum(dim=1) / weights.sum()
    unweighted_losses = per_sample_losses.mean(dim=1)
    loss = weighted_losses.sum()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), config.max_gradient_norm)
    optimizer.step()
    names = ("policy", "value", "reward", "constraint", "consistency")
    # Transfer diagnostics together rather than adding a CUDA fence per field.
    packed = torch.cat((weighted_losses.detach(), unweighted_losses.detach(),
        torch.stack((weights.mean(), weights.min(), weights.max(),
                     weights.sum().square() / weights.square().sum(), gradient_norm,
                     target_std)), legality_counts.flatten(), root_errors)).cpu().tolist()
    replay.update_priorities(samples, packed[56:], config.priority_alpha)
    ages = [max(0, policy_version - ep.policy_versions[index])
            for ep, start, _ in samples if ep.policy_versions is not None and policy_version is not None
            for index in range(start, min(start + config.unroll_steps + 1, len(ep.actions)))]
    metrics = {"learner/total_loss": sum(packed[:5]),
            "learner/total_loss_unweighted": sum(packed[5:10]),
            **{f"learner/{name}_loss": packed[index] for index, name in enumerate(names)},
            **{f"learner/{name}_loss_unweighted": packed[index + 5] for index, name in enumerate(names)},
            "learner/importance_weight_mean": packed[10],
            "learner/importance_weight_min": packed[11],
            "learner/importance_weight_max": packed[12],
            "learner/effective_sample_size": packed[13],
            "learner/gradient_norm": packed[14],
            "learner/consistency_target_std": packed[15],
            "learner/policy_target_age_mean": float(np.mean(ages)) if ages else 0.,
            "learner/policy_target_age_max": float(max(ages)) if ages else 0.,
            "learner/learning_rate": config.learning_rate}
    for action in range(10):
        true_positive, positive, false_positive, negative = [packed[16 + row * 10 + action] for row in range(4)]
        metrics.update({f"learner/legality_recall/{action}": true_positive / positive if positive else 0.,
                        f"learner/legality_false_positive_rate/{action}": false_positive / negative if negative else 0.,
                        f"learner/legality_positive_count/{action}": positive,
                        f"learner/legality_negative_count/{action}": negative})
    return metrics


def _reanalyse(network: MuZeroNetwork, replay: Replay, config: MuZeroConfig,
               rng: np.random.Generator, device: torch.device, *, batch_size: int | None = None,
               policy_version: int | None = None,
               samples: list[tuple[Episode, int, float]] | None = None) -> dict[str, float]:
    """Refresh complete sampled unrolls and their bootstrap values before learning.

    Requests deduplicate by episode identity and position. True terminal tails
    never bootstrap. Separate bootstrap positions can use direct value inference
    to preserve a bounded search budget while real policy targets use full MCTS.
    """
    network.eval()
    samples = replay.sample(config.reanalysis_positions, config.priority_beta, rng) if samples is None else samples
    policies, bootstraps = {}, {}
    for episode, position, _ in samples:
        count = len(episode.actions)
        for k in range(config.unroll_steps + 1):
            index = position + k
            if index >= count:
                continue
            policies[(id(episode), index)] = (episode, index)
            end = min(index + config.td_steps, count)
            if end < count or episode.truncated:
                bootstraps[(id(episode), end)] = (episode, end)
    search_targets = {**bootstraps, **policies} if config.reanalysis_bootstrap_search else policies
    chunk_size = config.reanalysis_batch_size if batch_size is None else batch_size
    targets = list(search_targets.values())
    for start in range(0, len(targets), chunk_size):
        chunk = targets[start:start + chunk_size]
        tensors, previous = batch_history_tensors(
            [episode.observations[:index + 1] for episode, index in chunk],
            [episode.actions[:index] for episode, index in chunk], network.config.history_length, device)
        results = search_many(network, tensors, previous,
            np.stack([episode.masks[index] for episode, index in chunk]), config, training=False)
        for (episode, index), result in zip(chunk, results, strict=True):
            if index == len(episode.actions):
                episode.tail_value = result.root_value
            else:
                episode.root_values[index] = result.root_value
                episode.policies[index] = result.policy
                if episode.policy_versions is not None and policy_version is not None:
                    episode.policy_versions[index] = policy_version
    value_targets = [target for key, target in bootstraps.items() if key not in search_targets]
    for start in range(0, len(value_targets), chunk_size):
        chunk = value_targets[start:start + chunk_size]
        tensors, previous = batch_history_tensors(
            [episode.observations[:index + 1] for episode, index in chunk],
            [episode.actions[:index] for episode, index in chunk], network.config.history_length, device)
        with torch.no_grad():
            _, _, logits = network.initial(tensors, previous)
            values = support_value(logits, network.config.support_size).cpu().tolist()
        for (episode, index), value in zip(chunk, values, strict=True):
            if index == len(episode.actions):
                episode.tail_value = float(value)
            else:
                episode.root_values[index] = float(value)
    errors = [ep.root_values[index] - ep.value_target(index, config.td_steps, config.gamma)
              for ep, index, _ in samples]
    replay.update_priorities(samples, errors, config.priority_alpha)
    return {"learner/reanalysis_sampled_starts": float(len(samples)),
            "learner/reanalysis_search_roots": float(len(targets)),
            "learner/reanalysis_value_roots": float(len(value_targets))}


def _learn_until(network: MuZeroNetwork, optimizer: torch.optim.Optimizer, replay: Replay,
                 config: MuZeroConfig, rng: np.random.Generator, device: torch.device,
                 updates: int, target_updates: int):
    """Pay update credit, refreshing a capped fraction of the actual training batch."""
    metrics = {}
    initial_updates = updates
    totals = {"learner/reanalysis_sampled_starts": 0., "learner/reanalysis_search_roots": 0.,
              "learner/reanalysis_value_roots": 0.}
    learner_seconds = reanalysis_seconds = 0.0
    while updates < target_updates:
        started = time.monotonic()
        samples = replay.sample(config.batch_size, config.priority_beta, rng)
        learner_seconds += time.monotonic() - started
        if config.reanalysis_fraction and (updates + 1) % config.reanalysis_interval_updates == 0:
            count = min(config.reanalysis_positions, max(1, int(np.ceil(len(samples) * config.reanalysis_fraction))))
            chosen = rng.choice(len(samples), size=count, replace=False)
            selected = [samples[int(index)] for index in chosen]
            started = time.monotonic()
            refreshed = _reanalyse(network, replay, config, rng, device,
                                  policy_version=updates, samples=selected)
            reanalysis_seconds += time.monotonic() - started
            for name, value in refreshed.items():
                totals[name] += value
        started = time.monotonic()
        metrics = _update(network, optimizer, replay, config, rng, device,
                          samples=samples, policy_version=updates)
        learner_seconds += time.monotonic() - started
        updates += 1
    metrics.update(totals)
    metrics["learner/reanalysis_sample_fraction"] = totals["learner/reanalysis_sampled_starts"] / max(1, (updates - initial_updates) * config.batch_size)
    return updates, metrics, learner_seconds, reanalysis_seconds


def _checkpoint(run_directory: Path, label: str, spec: ExperimentSpec, schemas: dict[str, str],
                network: MuZeroNetwork, optimizer: torch.optim.Optimizer, replay: Replay,
                rng: np.random.Generator, steps: int, updates: int, episodes_collected: int,
                update_credit: float, pending_collection_steps: int, *, collector_states=None) -> Path:
    destination = run_directory / "checkpoints" / label
    if destination.exists():
        shutil.rmtree(destination)
    native = destination / "native"
    native.mkdir(parents=True)
    torch.save(cpu_state(network.state_dict()), native / "model.pt")
    torch.save({"optimizer": optimizer.state_dict(), "replay": replay, "rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "steps": steps, "updates": updates, "episodes_collected": episodes_collected,
                "update_credit": update_credit, "pending_collection_steps": pending_collection_steps,
                "collectors": collector_states}, native / "trainer.pt")
    return write_checkpoint_sidecar(destination / "checkpoint.json", spec.as_dict(), spec.config_hash, schemas, environment_steps=steps)


def _resume(target: str | Path, spec: ExperimentSpec, network: MuZeroNetwork,
            optimizer: torch.optim.Optimizer, device: torch.device):
    path = Path(target).resolve()
    if path.is_dir() and (path / "manifest.json").is_file():
        path = latest_checkpoint(path)
    if path.is_dir():
        path /= "checkpoint.json"
    payload, native = validate_sidecar(path)
    old = dict(payload["experiment"])
    current = json.loads(json.dumps(spec.as_dict()))
    old_steps = old["algorithm_config"].pop("total_environment_steps")
    current_steps = current["algorithm_config"].pop("total_environment_steps")
    old["artifacts"].pop("resume_from", None)
    current["artifacts"].pop("resume_from", None)
    if old != current or current_steps < old_steps:
        raise ValueError("MuZero resume config differs beyond an increased training budget")
    network.load_state_dict(torch.load(native / "model.pt", map_location=device, weights_only=True))
    state = torch.load(native / "trainer.pt", map_location=device, weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if device.type == "cuda" and state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    random.setstate(state["python_rng"])
    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng"]
    return (path.parents[2], state["replay"], rng, int(state["steps"]), int(state["updates"]),
            int(state.get("episodes_collected", len(state["replay"].episodes))), float(state["update_credit"]),
            int(state.get("pending_collection_steps", 0)), state.get("collectors"))


def run_experiment(spec: ExperimentSpec | str | Path, registry: TrainingRegistry | None = None, *,
                   report: bool = False, component_registry=None,
                   resume_from: str | Path | None = None) -> Path:
    """Train and save checkpoints; run evaluation separately after training."""
    from ...training import load_experiment

    selected = create_training_registry() if registry is None else registry
    resolved = load_experiment(spec, registry=selected) if isinstance(spec, (str, Path)) else spec
    resolved = freeze_experiment(resolved, component_registry=component_registry)
    resolved.validate()
    schemas, _ = _preflight(resolved, selected, component_registry)
    config = resolved.algorithm_config
    resources = resolved.resources
    torch.set_num_threads(resources.torch_threads)
    device = torch.device(resources.device)
    random.seed(resolved.learner_seed)
    torch.manual_seed(resolved.learner_seed)
    rng = np.random.default_rng(resolved.learner_seed)
    network = MuZeroNetwork(resolved.model_config, environment_config(resolved).observation_config.window_size,
                            memory=resolved.environment.observation == "local_memory").to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, fused=device.type == "cuda")
    replay = Replay(config.replay_capacity)
    steps = updates = episodes_collected = 0
    update_credit = 0.0
    pending_collection_steps = 0
    previous_elapsed = 0.0
    collector_states = None
    run_directory = Path(resolved.artifacts.root).resolve() / resolved.experiment_id / resolved.trial_id
    resume_target = resume_from or resolved.artifacts.resume_from
    if resume_target:
        if resume_target == ".":
            resume_target = run_directory
        restored_dir, replay, rng, steps, updates, episodes_collected, update_credit, pending_collection_steps, collector_states = _resume(
            resume_target, resolved, network, optimizer, device)
        if restored_dir != run_directory:
            raise ValueError("MuZero resume checkpoint belongs to a different trial")
        existing = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
        run_id = str(existing["wandb_run_id"])
        previous_elapsed = float(existing.get("elapsed_seconds", 0.0))
    else:
        if run_directory.exists() and resolved.artifacts.reject_existing:
            raise FileExistsError(f"run already exists: {run_directory}")
        run_directory.mkdir(parents=True, exist_ok=True)
        run_id = new_wandb_run_id()
    import yaml
    (run_directory / "resolved_config.yaml").write_text(yaml.safe_dump(resolved.as_dict(), sort_keys=True), encoding="utf-8")
    _status(run_directory, "running")
    _manifest(run_directory, resolved, "running", steps, run_id)
    started = time.monotonic()
    initial_steps = steps
    session_initial_updates = updates
    wandb_run = None
    pool = None
    try:
        if resume_target and collector_states is None:
            raise ValueError("MuZero resume requires collector state")
        pool = RayCollectionPool(resolved, network, component_registry, collector_states)
        if resolved.tracking.mode != "disabled":
            import wandb
            if resolved.tracking.mode == "online" and not wandb.login(timeout=30):
                raise RuntimeError("W&B authentication failed")
            wandb_run = wandb.init(project=resolved.tracking.project, entity=resolved.tracking.entity,
                                   group=resolved.tracking.group, tags=list(resolved.tracking.tags),
                                   name=f"{resolved.experiment_id}-{resolved.trial_id}", id=run_id,
                                   resume="allow", mode=resolved.tracking.mode,
                                   dir=str(run_directory), config=resolved.as_dict())
        next_checkpoint = (steps // resolved.checkpoint.interval_environment_steps + 1) * resolved.checkpoint.interval_environment_steps
        last_metrics: dict[str, float] = {}
        episode_metrics: dict[str, float] = {}
        unfinished = [0] * resources.num_collectors
        profile = {"system/learner_seconds": 0.0, "system/reanalysis_seconds": 0.0,
                   "system/collection_wait_seconds": 0.0, "system/collector_search_seconds_sum": 0.0,
                   "system/collector_environment_seconds_sum": 0.0, "system/weight_publish_seconds": 0.0,
                   "system/search_roots": 0, "system/recurrent_calls": 0, "system/expanded_leaves": 0,
                   "system/max_inference_batch_size": 0, "system/max_policy_lag_updates": 0,
                   "system/reanalysis_sampled_starts": 0., "system/reanalysis_search_roots": 0.,
                   "system/reanalysis_value_roots": 0.}
        while steps < config.total_environment_steps:
            version = updates
            total_envs = resources.num_collectors * resources.num_envs_per_collector
            rounds = min(resources.collection_steps_per_env,
                         (config.total_environment_steps - steps + total_envs - 1) // total_envs)
            started_collection = time.monotonic()
            pool.start(version, rounds)
            profile["system/weight_publish_seconds"] += time.monotonic() - started_collection
            while pool.pending:
                started_collection = time.monotonic()
                result = pool.next_result()
                profile["system/collection_wait_seconds"] += time.monotonic() - started_collection
                steps += result.environment_steps
                pending_collection_steps += result.environment_steps
                for episode in result.episodes:
                    replay.add(episode, config.td_steps, config.gamma, config.priority_alpha)
                episodes_collected += len(result.episodes)
                if result.episode_metrics:
                    episode_metrics = result.episode_metrics[-1]
                unfinished[result.collector_id] = result.unfinished_steps
                profile["system/collector_search_seconds_sum"] += result.search_seconds
                profile["system/collector_environment_seconds_sum"] += result.environment_seconds
                profile["system/search_roots"] += result.search_stats.roots
                profile["system/recurrent_calls"] += result.search_stats.recurrent_calls
                profile["system/expanded_leaves"] += result.search_stats.expanded_leaves
                profile["system/max_inference_batch_size"] = max(profile["system/max_inference_batch_size"], result.search_stats.maximum_batch_size)
                profile["system/max_policy_lag_updates"] = max(profile["system/max_policy_lag_updates"], updates - result.policy_version)
                if pool.pending and pending_collection_steps >= config.rollout_batch_size and replay.steps >= config.learning_starts:
                    target = min(int(steps * config.updates_per_environment_step),
                                 version + resources.max_policy_lag_updates)
                    updates, latest, learner_seconds, reanalysis_seconds = _learn_until(
                        network, optimizer, replay, config, rng, device, updates, target)
                    last_metrics.update(latest)
                    profile["system/learner_seconds"] += learner_seconds
                    profile["system/reanalysis_seconds"] += reanalysis_seconds
                    for name in ("sampled_starts", "search_roots", "value_roots"):
                        profile[f"system/reanalysis_{name}"] += latest[f"learner/reanalysis_{name}"]
            if (pending_collection_steps >= config.rollout_batch_size or steps >= config.total_environment_steps) and replay.steps >= config.learning_starts:
                updates, latest, learner_seconds, reanalysis_seconds = _learn_until(
                    network, optimizer, replay, config, rng, device, updates,
                    int(steps * config.updates_per_environment_step))
                last_metrics.update(latest)
                profile["system/learner_seconds"] += learner_seconds
                profile["system/reanalysis_seconds"] += reanalysis_seconds
                for name in ("sampled_starts", "search_roots", "value_roots"):
                    profile[f"system/reanalysis_{name}"] += latest[f"learner/reanalysis_{name}"]
                pending_collection_steps = 0
            update_credit = steps * config.updates_per_environment_step - updates
            metrics = {"train/environment_steps": steps, "train/episodes": episodes_collected,
                       "train/invalid_actions": 0, "train/search_simulations": config.simulations,
                       "train/replay_steps": replay.steps, "train/learner_updates": updates,
                       "train/unfinished_episode_steps": sum(unfinished),
                       "train/update_credit": update_credit,
                       "train/reanalysis_start_fraction": profile["system/reanalysis_sampled_starts"] / max(1, (updates - session_initial_updates) * config.batch_size),
                       "train/actual_updates_per_environment_step": (updates / steps if steps else 0.0),
                       "system/environment_steps_per_second": (steps - initial_steps) / max(1e-6, time.monotonic() - started),
                       "system/elapsed_seconds": previous_elapsed + time.monotonic() - started,
                       "system/average_recurrent_batch_size": profile["system/expanded_leaves"] / max(1, profile["system/recurrent_calls"]),
                       **profile, **last_metrics, **episode_metrics}
            atomic_json(run_directory / "last_metrics.json", metrics)
            _manifest(run_directory, resolved, "running", steps, run_id,
                      elapsed=previous_elapsed + time.monotonic() - started)
            if wandb_run is not None:
                wandb_run.log(metrics, step=steps)
            if steps >= next_checkpoint:
                _checkpoint(run_directory, f"step_{steps:06d}", resolved, schemas,
                            network, optimizer, replay, rng, steps, updates, episodes_collected,
                            update_credit, pending_collection_steps,
                            collector_states=pool.get_states())
                checkpoints = numbered_checkpoints(run_directory)
                for expired in checkpoints[:-resolved.checkpoint.keep]:
                    shutil.rmtree(expired)
                next_checkpoint = (steps // resolved.checkpoint.interval_environment_steps + 1) * resolved.checkpoint.interval_environment_steps
        if replay.steps < config.learning_starts:
            raise RuntimeError("MuZero budget ended before completed episodes filled replay warmup; increase the budget")
        _checkpoint(run_directory, "final", resolved, schemas, network, optimizer,
                            replay, rng, steps, updates, episodes_collected, update_credit, pending_collection_steps,
                            collector_states=pool.get_states())
        if wandb_run is not None:
            wandb_run.finish()
            wandb_run = None
        if report and resolved.tracking.mode == "online":
            from ...training.reports import generate_report
            generate_report(run_directory)
        _manifest(run_directory, resolved, "completed", steps, run_id,
                  elapsed=previous_elapsed + time.monotonic() - started)
        _status(run_directory, "completed")
    except Exception as error:
        _manifest(run_directory, resolved, "failed", steps, run_id,
                  elapsed=previous_elapsed + time.monotonic() - started)
        _status(run_directory, "failed", f"{type(error).__name__}: {error}")
        raise
    finally:
        if pool is not None:
            pool.close()
        if wandb_run is not None:
            wandb_run.finish()
    return run_directory


__all__ = ["MuZeroFactory", "run_experiment"]
