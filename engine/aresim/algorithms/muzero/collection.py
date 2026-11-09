"""Collect bounded MuZero trajectory chunks in Ray CPU actors.

The driver owns the learner; collectors own real environments and immutable
search weights for each chunk. Checkpoints reconstruct unfinished environments
through deterministic action replay, including observation-builder memory.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ...factory import make_env
from ...registry import ComponentRegistry
from ...training.experiments import ExperimentSpec
from ...training.provenance import environment_config, experiment_seeds
from .checkpoint import batch_history_tensors
from .model import MuZeroNetwork, support_value
from .replay import Episode
from .search import SearchStats, choose_action, search_many


@dataclass
class PendingEpisode:
    """Real observations and targets belonging to one unfinished episode."""

    seed: int
    observations: list[dict[str, object]]
    masks: list[np.ndarray]
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    policies: list[np.ndarray] = field(default_factory=list)
    root_values: list[float] = field(default_factory=list)
    policy_versions: list[int] = field(default_factory=list)


@dataclass
class CollectorState:
    """Quiescent actor state; weights are republished from the learner on resume."""

    rng_states: list[dict[str, object]]
    episodes: list[PendingEpisode]


@dataclass
class CollectionResult:
    """A bounded chunk, completed replay episodes and collection timing."""

    collector_id: int
    policy_version: int
    environment_steps: int
    episodes: list[Episode]
    episode_metrics: list[dict[str, float]]
    unfinished_steps: int
    search_stats: SearchStats
    search_seconds: float
    environment_seconds: float


class Collector:
    """Own several real environments and batch their independent search trees."""

    def __init__(self, spec: ExperimentSpec, collector_id: int,
                 component_registry: ComponentRegistry | None = None,
                 state: CollectorState | None = None) -> None:
        self.spec = spec
        self.collector_id = collector_id
        config = environment_config(spec)
        self.seeds = experiment_seeds(spec).train
        torch.set_num_threads(spec.resources.collector_torch_threads)
        network = MuZeroNetwork(spec.model_config, config.observation_config.window_size,
                                memory=spec.environment.observation == "local_memory")
        self.network = network
        self.device = next(network.parameters()).device
        self.environments = [make_env(config, registry=component_registry,
            max_episode_steps=spec.environment.max_episode_steps, audit=False)
            for _ in range(spec.resources.num_envs_per_collector)]
        self.rngs = [np.random.default_rng(np.random.SeedSequence([spec.learner_seed, collector_id, index]))
                     for index in range(len(self.environments))]
        if state is None:
            self.episodes = [self._reset(index) for index in range(len(self.environments))]
        else:
            if len(state.episodes) != len(self.environments) or len(state.rng_states) != len(self.environments):
                raise ValueError("MuZero collector checkpoint has a different environment count")
            self.episodes = copy.deepcopy(state.episodes)
            for index, (environment, episode) in enumerate(zip(self.environments, self.episodes, strict=True)):
                self.rngs[index].bit_generator.state = state.rng_states[index]
                result = environment.reset(seed=episode.seed)
                for action in episode.actions:
                    result = environment.step(action)
                    if result.terminated or result.truncated:
                        raise ValueError("MuZero collector checkpoint contains a completed active episode")
                for key, value in episode.observations[-1].items():
                    np.testing.assert_array_equal(result.observation[key], value)
                np.testing.assert_array_equal(result.action_mask, episode.masks[-1])

    def _reset(self, index: int) -> PendingEpisode:
        seed = int(self.rngs[index].choice(self.seeds))
        reset = self.environments[index].reset(seed=seed)
        return PendingEpisode(seed, [reset.observation], [reset.action_mask.copy()])

    @torch.no_grad()
    def collect(self, weights: dict[str, torch.Tensor], policy_version: int,
                rounds: int) -> CollectionResult:
        """Advance each environment by ``rounds`` real steps using fixed weights."""
        if rounds < 1:
            raise ValueError("MuZero collection rounds must be positive")
        self.network.load_state_dict(weights)
        self.network.eval()
        completed, metrics = [], []
        stats = SearchStats()
        search_seconds = environment_seconds = 0.0
        config = self.spec.algorithm_config
        encoded = None
        reset_indices = []
        for _ in range(rounds):
            started = time.monotonic()
            observations, previous = batch_history_tensors(
                [episode.observations for episode in self.episodes],
                [episode.actions for episode in self.episodes], self.network.config.history_length, self.device)
            if encoded is None:
                batch, history = previous.shape
                flat = {key: value.flatten(0, 1) for key, value in observations.items()}
                encoded = self.network.encoder(flat).reshape(batch, history, -1)
            else:
                newest = self.network.encoder({key: value[:, -1] for key, value in observations.items()})
                encoded = torch.cat((encoded[:, 1:], newest[:, None]), 1)
                if reset_indices:
                    encoded[reset_indices] = newest[reset_indices, None].expand(-1, self.network.config.history_length, -1)
            results = search_many(self.network, observations, previous,
                np.stack([episode.masks[-1] for episode in self.episodes]), config,
                rngs=self.rngs, training=True, stats=stats, encoded_observations=encoded)
            search_seconds += time.monotonic() - started
            started = time.monotonic()
            ended = []
            reset_indices = []
            for index, (episode, result) in enumerate(zip(self.episodes, results, strict=True)):
                action = choose_action(result, training=True, temperature=config.temperature, rng=self.rngs[index])
                if not episode.masks[-1][action]:
                    raise RuntimeError("MuZero collector selected an illegal real action")
                step = self.environments[index].step(action)
                episode.actions.append(action)
                episode.rewards.append(float(step.reward))
                episode.policies.append(result.policy)
                episode.root_values.append(result.root_value)
                episode.policy_versions.append(policy_version)
                episode.observations.append(step.observation)
                episode.masks.append(step.action_mask.copy())
                if step.terminated or step.truncated:
                    ended.append((index, step))
            # Time limits retain the original model-value bootstrap, not search noise.
            truncated = [index for index, step in ended if step.truncated]
            tails = {}
            if truncated:
                observations, previous = batch_history_tensors(
                    [self.episodes[index].observations for index in truncated],
                    [self.episodes[index].actions for index in truncated], self.network.config.history_length, self.device)
                _, _, values = self.network.initial(observations, previous)
                tails = dict(zip(truncated, support_value(values, self.network.config.support_size).cpu().tolist(), strict=True))
            for index, step in ended:
                pending = self.episodes[index]
                episode = Episode(pending.observations, pending.masks, pending.actions, pending.rewards,
                    pending.policies, pending.root_values, step.terminated, step.truncated,
                    float(tails.get(index, 0.0)), pending.policy_versions)
                episode.validate()
                completed.append(episode)
                objective = step.info["objective_stats"]
                metrics.append({"train/episode_length": len(episode.actions),
                    "train/shaped_return": sum(episode.rewards), "train/success": int(bool(step.info["success"])),
                    "train/ice_delivered": float(objective["ice_delivered"]),
                    "train/samples_delivered": float(objective["samples_delivered"]),
                    "train/extract_opportunities": sum(int(bool(mask[6])) for mask in episode.masks[:-1]),
                    **{f"train/action_count/{action}": episode.actions.count(action) for action in range(10)}})
                self.episodes[index] = self._reset(index)
                reset_indices.append(index)
            environment_seconds += time.monotonic() - started
        return CollectionResult(self.collector_id, policy_version, rounds * len(self.environments),
            completed, metrics, sum(len(episode.actions) for episode in self.episodes),
            stats, search_seconds, environment_seconds)

    def get_state(self) -> CollectorState:
        """Copy actor RNGs and unfinished trajectories at a chunk boundary."""
        return copy.deepcopy(CollectorState([rng.bit_generator.state for rng in self.rngs], self.episodes))


class RayCollectionPool:
    """Overlap bounded CPU search chunks with a single driver-owned learner."""

    def __init__(self, spec: ExperimentSpec, network: MuZeroNetwork,
                 component_registry: ComponentRegistry | None,
                 states: list[CollectorState] | None) -> None:
        import ray

        if states is not None and len(states) != spec.resources.num_collectors:
            raise ValueError("Ray MuZero checkpoint has a different collector count")
        self.ray = ray
        self.network = network
        self.owns_runtime = not ray.is_initialized()
        self.actors = []
        self.tasks = []
        self.weights = None
        try:
            if self.owns_runtime:
                ray.init(include_dashboard=False, log_to_driver=False, num_gpus=0)
            required_cpus = spec.resources.num_collectors * spec.resources.collector_torch_threads
            if ray.available_resources().get("CPU", 0) < required_cpus:
                raise ValueError("MuZero collectors exceed the Ray CPU allocation")
            actor = ray.remote(num_cpus=spec.resources.collector_torch_threads, num_gpus=0)(Collector)
            self.actors = [actor.remote(spec, index, component_registry,
                          None if states is None else states[index])
                          for index in range(spec.resources.num_collectors)]
            # Surface actor construction/restore failures before dispatching collection.
            ray.get([worker.get_state.remote() for worker in self.actors])
        except Exception:
            self.close()
            raise

    @property
    def pending(self) -> bool:
        return bool(self.tasks)

    def start(self, policy_version: int, rounds: int) -> None:
        if self.pending:
            raise RuntimeError("MuZero collection already pending")
        self.weights = self.ray.put({key: value.detach().cpu().clone()
                                    for key, value in self.network.state_dict().items()})
        self.tasks = [worker.collect.remote(self.weights, policy_version, rounds) for worker in self.actors]

    def next_result(self) -> CollectionResult:
        if not self.tasks:
            raise RuntimeError("no MuZero collection pending")
        ready, self.tasks = self.ray.wait(self.tasks, num_returns=1)
        return self.ray.get(ready[0])

    def get_states(self) -> list[CollectorState]:
        if self.pending:
            raise RuntimeError("cannot checkpoint pending MuZero collection")
        return self.ray.get([worker.get_state.remote() for worker in self.actors])

    def close(self) -> None:
        for actor in self.actors:
            self.ray.kill(actor, no_restart=True)
        self.actors = []
        self.tasks = []
        self.weights = None
        if self.owns_runtime and self.ray.is_initialized():
            self.ray.shutdown()

