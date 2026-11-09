"""Bounded Ray CPU collector waves with batched recurrent actor inference.

Collectors own real environments and independent RNG/history state. The learner is single-owner;
policy publication and checkpoints occur between waves, never halfway through an action decision.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import time
import numpy as np
import torch
from ...factory import make_env
from ...core.engine import state_checksum
from ...algorithms.dreamerv3.model import WorldModel
from ...algorithms.dreamerv3.behavior import ActorCritic
from ...algorithms.dreamerv3.checkpoint import WorldModelAgent
from ...algorithms.baselines.scripted import ScriptedAgent
from ..provenance import environment_config, experiment_seeds
from .replay import EpisodeRecord, ReplayChunk


def movement_cycle_penalty(episode: EpisodeRecord, observation: dict, scale: float) -> float:
    """Charge only repeated 2–4 move cycles without collection, delivery or recovery.

    Three full repeats distinguish oscillation from ordinary return-path backtracking.
    Only public observations are consulted, and productive actions are exempt.
    """
    if not scale:
        return 0.0
    observations = episode.observations + [observation]
    for period in (2, 3, 4):
        count = 3 * period
        if len(episode.actions) < count or any(a not in (1, 2, 3, 4) for a in episode.actions[-count:]):
            continue
        window = observations[-count - 1:]
        positions = [tuple(obs["self"][:2]) for obs in window]
        if any(positions[i] != positions[i - period] for i in range(period, len(positions))):
            continue
        # Cargo changes, objective progress, and battery/health recovery break a cycle.
        productive = any(np.any(np.asarray(after["self"])[2:4] > np.asarray(before["self"])[2:4] + 1e-6)
                         or not np.array_equal(after["self"][4:7], before["self"][4:7])
                         or np.any(np.asarray(after["objectives"])[:4, 0] > np.asarray(before["objectives"])[:4, 0] + 1e-6)
                         for before, after in zip(window, window[1:]))
        if not productive:
            return float(scale)
    return 0.0


@dataclass
class CollectionResult:
    """Real-step suffixes and completed episode statistics from one collector wave."""

    chunks: list[ReplayChunk]
    environment_steps: int
    episode_metrics: list[dict[str, float]]
    inference_seconds: float
    environment_seconds: float
    reconstruction_seconds: float


class Collector:
    """Own a group of environments and shared CPU inference weights."""

    def __init__(self, spec, collector_id: int, component_registry=None, state=None, snapshot=None) -> None:
        self.spec = spec
        self.collector_id = collector_id
        self.pretraining = getattr(spec.algorithm_config, "phase", None) == "pretrain"
        torch.set_num_threads(spec.resources.collector_torch_threads)
        count = spec.resources.num_envs_per_collector
        config = environment_config(spec)
        self.seeds = experiment_seeds(spec).train
        self.environments = [make_env(config, registry=component_registry, max_episode_steps=spec.environment.max_episode_steps, audit=False) for _ in range(count)]
        self.model = WorldModel(spec.model_config, self.environments[0].observation_space)
        self.behavior = ActorCritic(self.model.feature_size, spec.model_config)
        self.agents = [WorldModelAgent(self.model, self.behavior, spec.algorithm) for _ in range(count)]
        fraction = getattr(spec.algorithm_config, "demonstration_fraction", 0.0)
        # Assign whole episodes evenly over the complete collection pool.
        total = count * spec.resources.num_collectors
        self.demonstrations = [((index * spec.resources.num_collectors + collector_id) < int(total * fraction)) for index in range(count)]
        self.teachers = [ScriptedAgent(config.task_config, config.engine.payload.capacity_kg) for _ in range(count)]
        self.rngs = [np.random.default_rng(np.random.SeedSequence([spec.learner_seed, collector_id, index])) for index in range(count)]
        self.counters = [-1] * count
        self.episodes = []
        self.episode_seeds = [0] * count
        self.version = -1
        if state is None:
            self.episodes = [self._reset(index) for index in range(count)]
        else:
            self._restore(state, snapshot)

    def _restore(self, state: dict, snapshot: dict | None) -> None:
        if len(state["episodes"]) != len(self.environments) or snapshot is None:
            raise ValueError("collector restore has incompatible environment count or missing snapshot")
        self._load_policy(snapshot)
        self.version = state["version"]
        self.counters = state["counters"]
        self.episodes = state["episodes"]
        self.episode_seeds = state["episode_seeds"]
        for index in range(len(self.environments)):
            self._restore_environment(index, state)

    def _restore_environment(self, index: int, state: dict) -> None:
        environment = self.environments[index]
        result = environment.reset(seed=self.episode_seeds[index])
        self.teachers[index].reset(self.episode_seeds[index])
        for action in self.episodes[index].actions:
            if self.demonstrations[index]:
                self.teachers[index].act(result.observation, result.action_mask)
            result = environment.step(action)
            if result.terminated or result.truncated:
                raise ValueError("collector active episode contains a terminal action")
        expected = self.episodes[index].observations[-1]
        if any(not np.array_equal(result.observation[key], value) for key, value in expected.items()):
            raise ValueError("collector restore observation differs")
        if not np.array_equal(result.action_mask, self.episodes[index].masks[-1]) or state_checksum(environment.world_state) != state["checksums"][index]:
            raise ValueError("collector restore mask/checksum differs")
        self.rngs[index].bit_generator.state = state["rngs"][index]
        agent = self.agents[index]
        saved = state["agents"][index]
        agent.state = saved["state"]
        agent.previous_action = saved["previous_action"]
        agent.generator.set_state(saved["action_rng"])
        agent.latent_generator.set_state(saved["latent_rng"])

    def _reset(self, index: int) -> EpisodeRecord:
        self.counters[index] += 1
        seed = int(self.rngs[index].choice(self.seeds))
        self.episode_seeds[index] = seed
        reset = self.environments[index].reset(seed=seed)
        self.teachers[index].reset(seed)
        agent_seed = int(np.random.SeedSequence([self.spec.learner_seed, self.collector_id, index, self.counters[index]]).generate_state(1, dtype=np.uint64)[0]) % (2**63 - 1)
        self.agents[index].reset(agent_seed)
        identity = f"{self.spec.experiment_id}:{self.spec.trial_id}:{self.collector_id}:{index}:{self.counters[index]}"
        return EpisodeRecord(identity, [reset.observation], [reset.action_mask.copy()])

    def _publish(self, snapshot: dict, version: int) -> float:
        if self.version == version:
            return 0.0
        started = time.monotonic()
        self._load_policy(snapshot)
        for index, (agent, episode) in enumerate(zip(self.agents, self.episodes, strict=True)):
            generator = torch.Generator().manual_seed(int(np.random.SeedSequence([
                self.spec.learner_seed, self.collector_id, index, self.counters[index], version]).generate_state(1)[0]))
            agent.reconstruct(episode.observations[:-1], episode.actions, generator)
        self.version = version
        return time.monotonic() - started

    def _load_policy(self, snapshot: dict) -> None:
        """Load strict inference submodules, accepting historical full-model snapshots too."""
        for prefix, module, source in (("encoder.", self.model.encoder, snapshot["model"]),
                                       ("rssm.", self.model.rssm, snapshot["model"]),
                                       ("actor.", self.behavior.actor, snapshot["behavior"])):
            module.load_state_dict({key[len(prefix):]: value for key, value in source.items() if key.startswith(prefix)})

    @staticmethod
    def _chunk(episode: EpisodeRecord, start: int) -> ReplayChunk:
        return ReplayChunk(start, EpisodeRecord(episode.episode_id, deepcopy(episode.observations[start:]),
            deepcopy(episode.masks[start:]), episode.actions[start:], episode.rewards[start:],
            episode.reward_valid[start:], episode.terminated[start:], episode.truncated[start:],
            episode.policy_versions[start:], episode.closed))

    def collect(self, snapshot: dict, version: int, counts: list[int], warmup: bool) -> CollectionResult:
        reconstruction = self._publish(snapshot, version)
        starts = [len(episode.actions) for episode in self.episodes]
        chunks, metrics = [], []
        inference_seconds = environment_seconds = 0.0
        for round_ in range(max(counts)):
            indices = [index for index, count in enumerate(counts) if round_ < count]
            started = time.monotonic()
            actions = WorldModelAgent.act_many([self.agents[i] for i in indices],
                [self.episodes[i].observations[-1] for i in indices], [self.episodes[i].masks[-1] for i in indices], uniform=warmup)
            for offset, index in enumerate(indices):
                if self.demonstrations[index]:
                    actions[offset] = self.teachers[index].act(self.episodes[index].observations[-1], self.episodes[index].masks[-1])
                    self.agents[index].previous_action = actions[offset]
            inference_seconds += time.monotonic() - started
            started = time.monotonic()
            self._advance(indices, actions, starts, chunks, metrics)
            environment_seconds += time.monotonic() - started
        for index, episode in enumerate(self.episodes):
            if len(episode.actions) > starts[index]:
                chunks.append(self._chunk(episode, starts[index]))
        return CollectionResult(chunks, sum(counts), metrics, inference_seconds, environment_seconds, reconstruction)

    def _advance(self, indices, actions, starts, chunks, metrics) -> None:
        for index, action in zip(indices, actions, strict=True):
            episode = self.episodes[index]
            if not episode.masks[-1][action]:
                raise RuntimeError("collector selected an illegal real action")
            result = self.environments[index].step(action)
            episode.actions.append(action)
            penalty = movement_cycle_penalty(episode, result.observation, getattr(self.spec.algorithm_config, "cycle_penalty", 0.0))
            episode.rewards.append(0.0 if self.pretraining else float(result.reward) - penalty)
            episode.reward_valid.append(not self.pretraining)
            episode.terminated.append(result.terminated)
            episode.truncated.append(result.truncated)
            episode.policy_versions.append(self.version)
            episode.observations.append(result.observation)
            episode.masks.append(result.action_mask.copy())
            if result.terminated or result.truncated:
                episode.closed = True
                chunks.append(self._chunk(episode, starts[index]))
                values = self._episode_metrics(episode, result)
                if self.demonstrations[index]:
                    values = {key.replace("train/", "demonstrations/", 1): value for key, value in values.items()}
                metrics.append(values)
                self.episodes[index] = self._reset(index)
                starts[index] = 0

    def _episode_metrics(self, episode: EpisodeRecord, result) -> dict[str, float]:
        positions = {(float(obs["self"][0]), float(obs["self"][1])) for obs in episode.observations}
        values = {"train/episode_length": float(len(episode.actions)), "train/observed_positions": float(len(positions)),
                  **{f"train/action_count/{action_id}": float(episode.actions.count(action_id)) for action_id in range(10)}}
        if not self.pretraining:
            values.update({"train/shaped_return": sum(episode.rewards), "train/success": float(result.info["success"]),
                "train/ice_delivered": float(result.info["objective_stats"]["ice_delivered"]),
                "train/samples_delivered": float(result.info["objective_stats"]["samples_delivered"])})
        return values

    def get_state(self) -> dict:
        """Save quiescent histories and RNGs for deterministic environment reconstruction."""
        return deepcopy({"episodes": self.episodes, "episode_seeds": self.episode_seeds, "counters": self.counters,
            "version": self.version, "rngs": [rng.bit_generator.state for rng in self.rngs],
            "checksums": [state_checksum(env.world_state) for env in self.environments],
            "agents": [{"state": agent.state, "previous_action": agent.previous_action,
                        "action_rng": agent.generator.get_state(), "latent_rng": agent.latent_generator.get_state()} for agent in self.agents]})


class RayCollectionPool:
    """One bounded task per actor, deterministic ingestion order, and owned-runtime cleanup."""

    def __init__(self, spec, component_registry=None, states=None, snapshot=None) -> None:
        import ray
        self.ray = ray
        self.spec = spec
        self.owns_runtime = not ray.is_initialized()
        self.actors = []
        self.tasks = []
        self.snapshot = snapshot
        if states is not None and len(states) != spec.resources.num_collectors:
            raise ValueError("resume collector count differs")
        try:
            if self.owns_runtime:
                ray.init(include_dashboard=False, log_to_driver=False, num_gpus=0)
            cpus = spec.resources.collector_torch_threads
            if ray.available_resources().get("CPU", 0) < spec.resources.num_collectors * cpus:
                raise ValueError("world-model collectors exceed available Ray CPUs")
            remote = ray.remote(num_cpus=cpus, num_gpus=0)(Collector)
            self.actors = [remote.remote(spec, index, component_registry,
                None if states is None else states[index], snapshot) for index in range(spec.resources.num_collectors)]
            ray.get([actor.get_state.remote() for actor in self.actors])
        except Exception:
            self.close()
            raise

    def collect(self, snapshot: dict, version: int, remaining: int, *, warmup: bool) -> list[CollectionResult]:
        total = len(self.actors) * self.spec.resources.num_envs_per_collector
        base, extra = divmod(remaining, total)
        counts = [min(self.spec.resources.collection_steps_per_env, base + int(index < extra)) for index in range(total)]
        self.snapshot = snapshot
        weights = self.ray.put(snapshot)
        size = self.spec.resources.num_envs_per_collector
        self.tasks = [actor.collect.remote(weights, version, counts[index * size:(index + 1) * size], warmup)
                      for index, actor in enumerate(self.actors) if any(counts[index * size:(index + 1) * size])]
        # ponytail: synchronous waves bound policy age; add bounded overlap only after measuring stalls.
        results = self.ray.get(self.tasks)
        self.tasks = []
        return results

    def get_states(self) -> list[dict]:
        if self.tasks:
            raise RuntimeError("cannot checkpoint pending collection")
        return self.ray.get([actor.get_state.remote() for actor in self.actors])

    def close(self) -> None:
        for actor in self.actors:
            self.ray.kill(actor, no_restart=True)
        self.actors = []
        if self.owns_runtime and self.ray.is_initialized():
            self.ray.shutdown()
