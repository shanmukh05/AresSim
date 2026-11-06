"""Build seeded RLlib environments from frozen AresSim configuration.

Workers consume the canonical Gym adapter; simulator rules remain in the engine.
"""
from __future__ import annotations
from collections.abc import Mapping
import gymnasium as gym
from ...factory import make_gym_env
from ..provenance import environment_from_payload

ENV_NAME = "aresim_gym_rllib_v1"

class _SeededTrainingEnv(gym.Env):
    """Cycle one RLlib worker through the fixed training seed list deterministically.

    Offsets seed selection per worker, vector index, and learner seed so parallel
    runners do not reset identical worlds every episode.
    """

    metadata = {"render_modes": []}

    def __init__(self, environment, seeds: tuple[int, ...], offset: int) -> None:
        if not seeds:
            raise ValueError("RLlib training requires at least one environment seed")
        self.environment = environment
        self.seeds = seeds
        self.offset = offset % len(seeds)
        self.episode_index = 0
        self.observation_space = environment.observation_space
        self.action_space = environment.action_space

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            selected = self.seeds[(self.offset + int(seed)) % len(self.seeds)]
        else:
            selected = self.seeds[(self.offset + self.episode_index) % len(self.seeds)]
            self.episode_index += 1
        return self.environment.reset(seed=selected, options=options)

    def step(self, action):
        return self.environment.step(action)

def _environment(env_config: Mapping[str, object]):
    values = env_config["environment"]
    config = environment_from_payload(values["resolved_config"])
    environment = make_gym_env(
        config,
        registry=env_config.get("component_registry"),
        max_episode_steps=int(values["max_episode_steps"]),
        audit=False,
    )
    seeds = tuple(int(seed) for seed in env_config["training_seeds"])
    worker_index = int(getattr(env_config, "worker_index", 0))
    vector_index = int(getattr(env_config, "vector_index", 0))
    learner_seed = int(env_config["learner_seed"])
    offset = learner_seed * 104729 + worker_index * 1009 + vector_index * 9176
    return _SeededTrainingEnv(environment, seeds, offset)
