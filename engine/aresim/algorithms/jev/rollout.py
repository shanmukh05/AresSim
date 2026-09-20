"""Run Jev from a small rollout YAML (not an RLlib experiment).

Used by ``aresim-rl rollout`` with ``configs/jev/*.yaml``. Does not start Ray.

**Last updated:** September 20, 2026

**Schema:** ``aresim.rollout.v1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...defaults import DEFAULT_ENVIRONMENT_CONFIG
from ...training.runner import EpisodeSpec, RolloutConfig, RolloutResult, RolloutRunner
from ...training.seeds import resolve_checkout_file
from ..common.config_decode import mapping, positive_integer
from .agent import JevAgent, make_test_jev_agent
from .config import JevAgentConfig, decode_config


ROLLOUT_SCHEMA = "aresim.rollout.v1"


@dataclass(frozen=True)
class JevRolloutSpec:
    """Checked-in Jev rollout plan independent of RLlib experiment YAML."""

    schema_version: str
    agent: str
    max_episode_steps: int
    episodes: tuple[EpisodeSpec, ...]
    fake_client: bool = False
    algorithm_config: object | None = None

    def validate(self) -> None:
        if self.schema_version != ROLLOUT_SCHEMA:
            raise ValueError(f"unsupported rollout schema: {self.schema_version}")
        if self.agent != "jev":
            raise ValueError("rollout agent must be jev")
        positive_integer(self.max_episode_steps, "max_episode_steps")
        if not self.episodes:
            raise ValueError("rollout requires at least one episode")
        if not isinstance(self.fake_client, bool):
            raise ValueError("fake_client must be a boolean")
        for episode in self.episodes:
            episode.validate()


def _episodes(value: object) -> tuple[EpisodeSpec, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("episodes must be a non-empty list")
    episodes: list[EpisodeSpec] = []
    for item in value:
        payload = mapping(item, "episode")
        episodes.append(
            EpisodeSpec(
                episode_id=str(payload.get("id", "")),
                environment_seed=positive_integer(payload.get("environment_seed"), "environment_seed", allow_zero=True),
                agent_seed=positive_integer(payload.get("agent_seed"), "agent_seed", allow_zero=True),
            )
        )
    return tuple(episodes)


def _jev_config(raw: object | None) -> JevAgentConfig:
    config = JevAgentConfig() if raw is None else decode_config(JevAgentConfig, raw, "algorithm_config")
    config.validate()
    return config


def load_jev_rollout(path: Path) -> JevRolloutSpec:
    """Decode one ``aresim.rollout.v1`` YAML file for Jev."""
    import yaml

    payload = yaml.safe_load(resolve_checkout_file(path, kind="rollout config").read_text(encoding="utf-8"))
    values = mapping(payload, "rollout")
    episodes = _episodes(values.pop("episodes", None))
    spec = JevRolloutSpec(
        schema_version=str(values.get("schema_version", "")),
        agent=str(values.get("agent", "")),
        max_episode_steps=positive_integer(values.get("max_episode_steps"), "max_episode_steps"),
        episodes=episodes,
        fake_client=bool(values.get("fake_client", False)),
        algorithm_config=values.get("algorithm_config"),
    )
    extra = set(values) - {"schema_version", "agent", "max_episode_steps", "fake_client", "algorithm_config"}
    if extra:
        raise ValueError(f"unknown rollout fields: {', '.join(sorted(extra))}")
    spec.validate()
    return spec


def run_jev_rollout(spec: JevRolloutSpec) -> RolloutResult:
    """Run ``RolloutRunner`` with a live or fake Jev client."""
    config = _jev_config(spec.algorithm_config)
    if spec.fake_client:
        agent = make_test_jev_agent(config=config)
    else:
        agent = JevAgent.from_environment(DEFAULT_ENVIRONMENT_CONFIG, config=config)
    plan = RolloutConfig(episodes=spec.episodes, max_episode_steps=spec.max_episode_steps)
    return RolloutRunner(plan, agent).run()


__all__ = ["JevRolloutSpec", "load_jev_rollout", "run_jev_rollout"]
