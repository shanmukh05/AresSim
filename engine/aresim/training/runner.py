"""Collect deterministic complete episodes from registered agents and environments.

Supports baselines, checkpoint evaluation, and optional trajectory export.
Online RLlib training collectors remain in :mod:`aresim.algorithms.ppo.train`.

**Last updated:** September 1, 2026

**Contains:** ``RolloutRunner``, ``RolloutConfig``, ``EpisodeSpec``, ``EpisodeSummary``,
``RolloutResult``.

**Default cutoff:** ``DEFAULT_MAX_EPISODE_STEPS`` (1200) unless the environment
terminates earlier.

**See also:** :mod:`aresim.training.trajectories` (dataset writer),
:mod:`aresim.training.evaluation` (checkpoint comparison).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..algorithms import Agent
from ..config import EnvironmentConfig
from ..defaults import DEFAULT_ENVIRONMENT_CONFIG
from ..factory import make_agent, make_env
from ..gameplay import TrajectoryRecorder
from ..integrations.ui import snapshot_from_state
from ..registry import ComponentRegistry, create_default_registry
from .trajectories import EpisodeTrajectory, TrajectoryWriter


DEFAULT_MAX_EPISODE_STEPS = 1200


@dataclass(frozen=True)
class EpisodeSpec:
    """Explicit environment and policy seeds for one reproducible episode."""

    episode_id: str
    environment_seed: int
    agent_seed: int

    def validate(self) -> None:
        """Reject empty identity or invalid NumPy generator seeds."""
        if not self.episode_id.strip():
            raise ValueError("episode_id cannot be empty")
        for label, value in (("environment_seed", self.environment_seed), ("agent_seed", self.agent_seed)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")


@dataclass(frozen=True)
class RolloutConfig:
    """Ordered episode plan and external cutoff for one baseline rollout."""

    episodes: tuple[EpisodeSpec, ...]
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS

    def validate(self) -> None:
        """Require a non-empty, uniquely identified, bounded episode sequence."""
        if not self.episodes:
            raise ValueError("rollout requires at least one episode")
        if not isinstance(self.max_episode_steps, int) or isinstance(self.max_episode_steps, bool) or self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be a positive integer")
        for episode in self.episodes:
            episode.validate()
        identifiers = [episode.episode_id for episode in self.episodes]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("rollout episode IDs must be unique")


@dataclass(frozen=True)
class EpisodeSummary:
    """Compact outcome for comparing or indexing one recorded episode."""

    episode_id: str
    policy_id: str
    scenario_id: str
    environment_seed: int
    agent_seed: int
    length: int
    episode_return: float
    engine_return: float
    terminated: bool
    truncated: bool
    ending_reason: str | None
    artifact_reference: str | None = None


@dataclass(frozen=True)
class RolloutResult:
    """Complete in-memory episodes, summaries, and optional dataset manifest."""

    episodes: tuple[EpisodeTrajectory[Any, Any], ...]
    summaries: tuple[EpisodeSummary, ...]
    transition_count: int
    artifact_manifest: str | None


class RolloutRunner:
    """Run one registered or direct agent over an explicit deterministic plan."""

    def __init__(
        self,
        rollout_config: RolloutConfig,
        agent: str | Agent[Any, Any],
        *,
        environment_config: EnvironmentConfig = DEFAULT_ENVIRONMENT_CONFIG,
        registry: ComponentRegistry | None = None,
    ) -> None:
        rollout_config.validate()
        environment_config.validate()
        for episode in rollout_config.episodes:
            if not environment_config.engine.world.seed_min <= episode.environment_seed <= environment_config.engine.world.seed_max:
                raise ValueError("environment seed is outside configured bounds")
        self.rollout_config = rollout_config
        self.environment_config = environment_config
        self.registry = create_default_registry() if registry is None else registry
        self.agent = make_agent(agent, environment_config, self.registry) if isinstance(agent, str) else agent
        if not isinstance(self.agent, Agent):
            raise TypeError("rollout agent does not implement Agent")

    def _require_compatible_schemas(self, observation_schema: str, action_schema: str) -> None:
        if self.agent.action_schema != action_schema:
            raise ValueError(
                f"agent action schema {self.agent.action_schema} is incompatible with {action_schema}"
            )
        if self.agent.observation_schema is not None and self.agent.observation_schema != observation_schema:
            raise ValueError(
                f"agent observation schema {self.agent.observation_schema} is incompatible with {observation_schema}"
            )

    def _record_transition(self, environment, action, current_mask: np.ndarray, trajectory_recorder: TrajectoryRecorder, observation_schema: str, action_schema: str):
        legal: bool | None = None
        if isinstance(action, (int, np.integer)) and 0 <= int(action) < current_mask.size:
            legal = bool(current_mask[int(action)])
        before_snapshot = snapshot_from_state(environment.world_state)
        result = environment.step(action)
        after_snapshot = snapshot_from_state(result.transition.state)
        trajectory_recorder.record(before_snapshot, after_snapshot)
        if str(result.info["observation_schema"]) != observation_schema or str(result.info["action_schema"]) != action_schema:
            raise RuntimeError("environment schema identifiers changed during an episode")
        return result, legal

    def _collect_episode(self, environment, spec: EpisodeSpec) -> EpisodeTrajectory[Any, Any]:
        reset = environment.reset(seed=spec.environment_seed)
        initial_snapshot = snapshot_from_state(environment.world_state)
        trajectory_recorder = TrajectoryRecorder.create(
            self.environment_config.engine.replay,
            initial_snapshot,
            checkpoint_mode="endpoints",
        )
        observation_schema = str(reset.info["observation_schema"])
        action_schema = str(reset.info["action_schema"])
        self._require_compatible_schemas(observation_schema, action_schema)
        self.agent.reset(spec.agent_seed)

        observations = [deepcopy(reset.observation)]
        action_masks = [reset.action_mask.copy()]
        actions: list[object] = []
        rewards: list[float] = []
        reward_breakdowns: list[dict[str, object]] = []
        engine_rewards: list[float] = []
        engine_reward_terms: list[dict[str, float]] = []
        terminated: list[bool] = []
        truncated: list[bool] = []
        action_legal: list[bool | None] = []
        effective_actions: list[str] = []
        events: list[tuple[str, ...]] = []
        terminal_reasons: list[str | None] = []
        state_checksums = [str(reset.info["state_checksum"])]

        while True:
            current_observation = deepcopy(observations[-1])
            current_mask = action_masks[-1].copy()
            action = self.agent.act(current_observation, current_mask)
            if not environment.action_space.contains(action):
                raise ValueError("agent returned an action outside the environment action space")
            result, legal = self._record_transition(
                environment, action, current_mask, trajectory_recorder, observation_schema, action_schema
            )
            actions.append(deepcopy(action))
            rewards.append(float(result.reward))
            reward_breakdowns.append(result.reward_breakdown.as_dict())
            engine_rewards.append(float(result.transition.reward))
            engine_reward_terms.append({key: float(value) for key, value in result.transition.reward_terms.items()})
            terminated.append(result.terminated)
            truncated.append(result.truncated)
            action_legal.append(legal)
            effective_actions.append(result.transition.effective_action.value)
            events.append(tuple(result.transition.events))
            terminal_reasons.append(
                result.info.get("terminal_reason") or result.info.get("truncation_reason")
            )
            observations.append(deepcopy(result.observation))
            action_masks.append(result.action_mask.copy())
            state_checksums.append(result.transition.after_checksum)
            if result.terminated or result.truncated:
                break

        return EpisodeTrajectory(
            episode_id=spec.episode_id,
            agent_id=reset.agent_id,
            policy_id=self.agent.policy_id,
            scenario_id=str(reset.info["scenario_id"]),
            task_id=str(reset.info["task_id"]),
            observation_schema=observation_schema,
            action_schema=action_schema,
            reward_profile=str(reset.info["reward_profile"]),
            environment_seed=spec.environment_seed,
            agent_seed=spec.agent_seed,
            environment_config=self.environment_config,
            observation_space=environment.observation_space,
            action_space=environment.action_space,
            action_mask_space=environment.action_mask_space,
            observations=tuple(observations),
            action_masks=tuple(action_masks),
            actions=tuple(actions),
            rewards=tuple(rewards),
            reward_breakdowns=tuple(reward_breakdowns),
            engine_rewards=tuple(engine_rewards),
            engine_reward_terms=tuple(engine_reward_terms),
            terminated=tuple(terminated),
            truncated=tuple(truncated),
            action_legal=tuple(action_legal),
            effective_actions=tuple(effective_actions),
            events=tuple(events),
            terminal_reasons=tuple(terminal_reasons),
            state_checksums=tuple(state_checksums),
            replay=trajectory_recorder.export_replay_projection(
                f"{spec.episode_id}.json",
                "algorithm",
                self.agent.policy_id,
            ),
        )

    @staticmethod
    def _summary(episode: EpisodeTrajectory[Any, Any]) -> EpisodeSummary:
        return EpisodeSummary(
            episode_id=episode.episode_id,
            policy_id=episode.policy_id,
            scenario_id=episode.scenario_id,
            environment_seed=episode.environment_seed,
            agent_seed=episode.agent_seed,
            length=episode.length,
            episode_return=episode.episode_return,
            engine_return=episode.engine_return,
            terminated=episode.terminated[-1],
            truncated=episode.truncated[-1],
            ending_reason=episode.ending_reason,
        )

    def run(self, writer: TrajectoryWriter | None = None) -> RolloutResult:
        """Collect all planned episodes and optionally finalize a JSONL dataset."""
        environment = make_env(
            self.environment_config,
            registry=self.registry,
            max_episode_steps=self.rollout_config.max_episode_steps,
        )
        episodes: list[EpisodeTrajectory[Any, Any]] = []
        summaries: list[EpisodeSummary] = []
        for spec in self.rollout_config.episodes:
            episode = self._collect_episode(environment, spec)
            episodes.append(episode)
            summaries.append(self._summary(episode))
            if writer is not None:
                writer.write_episode(episode)

        artifact_manifest: str | None = None
        if writer is not None:
            manifest_path = writer.finalize(max_episode_steps=self.rollout_config.max_episode_steps)
            artifact_manifest = str(Path(manifest_path).resolve())
            summaries = [replace(summary, artifact_reference=artifact_manifest) for summary in summaries]
        return RolloutResult(
            episodes=tuple(episodes),
            summaries=tuple(summaries),
            transition_count=sum(episode.length for episode in episodes),
            artifact_manifest=artifact_manifest,
        )


__all__ = [
    "DEFAULT_MAX_EPISODE_STEPS",
    "EpisodeSpec",
    "EpisodeSummary",
    "RolloutConfig",
    "RolloutResult",
    "RolloutRunner",
]
