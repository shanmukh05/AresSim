"""Bounded complete-episode replay and MuZero target construction.

Training samples start positions; rewards and root search values form n-step
targets. The terminal boundary is absorbing, while a time limit bootstraps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Episode:
    """One completed environment episode with T actions and T+1 observations."""

    observations: list[dict[str, object]]
    masks: list[np.ndarray]
    actions: list[int]
    rewards: list[float]
    policies: list[np.ndarray]
    root_values: list[float]
    terminated: bool
    truncated: bool
    tail_value: float
    policy_versions: list[int] | None = None

    def validate(self) -> None:
        steps = len(self.actions)
        if not steps or len(self.observations) != steps + 1 or len(self.masks) != steps + 1:
            raise ValueError("MuZero episode observation alignment is invalid")
        if any(len(items) != steps for items in (self.rewards, self.policies, self.root_values)):
            raise ValueError("MuZero episode targets are misaligned")
        if self.terminated == self.truncated:
            raise ValueError("MuZero episode must end by termination or truncation")
        if self.policy_versions is not None and len(self.policy_versions) != steps:
            raise ValueError("MuZero episode policy versions are misaligned")

    def value_target(self, index: int, td_steps: int, gamma: float) -> float:
        """Observed n-step rewards plus an available searched or tail bootstrap."""
        count = len(self.actions)
        if index >= count:
            return self.tail_value if self.truncated and index == count else 0.0
        end = min(count, index + td_steps)
        value = sum(gamma ** (step - index) * self.rewards[step] for step in range(index, end))
        if end == index + td_steps and end < count:
            value += gamma ** (end - index) * self.root_values[end]
        elif end == count and self.truncated:
            value += gamma ** (end - index) * self.tail_value
        return float(value)


def history(episode: Episode, index: int, length: int) -> tuple[list[dict[str, object]], list[int]]:
    """Left-pad initial episode history with its first observation and no-action ID."""
    observations = [episode.observations[max(0, index - length + 1 + offset)] for offset in range(length)]
    actions = [episode.actions[position - 1] if position > 0 else 10
               for position in (index - length + 1 + offset for offset in range(length))]
    return observations, actions


class Replay:
    """Keep the newest complete episodes within a transition budget."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.episodes: list[Episode] = []
        self.priorities: list[np.ndarray] = []
        self.steps = 0

    def add(self, episode: Episode, td_steps: int, gamma: float, alpha: float) -> None:
        episode.validate()
        self.episodes.append(episode)
        self.priorities.append(np.asarray([
            (abs(episode.root_values[index] - episode.value_target(index, td_steps, gamma)) + 1e-6) ** alpha
            for index in range(len(episode.actions))], dtype=np.float64))
        self.steps += len(episode.actions)
        while self.steps > self.capacity and len(self.episodes) > 1:
            removed = self.episodes.pop(0)
            self.priorities.pop(0)
            self.steps -= len(removed.actions)

    def sample(self, batch_size: int, beta: float,
               rng: np.random.Generator) -> list[tuple[Episode, int, float]]:
        """Prioritize root-search versus return error with importance correction."""
        totals = np.asarray([priority.sum() for priority in self.priorities], dtype=np.float64)
        grand_total = totals.sum()
        episode_indices = rng.choice(len(self.episodes), size=batch_size, p=totals / grand_total)
        selected = [(int(ep_index), int(rng.choice(len(self.priorities[ep_index]),
                         p=self.priorities[ep_index] / totals[ep_index]))) for ep_index in episode_indices]
        probabilities = np.asarray([self.priorities[ep_index][position] / grand_total
                                    for ep_index, position in selected])
        weights = (self.steps * probabilities) ** -beta
        weights /= weights.max()
        return [(self.episodes[ep_index], position, float(weight))
                for (ep_index, position), weight in zip(selected, weights, strict=True)]

    def update_priorities(self, samples, errors, alpha: float) -> None:
        """Refresh priorities from current learner errors, preserving episode identity."""
        indices = {id(episode): index for index, episode in enumerate(self.episodes)}
        for (episode, position, _), error in zip(samples, errors, strict=True):
            index = indices.get(id(episode))
            if index is not None:
                self.priorities[index][position] = (abs(float(error)) + 1e-6) ** alpha


__all__ = ["Episode", "Replay", "history"]
