"""Episode-aligned streaming replay and typed padded sequence batches.

Raw policy observations remain available for recurrent prefix reconstruction under current weights.
Replay is internal trainer state, independent of the portable trajectory format.
"""
from __future__ import annotations
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
import numpy as np
import torch
from torch import Tensor


@dataclass
class EpisodeRecord:
    """One contiguous history, containing one more observation/mask than actions."""

    episode_id: str
    observations: list[dict[str, object]]
    masks: list[np.ndarray]
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    reward_valid: list[bool] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)
    policy_versions: list[int] = field(default_factory=list)
    closed: bool = False

    def validate(self) -> None:
        count = len(self.actions)
        fields = (self.rewards, self.reward_valid, self.terminated, self.truncated, self.policy_versions)
        if len(self.observations) != count + 1 or len(self.masks) != count + 1 or any(len(v) != count for v in fields):
            raise ValueError("replay history is misaligned")
        if any(self.terminated[:-1]) or any(self.truncated[:-1]):
            raise ValueError("replay sequence crosses an episode boundary")
        if any(not 0 <= a < 10 or not self.masks[t][a] for t, a in enumerate(self.actions)):
            raise ValueError("replay contains an illegal action")


@dataclass
class ReplayChunk:
    """New contiguous suffix of one collector episode; starts at a real time index."""

    start: int
    episode: EpisodeRecord


@dataclass
class SequenceBatch:
    """B,T action-aligned training tensors plus an optional no-gradient prefix."""

    observations: dict[str, Tensor]
    masks: Tensor
    actions: Tensor
    previous_actions: Tensor
    rewards: Tensor
    reward_valid: Tensor
    terminated: Tensor
    truncated: Tensor
    valid: Tensor
    observation_valid: Tensor
    prefix_observations: dict[str, Tensor] | None
    prefix_previous_actions: Tensor | None
    prefix_lengths: Tensor


def _observations(histories: list[list[dict[str, object]]], device) -> dict[str, Tensor]:
    return {name: torch.as_tensor(np.stack([np.stack([np.asarray(obs[name]) for obs in history]) for history in histories]), device=device)
            for name in histories[0][0]}


class SequenceReplay:
    """Uniform-over-transition sequence starts with whole completed-episode eviction."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.episodes: OrderedDict[str, EpisodeRecord] = OrderedDict()
        self.steps = 0

    def add(self, chunk: ReplayChunk) -> None:
        chunk.episode.validate()
        suffix = chunk.episode
        previous = self.episodes.get(suffix.episode_id)
        if previous is None:
            if chunk.start != 0:
                raise ValueError("replay chunk is missing its episode prefix")
            self.episodes[suffix.episode_id] = deepcopy(suffix)
        else:
            if previous.closed or chunk.start != len(previous.actions):
                raise ValueError("replay chunk is duplicate or out of order")
            if any(not np.array_equal(previous.observations[-1][key], suffix.observations[0][key]) for key in previous.observations[-1]) or not np.array_equal(previous.masks[-1], suffix.masks[0]):
                raise ValueError("replay chunk boundary differs")
            previous.observations.extend(deepcopy(suffix.observations[1:]))
            previous.masks.extend(deepcopy(suffix.masks[1:]))
            for name in ("actions", "rewards", "reward_valid", "terminated", "truncated", "policy_versions"):
                getattr(previous, name).extend(getattr(suffix, name))
            previous.closed = suffix.closed
        self.steps += len(suffix.actions)
        self._evict()

    def _evict(self) -> None:
        for key in list(self.episodes):
            if self.steps <= self.capacity:
                return
            episode = self.episodes[key]
            if episode.closed:
                self.steps -= len(episode.actions)
                del self.episodes[key]
        if self.steps > self.capacity:
            raise ValueError("replay capacity cannot contain active episode histories")

    def resize(self, capacity: int) -> None:
        """Apply a new transition budget and evict completed histories as needed."""
        from ...algorithms.common.config_decode import positive_integer
        positive_integer(capacity, "replay capacity")
        active_steps = sum(len(episode.actions) for episode in self.episodes.values() if not episode.closed)
        if active_steps > capacity:
            raise ValueError("replay capacity cannot contain active episode histories")
        self.capacity = capacity
        self._evict()

    def sample(self, batch_size: int, length: int, rng: np.random.Generator, device) -> SequenceBatch:
        """Copy padded windows without crossing an ending; retain real final observations."""
        episodes = [ep for ep in self.episodes.values() if ep.actions]
        if not episodes:
            raise ValueError("cannot sample empty replay")
        sizes = np.asarray([len(ep.actions) for ep in episodes], dtype=float)
        choices = rng.choice(len(episodes), batch_size, p=sizes / sizes.sum())
        selected = [episodes[int(index)] for index in choices]
        starts = [int(rng.integers(len(ep.actions))) for ep in selected]
        windows = [_window(ep, start, length) for ep, start in zip(selected, starts, strict=True)]
        prefix_obs, prefix_prev = _prefix_tensors(selected, starts, device)
        tensors = _window_tensors(windows, device)
        return SequenceBatch(**tensors, prefix_observations=prefix_obs, prefix_previous_actions=prefix_prev,
                             prefix_lengths=torch.tensor(starts, device=device))

    def for_adaptation(self) -> SequenceReplay:
        """Keep pretraining experience as representation data without task labels or active IDs."""
        replay = deepcopy(self)
        for episode in replay.episodes.values():
            episode.reward_valid = [False] * len(episode.actions)
            episode.rewards = [0.0] * len(episode.actions)
            episode.closed = True
        return replay


def _padded(values: list, start: int, length: int, default) -> list:
    return [values[t] if t < len(values) else default for t in range(start, start + length)]


def _previous(episode: EpisodeRecord, positions: list[int]) -> list[int]:
    return [episode.actions[t - 1] if t > 0 else 10 for t in positions]


def _window(episode: EpisodeRecord, start: int, length: int) -> dict:
    count = len(episode.actions)
    positions = [min(start + k, count) for k in range(length + 1)]
    return {"observations": [episode.observations[t] for t in positions],
            "masks": [episode.masks[t] for t in positions], "previous_actions": _previous(episode, positions),
            "actions": _padded(episode.actions, start, length, 0),
            "rewards": _padded(episode.rewards, start, length, 0.0),
            "reward_valid": _padded(episode.reward_valid, start, length, False),
            "terminated": _padded(episode.terminated, start, length, True),
            "truncated": _padded(episode.truncated, start, length, False),
            "valid": [start + k < count for k in range(length)],
            "observation_valid": [start + k <= count for k in range(length + 1)]}


def _prefix_tensors(episodes: list[EpisodeRecord], starts: list[int], device):
    maximum = max(starts)
    if maximum == 0:
        return None, None
    histories = [[ep.observations[min(t, start)] for t in range(maximum)] for ep, start in zip(episodes, starts, strict=True)]
    # The padded prefix is ignored after each row's true start position.
    previous = [_previous(ep, [min(t, start) for t in range(maximum)]) for ep, start in zip(episodes, starts, strict=True)]
    return _observations(histories, device), torch.tensor(previous, device=device)


def _window_tensors(windows: list[dict], device) -> dict:
    dtypes = {"masks": torch.bool, "actions": torch.long, "previous_actions": torch.long,
              "rewards": torch.float32, "reward_valid": torch.bool, "terminated": torch.bool,
              "truncated": torch.bool, "valid": torch.bool, "observation_valid": torch.bool}
    tensors = {key: torch.as_tensor(np.asarray([row[key] for row in windows]), dtype=dtype, device=device)
               for key, dtype in dtypes.items()}
    tensors["observations"] = _observations([row["observations"] for row in windows], device)
    return tensors
