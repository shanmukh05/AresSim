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
    prefix_positions: Tensor | None = None


class SequenceReplay:
    """Uniform-over-transition sequence starts with whole completed-episode eviction."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.episodes: OrderedDict[str, EpisodeRecord] = OrderedDict()
        self.steps = 0
        self._columns = {}

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_columns", None)
        return state

    def add(self, chunk: ReplayChunk) -> None:
        chunk.episode.validate()
        suffix = chunk.episode
        getattr(self, "_columns", {}).pop(suffix.episode_id, None)
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
                getattr(self, "_columns", {}).pop(key, None)
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
        columns = [self._episode_columns(ep) for ep in selected]
        return _sequence_batch(columns, np.asarray(starts), length, device)

    def _episode_columns(self, episode: EpisodeRecord) -> dict:
        cache = getattr(self, "_columns", None)
        if cache is None:
            self._columns = cache = {}
        count = len(episode.actions)
        stored = cache.get(episode.episode_id)
        if stored is None or stored[0] != count:
            columns = {"observations": {name: np.stack([np.asarray(obs[name]) for obs in episode.observations])
                                        for name in episode.observations[0]},
                       "masks": np.stack(episode.masks)}
            for name in ("actions", "rewards", "reward_valid", "terminated", "truncated"):
                columns[name] = np.asarray(getattr(episode, name))
            cache[episode.episode_id] = count, columns
        return cache[episode.episode_id][1]

    def for_adaptation(self) -> SequenceReplay:
        """Keep pretraining experience as representation data without task labels or active IDs."""
        replay = deepcopy(self)
        for episode in replay.episodes.values():
            episode.reward_valid = [False] * len(episode.actions)
            episode.rewards = [0.0] * len(episode.actions)
            episode.closed = True
        return replay


def _sequence_batch(columns: list[dict], starts: np.ndarray, length: int, device) -> SequenceBatch:
    """Gather raw columns with the original padding, previous-action and ending semantics."""
    counts = np.asarray([len(row["actions"]) for row in columns])
    positions = starts[:, None] + np.arange(length + 1)
    indices = np.minimum(positions, counts[:, None])

    def observations(at):
        return {name: torch.as_tensor(np.stack([row["observations"][name][index] for row, index in zip(columns, at, strict=True)]), device=device)
                for name in columns[0]["observations"]}

    def gather(name, at):
        return np.stack([row[name][index] for row, index in zip(columns, at, strict=True)])

    def previous(at):
        values = gather("actions", np.maximum(at - 1, 0))
        return torch.as_tensor(np.where(at > 0, values, 10), dtype=torch.long, device=device)

    valid = positions[:, :-1] < counts[:, None]
    values = {}
    for name, dtype, default in (("actions", torch.long, 0), ("rewards", torch.float32, 0.),
                                  ("reward_valid", torch.bool, False), ("terminated", torch.bool, True),
                                  ("truncated", torch.bool, False)):
        raw = gather(name, np.minimum(indices[:, :-1], counts[:, None] - 1))
        values[name] = torch.as_tensor(np.where(valid, raw, default), dtype=dtype, device=device)
    maximum = int(starts.max())
    prefix_indices = np.minimum(np.arange(maximum)[None, :], starts[:, None])
    rows, times = np.nonzero(np.arange(maximum)[None, :] < starts[:, None])
    prefix = {name: torch.as_tensor(np.concatenate([row["observations"][name][:start] for row, start in zip(columns, starts, strict=True)]), device=device)
              for name in columns[0]["observations"]} if maximum else None
    # Only encoder inputs are packed; RSSM rows and RNG draws retain their original layout.
    return SequenceBatch(observations=observations(indices),
        masks=torch.as_tensor(gather("masks", indices), dtype=torch.bool, device=device),
        previous_actions=previous(indices), **values,
        valid=torch.as_tensor(valid, device=device),
        observation_valid=torch.as_tensor(positions <= counts[:, None], device=device),
        prefix_observations=prefix,
        prefix_previous_actions=previous(prefix_indices) if maximum else None,
        prefix_lengths=torch.as_tensor(starts, device=device),
        prefix_positions=torch.as_tensor(rows * maximum + times, dtype=torch.long, device=device) if maximum else None)
