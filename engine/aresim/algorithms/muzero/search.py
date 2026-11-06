"""Single-root and batched PUCT search over MuZero's learned latent transitions.

Only the real root receives an authoritative environment action mask. Imagined
nodes use model priors and rewards without querying or copying the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log, sqrt

import numpy as np
import torch
from torch import Tensor

from ..common.masks import require_mask
from .config import MuZeroConfig
from .model import MuZeroNetwork, support_value


@dataclass
class _Node:
    prior: float
    reward: float = 0.0
    latent: Tensor | None = None
    value_sum: float = 0.0
    visits: int = 0
    continuation: float = 1.0
    children: dict[int, _Node] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(frozen=True)
class SearchResult:
    policy: np.ndarray
    root_value: float
    visits: np.ndarray


@dataclass
class SearchStats:
    """Count real roots and batched recurrent inference, including terminal reuse."""

    roots: int = 0
    recurrent_calls: int = 0
    expanded_leaves: int = 0
    maximum_batch_size: int = 0


def _expand(node: _Node, logits: Tensor, legal: np.ndarray | None = None) -> None:
    scores = logits.detach().float().cpu().numpy().reshape(-1)
    _expand_scores(node, scores, legal)


def _expand_scores(node: _Node, scores: np.ndarray, legal: np.ndarray | None = None) -> None:
    indices = np.arange(10) if legal is None else np.flatnonzero(legal)
    shifted = scores[indices] - max(scores[indices])
    priors = np.exp(shifted)
    priors /= priors.sum()
    node.children = {int(action): _Node(float(prior)) for action, prior in zip(indices, priors, strict=True)}


def _select(node: _Node, minimum: float, maximum: float, config: MuZeroConfig) -> tuple[int, _Node]:
    total = sum(child.visits for child in node.children.values())
    coefficient = config.puct_c1 + log((total + config.puct_c2 + 1) / config.puct_c2)
    exploration_scale = sqrt(total + 1)
    def score(item: tuple[int, _Node]) -> float:
        child = item[1]
        q = child.reward + config.gamma * child.continuation * child.value if child.visits else 0.0
        normalized = (q - minimum) / (maximum - minimum) if maximum > minimum and child.visits else 0.0
        return normalized + child.prior * exploration_scale * coefficient / (child.visits + 1)
    return max(node.children.items(), key=score)


def search(
    network: MuZeroNetwork,
    observations: dict[str, Tensor],
    previous_actions: Tensor,
    action_mask: np.ndarray,
    config: MuZeroConfig,
    *,
    rng: np.random.Generator | None = None,
    training: bool = False,
) -> SearchResult:
    """Return visit policy and searched value for one real rover observation."""
    legal = require_mask(action_mask)
    simulations = config.simulations if training else config.evaluation_simulations
    with torch.no_grad():
        latent, logits, value_logits = network.initial(observations, previous_actions)
        root = _Node(1.0, latent=latent)
        _expand(root, logits, legal)
        initial_value = float(support_value(value_logits, network.config.support_size).item())
        if training:
            if rng is None:
                raise ValueError("training search requires a seeded generator")
            noise = rng.dirichlet(np.full(len(root.children), config.dirichlet_alpha))
            for child, sample in zip(root.children.values(), noise, strict=True):
                child.prior = (1 - config.dirichlet_fraction) * child.prior + config.dirichlet_fraction * float(sample)
        minimum, maximum = float("inf"), float("-inf")
        for _ in range(simulations):
            node = root
            path: list[_Node] = [root]
            while node.children:
                action, child = _select(node, minimum, maximum, config)
                path.append(child)
                if child.latent is None:
                    next_latent, reward_logits, child_logits, child_value_logits = network.recurrent(
                        node.latent, torch.tensor([action], device=latent.device)
                    )
                    child.latent = next_latent
                    child.reward = float(support_value(reward_logits, network.config.support_size).item())
                    imagined_mask = None
                    if network.config.predict_constraints:
                        mask_logits, continuation_logits = network.constraints(next_latent)
                        imagined_mask = (mask_logits.sigmoid().cpu().numpy().reshape(-1) >= 0.5)
                        # Wait is always valid, even when the remaining predictions disagree.
                        imagined_mask[0] = True
                        child.continuation = float(continuation_logits.sigmoid().item() >= 0.5)
                    if child.continuation:
                        _expand(child, child_logits, imagined_mask)
                        leaf_value = float(support_value(child_value_logits, network.config.support_size).item())
                    else:
                        leaf_value = 0.0
                    break
                node = child
            else:
                leaf_value = initial_value if node is root else 0.0
            value = leaf_value
            path[-1].value_sum += value
            path[-1].visits += 1
            for parent, child in zip(reversed(path[:-1]), reversed(path[1:]), strict=True):
                value = child.reward + config.gamma * child.continuation * value
                parent.value_sum += value
                parent.visits += 1
                minimum = min(minimum, value)
                maximum = max(maximum, value)
        visits = np.zeros(10, dtype=np.int32)
        for action, child in root.children.items():
            visits[action] = child.visits
        policy = visits.astype(np.float32) / visits.sum()
        return SearchResult(policy, root.value, visits)


@torch.no_grad()
def search_many(
    network: MuZeroNetwork,
    observations: dict[str, Tensor],
    previous_actions: Tensor,
    action_masks: np.ndarray,
    config: MuZeroConfig,
    *,
    rngs: list[np.random.Generator] | None = None,
    training: bool = False,
    stats: SearchStats | None = None,
) -> list[SearchResult]:
    """Search independent roots, batching one unvisited leaf per active tree.

    Every tree retains its own mask, exploration RNG and value normalization.
    Absorbing nodes consume visits without another model call. Model parameters
    must remain unchanged for the complete call; latents are never overwritten.
    """
    count = previous_actions.shape[0]
    if count < 1 or action_masks.shape != (count, 10):
        raise ValueError("MuZero search requires one ten-action mask per root")
    if training and (rngs is None or len(rngs) != count):
        raise ValueError("training search requires one seeded generator per root")
    legal_masks = [require_mask(mask) for mask in action_masks]
    simulations = config.simulations if training else config.evaluation_simulations
    latents, logits, value_logits = network.initial(observations, previous_actions)
    initial = torch.cat((logits.float(), support_value(value_logits, network.config.support_size)[:, None]), -1).cpu().numpy()
    roots = [_Node(1.0, latent=latents[index:index + 1]) for index in range(count)]
    for index, (root, legal) in enumerate(zip(roots, legal_masks, strict=True)):
        _expand_scores(root, initial[index, :10], legal)
        if training:
            noise = rngs[index].dirichlet(np.full(len(root.children), config.dirichlet_alpha))
            for child, sample in zip(root.children.values(), noise, strict=True):
                child.prior = (1 - config.dirichlet_fraction) * child.prior + config.dirichlet_fraction * float(sample)
    minimum = [float("inf")] * count
    maximum = [float("-inf")] * count
    if stats is not None:
        stats.roots += count
        stats.maximum_batch_size = max(stats.maximum_batch_size, count)
    for _ in range(simulations):
        paths: list[list[_Node]] = []
        pending: list[tuple[int, _Node]] = []
        parent_latents = []
        actions = []
        leaf_values = np.zeros(count)
        for index, root in enumerate(roots):
            node = root
            path = [root]
            while node.children:
                action, child = _select(node, minimum[index], maximum[index], config)
                path.append(child)
                if child.latent is None:
                    pending.append((index, child))
                    parent_latents.append(node.latent)
                    actions.append(action)
                    break
                node = child
            else:
                leaf_values[index] = initial[index, 10] if node is root else 0.0
            paths.append(path)
        if pending:
            next_latents, reward_logits, child_logits, child_values = network.recurrent(
                torch.cat(parent_latents), torch.tensor(actions, device=latents.device))
            # Decode on the model device and transfer the complete expansion once.
            outputs = [child_logits.float(),
                       support_value(reward_logits, network.config.support_size)[:, None],
                       support_value(child_values, network.config.support_size)[:, None]]
            if network.config.predict_constraints:
                mask_logits, continuation_logits = network.constraints(next_latents)
                outputs.extend((mask_logits.sigmoid(), continuation_logits.sigmoid()[:, None]))
            decoded = torch.cat(outputs, -1).cpu().numpy()
            if stats is not None:
                stats.recurrent_calls += 1
                stats.expanded_leaves += len(pending)
                stats.maximum_batch_size = max(stats.maximum_batch_size, len(pending))
            for row, (index, child) in enumerate(pending):
                # A view retains its own immutable inference batch, never a reused buffer.
                child.latent = next_latents[row:row + 1]
                child.reward = float(decoded[row, 10])
                imagined_mask = None
                if network.config.predict_constraints:
                    imagined_mask = decoded[row, 12:22] >= 0.5
                    imagined_mask[0] = True
                    child.continuation = float(decoded[row, 22] >= 0.5)
                if child.continuation:
                    _expand_scores(child, decoded[row, :10], imagined_mask)
                    leaf_values[index] = float(decoded[row, 11])
        for index, path in enumerate(paths):
            value = float(leaf_values[index])
            path[-1].value_sum += value
            path[-1].visits += 1
            for parent, child in zip(reversed(path[:-1]), reversed(path[1:]), strict=True):
                value = child.reward + config.gamma * child.continuation * value
                parent.value_sum += value
                parent.visits += 1
                minimum[index] = min(minimum[index], value)
                maximum[index] = max(maximum[index], value)
    results = []
    for root in roots:
        visits = np.zeros(10, dtype=np.int32)
        for action, child in root.children.items():
            visits[action] = child.visits
        results.append(SearchResult(visits.astype(np.float32) / visits.sum(), root.value, visits))
    return results


def choose_action(result: SearchResult, *, training: bool, temperature: float, rng: np.random.Generator | None = None) -> int:
    """Sample tempered visits when requested, otherwise select visit argmax.

    Training and stochastic evaluation both request sampling here. Root noise
    is controlled separately by ``search(training=...)``.
    """
    if not training:
        return int(np.argmax(result.visits))
    if rng is None:
        raise ValueError("training action selection requires a seeded generator")
    weights = result.visits.astype(np.float64) ** (1 / temperature)
    weights /= weights.sum()
    return int(rng.choice(10, p=weights))


__all__ = ["SearchResult", "SearchStats", "search", "search_many", "choose_action"]
