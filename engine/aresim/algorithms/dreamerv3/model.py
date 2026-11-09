"""Categorical recurrent world model and field-aware symbolic reconstruction.

The model learns only from policy-visible inputs and recorded actions/masks.
Real simulator transitions remain in the environment; imagination never invokes engine rules.
"""
from __future__ import annotations
from collections.abc import Mapping
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from ..common.encoder import LocalObservationEncoder
from .config import DreamerModelConfig
from .distributions import sample_latent, symlog

State = tuple[Tensor, Tensor]
CATEGORIES = {"terrain_type": 8, "pad_proximity": 3, "weather_type": 6, "objective_type": 9}
BINARY = {"cell_flags", "objective_mask", "visited"}
SPATIAL_FIELDS = {"terrain_type", "spatial", "cell_flags", "visited"}


def _hidden_layers(inputs: int, hidden: int) -> list[nn.Module]:
    return [nn.Linear(inputs, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU()]


def mlp(inputs: int, hidden: int, outputs: int, *, zero_output: bool = False) -> nn.Sequential:
    """Two normalized hidden layers shared by prediction and behavior heads."""
    network = nn.Sequential(*_hidden_layers(inputs, hidden), nn.Linear(hidden, outputs))
    if zero_output:
        nn.init.zeros_(network[-1].weight)
        nn.init.zeros_(network[-1].bias)
    return network


class RSSM(nn.Module):
    """GRU state plus factorized categorical posterior and prior distributions."""

    def __init__(self, config: DreamerModelConfig) -> None:
        super().__init__()
        self.config = config
        stochastic = config.stochastic_size * config.categories
        self.recurrent = nn.GRUCell(stochastic + 10, config.deterministic_size)
        self.prior = mlp(config.deterministic_size, config.hidden_size, stochastic)
        self.posterior = mlp(config.deterministic_size + config.fused_width, config.hidden_size, stochastic)

    def initial(self, batch: int, device) -> State:
        return (torch.zeros(batch, self.config.deterministic_size, device=device),
                torch.zeros(batch, self.config.stochastic_size, self.config.categories, device=device))

    def features(self, state: State) -> Tensor:
        return torch.cat((state[0], state[1].flatten(-2)), -1)

    def _logits(self, head: nn.Module, inputs: Tensor) -> Tensor:
        return head(inputs).reshape(*inputs.shape[:-1], self.config.stochastic_size, self.config.categories)

    def observe(self, state: State, previous_action: Tensor, embedding: Tensor, *, generator=None, deterministic=False):
        """Consume one real observation after the previous action; return state and KL logits."""
        hidden, posterior = self.posterior_step(state, previous_action, embedding)
        prior = self._logits(self.prior, hidden)
        stochastic = sample_latent(posterior, self.config.unimix, generator=generator, deterministic=deterministic)
        return (hidden, stochastic), prior, posterior

    def posterior_step(self, state: State, previous_action: Tensor, embedding: Tensor) -> tuple[Tensor, Tensor]:
        """Advance real-observation memory without computing an unused KL prior or sample."""
        hidden = self.recurrent(torch.cat((state[1].flatten(-2), previous_action), -1), state[0])
        return hidden, self._logits(self.posterior, torch.cat((hidden, embedding), -1))

    def imagine(self, state: State, action: Tensor, *, generator=None) -> State:
        hidden = self.recurrent(torch.cat((state[1].flatten(-2), action), -1), state[0])
        logits = self._logits(self.prior, hidden)
        return hidden, sample_latent(logits, self.config.unimix, generator=generator)


class StructuredDecoder(nn.Module):
    """Field-aware reconstruction with shared spatial/telemetry trunks or legacy heads."""

    def __init__(self, features: int, hidden: int, observation_space, *, layout: str = "independent") -> None:
        super().__init__()
        self.shapes = {name: tuple(space.shape) for name, space in observation_space.spaces.items()}
        self.groups = {name: "spatial" if name in SPATIAL_FIELDS else "telemetry" for name in self.shapes}
        self.trunks = nn.ModuleDict()
        if layout == "grouped":
            self.trunks.update({group: nn.Sequential(*_hidden_layers(features, hidden)) for group in sorted(set(self.groups.values()))})
            self.heads = nn.ModuleDict({name: nn.Linear(hidden, math.prod(shape) * CATEGORIES.get(name, 1))
                                       for name, shape in self.shapes.items()})
        elif layout == "independent":
            self.heads = nn.ModuleDict({name: mlp(features, hidden, math.prod(shape) * CATEGORIES.get(name, 1))
                                       for name, shape in self.shapes.items()})
        else:
            raise ValueError("decoder_layout must be independent or grouped")

    def losses(self, features: Tensor, observations: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Return per-position losses; caller applies real-observation validity masks."""
        leading = features.shape[:-1]
        losses = {}
        contexts = {group: trunk(features) for group, trunk in self.trunks.items()}
        for name, head in self.heads.items():
            shape = self.shapes[name]
            inputs = contexts[self.groups[name]] if contexts else features
            output = head(inputs).reshape(*leading, *shape, *([CATEGORIES[name]] if name in CATEGORIES else []))
            target = observations[name]
            if name in CATEGORIES:
                loss = -output.log_softmax(-1).gather(-1, target.long().unsqueeze(-1)).squeeze(-1)
            elif name in BINARY:
                loss = F.binary_cross_entropy_with_logits(output, target.float(), reduction="none")
            else:
                loss = (output - symlog(target.float())).square()
            if name in ("objective_type", "objectives"):
                mask = observations["objective_mask"].float()
                if name == "objectives":
                    mask = mask.unsqueeze(-1).expand_as(loss)
                axes = tuple(range(len(leading), loss.ndim))
                losses[name] = (loss * mask).sum(axes) / mask.sum(axes).clamp(min=1)
            else:
                losses[name] = loss.reshape(*leading, -1).mean(-1)
        return losses


class WorldModel(nn.Module):
    """Shared Dreamer/Plan2Explore model, with no task-controller parameters."""

    def __init__(self, config: DreamerModelConfig, observation_space) -> None:
        super().__init__()
        self.config = config
        size = observation_space["terrain_type"].shape[0]
        self.encoder = LocalObservationEncoder(config, size, memory="navigation" in observation_space.spaces)
        self.rssm = RSSM(config)
        self.feature_size = config.deterministic_size + config.stochastic_size * config.categories
        self.decoder = StructuredDecoder(self.feature_size, config.hidden_size, observation_space, layout=config.decoder_layout)
        self.reward = mlp(self.feature_size, config.hidden_size, config.value_bins, zero_output=True)
        self.continuation = mlp(self.feature_size, config.hidden_size, 1)
        self.legality = mlp(self.feature_size, config.hidden_size, 10)

    def encode_sequence(self, observations: Mapping[str, Tensor]) -> Tensor:
        """Flatten B,T only for the stateless encoder, preserving categorical field dtypes."""
        batch, steps = observations["terrain_type"].shape[:2]
        flat = {key: value.reshape(batch * steps, *value.shape[2:]) for key, value in observations.items()}
        return self.encoder(flat).reshape(batch, steps, -1)

    def imagined_mask(self, features: Tensor, threshold: float) -> Tensor:
        mask = self.legality(features).sigmoid() >= threshold
        mask[..., 0] = True
        return mask
