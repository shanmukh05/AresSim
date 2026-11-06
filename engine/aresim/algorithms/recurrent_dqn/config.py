"""Typed recurrent Double/Dueling DQN and encoder settings.

RLlib supplies sequence replay and the TD learner; these settings add explicit memory and burn-in.
"""
from __future__ import annotations
from dataclasses import dataclass
from ..dqn.config import MaskedDQNConfig
from ..common.model_config import ModelConfig
from ..common.config_decode import positive_integer


@dataclass(frozen=True)
class RecurrentDQNConfig(MaskedDQNConfig):
    """One-step recurrent DQN; batches are timestep budgets including burn-in."""

    sequence_length: int = 64
    burn_in_length: int = 16

    def validate(self) -> None:
        super().validate()
        positive_integer(self.sequence_length, "sequence_length")
        positive_integer(self.burn_in_length, "burn_in_length", allow_zero=True)
        if self.n_step != 1:
            raise ValueError("recurrent DQN sequence replay requires n_step=1")
        if self.burn_in_length >= self.sequence_length:
            raise ValueError("burn_in_length must be smaller than sequence_length")
        if not self.double_q:
            raise ValueError("recurrent DQN requires double_q")


@dataclass(frozen=True)
class RecurrentModelConfig(ModelConfig):
    """Existing observation encoder followed by one LSTM layer."""

    lstm_hidden_size: int = 256

    def validate(self) -> None:
        super().validate()
        positive_integer(self.lstm_hidden_size, "lstm_hidden_size")
