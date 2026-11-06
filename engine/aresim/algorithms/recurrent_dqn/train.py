"""Masked recurrent Double/Dueling DQN on RLlib's stateful replay path.

The LSTM owns episode memory; shared RLlib infrastructure owns collection and trial execution.
One-step targets use contiguous online/target unrolls, not independently reset next-state calls.
"""
from __future__ import annotations
from collections.abc import Mapping
import numpy as np
import torch
from torch import Tensor, nn
from ray.rllib.algorithms.dqn import DQN, DQNConfig
from ray.rllib.algorithms.dqn.torch.dqn_torch_learner import DQNTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.utils.replay_buffers.episode_replay_buffer import EpisodeReplayBuffer
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ..common.encoder import LocalObservationEncoder, combine_q_values
from ..dqn.train import AresMaskedDQNRLModule, MaskedDQNFactory, _q_heads, _heads, _require_local_spaces, _dqn_runtime_flags
from .config import RecurrentDQNConfig, RecurrentModelConfig
from ..registry import TrainingContext


class RecurrentEpisodeReplayBuffer(EpisodeReplayBuffer):
    """Fail clearly when every recorded episode is too short for configured burn-in."""

    def __init__(self, *, seed: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.rng = np.random.default_rng(seed)

    def get_state(self) -> dict:
        state = super().get_state()
        state["aresim_rng"] = self.rng.bit_generator.state
        return state

    def set_state(self, state: dict) -> None:
        super().set_state(state)
        self.rng.bit_generator.state = state["aresim_rng"]

    def sample(self, *args, **kwargs):
        minimum = kwargs.get("min_batch_length_T", 0)
        if minimum and not any(len(episode) > minimum for episode in self.episodes):
            raise ValueError("no replay sequence is longer than recurrent burn-in; lower burn_in_length")
        return super().sample(*args, **kwargs)


class RecurrentDQNLearner(DQNTorchLearner):
    """Keep native Huber TD learning and expose the actual sequence/burn-in counts."""

    def compute_loss_for_module(self, *, module_id, config, batch, fwd_out):
        loss = super().compute_loss_for_module(module_id=module_id, config=config, batch=batch, fwd_out=fwd_out)
        valid = batch[Columns.LOSS_MASK]
        learned = valid.clone()
        learned[:, :config.burn_in_len] = False
        self.metrics.log_value((module_id, "recurrent_learned_timesteps"), learned.sum().float())
        self.metrics.log_value((module_id, "recurrent_burn_in_timesteps"), (valid & ~learned).sum().float())
        self.metrics.log_value((module_id, "recurrent_padding_fraction"), (~valid).float().mean())
        return loss


class RecurrentDQN(DQN):
    """Construct the episode buffer directly around Ray 2.58's class/string dispatch bug."""

    def _create_local_replay_buffer_if_necessary(self, config):
        values = dict(config["replay_buffer_config"])
        values.pop("type")
        values["metrics_num_episodes_for_smoothing"] = self.config.metrics_num_episodes_for_smoothing
        return RecurrentEpisodeReplayBuffer(**values)


class RecurrentEncoder(nn.Module):
    """Encode B,T observations and carry independent LSTM hidden/cell state per batch row."""

    def __init__(self, config: RecurrentModelConfig, size: int, *, memory: bool) -> None:
        super().__init__()
        self.encoder = LocalObservationEncoder(config, size, memory=memory)
        self.lstm = nn.LSTM(config.fused_width, config.lstm_hidden_size, batch_first=True)
        self.hidden_size = config.lstm_hidden_size

    def get_initial_state(self) -> dict[str, Tensor]:
        return {"h": torch.zeros(self.hidden_size), "c": torch.zeros(self.hidden_size)}

    def sequence(self, observations: Mapping[str, Tensor], state: dict[str, Tensor] | None, burn_in: int = 0):
        terrain = observations["terrain_type"]
        single = terrain.ndim == 3
        observations = {key: value.unsqueeze(1) if single else value for key, value in observations.items()}
        batch, length = observations["terrain_type"].shape[:2]
        flat = {key: value.reshape(batch * length, *value.shape[2:]) for key, value in observations.items()}
        embedded = self.encoder(flat).reshape(batch, length, -1)
        recurrent = self._state(state, batch, embedded.device)
        output, recurrent = self._unroll(embedded, recurrent, burn_in)
        state_out = {key: value.squeeze(0) for key, value in zip(("h", "c"), recurrent, strict=True)}
        return (output[:, 0] if single else output), state_out

    def _state(self, state, batch: int, device):
        if state is None:
            return tuple(torch.zeros(1, batch, self.hidden_size, device=device) for _ in range(2))
        return tuple(state[key].to(device).reshape(1, batch, self.hidden_size) for key in ("h", "c"))

    def _unroll(self, embedded: Tensor, recurrent, burn_in: int):
        length = embedded.shape[1]
        prefix = min(burn_in, length)
        outputs = []
        if prefix:
            with torch.no_grad():
                initial, recurrent = self.lstm(embedded[:, :prefix].detach(), recurrent)
            recurrent = tuple(value.detach() for value in recurrent)
            outputs.append(initial)
        if prefix < length:
            learned, recurrent = self.lstm(embedded[:, prefix:], recurrent)
            outputs.append(learned)
        return torch.cat(outputs, 1), recurrent


class AresRecurrentDQNRLModule(AresMaskedDQNRLModule):
    """Keep state in all RLlib forward modes and exclude padded/burn-in TD targets."""

    def setup(self) -> None:
        super().setup()
        local_space, actions = _require_local_spaces(self.observation_space, self.action_space)
        values = dict(self.model_config)
        config = RecurrentModelConfig(**{key: value for key, value in values.items() if key in RecurrentModelConfig.__dataclass_fields__})
        self.encoder = RecurrentEncoder(config, int(local_space["terrain_type"].shape[0]), memory="navigation" in local_space.spaces)
        self.af, value = _q_heads(config.lstm_hidden_size, actions, self.uses_dueling)
        if value is not None:
            self.vf = value
        self.burn_in = int(values["burn_in_length"])

    def _qf_forward_helper(self, batch, encoder, head):
        inputs = batch[Columns.OBS]
        features, state = encoder.sequence(inputs["observation"], batch.get(Columns.STATE_IN))
        advantage, value = _heads(head)
        mask = self._safe_mask(inputs["action_mask"])
        return {"qf_preds": combine_q_values(features, advantage, value, mask), Columns.STATE_OUT: state}

    @staticmethod
    def _safe_mask(mask: Tensor) -> Tensor:
        # RLlib pads all fields with zero. Padded rows have no loss, but still need finite heads.
        legal = mask.bool().clone()
        legal[..., 0] |= ~legal.any(-1)
        return legal

    def _forward_exploration(self, batch, t: int, **kwargs):
        output = self.compute_q_values(batch)
        values = output["qf_preds"]
        mask = self._safe_mask(batch[Columns.OBS]["action_mask"])
        random_actions = torch.multinomial(mask.float().reshape(-1, mask.shape[-1]), 1).reshape(values.shape[:-1])
        self.epsilon_schedule.update(t)
        epsilon = self.epsilon_schedule.get_current_value()
        actions = torch.where(torch.rand(values.shape[:-1], device=values.device) < epsilon, random_actions, values.argmax(-1))
        return {Columns.ACTIONS: actions, Columns.STATE_OUT: output[Columns.STATE_OUT]}

    def _forward_train(self, batch):
        current = batch[Columns.OBS]
        following = batch[Columns.NEXT_OBS]
        # NEXT_OBS retains the real final frame even when the rest of the sequence is padded.
        observations = {key: torch.cat((value[:, :1], following["observation"][key]), 1) for key, value in current["observation"].items()}
        masks = self._safe_mask(torch.cat((current["action_mask"][:, :1], following["action_mask"]), 1))
        online, state = self.encoder.sequence(observations, batch.get(Columns.STATE_IN), self.burn_in)
        q = combine_q_values(online, self.af, getattr(self, "vf", None), masks)
        with torch.no_grad():
            target, target_state = self._target_encoder.sequence(observations, batch.get(Columns.STATE_IN), self.burn_in)
            target_q = combine_q_values(target, self._target_af, getattr(self, "_target_vf", None), masks)
        return {"qf_preds": q[:, :-1], "qf_next_preds": q[:, 1:], "qf_target_next_preds": target_q[:, 1:],
                Columns.STATE_OUT: state, Columns.NEXT_STATE_OUT: target_state}


class RecurrentDQNFactory(MaskedDQNFactory):
    """Extend the existing RLlib DQN factory with explicit recurrent replay settings."""

    algorithm_id = "recurrent_dqn"
    trainable = RecurrentDQN
    checkpoint_loader_id = "rllib_recurrent_dqn"

    def decode_config(self, payload: object) -> RecurrentDQNConfig:
        from ..common.config_decode import decode_dataclass
        return decode_dataclass(RecurrentDQNConfig, payload, "algorithm_config")

    def build(self, context: TrainingContext) -> DQNConfig:
        spec = context.experiment
        if not isinstance(spec.algorithm_config, RecurrentDQNConfig):
            raise TypeError("recurrent_dqn requires RecurrentDQNConfig")
        if spec.environment.max_episode_steps <= spec.algorithm_config.burn_in_length:
            raise ValueError("episode limit must exceed recurrent burn-in")
        config = super().build(context)
        config.algo_class = RecurrentDQN
        values = dict(config.rl_module_spec.model_config)
        values.update(max_seq_len=spec.algorithm_config.sequence_length, burn_in_length=spec.algorithm_config.burn_in_length)
        return (config.training(burn_in_len=spec.algorithm_config.burn_in_length, store_buffer_in_checkpoints=True,
                    replay_buffer_config={"type": "aresim.algorithms.recurrent_dqn.train.RecurrentEpisodeReplayBuffer", "capacity": spec.algorithm_config.replay_capacity, "seed": spec.learner_seed})
                .learners(learner_class=RecurrentDQNLearner)
                .rl_module(model_config=values, rl_module_spec=RLModuleSpec(module_class=AresRecurrentDQNRLModule, model_config=values)))
