"""Masked recurrent Double/Dueling DQN on RLlib's stateful replay path.

The LSTM owns episode memory; shared RLlib infrastructure owns collection and trial execution.
Multi-step targets use contiguous online/target unrolls, not independently reset next-state calls.
"""
from __future__ import annotations
from collections.abc import Mapping
import numpy as np
import torch
from torch import Tensor, nn
from ray.rllib.algorithms.dqn import DQN, DQNConfig
from ray.rllib.algorithms.dqn.torch.dqn_torch_learner import DQNTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.env.utils.infinite_lookback_buffer import InfiniteLookbackBuffer
from ray.rllib.utils.replay_buffers.episode_replay_buffer import EpisodeReplayBuffer
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ..common.encoder import LocalObservationEncoder, combine_q_values
from ..dqn.train import AresMaskedDQNRLModule, MaskedDQNFactory, _q_heads, _heads, _require_local_spaces, _dqn_runtime_flags
from .config import RecurrentDQNConfig, RecurrentModelConfig
from ..registry import TrainingContext

REPLAY_BURN_IN = "recurrent_burn_in"


class RecurrentEpisodeReplayBuffer(EpisodeReplayBuffer):
    """Sample learning anchors uniformly, then prepend available recurrent history.

    Episode starts have zero burn-in; middle-of-episode windows carry their
    recorded state and up to burn_in_length preceding frames. Short episodes
    and terminal transitions remain eligible for learning.
    """

    def __init__(self, *, seed: int = 0, burn_in_length: int = 16, **kwargs) -> None:
        super().__init__(**kwargs)
        self.rng = np.random.default_rng(seed)
        self.burn_in_length = burn_in_length

    def get_state(self) -> dict:
        state = super().get_state()
        state["aresim_rng"] = self.rng.bit_generator.state
        return state

    def set_state(self, state: dict) -> None:
        super().set_state(state)
        self.rng.bit_generator.state = state["aresim_rng"]

    def sample(self, *args, **kwargs):
        if not kwargs.get("sample_episodes") or not kwargs.get("batch_length_T"):
            return super().sample(*args, **kwargs)
        kwargs["min_batch_length_T"] = 0
        kwargs["n_step"] = 1  # The learner computes returns from contiguous frames.
        samples = super().sample(*args, **kwargs)
        lookback = kwargs.get("lookback", 0)
        windows = []
        for sample in samples:
            index = self.episode_id_to_index[sample.id_] - self._num_episodes_evicted
            episode = self.episodes[index]
            anchor = sample.t_started - episode.t_started
            burn_in = min(self.burn_in_length, anchor)
            first = anchor - burn_in
            end = min(len(episode), first + kwargs["batch_length_T"])
            window = episode.slice(slice(first, end), len_lookback_buffer=lookback)
            for name, value in ((REPLAY_BURN_IN, burn_in), ("n_step", 1), ("weights", 1.0)):
                window.extra_model_outputs[name] = InfiniteLookbackBuffer(
                    np.full(len(window) + lookback, value, dtype=np.float32 if name == "weights" else np.int32), lookback=lookback)
            windows.append(window)
        return windows


class RecurrentDQNLearner(DQNTorchLearner):
    """Learn masked multi-step Double Q targets with per-window burn-in."""

    def configure_optimizers_for_module(self, module_id, config=None) -> None:
        """Keep RLlib's learning-rate scheduler while fusing CUDA Adam updates."""
        params = self.get_parameters(self._module[module_id])
        optimizer = torch.optim.Adam(params, fused=params[0].device.type == "cuda")
        self.register_optimizer(module_id=module_id, optimizer=optimizer, params=params, lr_or_lr_schedule=config.lr)

    def compute_loss_for_module(self, *, module_id, config, batch, fwd_out):
        valid = batch[Columns.LOSS_MASK].bool()
        burn_in = batch[REPLAY_BURN_IN][:, 0].long()
        learned = valid & (torch.arange(valid.shape[1], device=valid.device)[None] >= burn_in[:, None])
        selected = fwd_out["qf_preds"].gather(-1, batch[Columns.ACTIONS].long().unsqueeze(-1)).squeeze(-1)
        next_actions = fwd_out["qf_next_preds"].argmax(-1, keepdim=True)
        next_values = fwd_out["qf_target_next_preds"].gather(-1, next_actions).squeeze(-1)
        targets, horizons = recurrent_returns(
            batch[Columns.REWARDS], batch[Columns.TERMINATEDS], batch[Columns.TRUNCATEDS],
            valid, next_values, config.gamma, int(config.model_config.get("return_n", 1)))
        td_error = (selected - targets).abs()
        errors = nn.functional.huber_loss(selected, targets, reduction="none") * batch["weights"]
        loss = errors[learned].mean()
        self.metrics.log_value((module_id, "td_error"), td_error.detach(), reduce="item_series")
        self.metrics.log_dict({"qf_loss": loss, "qf_mean": selected[learned].mean(),
            "qf_min": selected[learned].min(), "qf_max": selected[learned].max(),
            "td_error_mean": td_error[learned].mean()}, key=module_id, window=1)
        self.metrics.log_value((module_id, "recurrent_learned_timesteps"), learned.sum().float())
        self.metrics.log_value((module_id, "recurrent_burn_in_timesteps"), (valid & ~learned).sum().float())
        self.metrics.log_value((module_id, "recurrent_padding_fraction"), (~valid).float().mean())
        self.metrics.log_value((module_id, "recurrent_effective_n_step"), horizons[learned].mean())
        self.metrics.log_value((module_id, "recurrent_zero_burn_sequences"), (burn_in == 0).sum().float())
        return loss


def recurrent_returns(
    rewards: Tensor, terminated: Tensor, truncated: Tensor, valid: Tensor,
    next_values: Tensor, gamma: float, n_step: int,
) -> tuple[Tensor, Tensor]:
    """Build shortened n-step returns at episode or sampled-window boundaries.

    True terminals suppress bootstrap; time limits retain the final real frame.
    Every bootstrap Q comes from the same contiguous recurrent unroll.
    """
    returns = torch.zeros_like(rewards)
    bootstrap = torch.zeros_like(rewards)
    discount = torch.ones_like(rewards)
    horizons = torch.zeros_like(rewards)
    active = valid.clone()
    width = rewards.shape[1]
    with torch.no_grad():
        for offset in range(min(n_step, width)):
            def shifted(value):
                return nn.functional.pad(value[:, offset:], (0, offset))
            active = active & shifted(valid)
            returns += torch.where(active, discount * shifted(rewards), 0.)
            discount = torch.where(active, discount * gamma, discount)
            bootstrap = torch.where(active, discount * shifted(next_values) * (~shifted(terminated).bool()), bootstrap)
            horizons += active.float()
            active = active & ~shifted(terminated).bool() & ~shifted(truncated).bool()
    return returns + bootstrap, horizons


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

    def sequence(self, observations: Mapping[str, Tensor], state: dict[str, Tensor] | None, burn_in: int | Tensor = 0):
        if isinstance(burn_in, Tensor):
            return self._grouped_sequence(observations, state, burn_in)
        terrain = observations["terrain_type"]
        single = terrain.ndim == 3
        observations = {key: value.unsqueeze(1) if single else value for key, value in observations.items()}
        batch, length = observations["terrain_type"].shape[:2]
        prefix = min(burn_in, length)
        if prefix and torch.is_grad_enabled():
            with torch.no_grad():
                initial = self._encode(observations, 0, prefix)
            embedded = torch.cat((initial, self._encode(observations, prefix, length)), 1) if prefix < length else initial
        else:
            embedded = self._encode(observations, 0, length)
        recurrent = self._state(state, batch, embedded.device)
        output, recurrent = self._unroll(embedded, recurrent, burn_in)
        state_out = {key: value.squeeze(0) for key, value in zip(("h", "c"), recurrent, strict=True)}
        return (output[:, 0] if single else output), state_out

    def _grouped_sequence(self, observations, state, burn_in: Tensor):
        # Most rows share the full prefix; boundary rows use their real history.
        groups = {}
        for row, length in enumerate(burn_in.detach().cpu().tolist()):
            groups.setdefault(int(length), []).append(row)
        outputs, states, order = [], [], []
        for length, rows in groups.items():
            indices = torch.tensor(rows, device=burn_in.device)
            subset = {key: value.index_select(0, indices) for key, value in observations.items()}
            initial = {key: value.index_select(0, indices) for key, value in state.items()} if state is not None else None
            output, final = self.sequence(subset, initial, length)
            outputs.append(output)
            states.append(final)
            order.extend(rows)
        restore = torch.argsort(torch.tensor(order, device=burn_in.device))
        return torch.cat(outputs).index_select(0, restore), {
            key: torch.cat([value[key] for value in states]).index_select(0, restore) for key in ("h", "c")}

    def _encode(self, observations, first: int, end: int) -> Tensor:
        batch = observations["terrain_type"].shape[0]
        flat = {key: value[:, first:end].reshape(batch * (end - first), *value.shape[2:]) for key, value in observations.items()}
        return self.encoder(flat).reshape(batch, end - first, -1)

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
        return (outputs[0] if len(outputs) == 1 else torch.cat(outputs, 1)), recurrent


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
        return {"qf_preds": combine_q_values(features, advantage, value, mask, validate_mask=self._validate_masks), Columns.STATE_OUT: state}

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
        burn_in = batch[REPLAY_BURN_IN][:, 0].long() if REPLAY_BURN_IN in batch else self.burn_in
        online, state = self.encoder.sequence(observations, batch.get(Columns.STATE_IN), burn_in)
        q = combine_q_values(online, self.af, getattr(self, "vf", None), masks, validate_mask=self._validate_masks)
        with torch.no_grad():
            target, target_state = self._target_encoder.sequence(observations, batch.get(Columns.STATE_IN), burn_in)
            target_q = combine_q_values(target, self._target_af, getattr(self, "_target_vf", None), masks, validate_mask=self._validate_masks)
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
        config = super().build(context)
        config.algo_class = RecurrentDQN
        values = dict(config.rl_module_spec.model_config)
        values.update(max_seq_len=spec.algorithm_config.sequence_length, burn_in_length=spec.algorithm_config.burn_in_length,
                      return_n=spec.algorithm_config.n_step)
        return (config.training(n_step=1, burn_in_len=0, store_buffer_in_checkpoints=True,
                    replay_buffer_config={"type": "aresim.algorithms.recurrent_dqn.train.RecurrentEpisodeReplayBuffer", "capacity": spec.algorithm_config.replay_capacity, "seed": spec.learner_seed,
                                          "burn_in_length": spec.algorithm_config.burn_in_length})
                .learners(learner_class=RecurrentDQNLearner)
                .rl_module(model_config=values, rl_module_spec=RLModuleSpec(module_class=AresRecurrentDQNRLModule, model_config=values)))
