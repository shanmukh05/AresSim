"""Dreamer sequence-model and isolated imagined-behavior optimizer steps.

Plan2Explore adds ensemble fitting and selects its separate exploration behavior during pretraining.
The shared model never receives behavior gradients or pretraining task rewards.
"""
from __future__ import annotations
import time
import torch
from torch import Tensor
import torch.nn.functional as F
from ...algorithms.dreamerv3.model import WorldModel, State
from ...algorithms.dreamerv3.behavior import ActorCritic
from ...algorithms.dreamerv3.distributions import latent_probs, policy_probs, scalar_prediction, twohot_loss, lambda_returns
from ...algorithms.plan2explore.config import Plan2ExploreConfig
from ...algorithms.plan2explore.ensemble import DisagreementEnsemble
from .replay import SequenceBatch


def onehot(actions: Tensor) -> Tensor:
    """The no-previous-action sentinel 10 maps to an all-zero action vector."""
    return F.one_hot(actions, 11)[..., :10].float()


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    return values[mask].mean() if mask.any() else values.sum() * 0


class DreamerLearner:
    """Single-device model and independent task/exploration actor-critic optimizers."""

    def __init__(self, spec, observation_space) -> None:
        self.spec = spec
        self.config = config = spec.algorithm_config
        self.device = torch.device(spec.resources.device)
        self.model = WorldModel(spec.model_config, observation_space).to(self.device)
        self.task = ActorCritic(self.model.feature_size, spec.model_config).to(self.device)
        self.exploration = None
        self.ensemble = None
        self.pretraining = isinstance(config, Plan2ExploreConfig) and config.phase == "pretrain"
        if isinstance(config, Plan2ExploreConfig):
            self.exploration = ActorCritic(self.model.feature_size, spec.model_config).to(self.device)
            self.ensemble = DisagreementEnsemble(self.model.feature_size, spec.model_config.fused_width,
                                                spec.model_config.hidden_size, config.ensemble_size).to(self.device)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=config.model_learning_rate, eps=1e-8)
        self.optimizers = {}
        for name, behavior in (("task", self.task), ("exploration", self.exploration)):
            if behavior is not None:
                self.optimizers[name + "_actor"] = torch.optim.Adam(behavior.actor.parameters(), lr=config.actor_learning_rate, eps=1e-8)
                self.optimizers[name + "_critic"] = torch.optim.Adam(behavior.critic.parameters(), lr=config.critic_learning_rate, eps=1e-8)
        if self.ensemble is not None:
            self.optimizers["ensemble"] = torch.optim.Adam(self.ensemble.parameters(), lr=config.model_learning_rate, eps=1e-8)

    @property
    def behavior(self) -> ActorCritic:
        return self.exploration if self.pretraining else self.task

    @torch.no_grad()
    def _prefix(self, batch: SequenceBatch) -> State:
        # ponytail: full-prefix reconstruction is O(episode length); cache validated states only after profiling.
        state = self.model.rssm.initial(len(batch.actions), self.device)
        if batch.prefix_observations is None:
            return state
        embeddings = self.model.encode_sequence(batch.prefix_observations)
        actions = onehot(batch.prefix_previous_actions)
        for t in range(embeddings.shape[1]):
            advanced, _, _ = self.model.rssm.observe(state, actions[:, t], embeddings[:, t])
            active = batch.prefix_lengths > t
            state = tuple(torch.where(active.reshape(-1, *([1] * (new.ndim - 1))), new, old) for new, old in zip(advanced, state, strict=True))
        return state

    def _observe(self, batch: SequenceBatch):
        started = time.monotonic()
        state = self._prefix(batch)
        prefix_seconds = time.monotonic() - started
        embeddings = self.model.encode_sequence(batch.observations)
        previous = onehot(batch.previous_actions)
        hidden, stochastic, priors, posteriors = [], [], [], []
        for t in range(embeddings.shape[1]):
            state, prior, posterior = self.model.rssm.observe(state, previous[:, t], embeddings[:, t])
            hidden.append(state[0]); stochastic.append(state[1])
            priors.append(prior); posteriors.append(posterior)
        states = (torch.stack(hidden, 1), torch.stack(stochastic, 1))
        features = self.model.rssm.features(states)
        return states, features, embeddings, torch.stack(priors, 1), torch.stack(posteriors, 1), prefix_seconds

    def _model_loss(self, batch: SequenceBatch, features: Tensor, prior: Tensor, posterior: Tensor):
        config = self.config
        valid = batch.observation_valid
        reconstruction = self.model.decoder.losses(features, batch.observations)
        reconstruction_loss = sum(masked_mean(value, valid) for value in reconstruction.values())
        p = latent_probs(prior, self.spec.model_config.unimix)
        q = latent_probs(posterior, self.spec.model_config.unimix)
        dynamics_kl = (q.detach() * (q.detach().log() - p.log())).sum((-1, -2))
        representation_kl = (q * (q.log() - p.detach().log())).sum((-1, -2))
        dynamics = masked_mean(dynamics_kl.clamp(min=config.free_nats), valid)
        representation = masked_mean(representation_kl.clamp(min=config.free_nats), valid)
        legality_losses = F.binary_cross_entropy_with_logits(self.model.legality(features), batch.masks.float(), reduction="none",
            pos_weight=torch.tensor(config.legality_positive_weights, device=self.device)).mean(-1)
        legality = masked_mean(legality_losses, valid)
        task_labels = batch.valid & batch.reward_valid
        reward = masked_mean(twohot_loss(self.model.reward(features[:, 1:]), batch.rewards), task_labels)
        continuation_valid = batch.valid if self.pretraining else task_labels
        continuation = masked_mean(F.binary_cross_entropy_with_logits(self.model.continuation(features[:, 1:]).squeeze(-1),
            (~batch.terminated).float(), reduction="none"), continuation_valid)
        loss = reconstruction_loss + config.dynamics_loss_weight * dynamics + config.representation_loss_weight * representation + legality + continuation
        if not self.pretraining:
            loss = loss + reward
        metrics = {"learner/reconstruction_loss": reconstruction_loss.detach(), "learner/dynamics_kl": dynamics.detach(),
                   "learner/representation_kl": representation.detach(), "learner/reward_loss": reward.detach(),
                   "learner/continuation_loss": continuation.detach(), "learner/legality_loss": legality.detach()}
        metrics.update({f"learner/reconstruction/{key}": masked_mean(value, valid).detach() for key, value in reconstruction.items()})
        with torch.no_grad():
            predicted = self.model.legality(features).sigmoid() >= config.legality_threshold
            for action in range(10):
                positive = batch.masks[..., action] & valid
                negative = ~batch.masks[..., action] & valid
                metrics[f"learner/legality_recall/{action}"] = (predicted[..., action] & positive).sum() / positive.sum().clamp(min=1)
                metrics[f"learner/legality_false_positive_rate/{action}"] = (predicted[..., action] & negative).sum() / negative.sum().clamp(min=1)
        return loss, metrics

    def _step(self, loss: Tensor, optimizer, parameters) -> Tensor:
        if not torch.isfinite(loss):
            raise FloatingPointError("world-model training produced a non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.max_gradient_norm, error_if_nonfinite=True)
        optimizer.step()
        return norm.detach()

    @torch.no_grad()
    def _imagine(self, roots: State, root_masks: Tensor):
        state = tuple(value.detach() for value in roots)
        features = [self.model.rssm.features(state)]
        actions, rewards, discounts, masks = [], [], [], []
        behavior = self.behavior
        for index in range(self.config.imagination_horizon):
            mask = root_masks if index == 0 else self.model.imagined_mask(features[-1], self.config.legality_threshold)
            probabilities = policy_probs(behavior.actor(features[-1]), mask, self.spec.model_config.unimix)
            action = torch.multinomial(probabilities, 1).squeeze(-1)
            encoded = onehot(action)
            intrinsic = self.ensemble.reward(features[-1], encoded) * self.config.intrinsic_scale if self.pretraining else None
            state = self.model.rssm.imagine(state, encoded)
            next_features = self.model.rssm.features(state)
            reward = intrinsic if self.pretraining else scalar_prediction(self.model.reward(next_features))
            actions.append(action); rewards.append(reward)
            discounts.append(self.config.gamma * self.model.continuation(next_features).sigmoid().squeeze(-1))
            masks.append(mask); features.append(next_features)
        return torch.stack(features, 1), torch.stack(actions, 1), torch.stack(rewards, 1), torch.stack(discounts, 1), torch.stack(masks, 1)

    def _behavior_update(self, states: State, features: Tensor, batch: SequenceBatch):
        behavior = self.behavior
        valid = batch.valid
        roots = tuple(value[:, :-1][valid] for value in states)
        imagined, actions, rewards, discounts, masks = self._imagine(roots, batch.masks[:, :-1][valid])
        with torch.no_grad():
            values = behavior.value(imagined)
            returns = lambda_returns(rewards, discounts, values, self.config.lambda_return)
            weights = torch.cumprod(torch.cat((torch.ones_like(discounts[:, :1]), discounts[:, :-1]), -1), -1)
            scale = behavior.return_scale(returns.flatten())
            advantages = (returns - values[:, :-1]) / scale
        probabilities = policy_probs(behavior.actor(imagined[:, :-1]), masks, self.spec.model_config.unimix)
        log_probs = probabilities.clamp(min=1e-8).log()
        chosen = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(probabilities * log_probs).sum(-1)
        actor_loss = -(weights * (chosen * advantages + self.config.entropy_coefficient * entropy)).mean()
        key = "exploration" if self.pretraining else "task"
        actor_norm = self._step(actor_loss, self.optimizers[key + "_actor"], behavior.actor.parameters())
        predicted = behavior.critic(imagined[:, :-1])
        with torch.no_grad():
            slow_values = scalar_prediction(behavior.slow_critic(imagined[:, :-1]))
        critic_loss = (weights * (twohot_loss(predicted, returns) + self.config.slow_regularization_weight * twohot_loss(predicted, slow_values))).mean()
        if not self.pretraining:
            replay_valid = valid & batch.reward_valid
            with torch.no_grad():
                bootstraps = behavior.value(features.detach())
                root_values = bootstraps[:, :-1].clone()
                root_values[valid] = returns[:, 0]
                bootstrap_values = torch.cat((root_values, bootstraps[:, -1:]), 1)
                targets = lambda_returns(batch.rewards, self.config.gamma * (~batch.terminated).float() * valid.float(), bootstrap_values, self.config.lambda_return, valid=valid)
            replay_losses = twohot_loss(behavior.critic(features[:, :-1].detach()), targets)
            critic_loss = critic_loss + self.config.replay_value_loss_weight * masked_mean(replay_losses, replay_valid)
        critic_norm = self._step(critic_loss, self.optimizers[key + "_critic"], behavior.critic.parameters())
        behavior.update_slow(self.config.slow_critic_rate)
        return {"learner/actor_loss": actor_loss.detach(), "learner/value_loss": critic_loss.detach(),
                "learner/entropy": entropy.mean().detach(), "learner/imagined_reward": rewards.mean(),
                "learner/actor_gradient_norm": actor_norm, "learner/critic_gradient_norm": critic_norm,
                "learner/return_scale": scale.detach(), "learner/imagined_mask_density": masks.float().mean()}

    def update(self, batch: SequenceBatch) -> dict[str, float]:
        """Fit one model batch then update the active behavior with isolated gradients."""
        states, features, embeddings, prior, posterior, prefix_seconds = self._observe(batch)
        loss, metrics = self._model_loss(batch, features, prior, posterior)
        metrics["learner/gradient_norm"] = self._step(loss, self.model_optimizer, self.model.parameters())
        metrics["learner/total_loss"] = loss.detach()
        detached_states = tuple(value.detach() for value in states)
        if self.pretraining:
            ensemble_loss = self.ensemble.loss(features[:, :-1][batch.valid].detach(), onehot(batch.actions)[batch.valid], embeddings[:, 1:][batch.valid].detach())
            self._step(ensemble_loss, self.optimizers["ensemble"], self.ensemble.parameters())
            metrics["learner/disagreement_loss"] = ensemble_loss.detach()
        metrics.update(self._behavior_update(detached_states, features.detach(), batch))
        metrics["system/prefix_reconstruction_seconds"] = torch.tensor(prefix_seconds)
        metrics["learner/learned_timesteps"] = batch.valid.sum()
        return {key: float(value.detach().cpu()) for key, value in metrics.items()}

    def model_state(self) -> dict:
        return {"model": self.model.state_dict(), "task": self.task.state_dict(),
                "exploration": self.exploration.state_dict() if self.exploration is not None else None,
                "ensemble": self.ensemble.state_dict() if self.ensemble is not None else None}

    def load_model_state(self, state: dict, *, transfer: bool = False) -> None:
        self.model.load_state_dict(state["model"])
        if not transfer:
            self.task.load_state_dict(state["task"])
        if self.exploration is not None:
            self.exploration.load_state_dict(state["exploration"])
            self.ensemble.load_state_dict(state["ensemble"])
        if transfer:
            # Downstream reward is unavailable during pretraining; start its predictor fresh.
            for module in self.model.reward.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            torch.nn.init.zeros_(self.model.reward[-1].weight)
            torch.nn.init.zeros_(self.model.reward[-1].bias)

    def optimizer_state(self) -> dict:
        return {"model": self.model_optimizer.state_dict(), **{name: optimizer.state_dict() for name, optimizer in self.optimizers.items()}}

    def load_optimizer_state(self, state: dict) -> None:
        self.model_optimizer.load_state_dict(state["model"])
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state[name])
