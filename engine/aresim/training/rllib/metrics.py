"""Translate authoritative environment and RLlib diagnostics to stable W&B names.

These callbacks observe transitions; they never calculate simulator rewards or legality.
"""
from __future__ import annotations
import math
from collections.abc import Mapping
import numpy as np
from ray.rllib.callbacks.callbacks import RLlibCallback

def _nested_number(payload: Mapping[str, object], *paths: tuple[str, ...]) -> float | None:
    for path in paths:
        value: object = payload
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                break
            value = value[key]
        else:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None

def canonicalize_rllib_metrics(result: Mapping[str, object]) -> dict[str, float]:
    """Map unstable RLlib Tune result paths to stable AresSim W&B metric names.

    Also rewrites ``ares/*`` env-runner diagnostics to ``train/*`` for dashboards.
    Missing paths are omitted rather than invented.
    """
    paths = {
        "train/environment_steps": (("num_env_steps_sampled_lifetime",), ("env_runners", "num_env_steps_sampled_lifetime")),
        "train/episodes": (("env_runners", "num_episodes_lifetime"), ("episodes_total",)),
        "train/episode_length": (("env_runners", "episode_len_mean"), ("episode_len_mean",)),
        "train/shaped_return": (("env_runners", "episode_return_mean"), ("episode_reward_mean",)),
        "system/environment_steps_per_second": (("env_runners", "num_env_steps_sampled_per_second"),),
        "learner/total_loss": (("learners", "default_policy", "total_loss"), ("learners", "default_policy", "qf_loss")),
        "learner/policy_loss": (("learners", "default_policy", "policy_loss"),),
        "learner/value_loss": (("learners", "default_policy", "vf_loss"),),
        "learner/entropy": (("learners", "default_policy", "entropy"),),
        "learner/approx_kl": (("learners", "default_policy", "mean_kl_loss"),),
        "learner/clip_fraction": (("learners", "default_policy", "clip_fraction"),),
        "learner/learning_rate": (("learners", "default_policy", "default_optimizer_learning_rate"),),
        "learner/gradient_norm": (("learners", "default_policy", "gradients_default_optimizer_global_norm"),),
        "learner/explained_variance": (("learners", "default_policy", "vf_explained_var"),),
        "learner/td_error": (("learners", "default_policy", "td_error_mean"),),
        "learner/q_mean": (("learners", "default_policy", "qf_mean"),),
        "learner/q_max": (("learners", "default_policy", "qf_max"),),
        "learner/q_min": (("learners", "default_policy", "qf_min"),),
        "learner/recurrent_learned_timesteps": (("learners", "default_policy", "recurrent_learned_timesteps"),),
        "learner/recurrent_burn_in_timesteps": (("learners", "default_policy", "recurrent_burn_in_timesteps"),),
        "learner/recurrent_padding_fraction": (("learners", "default_policy", "recurrent_padding_fraction"),),
        "learner/recurrent_effective_n_step": (("learners", "default_policy", "recurrent_effective_n_step"),),
        "learner/recurrent_zero_burn_sequences": (("learners", "default_policy", "recurrent_zero_burn_sequences"),),
    }
    metrics = {
        name: value
        for name, candidates in paths.items()
        if (value := _nested_number(result, *candidates)) is not None
    }
    runner_metrics = result.get("env_runners")
    if isinstance(runner_metrics, Mapping):
        metrics.update(_environment_metrics(runner_metrics))
    return metrics

def _environment_metrics(values: Mapping[str, object]) -> dict[str, float]:
    return {
        f"train/{str(name)[5:]}": float(value)
        for name, value in values.items()
        if str(name).startswith("ares/")
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    }

def _log_value(metrics_logger, name: str, value: object, *, reduce: str = "sum") -> None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        metrics_logger.log_value(name, float(value), reduce=reduce)

def _log_transition(metrics_logger, episode, info: Mapping[str, object]) -> None:
    action = str(info.get("effective_action", "unknown")).lower()
    for name in ("wait", "move", "scan", "extract", "build", "service", "unload", "invalid"):
        metrics_logger.log_value(f"ares/action_count/{name}", float(action == name), reduce="sum")
    metrics_logger.log_value("ares/invalid_actions", float(action == "invalid"), reduce="sum")
    _log_value(metrics_logger, "ares/engine_reward", info.get("engine_reward"))
    breakdown = info.get("reward_breakdown")
    if not isinstance(breakdown, Mapping) or not isinstance(breakdown.get("terms"), Mapping):
        return
    for name, term in breakdown["terms"].items():
        if isinstance(term, Mapping):
            _log_value(metrics_logger, f"ares/reward_term/{name}", term.get("value", term.get("contribution", term.get("weighted"))))

def _log_mask_violation(metrics_logger, episode) -> None:
    action = int(episode.get_actions(-1))
    previous = episode.get_observations(-2)
    if not isinstance(previous, Mapping) or "action_mask" not in previous:
        return
    mask = np.asarray(previous["action_mask"])
    violation = action < 0 or action >= mask.size or not bool(mask[action])
    metrics_logger.log_value("ares/mask_violations", float(violation), reduce="sum")

def _log_telemetry(metrics_logger, episode) -> None:
    current = episode.get_observations(-1)
    if not isinstance(current, Mapping) or not isinstance(current.get("observation"), Mapping):
        return
    observation = current["observation"]
    rover = np.asarray(observation["self"], dtype=float)
    colony = np.asarray(observation["colony"], dtype=float)
    values = {
        "battery": rover[2], "health": rover[3], "payload_remaining": rover[7],
        "power_margin": colony[2], "colony_battery": colony[3], "water": colony[4],
        "oxygen": colony[5], "livability": colony[6],
    }
    for name, value in values.items():
        metrics_logger.log_value(f"ares/telemetry/{name}", float(value), reduce="mean")

class AresMetricsCallback(RLlibCallback):
    """Forward authoritative env info and canonical learner metrics to W&B.

    On each episode step, logs action counts, invalid actions, engine reward terms,
    mask violations, and selected telemetry means. On train results, injects
    :func:`canonicalize_rllib_metrics` before Ray's logger runs.
    """

    def on_episode_step(self, *, episode, metrics_logger=None, **kwargs) -> None:
        """Log per-step environment diagnostics when a metrics logger is available."""
        if metrics_logger is None:
            return
        info = episode.get_infos(-1)
        if not isinstance(info, Mapping):
            return
        _log_transition(metrics_logger, episode, info)
        try:
            _log_mask_violation(metrics_logger, episode)
            _log_telemetry(metrics_logger, episode)
        except (IndexError, KeyError, TypeError, ValueError):
            return

    def on_episode_end(self, *, episode, metrics_logger=None, **kwargs) -> None:
        if metrics_logger is None:
            return
        info = episode.get_infos(-1)
        if isinstance(info, Mapping):
            _log_value(metrics_logger, "ares/success", float(bool(info.get("success"))), reduce="mean")
            for name, value in info.get("objective_stats", {}).items():
                _log_value(metrics_logger, f"ares/{name}", value, reduce="mean")

    def on_train_result(self, *, algorithm, result: dict, **kwargs) -> None:
        """Add stable names before Ray's W&B callback sees the result."""
        result.update(canonicalize_rllib_metrics(result))
