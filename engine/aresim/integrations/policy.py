"""Live-session policy bridge: observation build, agent act, and action decode.

Connects registered baseline agents and checkpoint loaders to one session
``AresEngine`` without a second environment instance. Checkpoint agents are
cached in-process for efficient UI autoplay.

**Last updated:** September 20, 2026

**Contains:** ``PolicyBridge``, ``PolicyStepResult``, policy catalog helpers,
checkpoint path validation, and a small LRU checkpoint cache.

**See also:** :mod:`aresim.service` (attach/agent-step use cases),
:mod:`aresim.algorithms.ppo.checkpoint` (masked PPO loader),
:mod:`aresim.algorithms.dqn.checkpoint` (masked DQN loader),
:mod:`aresim.algorithms.llm_dm.agent` (DecisionModels rover agent).
"""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..algorithms.base import Agent, TransitionAwareAgent
from ..algorithms.catalog import BASELINE_ALGORITHMS, CHECKPOINT_ALGORITHMS, LLM_ALGORITHMS, POLICY_CATALOG
from ..config import EngineConfig, EnvironmentConfig
from ..core.engine import AresEngine
from ..defaults import DEFAULT_ENVIRONMENT_CONFIG
from ..factory import make_agent
from ..registry import ComponentBuildContext, ComponentRegistry, create_default_registry
from ..types import ActionCommand, Actor, EngineTransition, GameStatus

_CHECKPOINT_CACHE_MAX = 4
_checkpoint_cache: OrderedDict[str, Agent[Any, Any]] = OrderedDict()
_checkpoint_cache_lock = threading.Lock()


class PolicyError(Exception):
    """Policy resolution or inference failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def rllib_available() -> bool:
    """Return whether optional RLlib checkpoint loading is importable."""
    try:
        from ..training.rllib.checkpoint import RLlibCheckpointAgent  # noqa: F401
    except ImportError:
        return False
    return True


def muzero_available() -> bool:
    """Return whether the optional PyTorch MuZero checkpoint loader is importable."""
    try:
        from ..algorithms.muzero.checkpoint import BuiltinCheckpointLoader  # noqa: F401
    except ImportError:
        return False
    return True



def world_models_available() -> bool:
    """Return whether frozen world-model inference dependencies are installed (no Ray startup)."""
    try:
        from ..algorithms.dreamerv3.checkpoint import WorldModelAgent  # noqa: F401
    except ImportError:
        return False
    return True


def llm_dm_available() -> bool:
    """Report configured frozen-bundle readiness without calls or credential disclosure."""
    import os
    from importlib.util import find_spec
    path = os.environ.get("ARESIM_LLM_DM_CONFIG")
    if not path or find_spec("jsonschema") is None:
        return False
    try:
        from ..algorithms.llm_dm.config import load_llm_dm_run
        config = load_llm_dm_run(Path(path))
        return config.execution_mode == "frozen_evaluation" and (config.transport == "fake" or bool(os.environ.get(config.actor.credential_env)))
    except (ValueError, TypeError, OSError, KeyError):
        return False


def _is_safe_resolved_checkpoint(candidate: Path) -> bool:
    if ".." in candidate.parts:
        return False
    return candidate.is_file() and candidate.name == "checkpoint.json"


def resolve_checkpoint_path(path: str) -> Path:
    """Resolve and validate one absolute checkpoint sidecar path."""
    if not path.strip():
        raise PolicyError("CHECKPOINT_NOT_FOUND", "Checkpoint path is required for this learned policy.")
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise PolicyError("CHECKPOINT_NOT_FOUND", "Checkpoint path must be absolute.")
    candidate = raw.resolve()
    if not _is_safe_resolved_checkpoint(candidate):
        if candidate.name != "checkpoint.json":
            raise PolicyError("CHECKPOINT_NOT_FOUND", "Checkpoint path must point to checkpoint.json.")
        raise PolicyError("CHECKPOINT_NOT_FOUND", f"Checkpoint file does not exist: {candidate}")
    return candidate


def _cache_key(sidecar: Path, *, deterministic: bool = True) -> str:
    return f"{sidecar}:{sidecar.stat().st_mtime_ns}:{deterministic}"


def _cache_get(key: str) -> Agent[Any, Any] | None:
    with _checkpoint_cache_lock:
        agent = _checkpoint_cache.get(key)
        if agent is None:
            return None
        _checkpoint_cache.move_to_end(key)
        return agent


def _cache_put(key: str, agent: Agent[Any, Any]) -> None:
    with _checkpoint_cache_lock:
        _checkpoint_cache[key] = agent
        _checkpoint_cache.move_to_end(key)
        while len(_checkpoint_cache) > _CHECKPOINT_CACHE_MAX:
            _checkpoint_cache.popitem(last=False)


def load_checkpoint_agent(path: str, *, deterministic: bool = True) -> Agent[Any, Any]:
    """Load one checkpoint sidecar, using the process-level LRU cache."""
    sidecar = resolve_checkpoint_path(path)
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (ValueError, json.JSONDecodeError) as error:
        raise PolicyError("CHECKPOINT_INCOMPATIBLE", str(error)) from error
    from ..algorithms.registry import create_training_registry
    try:
        extra = create_training_registry().checkpoint_extra(str(metadata.get("loader_id", "rllib_masked_ppo")))
    except ValueError as error:
        raise PolicyError("CHECKPOINT_INCOMPATIBLE", str(error)) from error
    available = {"rllib": rllib_available, "muzero": muzero_available, "world_models": world_models_available}
    error_code = {"rllib": "RLLIB_UNAVAILABLE", "muzero": "MUZERO_UNAVAILABLE", "world_models": "WORLD_MODELS_UNAVAILABLE"}[extra]
    if not available[extra]():
        raise PolicyError(error_code, f"Checkpoint requires the aresim[{extra}] extra on the API host.")
    key = _cache_key(sidecar, deterministic=deterministic)
    cached = _cache_get(key)
    if cached is not None:
        return deepcopy(cached)
    try:
        from ..training import make_checkpoint_agent

        agent = make_checkpoint_agent(sidecar, deterministic=deterministic)
    except (ImportError, OSError) as error:
        raise PolicyError(error_code, "Checkpoint dependencies are unavailable on the API host.") from error
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        raise PolicyError("CHECKPOINT_INCOMPATIBLE", str(error)) from error
    _cache_put(key, agent)
    return deepcopy(agent)


def validate_checkpoint_path(path: str) -> dict[str, object]:
    """Inspect one checkpoint sidecar without attaching it to a session."""
    try:
        sidecar = resolve_checkpoint_path(path)
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        return {
            "valid": True,
            "policyId": payload.get("policy_id"),
            "loaderId": payload.get("loader_id"),
            "resolvedPath": str(sidecar),
        }
    except PolicyError as error:
        return {"valid": False, "error": error.message, "code": error.code}
    except (ValueError, json.JSONDecodeError) as error:
        return {"valid": False, "error": str(error), "code": "CHECKPOINT_INCOMPATIBLE"}


def resolve_action_mode(algorithm_id: str, action_mode: str | None = None) -> str:
    """Default learned live controllers to their seeded sampling method."""
    supports_sampling = algorithm_id in CHECKPOINT_ALGORITHMS
    mode = action_mode or ("stochastic" if supports_sampling else "deterministic")
    if mode not in {"stochastic", "deterministic"}:
        raise PolicyError("INVALID_ACTION_MODE", "Unknown policy action mode.")
    if mode == "stochastic" and not supports_sampling:
        raise PolicyError("UNSUPPORTED_ACTION_MODE", "Sampling is available for learned checkpoints.")
    return mode


def resolve_agent(algorithm_id: str, checkpoint_path: str | None = None, *, action_mode: str | None = None) -> tuple[Agent[Any, Any], str, str | None]:
    """Build or load one agent for ``algorithm_id`` and return ``(agent, policy_id, resolved_path)``."""
    if algorithm_id in BASELINE_ALGORITHMS:
        agent = make_agent(algorithm_id)
        return agent, agent.policy_id, None
    if algorithm_id == "llm_dm":
        try:
            agent = make_agent("llm_dm")
        except (ValueError, OSError, ImportError) as error:
            raise PolicyError("LLM_DM_UNAVAILABLE", str(error)) from error
        return agent, agent.policy_id, None
    if algorithm_id in CHECKPOINT_ALGORITHMS:
        if checkpoint_path is None or not checkpoint_path.strip():
            raise PolicyError("CHECKPOINT_NOT_FOUND", "Checkpoint path is required for this learned policy.")
        sidecar = resolve_checkpoint_path(checkpoint_path)
        try:
            checkpoint_algorithm = json.loads(sidecar.read_text(encoding="utf-8")).get("algorithm_id")
        except (ValueError, json.JSONDecodeError) as error:
            raise PolicyError("CHECKPOINT_INCOMPATIBLE", str(error)) from error
        if checkpoint_algorithm != algorithm_id:
            raise PolicyError("CHECKPOINT_INCOMPATIBLE", "Checkpoint algorithm does not match the selected policy.")
        mode = resolve_action_mode(algorithm_id, action_mode)
        agent = load_checkpoint_agent(checkpoint_path, deterministic=mode == "deterministic")
        return agent, agent.policy_id, str(sidecar)
    raise PolicyError("UNKNOWN_ALGORITHM", f"Unknown algorithm id: {algorithm_id}")


@dataclass(frozen=True)
class PolicyStepResult:
    """Outcome of one server-side policy step on the live engine."""

    transition: EngineTransition
    command: ActionCommand
    action_index: int
    policy_id: str
    algorithm_id: str
    action_mask: list[int]
    info: dict[str, object] | None = None
    episode_done: bool = False
    ending_reason: str | None = None


class PolicyBridge:
    """Build policy observations and decode discrete actions for one live engine."""

    def __init__(self, env_config: EnvironmentConfig, registry: ComponentRegistry | None = None) -> None:
        env_config.validate()
        self.env_config = env_config
        components = create_default_registry() if registry is None else registry
        self._components = components
        self._last_policy_step: int | None = None
        self._learning_environment = None
        self._learning_steps = 0
        self._bound_agent = None
        context = ComponentBuildContext(env_config, env_config.engine)
        self._observation = components.build_observation(env_config.observation, context)
        self._actions = components.build_action(env_config.action, context)

    @classmethod
    def for_engine(cls, engine_config: EngineConfig, registry: ComponentRegistry | None = None) -> PolicyBridge:
        """Construct a bridge using default RL component wiring for ``engine_config``."""
        env_config = replace(DEFAULT_ENVIRONMENT_CONFIG, engine=engine_config)
        return cls(env_config, registry=registry)

    def prepare(self, engine: AresEngine, agent: Agent[Any, Any]) -> tuple[object, np.ndarray]:
        """Build input under the session lock; provider inference happens outside it."""
        state = engine.state
        if state.game_status != GameStatus.RUNNING:
            raise PolicyError("POLICY_NOT_RUNNING", "Resume a running session before requesting agent actions.")
        if isinstance(agent, TransitionAwareAgent):
            if not getattr(agent, "episode_open", True):
                raise PolicyError("POLICY_EPISODE_ENDED", "The policy episode ended; reset before continuing.")
            if self._learning_environment is None or self._bound_agent is not agent:
                from ..factory import make_env
                config = getattr(agent, "environment_config", self.env_config)
                self.env_config = replace(config, engine=self.env_config.engine)
                self._learning_environment = make_env(self.env_config, max_episode_steps=getattr(agent, "max_episode_steps", 1200))
                reset = self._learning_environment.bind_engine(engine)
                self._bound_agent = agent
                self._learning_steps = 0
                agent.begin_episode(f"live-attachment-step-{state.step}")
                return reset.observation, reset.action_mask
            if self._learning_environment._done:
                raise PolicyError("POLICY_EPISODE_ENDED", "The policy episode ended; reset the game before continuing.")
            current = self._learning_environment.current_input()
            return current.observation, current.action_mask
        if self._last_policy_step is not None and self._last_policy_step != state.step:
            reset_memory = getattr(agent, "reset_memory", None)
            if callable(reset_memory):
                reset_memory()
        declared = agent.observation_schema
        if declared is not None and declared != self._observation.schema:
            name = "local_memory" if declared == "aresim.obs.local_memory.v2" else "local"
            config = getattr(agent, "environment_config", self.env_config)
            self.env_config = replace(config, engine=self.env_config.engine, observation=name)
            context = ComponentBuildContext(self.env_config, self.env_config.engine)
            self._observation = self._components.build_observation(name, context)
            if self._observation.schema != declared:
                raise PolicyError("CHECKPOINT_INCOMPATIBLE", "Unsupported observation schema.")
        return self._observation.build(state, self.env_config.engine), self._actions.mask(state, self.env_config.engine)

    def commit(self, engine: AresEngine, agent: Agent[Any, Any], action_index: int,
               action_mask: np.ndarray, *, algorithm_id: str) -> PolicyStepResult:
        """Dispatch only after the service rechecks its session revision and pending request."""
        if not self._actions.space.contains(action_index):
            raise PolicyError("INVALID_POLICY_ACTION", "Policy returned an action outside Discrete(10).")
        command = self._actions.decode(engine.state, action_index)
        done, reason = False, None
        if isinstance(agent, TransitionAwareAgent):
            result = self._learning_environment.step(action_index)
            transition = result.transition
            decision = getattr(agent, "last_info", None) or {}
            reason = result.info.get("terminal_reason") or result.info.get("truncation_reason")
            agent.observe_transition({"schema_version": "aresim.llm.feedback.v1", "episode_id": decision.get("episode_id"),
                                      "decision_id": decision.get("decision_id"), "step_index": self._learning_steps,
                                      "issued_action_id": action_index, "observation": result.observation,
                                      "action_mask": result.action_mask, "reward": float(result.reward),
                                      "effective_action": transition.effective_action.value,
                                      "terminated": result.terminated, "truncated": result.truncated, "ending_reason": reason})
            self._learning_steps += 1
            done = result.terminated or result.truncated
            if done:
                agent.finish_episode({"transitions": self._learning_steps, "terminated": result.terminated,
                                      "truncated": result.truncated, "ending_reason": reason})
        else:
            transition = engine.step(command, Actor.AGENT)
        self._last_policy_step = transition.state.step
        return PolicyStepResult(transition, command, action_index, agent.policy_id, algorithm_id,
                                [int(value) for value in action_mask.tolist()], _agent_info(agent), done, reason)

    def step(self, engine: AresEngine, agent: Agent[Any, Any], *, algorithm_id: str) -> PolicyStepResult:
        """Synchronous convenience path; service uses prepare/commit to allow cancellation."""
        observation, mask = self.prepare(engine, agent)
        return self.commit(engine, agent, int(agent.act(observation, mask)), mask, algorithm_id=algorithm_id)


def command_to_ui_payload(command: ActionCommand) -> dict[str, object]:
    """Project one engine command into the UI action JSON shape."""
    payload: dict[str, object] = {"type": command.type.value}
    if command.target is not None:
        payload["target"] = {"x": command.target.x, "y": command.target.y}
    return payload


def _agent_info(agent: object) -> dict[str, object] | None:
    info = getattr(agent, "last_info", None)
    if not isinstance(info, dict) or not info:
        return None
    return dict(info)


def policy_meta_from_result(result: PolicyStepResult) -> dict[str, object]:
    """Build camelCase policy metadata for one agent-step API response."""
    meta: dict[str, object] = {
        "algorithmId": result.algorithm_id,
        "policyId": result.policy_id,
        "actionIndex": result.action_index,
        "action": command_to_ui_payload(result.command),
        "actionMask": result.action_mask,
    }
    if result.algorithm_id == "llm_dm":
        meta["episodeDone"] = result.episode_done
        meta["endingReason"] = result.ending_reason
        meta["llmDm"] = dict(result.info or {})
    return meta


__all__ = [
    "BASELINE_ALGORITHMS",
    "CHECKPOINT_ALGORITHMS",
    "LLM_ALGORITHMS",
    "POLICY_CATALOG",
    "PolicyBridge",
    "PolicyError",
    "PolicyStepResult",
    "command_to_ui_payload",
    "load_checkpoint_agent",
    "policy_meta_from_result",
    "resolve_agent",
    "resolve_checkpoint_path",
    "rllib_available",
    "muzero_available",
    "world_models_available",
    "validate_checkpoint_path",
]
