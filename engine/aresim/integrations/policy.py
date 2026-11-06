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
:mod:`aresim.algorithms.jev.agent` (Jev rover agent).
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

from ..algorithms.base import Agent
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
        from ..algorithms.dreamerv3.checkpoint import WorldModelAgent
    except ImportError:
        return False
    return True


def jev_available() -> bool:
    """Return whether the TypeSafe SDK and an API key are available for Jev."""
    from ..algorithms.jev.client import jev_available as live_jev_available

    return live_jev_available()


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
    if algorithm_id in LLM_ALGORITHMS:
        if not jev_available():
            raise PolicyError(
                "JEV_UNAVAILABLE",
                "Jev requires aresim[jev] and JEV_API_KEY.",
            )
        agent = make_agent(algorithm_id)
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


class PolicyBridge:
    """Build policy observations and decode discrete actions for one live engine."""

    def __init__(self, env_config: EnvironmentConfig, registry: ComponentRegistry | None = None) -> None:
        env_config.validate()
        self.env_config = env_config
        components = create_default_registry() if registry is None else registry
        self._components = components
        self._last_policy_step: int | None = None
        context = ComponentBuildContext(env_config, env_config.engine)
        self._observation = components.build_observation(env_config.observation, context)
        self._actions = components.build_action(env_config.action, context)

    @classmethod
    def for_engine(cls, engine_config: EngineConfig, registry: ComponentRegistry | None = None) -> PolicyBridge:
        """Construct a bridge using default RL component wiring for ``engine_config``."""
        env_config = replace(DEFAULT_ENVIRONMENT_CONFIG, engine=engine_config)
        return cls(env_config, registry=registry)

    def step(self, engine: AresEngine, agent: Agent[Any, Any], *, algorithm_id: str) -> PolicyStepResult:
        """Run one policy decision and apply it to ``engine`` as ``Actor.AGENT``."""
        state = engine.state
        if state.game_status != GameStatus.RUNNING:
            raise PolicyError("POLICY_NOT_RUNNING", "Resume a running session before requesting agent actions.")
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
        observation = self._observation.build(state, self.env_config.engine)
        action_mask = self._actions.mask(state, self.env_config.engine)
        action_index = int(agent.act(observation, action_mask))
        command = self._actions.decode(state, action_index)
        transition = engine.step(command, Actor.AGENT)
        self._last_policy_step = transition.state.step
        info = _agent_info(agent)
        return PolicyStepResult(
            transition=transition,
            command=command,
            action_index=action_index,
            policy_id=agent.policy_id,
            algorithm_id=algorithm_id,
            action_mask=[int(value) for value in action_mask.tolist()],
            info=info,
        )


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
    if result.info:
        meta["jev"] = {
            "choice": result.info.get("choice"),
            "confidence": result.info.get("confidence"),
            "probabilities": result.info.get("probabilities") or {},
            "action": result.info.get("action"),
            "fallback": result.info.get("fallback"),
            "waitRecharges": result.info.get("wait_recharges"),
        }
    return meta


__all__ = [
    "BASELINE_ALGORITHMS",
    "CHECKPOINT_ALGORITHMS",
    "LLM_ALGORITHMS",
    "POLICY_CATALOG",
    "jev_available",
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
