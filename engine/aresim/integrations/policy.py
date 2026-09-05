"""Live-session policy bridge: observation build, agent act, and action decode.

Connects registered baseline agents and checkpoint loaders to one session
``AresEngine`` without a second environment instance. Checkpoint agents are
cached in-process for efficient UI autoplay.

**Last updated:** September 2, 2026

**Contains:** ``PolicyBridge``, ``PolicyStepResult``, policy catalog helpers,
checkpoint path validation, and a small LRU checkpoint cache.

**See also:** :mod:`aresim.service` (attach/agent-step use cases),
:mod:`aresim.algorithms.ppo.checkpoint` (masked PPO loader).
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..algorithms.base import Agent
from ..config import EngineConfig, EnvironmentConfig
from ..core.engine import AresEngine
from ..defaults import DEFAULT_ENVIRONMENT_CONFIG
from ..factory import make_agent
from ..registry import ComponentBuildContext, ComponentRegistry, create_default_registry
from ..types import ActionCommand, Actor, EngineTransition

BASELINE_ALGORITHMS = frozenset({"random", "random_valid", "wait", "scripted"})
CHECKPOINT_ALGORITHM = "masked_ppo"

POLICY_CATALOG: list[dict[str, object]] = [
    {"id": "random", "label": "Random", "kind": "baseline", "requiresPath": False},
    {"id": "random_valid", "label": "Random (valid)", "kind": "baseline", "requiresPath": False},
    {"id": "wait", "label": "Wait", "kind": "baseline", "requiresPath": False},
    {"id": "scripted", "label": "Scripted", "kind": "baseline", "requiresPath": False},
    {"id": CHECKPOINT_ALGORITHM, "label": "Masked PPO", "kind": "checkpoint", "requiresPath": True},
]

# ponytail: max 4 cached RLModules; single-user local API; bump for multi-session prod
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
        from ..algorithms.ppo.checkpoint import make_checkpoint_agent  # noqa: F401
    except ImportError:
        return False
    return True


def _is_safe_resolved_checkpoint(candidate: Path) -> bool:
    if ".." in candidate.parts:
        return False
    return candidate.is_file() and candidate.name == "checkpoint.json"


def resolve_checkpoint_path(path: str) -> Path:
    """Resolve and validate one absolute checkpoint sidecar path."""
    if not path.strip():
        raise PolicyError("CHECKPOINT_NOT_FOUND", "Checkpoint path is required for masked PPO.")
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise PolicyError("CHECKPOINT_NOT_FOUND", "Checkpoint path must be absolute.")
    candidate = raw.resolve()
    if not _is_safe_resolved_checkpoint(candidate):
        if candidate.name != "checkpoint.json":
            raise PolicyError("CHECKPOINT_NOT_FOUND", "Checkpoint path must point to checkpoint.json.")
        raise PolicyError("CHECKPOINT_NOT_FOUND", f"Checkpoint file does not exist: {candidate}")
    return candidate


def _cache_key(sidecar: Path) -> str:
    return f"{sidecar}:{sidecar.stat().st_mtime_ns}"


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


def load_checkpoint_agent(path: str) -> Agent[Any, Any]:
    """Load one checkpoint sidecar, using the process-level LRU cache."""
    if not rllib_available():
        raise PolicyError("RLLIB_UNAVAILABLE", "Masked PPO requires the aresim[rllib] extra on the API host.")
    sidecar = resolve_checkpoint_path(path)
    key = _cache_key(sidecar)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        from ..algorithms.ppo.checkpoint import make_checkpoint_agent

        agent = make_checkpoint_agent(sidecar)
    except (ImportError, OSError) as error:
        raise PolicyError("RLLIB_UNAVAILABLE", "Masked PPO requires the aresim[rllib] extra on the API host.") from error
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        raise PolicyError("CHECKPOINT_INCOMPATIBLE", str(error)) from error
    _cache_put(key, agent)
    return agent


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


def resolve_agent(algorithm_id: str, checkpoint_path: str | None = None) -> tuple[Agent[Any, Any], str, str | None]:
    """Build or load one agent for ``algorithm_id`` and return ``(agent, policy_id, resolved_path)``."""
    if algorithm_id in BASELINE_ALGORITHMS:
        agent = make_agent(algorithm_id)
        return agent, agent.policy_id, None
    if algorithm_id == CHECKPOINT_ALGORITHM:
        if checkpoint_path is None or not checkpoint_path.strip():
            raise PolicyError("CHECKPOINT_NOT_FOUND", "Checkpoint path is required for masked PPO.")
        agent = load_checkpoint_agent(checkpoint_path)
        return agent, agent.policy_id, str(resolve_checkpoint_path(checkpoint_path))
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


class PolicyBridge:
    """Build policy observations and decode discrete actions for one live engine."""

    def __init__(self, env_config: EnvironmentConfig, registry: ComponentRegistry | None = None) -> None:
        env_config.validate()
        self.env_config = env_config
        components = create_default_registry() if registry is None else registry
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
        observation = self._observation.build(state, self.env_config.engine)
        action_mask = self._actions.mask(state, self.env_config.engine)
        action_index = int(agent.act(observation, action_mask))
        command = self._actions.decode(state, action_index)
        transition = engine.step(command, Actor.AGENT)
        return PolicyStepResult(
            transition=transition,
            command=command,
            action_index=action_index,
            policy_id=agent.policy_id,
            algorithm_id=algorithm_id,
            action_mask=[int(value) for value in action_mask.tolist()],
        )


def command_to_ui_payload(command: ActionCommand) -> dict[str, object]:
    """Project one engine command into the UI action JSON shape."""
    payload: dict[str, object] = {"type": command.type.value}
    if command.target is not None:
        payload["target"] = {"x": command.target.x, "y": command.target.y}
    return payload


def policy_meta_from_result(result: PolicyStepResult) -> dict[str, object]:
    """Build camelCase policy metadata for one agent-step API response."""
    return {
        "algorithmId": result.algorithm_id,
        "policyId": result.policy_id,
        "actionIndex": result.action_index,
        "action": command_to_ui_payload(result.command),
        "actionMask": result.action_mask,
    }


__all__ = [
    "BASELINE_ALGORITHMS",
    "CHECKPOINT_ALGORITHM",
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
    "validate_checkpoint_path",
]
