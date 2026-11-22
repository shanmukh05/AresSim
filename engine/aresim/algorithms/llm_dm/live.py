"""Live interactive LLM + Decision Model runtime for the browser UI.

Owns the supported provider metadata, safe dev-mode `.env.local` discovery,
in-memory per-session API key binding (never written to disk or global env),
and live on-demand Learner reflection.

See also: :mod:`aresim.algorithms.llm_dm.agent` (Actor lifecycle) and
:mod:`aresim.algorithms.llm_dm.learner` (between-episode proposals).
"""

from __future__ import annotations

import json
import os
import shlex
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

from ...config import EnvironmentConfig
from ...defaults import DEFAULT_ENVIRONMENT_CONFIG
from ..common.learnings import learning_statement, merge_learnings
from .agent import LLMDMAgent
from .artifacts import RunArtifacts
from .budget import BudgetLedger
from .calls import ModelCalls
from .config import CHOICE_PROVIDERS, ROOT, BudgetConfig, HistoryConfig, RoleConfig
from .contracts import Json
from .knowledge import candidate_bundle, make_bundle, run_learnings
from .learner import propose
from .prompts import render_prompt
from .providers import make_provider

PROVIDER_CREDENTIAL_ENV: dict[str, str] = {
    "decision_models": "JEV_API_KEY",
    "openrouter_decisions": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

DM_PROVIDERS: list[dict[str, object]] = [
    {
        "id": "decision_models",
        "label": "Jev",
        "credentialEnv": "JEV_API_KEY",
        "defaultModel": "jev-1.14.0",
        "placeholder": "jev-1.14.0",
    },
    {
        "id": "openrouter_decisions",
        "label": "OpenRouter",
        "credentialEnv": "OPENROUTER_API_KEY",
        "defaultModel": "liquid/d1.2-ultra",
        "placeholder": "liquid/d1.2-ultra or openai/gpt-6-luna-decisions",
    },
    {
        "id": "openai",
        "label": "OpenAI",
        "credentialEnv": "OPENAI_API_KEY",
        "defaultModel": "gpt-6-luna",
        "placeholder": "gpt-6-luna",
    },
]

LLM_PROVIDERS: list[dict[str, object]] = [
    {
        "id": "anthropic",
        "label": "Anthropic",
        "credentialEnv": "ANTHROPIC_API_KEY",
        "defaultModel": "claude-haiku-5-5",
        "placeholder": "claude-haiku-5-5",
    },
    {
        "id": "openai",
        "label": "OpenAI",
        "credentialEnv": "OPENAI_API_KEY",
        "defaultModel": "gpt-6-luna",
        "placeholder": "gpt-6-luna",
    },
    {
        "id": "gemini",
        "label": "Gemini",
        "credentialEnv": "GEMINI_API_KEY",
        "defaultModel": "gemini-4-neon",
        "placeholder": "gemini-4-neon",
    },
    {
        "id": "deepseek",
        "label": "DeepSeek",
        "credentialEnv": "DEEPSEEK_API_KEY",
        "defaultModel": "deepseek-flash",
        "placeholder": "deepseek-flash",
    },
    {
        "id": "openrouter",
        "label": "OpenRouter",
        "credentialEnv": "OPENROUTER_API_KEY",
        "defaultModel": "deepseek/deepseek-v4.1-flash-1017",
        "defaultRoute": "wafer/us",
        "placeholder": "deepseek/deepseek-v4.1-flash-1017",
    },
]


def _parse_dotenv_keys(env_file: Path) -> dict[str, str]:
    """Read known provider keys from dotenv text without shell evaluation."""
    allowed = set(PROVIDER_CREDENTIAL_ENV.values())
    found: dict[str, str] = {}
    if not env_file.is_file():
        return found
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return found
    for raw in lines:
        line = raw.strip().removeprefix("export ")
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or name not in allowed:
            continue
        try:
            parts = shlex.split(value.strip(), comments=True, posix=True)
        except ValueError:
            continue
        if len(parts) == 1 and parts[0]:
            found[name] = parts[0]
    return found


def resolve_dev_env_key(credential_env: str, *, deployed: bool) -> str | None:
    """Resolve one credential from env or `.env.local` in dev mode only."""
    if deployed or not credential_env:
        return None
    direct = os.environ.get(credential_env, "").strip()
    if direct:
        return direct
    for candidate in (ROOT / ".env.local", ROOT / ".env"):
        parsed = _parse_dotenv_keys(candidate)
        if parsed.get(credential_env):
            return parsed[credential_env]
    return None


def dev_env_credentials_available(*, deployed: bool) -> dict[str, bool]:
    """Report which credential env names are configured (booleans only, never values)."""
    names = sorted(set(PROVIDER_CREDENTIAL_ENV.values()))
    if deployed:
        return {name: False for name in names}
    return {name: bool(resolve_dev_env_key(name, deployed=False)) for name in names}


def _load_price_table() -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = {}
    for rel in ("configs/llm_dm/prices.json", "configs/llm_dm/ablation-prices.json"):
        path = ROOT / rel
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        models = payload.get("models")
        if isinstance(models, dict):
            for key, value in models.items():
                if isinstance(value, dict) and "input_per_million" in value and "output_per_million" in value:
                    merged[str(key)] = {
                        "input_per_million": float(value["input_per_million"]),
                        "output_per_million": float(value["output_per_million"]),
                    }
    return merged


class LiveBudgetLedger(BudgetLedger):
    """Per-session UI budget tracker that falls back gracefully for custom model IDs."""

    def __init__(self, config: BudgetConfig) -> None:
        super().__init__(config, fake=True)
        self.fake = False
        self.prices = _load_price_table()

    def price(self, config: RoleConfig) -> Json:
        key = f"{config.provider}/{config.model}" + (
            f"@{config.route}" if config.route else ""
        )
        raw = self.prices.get(key)
        if isinstance(raw, dict):
            return {
                "input_per_million": float(raw["input_per_million"]),
                "output_per_million": float(raw["output_per_million"]),
            }
        return {"input_per_million": 0.0, "output_per_million": 0.0}


def llm_dm_catalog(*, deployed: bool) -> dict[str, object]:
    """Return UI catalog for supported DM/LLM providers and dev-mode credential availability."""
    return {
        "dmProviders": DM_PROVIDERS,
        "llmProviders": LLM_PROVIDERS,
        "envKeysConfigured": dev_env_credentials_available(deployed=deployed),
    }


@dataclass
class LiveLLMDMConfig:
    """Validated request parameters for attaching a live LLM-DM agent from the UI."""

    actor_provider: str = "decision_models"
    actor_model: str = "jev-1.14.0"
    actor_route: str | None = None
    actor_api_key: str | None = None
    actor_context_window: int = 131072
    actor_max_output_tokens: int = 2048
    learner_enabled: bool = True
    learner_provider: str = "gemini"
    learner_model: str = "gemini-4-neon"
    learner_route: str | None = None
    learner_api_key: str | None = None
    learner_context_window: int = 262144
    learner_max_output_tokens: int = 16384
    history_mode: str = "focused"
    advice_limit: int = 3
    attempt_timeout_seconds: float = 60.0
    max_episode_steps: int = 256
    carry_learnings: list[dict[str, object]] = field(default_factory=list)


@dataclass
class LiveLLMDMController:
    """Binds a live LLMDMAgent with optional on-demand Learner reflection and UI feed state."""

    agent: LLMDMAgent
    learner_calls: ModelCalls | None
    learner_system: str | None
    artifacts: RunArtifacts
    temp_dir: tempfile.TemporaryDirectory[str]
    actor_provider_name: str
    actor_model_name: str
    learner_provider_name: str | None
    learner_model_name: str | None
    learning_round: int = 0
    learning_notes: dict[str, dict[str, object]] = field(default_factory=dict)

    def learnings_payload(self) -> list[dict[str, object]]:
        """Project current bundle learnings into UI-friendly cards (L1..Ln)."""
        items: list[dict[str, object]] = []
        for idx, row in enumerate(self.agent.learnings, 1):
            statement_id = str(row.get("id", f"l-{idx}"))
            meta = self.learning_notes.get(statement_id, {})
            items.append(
                {
                    "code": f"L{idx}",
                    "id": statement_id,
                    "target": row.get("target", "lesson"),
                    "text": row.get("text", ""),
                    "conditions": row.get("conditions", []),
                    "status": row.get("source_status", "tentative"),
                    "reason": meta.get("reason"),
                    "expectedBenefit": meta.get("expected_benefit"),
                    "failureCase": meta.get("failure_case"),
                    "author": meta.get("author"),
                    "round": meta.get("round"),
                }
            )
        return items

    def attach_meta(self) -> dict[str, object]:
        """Metadata returned immediately on attach so the UI initializes its live feed."""
        return {
            "actorProvider": self.actor_provider_name,
            "actorModel": self.actor_model_name,
            "learnerConfigured": self.learner_calls is not None,
            "learnerProvider": self.learner_provider_name,
            "learnerModel": self.learner_model_name,
            "learnings": self.learnings_payload(),
        }

    def step_meta(self, base_info: dict[str, object] | None) -> dict[str, object]:
        """Enrich per-step agent info with current learnings and learner metadata."""
        payload = dict(base_info or {})
        payload["learnerConfigured"] = self.learner_calls is not None
        payload["learnerProvider"] = self.learner_provider_name
        payload["learnerModel"] = self.learner_model_name
        payload["learnings"] = self.learnings_payload()
        payload["episodeTransitions"] = len(self.agent._evidence)
        return payload

    def reflect_and_learn(self) -> dict[str, object]:
        """Run one Learner review over the current/completed episode evidence and stage new advice."""
        if self.learner_calls is None or self.learner_system is None:
            raise ValueError("Configure a Learner model and API key to generate learnings.")
        episodes: list[Json] = []
        if self.agent.completed_evidence:
            episodes = [self.agent.completed_evidence[-1]]
        elif self.agent._evidence:
            episodes = [
                {
                    "episode_id": self.agent.episode_id,
                    "bundle_id": self.agent.bundle["bundle_id"],
                    "outcome": {
                        "transitions": len(self.agent._evidence),
                        "terminated": False,
                        "truncated": False,
                        "ending_reason": "in_progress_reflection",
                    },
                    "transitions": deepcopy(self.agent._evidence),
                }
            ]
        if not episodes or not episodes[0]["transitions"]:
            raise ValueError("Run at least one step with the LLM/DM agent before generating learnings.")

        item, proposal = propose(
            self.learner_calls,
            self.learner_system,
            self.agent.bundle,
            episodes,
            self.artifacts,
            self.learning_round,
        )
        self.learning_round += 1
        added_learning: dict[str, object] | None = None
        if item is not None:
            updated_bundle = candidate_bundle(self.agent.bundle, item)
            self.agent.bundle = updated_bundle
            self.agent.evaluating = True
            self.agent.learnings = run_learnings(updated_bundle, include_tentative=True)
            if self.agent.focus is not None:
                self.agent.focus.learnings = self.agent.learnings
            new_statement = self.agent.learnings[-1]
            statement_id = str(new_statement["id"])
            self.learning_notes[statement_id] = {
                "reason": proposal.get("reason"),
                "expected_benefit": proposal.get("expected_benefit"),
                "failure_case": proposal.get("failure_case"),
                "author": f"{self.learner_provider_name}/{self.learner_model_name}",
                "round": self.learning_round,
            }
            added_learning = self.learnings_payload()[-1]

        return {
            "outcome": proposal.get("outcome", "no_change"),
            "reason": proposal.get("reason"),
            "expectedBenefit": proposal.get("expected_benefit"),
            "failureCase": proposal.get("failure_case"),
            "round": self.learning_round,
            "learnerProvider": self.learner_provider_name,
            "learnerModel": self.learner_model_name,
            "addedLearning": added_learning,
            "learnings": self.learnings_payload(),
        }

    def close(self) -> None:
        try:
            self.agent.calls.provider.close()
        except Exception:
            pass
        if self.learner_calls is not None:
            try:
                self.learner_calls.provider.close()
            except Exception:
                pass
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass


def build_live_llm_dm_controller(
    cfg: LiveLLMDMConfig,
    *,
    environment: EnvironmentConfig = DEFAULT_ENVIRONMENT_CONFIG,
    deployed: bool = False,
) -> LiveLLMDMController:
    """Build an interactive live LLMDMAgent + optional Learner using UI or dev-env credentials."""
    actor_cred_env = PROVIDER_CREDENTIAL_ENV.get(cfg.actor_provider)
    if not actor_cred_env:
        raise ValueError(f"Unsupported actor provider: {cfg.actor_provider}")
    actor_model = cfg.actor_model.strip()
    if not actor_model:
        raise ValueError("Enter an Actor model ID.")

    actor_output_mode = (
        "choice" if cfg.actor_provider in CHOICE_PROVIDERS else "native_schema"
    )
    actor_ctx = max(8192, int(cfg.actor_context_window))
    actor_out = max(256, min(int(cfg.actor_max_output_tokens), actor_ctx // 2))
    actor_in = max(4096, actor_ctx - actor_out)
    actor_timeout = max(10.0, float(cfg.attempt_timeout_seconds))
    actor_route = (cfg.actor_route or "").strip() or ("wafer/us" if cfg.actor_provider == "openrouter" else None)

    actor_role = RoleConfig(
        provider=cfg.actor_provider,
        model=actor_model,
        credential_env=actor_cred_env,
        output_mode=actor_output_mode,
        generation={},
        max_input_tokens=actor_in,
        max_output_tokens=actor_out,
        attempt_timeout_seconds=actor_timeout,
        decision_deadline_seconds=max(actor_timeout * 2, 60.0),
        context_window=actor_ctx,
        route=actor_route if cfg.actor_provider == "openrouter" else None,
    )
    actor_role.validate()

    resolved_actor_key = (cfg.actor_api_key or "").strip() or resolve_dev_env_key(
        actor_cred_env, deployed=deployed
    )
    if not resolved_actor_key:
        if deployed:
            raise ValueError(
                f"Enter an API key for {cfg.actor_provider} ({actor_cred_env}) to run this Actor."
            )
        raise ValueError(
            f"Missing {actor_cred_env}. Add it to .env.local or enter an API key in LLM-DM settings."
        )

    history_mode = cfg.history_mode if cfg.history_mode in {"none", "recent", "facts_recent", "focused"} else "focused"
    history = HistoryConfig(
        mode=history_mode,
        goal_horizon=8,
        stall_steps=6,
        target_limit=6,
        advice_limit=max(1, min(10, int(cfg.advice_limit))),
    )
    history.validate()

    carried_statements: list[Json] = []
    notes: dict[str, dict[str, object]] = {}
    if cfg.carry_learnings:
        for entry in cfg.carry_learnings:
            if not isinstance(entry, dict) or not isinstance(entry.get("text"), str):
                continue
            stmt = learning_statement(
                {
                    "text": entry["text"],
                    "target": entry.get("target", "lesson"),
                    "conditions": entry.get("conditions", []),
                    "status": entry.get("status", "evaluation"),
                }
            )
            carried_statements.append(stmt)
            notes[str(stmt["id"])] = {
                "reason": entry.get("reason"),
                "expected_benefit": entry.get("expectedBenefit"),
                "failure_case": entry.get("failureCase"),
                "author": entry.get("author") or "Session Learner",
                "round": entry.get("round"),
            }
        carried_statements = merge_learnings(carried_statements)

    actor_prompt = render_prompt(
        "actor", actor_role, environment, focused=history.mode == "focused"
    )
    bundle = make_bundle(
        actor_role,
        history,
        environment,
        actor_prompt,
        include_rl_reward=True,
        learnings=carried_statements,
    )

    budget_cfg = BudgetConfig(
        max_provider_attempts=1200,
        max_total_tokens=12_000_000,
        max_wall_seconds=14400,
        max_usd=10.0,
    )
    budget = LiveBudgetLedger(budget_cfg)
    temp_dir = tempfile.TemporaryDirectory(prefix="aresim-live-llm-")
    artifacts = RunArtifacts(Path(temp_dir.name) / "run", "live-ui", debug_trace=False)

    actor_provider = make_provider(
        actor_role,
        api_key=resolved_actor_key,
        allow_env_fallback=False,
    )
    actor_provider.preflight()
    actor_calls = ModelCalls("actor", actor_role, actor_provider, budget, artifacts)

    learner_calls: ModelCalls | None = None
    learner_system: str | None = None
    learner_provider_name: str | None = None
    learner_model_name: str | None = None

    if cfg.learner_enabled and cfg.learner_provider and cfg.learner_model.strip():
        learner_cred_env = PROVIDER_CREDENTIAL_ENV.get(cfg.learner_provider)
        if not learner_cred_env or cfg.learner_provider in CHOICE_PROVIDERS:
            raise ValueError(
                "Learner must use a supported LLM provider (Anthropic, OpenAI, Gemini, DeepSeek, or OpenRouter)."
            )
        learner_ctx = max(16384, int(cfg.learner_context_window))
        learner_out = max(1024, min(int(cfg.learner_max_output_tokens), learner_ctx // 2))
        learner_in = max(8192, learner_ctx - learner_out)
        learner_timeout = max(60.0, actor_timeout * 2)
        learner_route = (cfg.learner_route or "").strip() or (
            "wafer/us" if cfg.learner_provider == "openrouter" else None
        )
        learner_role = RoleConfig(
            provider=cfg.learner_provider,
            model=cfg.learner_model.strip(),
            credential_env=learner_cred_env,
            output_mode="native_schema",
            generation={},
            max_input_tokens=learner_in,
            max_output_tokens=learner_out,
            attempt_timeout_seconds=learner_timeout,
            decision_deadline_seconds=learner_timeout * 2,
            context_window=learner_ctx,
            route=learner_route if cfg.learner_provider == "openrouter" else None,
            guidance=(
                "Live interactive session. Propose one concise, supported transferable lesson "
                "from the observed episode or return no_change if evidence is insufficient."
            ),
        )
        learner_role.validate()
        learner_key_candidate = (cfg.learner_api_key or "").strip()
        if not learner_key_candidate and learner_cred_env == actor_cred_env:
            learner_key_candidate = resolved_actor_key
        resolved_learner_key = learner_key_candidate or resolve_dev_env_key(
            learner_cred_env, deployed=deployed
        )
        if resolved_learner_key:
            learner_provider = make_provider(
                learner_role,
                api_key=resolved_learner_key,
                allow_env_fallback=False,
            )
            learner_provider.preflight()
            learner_calls = ModelCalls(
                "learner", learner_role, learner_provider, budget, artifacts
            )
            learner_system = render_prompt("learner", learner_role, environment)
            learner_provider_name = cfg.learner_provider
            learner_model_name = cfg.learner_model.strip()

    agent = LLMDMAgent(
        bundle,
        environment,
        actor_calls,
        artifacts,
        evaluating=bool(carried_statements),
        phase="live",
    )
    agent.max_episode_steps = max(16, min(2000, int(cfg.max_episode_steps)))

    return LiveLLMDMController(
        agent=agent,
        learner_calls=learner_calls,
        learner_system=learner_system,
        artifacts=artifacts,
        temp_dir=temp_dir,
        actor_provider_name=cfg.actor_provider,
        actor_model_name=actor_model,
        learner_provider_name=learner_provider_name,
        learner_model_name=learner_model_name,
        learning_notes=notes,
    )
