"""Strict configuration for LLM roles, history, shared budgets and learning runs.

The loader resolves public environment dataclasses once, rejecting unknown fields
and unresolved placeholders. Paths are rooted at the checkout or config directory.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from ...config import (
    EnvironmentConfig,
    ObservationConfig,
    RewardProfileConfig,
    TaskConfig,
)
from ...defaults import DEFAULT_ENVIRONMENT_CONFIG
from ...training.runner import EpisodeSpec
from ...training.seeds import resolve_checkout_file
from ..common.config_decode import (
    decode_dataclass,
    finite_number,
    mapping,
    positive_integer,
)
from ..common.learnings import resolve_learnings_file

ROOT = Path(__file__).resolve().parents[4]


def positive(value: object, label: str) -> float:
    number = finite_number(value, label)
    if number <= 0:
        raise ValueError(f"{label} must be positive")
    return number


def validate_experiment_name(value: object) -> str:
    """Accept a readable folder name without allowing paths or traversal."""
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value
    ):
        raise ValueError(
            "experiment_name must be 1–80 letters, digits, hyphens or underscores, starting with a letter or digit"
        )
    return value


@dataclass(frozen=True)
class RoleConfig:
    """One explicit model endpoint/profile; credentials are environment variable names only."""

    provider: str
    model: str
    credential_env: str = ""
    output_mode: str = "native_schema"
    generation: dict[str, object] = field(default_factory=dict)
    max_input_tokens: int = 32768
    max_output_tokens: int = 1024
    attempt_timeout_seconds: float = 15
    decision_deadline_seconds: float = 30
    transient_retries: int = 1
    retry_backoff_seconds: float = 1
    route: str | None = None
    context_window: int = 131072
    prompt_path: str | None = None
    goal_prompt_path: str | None = None
    guidance: str = ""

    def validate(self) -> None:
        from .providers.base import provider_known

        if not provider_known(self.provider):
            raise ValueError("unknown LLM provider")
        if (
            not isinstance(self.model, str)
            or not self.model.strip()
            or "${" in self.model
        ):
            raise ValueError("model must be explicit and resolved")
        expected_mode = (
            "choice" if self.provider == "decision_models" else "native_schema"
        )
        if self.output_mode != expected_mode:
            raise ValueError(f"{self.provider} requires {expected_mode} output")
        for name in ("max_input_tokens", "max_output_tokens", "context_window"):
            positive_integer(getattr(self, name), name)
        if self.max_input_tokens + self.max_output_tokens > self.context_window:
            raise ValueError(
                "input plus output allowance exceeds configured model context window"
            )
        for name in ("attempt_timeout_seconds", "decision_deadline_seconds"):
            positive(getattr(self, name), name)
        if self.attempt_timeout_seconds > self.decision_deadline_seconds:
            raise ValueError("attempt timeout exceeds decision deadline")
        positive_integer(self.transient_retries, "transient_retries", allow_zero=True)
        if (
            self.transient_retries > 1
            or finite_number(self.retry_backoff_seconds, "retry_backoff_seconds") < 0
        ):
            raise ValueError("v1 permits at most one retry and nonnegative backoff")
        if not isinstance(self.credential_env, str) or (
            self.provider != "fake"
            and not re.fullmatch(r"[A-Z][A-Z0-9_]+", self.credential_env)
        ):
            raise ValueError("credential_env must name an environment variable")
        if not isinstance(self.generation, dict):
            raise ValueError("generation must be an object")
        supported = {
            "openai": {"temperature", "top_p", "reasoning"},
            "deepseek": {"reasoning"},
            "anthropic": {"temperature", "top_p", "output_config"},
            "gemini": {"temperature", "topP", "thinkingConfig"},
            "openrouter": {"temperature", "top_p", "seed", "reasoning"},
            "fake": set(),
            "decision_models": set(),
        }.get(self.provider)
        if supported is not None and set(self.generation) - supported:
            raise ValueError(
                "unsupported generation parameters; implicit dropping/overrides are forbidden"
            )
        if self.provider == "deepseek" and "reasoning" in self.generation:
            reasoning = mapping(self.generation["reasoning"], "reasoning")
            if set(reasoning) != {"effort"} or reasoning["effort"] not in {
                "none",
                "low",
                "high",
                "max",
            }:
                raise ValueError(
                    "DeepSeek reasoning must specify effort: none, low, high or max"
                )
        if self.provider == "anthropic" and "output_config" in self.generation:
            value = mapping(self.generation["output_config"], "output_config")
            if set(value) - {"effort"}:
                raise ValueError(
                    "output_config only allows effort; output format is runtime-owned"
                )
        if self.provider == "openrouter" and not self.route:
            raise ValueError("OpenRouter requires one explicit upstream route")
        if self.provider != "openrouter" and self.route is not None:
            raise ValueError("route is only supported by OpenRouter")
        if not isinstance(self.guidance, str) or len(self.guidance) > 4000:
            raise ValueError(
                "role guidance must be a string of at most 4000 characters"
            )


@dataclass(frozen=True)
class HistoryConfig:
    """Deterministic episode memory and explicit packet size limits."""

    mode: str = "focused"
    recent_limit: int = 8
    fact_limit: int = 16
    max_bytes: int = 8192
    goal_horizon: int = 8
    stall_steps: int = 6
    target_limit: int = 6
    advice_limit: int = 3

    def validate(self) -> None:
        if self.mode not in {"none", "recent", "facts_recent", "focused"}:
            raise ValueError("unsupported history mode")
        for name in (
            "recent_limit",
            "fact_limit",
            "max_bytes",
            "goal_horizon",
            "stall_steps",
            "target_limit",
            "advice_limit",
        ):
            positive_integer(getattr(self, name), name)


@dataclass(frozen=True)
class BudgetConfig:
    """Run ceilings include actor, Learner, failed attempts and candidate comparisons."""

    max_provider_attempts: int
    max_total_tokens: int
    max_wall_seconds: float
    max_usd: float
    price_sheet: str | None = None

    def validate(self) -> None:
        positive_integer(self.max_provider_attempts, "max_provider_attempts")
        positive_integer(self.max_total_tokens, "max_total_tokens")
        positive(self.max_wall_seconds, "max_wall_seconds")
        positive(self.max_usd, "max_usd")


@dataclass(frozen=True)
class LearnerConfig:
    """Required development role, fixed batch schedule and disjoint comparison episodes."""

    role: RoleConfig
    batch_episodes: int = 1
    max_rounds: int = 1
    max_proposals_per_batch: int = 1
    evaluation_episodes: tuple[EpisodeSpec, ...] = ()


@dataclass(frozen=True)
class RunConfig:
    """Resolved executable run; frozen modes require a previously saved policy bundle."""

    actor: RoleConfig
    learner: LearnerConfig | None
    budgets: BudgetConfig
    episodes: tuple[EpisodeSpec, ...]
    output_directory: Path
    environment: EnvironmentConfig = DEFAULT_ENVIRONMENT_CONFIG
    history: HistoryConfig = HistoryConfig()
    max_episode_steps: int = 1200
    execution_mode: str = "development"
    split: str = "development"
    transport: str = "live"
    include_rl_reward: bool = True
    policy_bundle: Path | None = None
    bootstrap_seed: int = 0
    debug_trace: bool = False
    learnings_file: Path | None = None
    experiment_name: str = "experiment"


def _episodes(raw: object, label: str) -> tuple[EpisodeSpec, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{label} must be a nonempty list")
    episodes = []
    for item in raw:
        values = mapping(item, label)
        if set(values) != {"id", "environment_seed", "agent_seed"}:
            raise ValueError(f"{label} requires only id, environment_seed, agent_seed")
        if not isinstance(values["id"], str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+", values["id"]
        ):
            raise ValueError("episode id must be a safe nonempty slug")
        episode = EpisodeSpec(
            values["id"], values["environment_seed"], values["agent_seed"]
        )
        episode.validate()
        episodes.append(episode)
    if len({e.episode_id for e in episodes}) != len(episodes):
        raise ValueError(f"duplicate {label} IDs")
    return tuple(episodes)


def _resolve(value: object) -> object:
    if isinstance(value, str):

        def substitute(match: re.Match) -> str:
            found = os.environ.get(match.group(1), "")
            if not found:
                raise ValueError(f"unresolved configuration variable: {match.group(1)}")
            return found

        return re.sub(r"\$\{([A-Z][A-Z0-9_]*)\}", substitute, value)
    if isinstance(value, dict):
        return {key: _resolve(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item) for item in value]
    return value


def role_config(raw: object, base: Path) -> RoleConfig:
    """Decode a role without loading credentials or selecting a mutable model default."""
    values = mapping(raw, "role")
    for name in ("prompt_path", "goal_prompt_path"):
        if values.get(name):
            values[name] = str((base / str(values[name])).resolve())
    role = decode_dataclass(RoleConfig, values, "role")
    role.validate()
    return role


def environment_config(raw: object) -> EnvironmentConfig:
    """Reuse public environment dataclasses; engine internals cannot be overridden in run YAML."""
    values = mapping(raw or {}, "environment")
    if values.pop("preset", "default") != "default":
        raise ValueError("unknown environment preset")
    allowed = {
        "observation",
        "task",
        "reward",
        "scenario_id",
        "task_config",
        "observation_config",
        "reward_config",
    }
    if set(values) - allowed:
        raise ValueError("unknown environment fields")
    for name, cls in (
        ("task_config", TaskConfig),
        ("observation_config", ObservationConfig),
        ("reward_config", RewardProfileConfig),
    ):
        if name in values:
            base = getattr(DEFAULT_ENVIRONMENT_CONFIG, name)
            merged = {f.name: getattr(base, f.name) for f in fields(base)} | mapping(
                values[name], name
            )
            values[name] = decode_dataclass(cls, merged, name)
    config = replace(DEFAULT_ENVIRONMENT_CONFIG, **values)
    config.validate()
    if (
        config.observation not in {"local", "local_memory"}
        or config.observation_config.window_size < 3
    ):
        raise ValueError(
            "LLM requires local/local_memory with a crop covering all adjacent moves"
        )
    if config.task not in {"resource_mission", "open_exploration"}:
        raise ValueError("unsupported LLM task")
    return config


def load_llm_dm_run(path: Path) -> RunConfig:
    """Read and completely validate YAML before creating artifacts or calling a provider."""
    import yaml

    source = resolve_checkout_file(path, kind="LLM run config")
    v = mapping(_resolve(yaml.safe_load(source.read_text())), "LLM run")
    allowed = {
        "schema_version",
        "agent",
        "actor",
        "learner",
        "budgets",
        "episodes",
        "output_directory",
        "experiment_name",
        "environment",
        "history",
        "feedback",
        "execution_mode",
        "transport",
        "split",
        "policy_bundle",
        "max_episode_steps",
        "bootstrap_seed",
        "debug_trace",
        "learnings_file",
    }
    if (
        set(v) - allowed
        or v.get("schema_version") != "aresim.llm_dm.run.v1"
        or v.get("agent") != "llm_dm"
    ):
        raise ValueError("unknown fields or unsupported LLM run schema/agent")
    base = source.parent
    learner = None
    if v.get("learner") is not None:
        lv = mapping(v["learner"], "learner")
        if lv.pop("enabled", True) is not True:
            raise ValueError(
                "Learner cannot be disabled; choose an explicit frozen execution mode"
            )
        batch = positive_integer(lv.pop("batch_episodes", 1), "batch_episodes")
        rounds = positive_integer(lv.pop("max_rounds", 1), "max_rounds")
        proposals = positive_integer(
            lv.pop("max_proposals_per_batch", 1), "max_proposals_per_batch"
        )
        comparison = lv.pop("evaluation_episodes", None)
        manifest = lv.pop("evaluation_manifest", None)
        if manifest:
            if comparison is not None:
                raise ValueError(
                    "select evaluation_manifest or evaluation_episodes, not both"
                )
            comparison = yaml.safe_load((base / str(manifest)).read_text())
        learner = LearnerConfig(
            role_config(lv, base),
            batch,
            rounds,
            proposals,
            _episodes(comparison, "evaluation episodes")
            if comparison is not None
            else (),
        )
        if learner.role.provider == "decision_models":
            raise ValueError(
                "Decision Models can act only as Actor; the Learner must be an LLM"
            )
        if proposals != 1:
            raise ValueError("v1 requires one proposal per batch")
    frozen = None
    if v.get("policy_bundle"):
        from .contracts import strict_json

        frozen = strict_json((base / str(v["policy_bundle"])).resolve().read_text())
        for key in ("actor", "history"):
            if key not in v:
                v[key] = frozen[key]
        if "feedback" not in v:
            v["feedback"] = {"include_rl_reward": frozen["include_rl_reward"]}
    budget_values = mapping(v.get("budgets"), "budgets")
    if budget_values.get("price_sheet"):
        budget_values["price_sheet"] = str(
            (base / str(budget_values["price_sheet"])).resolve()
        )
    budgets = decode_dataclass(BudgetConfig, budget_values, "budgets")
    budgets.validate()
    history = decode_dataclass(HistoryConfig, v.get("history", {}), "history")
    history.validate()
    feedback = mapping(v.get("feedback", {}), "feedback")
    if set(feedback) - {"include_rl_reward"} or not isinstance(
        feedback.get("include_rl_reward", True), bool
    ):
        raise ValueError("feedback only accepts boolean include_rl_reward")
    if frozen is not None and "environment" not in v:
        from ...training.provenance import environment_from_payload

        resolved_environment = environment_from_payload(frozen["environment"])
    else:
        resolved_environment = environment_config(v.get("environment"))
    config = RunConfig(
        actor=role_config(v["actor"], base),
        learner=learner,
        budgets=budgets,
        episodes=_episodes(v.get("episodes"), "episodes"),
        output_directory=(ROOT / str(v["output_directory"])).resolve(),
        experiment_name=validate_experiment_name(v.get("experiment_name", source.stem)),
        environment=resolved_environment,
        history=history,
        max_episode_steps=positive_integer(
            v.get("max_episode_steps", 1200), "max_episode_steps"
        ),
        execution_mode=str(v.get("execution_mode", "development")),
        split=str(v.get("split", "development")),
        transport=str(v.get("transport", "live")),
        include_rl_reward=feedback.get("include_rl_reward", True),
        policy_bundle=(base / str(v["policy_bundle"])).resolve()
        if v.get("policy_bundle")
        else None,
        bootstrap_seed=positive_integer(
            v.get("bootstrap_seed", 0), "bootstrap_seed", allow_zero=True
        ),
        debug_trace=v.get("debug_trace", False),
        learnings_file=resolve_learnings_file(v.get("learnings_file"), base),
    )
    if not isinstance(config.debug_trace, bool):
        raise ValueError("debug_trace must be a boolean")
    if config.execution_mode not in {
        "development",
        "smoke",
        "sequential_diagnostic",
        "frozen_evaluation",
        "actor_ablation",
    }:
        raise ValueError("unknown execution_mode")
    if config.transport not in {"live", "fake"}:
        raise ValueError("unknown transport")
    if config.transport == "live" and (
        config.actor.provider == "fake" or (learner and learner.role.provider == "fake")
    ):
        raise ValueError("fake provider requires explicit fake transport")
    if config.execution_mode == "sequential_diagnostic":
        if learner is None or learner.batch_episodes != 1 or len(config.episodes) < 2:
            raise ValueError(
                "sequential_diagnostic requires a Learner, batch_episodes=1 and at least two games"
            )
        if len(config.episodes) - 1 > learner.max_rounds:
            raise ValueError(
                "sequential games exceed the between-game learning-round capacity"
            )
        if learner.evaluation_episodes:
            raise ValueError(
                "sequential_diagnostic uses exactly the scheduled games, without evaluation episodes"
            )
    elif config.execution_mode in {"development", "smoke"}:
        if learner is None:
            raise ValueError("development and smoke require a configured Learner")
        if len(config.episodes) > learner.batch_episodes * learner.max_rounds:
            raise ValueError(
                "development episodes exceed required learning-round capacity"
            )
        comparisons = learner.evaluation_episodes
        if not comparisons:
            raise ValueError("development and smoke require comparison episodes")
        if len({e.environment_seed for e in comparisons}) != len(comparisons):
            raise ValueError("comparison environment seeds must be unique")
        if {e.environment_seed for e in config.episodes} & {
            e.environment_seed for e in comparisons
        }:
            raise ValueError("development and comparison seeds must be disjoint")
        if (
            config.execution_mode == "development"
            and len({e.environment_seed for e in comparisons}) < 30 * learner.max_rounds
        ):
            raise ValueError(
                "development requires 30 fresh paired comparison seeds per round"
            )
    elif config.policy_bundle is None:
        raise ValueError("frozen/ablation mode requires a saved policy_bundle")
    for episode in config.episodes + (learner.evaluation_episodes if learner else ()):
        if (
            not config.environment.engine.world.seed_min
            <= episode.environment_seed
            <= config.environment.engine.world.seed_max
        ):
            raise ValueError("episode seed outside supported bounds")
    return config


def load_role_credentials(config: RunConfig, env_file: Path) -> None:
    """Read only configured credential names from dotenv text, without executing shell code.

    Existing process variables take precedence. Neither values nor file contents
    become run configuration or artifacts.
    """
    roles = [config.actor] + ([config.learner.role] if config.learner else [])
    names = {role.credential_env for role in roles if role.credential_env}
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip().removeprefix("export ")
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or name not in names or os.environ.get(name, "").strip():
            continue
        parts = shlex.split(value.strip(), comments=True, posix=True)
        if len(parts) != 1 or not parts[0]:
            raise ValueError(
                f"invalid nonempty credential value for {name} in env file"
            )
        os.environ[name] = parts[0]
