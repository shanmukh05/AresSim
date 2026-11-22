"""Coordinate budgeted Actor/Memory/Learner runs through the shared rollout runner.

Owns preflight, immutable bundle selection, required learning batches and fresh
paired evaluation. Existing trajectory writers own replayable game artifacts.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from itertools import count

from ...algorithms.base import EpisodeStop
from ...config import EnvironmentConfig
from ...training.provenance import source_hashes
from ...training.runner import EpisodeSpec, RolloutConfig, RolloutRunner
from ..common.learnings import load_learnings_file
from .agent import LLMDMAgent
from .artifacts import ReplayWriter, RunArtifacts
from .budget import BudgetLedger
from .calls import ModelCalls
from .config import (
    CHOICE_PROVIDERS,
    ROOT,
    RoleConfig,
    RunConfig,
    HistoryConfig,
    load_llm_dm_run,
    validate_experiment_name,
)
from .contracts import Json, json_value
from .evaluation import compare
from .knowledge import (
    candidate_bundle,
    load_bundle,
    make_bundle,
    publish_bundle,
    run_learnings,
    with_learnings,
)
from .learner import propose
from .prompts import render_prompt
from .providers import make_provider

logger = logging.getLogger(__name__)


def _run_artifacts(config: RunConfig) -> RunArtifacts:
    """Atomically allocate a named run, adding a readable suffix on repeat runs."""
    name = validate_experiment_name(config.experiment_name)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    for index in count(1):
        run_id = name if index == 1 else f"{name}-{index:03d}"
        try:
            return RunArtifacts(
                config.output_directory / run_id, run_id, debug_trace=config.debug_trace
            )
        except FileExistsError:
            continue


def _acting_config(
    config: RunConfig, imported: Json | None = None
) -> tuple[Json, RoleConfig]:
    imported = (
        load_learnings_file(config.learnings_file) if imported is None else imported
    )
    if config.policy_bundle:
        bundle = load_bundle(config.policy_bundle, config.environment)
        actor = RoleConfig(**bundle["actor"])
        actor.validate()
        if (
            json_value(config.actor) != json_value(actor)
            or json_value(config.history)
            != json_value(HistoryConfig(**bundle["history"]))
            or config.include_rl_reward != bundle["include_rl_reward"]
        ):
            raise ValueError("inline acting settings conflict with the loaded bundle")
        return with_learnings(bundle, imported["statements"]), actor
    prompt = render_prompt(
        "actor",
        config.actor,
        config.environment,
        focused=config.history.mode == "focused",
    )
    return (
        make_bundle(
            config.actor,
            config.history,
            config.environment,
            prompt,
            config.include_rl_reward,
            learnings=imported["statements"],
        ),
        config.actor,
    )


def _provenance() -> Json:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    )
    from hashlib import sha256

    return {
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_dirty": bool(dirty.stdout.strip()),
        "environment_sources": source_hashes(),
        "llm_sources": {
            str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.rglob("*"))
            if p.suffix in {".py", ".txt"}
        },
    }


def _collect(
    config: RunConfig,
    bundle: Json,
    calls: ModelCalls,
    artifacts: RunArtifacts,
    episodes: tuple[EpisodeSpec, ...],
    label: str,
    *,
    evaluating: bool = False,
) -> LLMDMAgent:
    agent = LLMDMAgent(
        bundle, config.environment, calls, artifacts, evaluating=evaluating, phase=label
    )
    writer = ReplayWriter(artifacts, label)
    runner = RolloutRunner(
        RolloutConfig(episodes, config.max_episode_steps),
        agent,
        environment_config=config.environment,
    )
    if len(episodes) == 1:
        logger.info("Starting %s: %s", label, episodes[0].episode_id)
    else:
        logger.info("Starting %s: %d episodes", label, len(episodes))
    artifacts.trace(
        "collection",
        {
            "event": "collection_started",
            "phase": label,
            "bundle_id": bundle["bundle_id"],
            "planned_episodes": json_value(episodes),
        },
    )
    result = runner.run(writer)
    artifacts.trace(
        "collection",
        {
            "event": "collection_completed",
            "phase": label,
            "transition_count": result.transition_count,
            "dataset_manifest": result.artifact_manifest,
            "summaries": json_value(result.summaries),
            "interruption_reason": runner.interruption_reason,
        },
    )
    if runner.interruption_reason:
        raise EpisodeStop(runner.interruption_reason)
    return agent


def _sequential_trial(
    config: RunConfig,
    bundle: Json,
    actor: ModelCalls,
    learner: ModelCalls,
    learner_system: str,
    artifacts: RunArtifacts,
    cycles: list[Json],
    checkpoint: Json | None = None,
) -> list[Json]:
    trial = checkpoint["trial"] if checkpoint is not None else bundle
    start = checkpoint["next_game"] if checkpoint is not None else 0

    def save(**updates):
        if checkpoint is not None:
            checkpoint.update(updates, trial=trial, cycles=cycles)
            artifacts.write("resume.json", checkpoint)

    for index in range(start, len(config.episodes)):
        episode = config.episodes[index]
        pending = checkpoint.get("pending_evidence") if checkpoint is not None else None
        if pending is not None:
            evidence = pending
        else:
            label = "development-0000" if index == 0 else f"candidate-{index - 1:04d}"
            game = _collect(
                config, trial, actor, artifacts, (episode,), label, evaluating=index > 0
            )
            metrics = game.metrics[0]
            evidence = game.completed_evidence
            logger.info(
                "Game %d learning context: %d retained statements; advice supplied on %d action decisions",
                index + 1,
                metrics["learning_statement_count"],
                metrics["learning_context_decisions"],
            )
            logger.info(
                "Game %d/%d: %s after %d turns; tentative advice conditions matched on %d decisions",
                index + 1,
                len(config.episodes),
                metrics["ending_reason"],
                metrics["steps"],
                metrics["candidate_advice_exposed_decisions"],
            )
            if cycles:
                cycles[-1]["candidate_advice_exposed_decisions"] = metrics[
                    "candidate_advice_exposed_decisions"
                ]
                cycles[-1]["learning_statement_count"] = metrics[
                    "learning_statement_count"
                ]
                cycles[-1]["learning_context_decisions"] = metrics[
                    "learning_context_decisions"
                ]
                artifacts.learning_record(
                    {
                        "event": "sequential_trial_completed",
                        **cycles[-1],
                        "note": "Diagnostic only; selected initial policy retained.",
                    }
                )
            save(game_ids=list(artifacts.games), pending_evidence=evidence)
        if index == len(config.episodes) - 1:
            save(next_game=index + 1, pending_evidence=None)
            break
        item, proposal = propose(
            learner, learner_system, trial, evidence, artifacts, index
        )
        parent_id = trial["bundle_id"]
        trial = candidate_bundle(trial, item) if item else trial
        artifacts.remember_bundle(trial)
        cycles.append(
            {
                "batch": index,
                "parent_bundle_id": parent_id,
                "candidate_bundle_id": trial["bundle_id"],
                "proposal_outcome": proposal["outcome"],
                "promoted": False,
                "selected_bundle_id": bundle["bundle_id"],
            }
        )
        artifacts.learning_record(
            {
                "event": "candidate_trial_started",
                "batch": index,
                "candidate_bundle_id": trial["bundle_id"],
                "proposal_outcome": proposal["outcome"],
                "carried_learning_ids": [
                    entry["id"]
                    for entry in run_learnings(trial, include_tentative=True)
                ],
                "note": "All earlier statements remain in subsequent game context. Promotion is separate from within-run reuse.",
            }
        )
        save(next_game=index + 1, pending_evidence=None)
        logger.info("Learner after game %d: %s", index + 1, proposal["outcome"])
    return cycles


def run_llm_dm(
    config: RunConfig, *, sequential_diagnostic: bool = False, resume: bool = False
) -> Path:
    """Run with optional game/review checkpoints and one writer per experiment."""
    if not resume:
        return _run_llm_dm(config, sequential_diagnostic=sequential_diagnostic)
    if config.execution_mode != "sequential_diagnostic":
        raise ValueError("--resume requires sequential_diagnostic")
    import fcntl

    name = validate_experiment_name(config.experiment_name)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    lock_path = config.output_directory / f".{name}.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"{name} is already running") from error
        return _run_llm_dm(config, resume_directory=config.output_directory / name)


def _resume_settings(config: Json, *, allow_thinking_change: bool) -> Json:
    """Compare policy settings separately from operational limits."""
    settings = {key: value for key, value in config.items() if key != "budgets"}
    learner = settings["learner"]
    if learner is not None:
        role = {
            key: value
            for key, value in learner["role"].items()
            if key not in {
                "attempt_timeout_seconds",
                "decision_deadline_seconds",
                "max_output_tokens",
            }
        }
        if allow_thinking_change and role["provider"] == "gemini":
            generation = role["generation"]
            thinking = generation.get("thinkingConfig")
            if isinstance(thinking, dict):
                role["generation"] = {
                    **generation,
                    "thinkingConfig": {
                        key: value
                        for key, value in thinking.items()
                        if key != "thinkingLevel"
                    },
                }
        settings["learner"] = {**learner, "role": role}
    return settings


def _run_llm_dm(
    config: RunConfig,
    *,
    sequential_diagnostic: bool = False,
    resume_directory: Path | None = None,
) -> Path:
    if config.learner and config.learner.role.provider in CHOICE_PROVIDERS:
        raise ValueError("Decision Models cannot act as Learner")
    sequential_diagnostic = (
        sequential_diagnostic or config.execution_mode == "sequential_diagnostic"
    )
    if sequential_diagnostic and (
        config.execution_mode not in {"smoke", "sequential_diagnostic"}
        or len(config.episodes) < 2
        or config.learner is None
        or config.learner.batch_episodes != 1
        or config.learner.max_rounds < len(config.episodes) - 1
    ):
        raise ValueError(
            "sequential diagnostic requires at least two games and a Learner round between games"
        )
    imported = load_learnings_file(config.learnings_file)
    bundle, actor = _acting_config(config, imported)
    budget = BudgetLedger(config.budgets, fake=config.transport == "fake")
    saved = None
    checkpoint = None
    pricing = (
        json.loads(Path(config.budgets.price_sheet).read_text())
        if config.transport == "live"
        else None
    )
    if resume_directory is not None:
        manifest_path = resume_directory / "manifest.json"
        resume_path = resume_directory / "resume.json"
        if manifest_path.exists():
            if not resume_path.exists():
                raise ValueError(
                    "Existing run has no resume checkpoint. Choose another experiment_name."
                )
            saved = json.loads(manifest_path.read_text())
            checkpoint = json.loads(resume_path.read_text())
            if checkpoint.get("schema") != 1:
                raise ValueError("Unsupported resume checkpoint")
            requested = json_value(config)
            prior = saved["config"]
            allow_thinking_change = (
                saved["status"] == "incomplete"
                and checkpoint["next_game"] == 0
                and checkpoint["pending_evidence"] is not None
                and not checkpoint["cycles"]
            )
            if _resume_settings(
                prior, allow_thinking_change=allow_thinking_change
            ) != _resume_settings(requested, allow_thinking_change=allow_thinking_change):
                raise ValueError(
                    "Run settings changed. Choose another experiment_name."
                )
            previous_output_limit = prior["learner"]["role"]["max_output_tokens"]
            if requested["learner"]["role"]["max_output_tokens"] < previous_output_limit:
                raise ValueError("Learner output limit can only increase on resume.")
            if (
                prior["budgets"]["price_sheet"] != requested["budgets"]["price_sheet"]
                or saved.get("pricing") != pricing
            ):
                raise ValueError(
                    "Run pricing changed. Restore the original price sheet to resume."
                )
            if saved["status"] == "completed":
                logger.info("Skipping completed experiment %s", config.experiment_name)
                return manifest_path
            if saved["initial_bundle_id"] != bundle["bundle_id"]:
                raise ValueError(
                    "Actor prompts or imported advice changed. Choose another experiment_name."
                )
            budget.restore(saved["budget"])
        else:
            if resume_directory.exists() and any(
                p.name
                not in {
                    "resume.json",
                    "resume.json.partial",
                    "manifest.json.partial",
                    "selected-bundle.json",
                    "selected-bundle.json.partial",
                }
                for p in resume_directory.iterdir()
            ):
                raise ValueError("Existing experiment directory has no manifest")
            checkpoint = {
                "schema": 1,
                "next_game": 0,
                "game_ids": [],
                "cycles": [],
                "pending_evidence": None,
                "trial": bundle,
            }
    actor_provider = make_provider(actor, fake=config.transport == "fake")
    learner_provider = (
        make_provider(config.learner.role, fake=config.transport == "fake")
        if config.learner
        else None
    )
    capabilities = {"actor": actor_provider.preflight()}
    budget.price(actor)
    learner_system = None
    if config.learner:
        capabilities["learner"] = learner_provider.preflight()
        budget.price(config.learner.role)
        learner_system = render_prompt(
            "learner", config.learner.role, config.environment
        )
    if saved and saved.get("learner_system_prompt") != learner_system:
        raise ValueError("Learner prompt changed. Choose another experiment_name.")
    artifacts = (
        RunArtifacts(
            resume_directory,
            config.experiment_name,
            debug_trace=config.debug_trace,
            resumable=True,
        )
        if resume_directory is not None
        else _run_artifacts(config)
    )
    run_id = artifacts.run_id
    manifest = {
        "schema_version": "aresim.llm.run.v1",
        "run_id": run_id,
        "experiment_name": config.experiment_name,
        "status": "running",
        "config": json_value(config),
        "capabilities": capabilities,
        "provenance": _provenance(),
        "initial_bundle_id": bundle["bundle_id"],
        "selected_bundle_id": bundle["bundle_id"],
        "learning_cycles": [],
        "budget": budget.snapshot(),
        "learner_system_prompt": learner_system,
        "experiment": "sequential_diagnostic" if sequential_diagnostic else "standard",
        "imported_learnings": {
            key: value for key, value in imported.items() if key != "statements"
        },
        "imported_learning_count": len(imported["statements"]),
        "initial_learning_count": len(run_learnings(bundle)),
    }
    if pricing is not None:
        manifest["pricing"] = pricing
    if saved is not None:
        artifacts.restore(saved, checkpoint["game_ids"])
        manifest["provenance"] = saved["provenance"]
        manifest["resume_count"] = saved.get("resume_count", 0) + 1
        manifest["learning_cycles"] = checkpoint["cycles"]
        artifacts.record(
            "games.jsonl",
            {
                "event": "run_resumed",
                "next_game_index": checkpoint["next_game"],
                "retained_episode_ids": checkpoint["game_ids"],
            },
        )
    elif checkpoint is not None:
        artifacts.write("resume.json", checkpoint)
    artifacts.select_bundle(bundle)
    artifacts.set_manifest(manifest)
    if saved is not None:
        prior_generation = saved["config"]["learner"]["role"]["generation"]
        generation = manifest["config"]["learner"]["role"]["generation"]
        output_limit = config.learner.role.max_output_tokens
        if previous_output_limit != output_limit:
            artifacts.learning_record(
                {
                    "event": "learner_output_limit_increased",
                    "before_game_index": checkpoint["next_game"],
                    "previous": previous_output_limit,
                    "current": output_limit,
                }
            )
        if prior_generation != generation:
            artifacts.learning_record(
                {
                    "event": "learner_generation_changed_before_first_review",
                    "previous": prior_generation,
                    "current": generation,
                }
            )
    if config.learnings_file and saved is None:
        artifacts.learning_record(
            {
                "event": "learnings_imported",
                "count": len(imported["statements"]),
                "source": manifest["imported_learnings"],
            }
        )
        logger.info(
            "Imported %d reusable learning statements from %s",
            len(imported["statements"]),
            config.learnings_file,
        )
    logger.info("Run artifacts: %s", artifacts.directory)
    actor_calls = ModelCalls("actor", actor, actor_provider, budget, artifacts)
    learner_calls = (
        ModelCalls("learner", config.learner.role, learner_provider, budget, artifacts)
        if config.learner
        else None
    )
    try:
        if sequential_diagnostic:
            manifest["learning_cycles"] = _sequential_trial(
                config,
                bundle,
                actor_calls,
                learner_calls,
                learner_system,
                artifacts,
                manifest["learning_cycles"],
                checkpoint,
            )
        elif config.execution_mode in {"frozen_evaluation", "actor_ablation"}:
            _collect(config, bundle, actor_calls, artifacts, config.episodes, "frozen")
        else:
            learner = config.learner
            for batch in range(
                (len(config.episodes) + learner.batch_episodes - 1)
                // learner.batch_episodes
            ):
                start = batch * learner.batch_episodes
                development = _collect(
                    config,
                    bundle,
                    actor_calls,
                    artifacts,
                    config.episodes[start : start + learner.batch_episodes],
                    f"development-{batch:04d}",
                )
                item, proposal = propose(
                    learner_calls,
                    learner_system,
                    bundle,
                    development.completed_evidence,
                    artifacts,
                    batch,
                )
                cycle = {
                    "batch": batch,
                    "parent_bundle_id": bundle["bundle_id"],
                    "proposal_outcome": proposal["outcome"],
                    "promoted": False,
                }
                if item:
                    candidate = candidate_bundle(bundle, item)
                    artifacts.remember_bundle(candidate)
                    comparisons = learner.evaluation_episodes
                    if config.execution_mode == "development":
                        comparisons = comparisons[batch * 30 : (batch + 1) * 30]
                    parent = _collect(
                        config,
                        bundle,
                        actor_calls,
                        artifacts,
                        comparisons,
                        f"parent-{batch:04d}",
                    )
                    child = _collect(
                        config,
                        candidate,
                        actor_calls,
                        artifacts,
                        comparisons,
                        f"candidate-{batch:04d}",
                        evaluating=True,
                    )
                    evaluation = compare(
                        parent.metrics,
                        child.metrics,
                        smoke=config.execution_mode == "smoke",
                        bootstrap_seed=config.bootstrap_seed,
                    )
                    evaluation["parent_bundle_id"] = bundle["bundle_id"]
                    evaluation["candidate_bundle_id"] = candidate["bundle_id"]
                    evaluation["budget"] = budget.snapshot()
                    evaluation_record = {
                        key: value
                        for key, value in evaluation.items()
                        if key not in {"parent_episodes", "candidate_episodes"}
                    }
                    evaluation_record["parent_episode_ids"] = [
                        row["episode_id"] for row in parent.metrics
                    ]
                    evaluation_record["candidate_episode_ids"] = [
                        row["episode_id"] for row in child.metrics
                    ]
                    reference = artifacts.learning_record(
                        {
                            "event": "evaluation_completed",
                            "batch": batch,
                            "evaluation": evaluation_record,
                        }
                    )
                    if evaluation["promote"]:
                        bundle = publish_bundle(candidate, reference)
                        artifacts.remember_bundle(bundle)
                    cycle.update(
                        promoted=evaluation["promote"],
                        candidate_bundle_id=candidate["bundle_id"],
                        evaluation_reference=reference,
                        selected_bundle_id=bundle["bundle_id"],
                    )
                    artifacts.learning_record(
                        {
                            "event": "bundle_published"
                            if evaluation["promote"]
                            else "candidate_rejected",
                            **cycle,
                            "checks": evaluation["checks"],
                            "budget": budget.snapshot(),
                        }
                    )
                manifest["learning_cycles"].append(cycle)
                manifest["selected_bundle_id"] = bundle["bundle_id"]
                artifacts.select_bundle(bundle)
                artifacts.set_manifest(manifest)
        manifest["status"] = "completed"
    except KeyboardInterrupt:
        manifest.update(status="incomplete", ending_reason="interrupted")
        artifacts.learning_record(
            {
                "event": "run_stopped",
                "reason": "interrupted",
                "parent_retained": bundle["bundle_id"],
            }
        )
        raise
    except EpisodeStop as error:
        logger.info("Run stopped: %s", error.reason)
        manifest.update(status="incomplete", ending_reason=error.reason)
        artifacts.learning_record(
            {
                "event": "run_stopped",
                "reason": error.reason,
                "parent_retained": bundle["bundle_id"],
            }
        )
    except Exception as error:
        # Orchestration boundary preserves failure evidence, then re-raises programming/storage errors.
        manifest.update(status="failed", ending_reason=type(error).__name__)
        raise
    finally:
        manifest["budget"] = budget.snapshot()
        manifest["selected_bundle_id"] = bundle["bundle_id"]
        artifacts.select_bundle(bundle)
        artifacts.set_manifest(manifest)
        actor_provider.close()
        if learner_provider:
            learner_provider.close()
    return artifacts.directory / "manifest.json"


def build_frozen_agent(path: Path, *, environment: EnvironmentConfig) -> LLMDMAgent:
    """Attach an already selected policy to a live session; never run the Learner inline."""
    config = load_llm_dm_run(path)
    if (
        config.execution_mode != "frozen_evaluation"
        or config.environment.engine != environment.engine
    ):
        raise ValueError(
            "live attach requires a frozen_evaluation config with matching engine"
        )
    imported = load_learnings_file(config.learnings_file)
    bundle, actor = _acting_config(config, imported)
    budget = BudgetLedger(config.budgets, fake=config.transport == "fake")
    provider = make_provider(actor, fake=config.transport == "fake")
    capabilities = provider.preflight()
    budget.price(actor)
    artifacts = _run_artifacts(config)
    run_id = artifacts.run_id
    manifest = {
        "schema_version": "aresim.llm.run.v1",
        "run_id": run_id,
        "experiment_name": config.experiment_name,
        "source": "live-session-attachment",
        "config": json_value(config),
        "provenance": _provenance(),
        "status": "running",
        "bundle_id": bundle["bundle_id"],
        "capabilities": {"actor": capabilities},
        "budget": budget.snapshot(),
        "imported_learnings": {
            key: value for key, value in imported.items() if key != "statements"
        },
        "imported_learning_count": len(imported["statements"]),
        "initial_learning_count": len(run_learnings(bundle)),
    }
    if config.transport != "fake":
        manifest["pricing"] = json.loads(Path(config.budgets.price_sheet).read_text())
    artifacts.select_bundle(bundle)
    artifacts.set_manifest(manifest)
    agent = LLMDMAgent(
        bundle,
        config.environment,
        ModelCalls("actor", actor, provider, budget, artifacts),
        artifacts,
        phase="live",
    )
    agent.max_episode_steps = config.max_episode_steps
    return agent
