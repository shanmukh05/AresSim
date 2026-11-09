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
) -> list[Json]:
    trial = bundle
    for index, episode in enumerate(config.episodes):
        label = "development-0000" if index == 0 else f"candidate-{index - 1:04d}"
        game = _collect(
            config, trial, actor, artifacts, (episode,), label, evaluating=index > 0
        )
        metrics = game.metrics[0]
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
            cycles[-1]["learning_statement_count"] = metrics["learning_statement_count"]
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
        if index == len(config.episodes) - 1:
            break
        item, proposal = propose(
            learner, learner_system, trial, game.completed_evidence, artifacts, index
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
        logger.info("Learner after game %d: %s", index + 1, proposal["outcome"])
    return cycles


def run_llm_dm(config: RunConfig, *, sequential_diagnostic: bool = False) -> Path:
    """Run learning/evaluation or an explicitly selected sequential diagnostic."""
    if config.learner and config.learner.role.provider == "decision_models":
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
    artifacts = _run_artifacts(config)
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
    if config.transport != "fake":
        manifest["pricing"] = json.loads(Path(config.budgets.price_sheet).read_text())
    artifacts.select_bundle(bundle)
    artifacts.set_manifest(manifest)
    if config.learnings_file:
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
