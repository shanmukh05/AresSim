"""Explain measured game progression and Learner decisions in one readable summary.

These deterministic renderers summarize saved evidence. They never invent model
reasoning or equate collection with delivery. Replays retain the full game data.
"""

from __future__ import annotations

from .contracts import Json


def fractions(state: Json) -> Json:
    progress = {
        row["type_name"]: row["current_fraction"] for row in state["objectives"]
    }
    return {"ice": progress.get("deliver_ice"), "samples": progress.get("scan")}


def compact_turn(event: Json) -> Json:
    """Keep the issued action and observed outcome needed to follow one real tick."""
    after, before = event["state_after"], event["state_before"]
    rover, colony = after["rover"], after["colony"]
    effects = []
    for key in ("ice_kg", "samples_kg", "ore_kg"):
        delta = rover["cargo"][key] - before["rover"]["cargo"][key]
        if abs(delta) > 0.00001:
            effects.append(f"cargo {key} {delta:+g}")
    previous_progress = fractions(before)
    for key, value in fractions(after).items():
        previous = previous_progress[key]
        if value is not None and previous is not None and value > previous:
            effects.append(
                f"delivered {key} {100 * previous:.0f}% -> {100 * value:.0f}%"
            )
    if colony["pad_needs_service"] != before["colony"]["pad_needs_service"]:
        effects.append(
            "pad service needed" if colony["pad_needs_service"] else "pad service safe"
        )
    return {
        "decision_id": event["decision_id"],
        "transition_id": event["transition_id"],
        "proposed_action": event["proposed_action"],
        "issued_action": event["issued_action"],
        "effective_action": event["effective_action"],
        "fallback": event["fallback"],
        "position": rover["position_cells"],
        "battery_pct": rover["battery_pct"],
        "health_pct": rover["health_pct"],
        "cargo": rover["cargo"],
        "pad_proximity": rover["pad_proximity"],
        "pad_needs_service": colony["pad_needs_service"],
        "livability_pct": colony["livability_pct"],
        "power_margin": colony["power"]["margin"],
        "water_reserve": colony["water_reserve"],
        "oxygen_reserve": colony["oxygen_reserve"],
        "weather": after["weather"],
        "delivered_fractions": fractions(after),
        "effects": effects,
        "rl_reward": event["rl_reward"],
        "terminated": event["terminated"],
        "truncated": event["truncated"],
        "ending_reason": event["ending_reason"],
        "actor_details": event.get("actor_details"),
        "history_omitted": event.get("history_omitted"),
        "history_trace_steps": event.get("history_trace_steps"),
        "goal": (event.get("decision_context") or {}).get("goal"),
        "advice_ids": (event.get("decision_context") or {}).get("advice_ids", []),
        "deferred_advice_count": (event.get("decision_context") or {}).get(
            "deferred_advice_count", 0
        ),
    }


def _quota(value: object) -> str:
    return "n/a" if value is None else f"{100 * float(value):.0f}%"


def _label(game: Json) -> str:
    kind, _, batch = game["phase"].partition("-")
    label = {
        "development": "Development batch",
        "parent": "Parent evaluation",
        "candidate": "Candidate evaluation",
        "frozen": "Frozen evaluation",
        "live": "Live game",
    }.get(kind, kind)
    if batch.isdigit():
        label += f" {int(batch) + 1}"
    seed = f" (seed {game['environment_seed']})" if "environment_seed" in game else ""
    return f"{label} / {game['source_episode_id']}{seed}"


def _source_label(reference: str, games: list[Json]) -> str:
    identity, _, step = reference.partition(":transition:")
    game = next((g for g in games if g["episode_id"] == identity), None)
    if game:
        return _label(game) + (f" turn {int(step) + 1}" if step else "")
    return reference


def render_summary(
    manifest: Json, learning: list[Json], progress: dict[str, list[Json]]
) -> str:
    """Render results, limitations, source-linked lessons and meaningful game milestones."""
    config, budget = manifest.get("config", {}), manifest.get("budget", {})
    actor, games = config.get("actor", {}), manifest.get("games", [])
    closed = [game for game in games if game.get("status") == "closed"]
    mode = config.get("execution_mode", manifest.get("source", "unknown"))
    lines = [
        "ARESSIM — RUN SUMMARY",
        f"Experiment: {manifest.get('experiment_name', config.get('experiment_name', '?'))}",
        f"Run: {manifest.get('run_id', '?')}",
        f"Status: {manifest.get('status', 'running')} | mode: {mode} | transport: {config.get('transport', '?')}",
        f"Actor: {actor.get('provider', '?')} / {actor.get('model', '?')}",
    ]
    if manifest.get("ending_reason"):
        lines.append(f"Stopped because: {manifest['ending_reason']}")
    failure = budget.get("input_budget_failure")
    if failure:
        lines.append(
            f"{failure['role'].title()} input: conservative reservation {failure['estimated_input_tokens']:,} "
            f"exceeded max_input_tokens={failure['max_input_tokens']:,}; "
            f"system {failure['system_bytes']:,} bytes + packet {failure['packet_bytes']:,} bytes + 1,024 overhead. "
            "This is a local input allowance, not provider-reported tokens or a spending limit."
        )
    imported = manifest.get("imported_learnings", {})
    if imported.get("path"):
        lines.append(
            f"Imported learnings: {manifest.get('imported_learning_count', 0)} distinct statements from {imported['path']}; "
            f"{imported.get('no_change_records', 0)} no_change records added no statements."
        )
    if budget:
        lines.append(
            f"Usage: {budget.get('attempts', 0)} attempts; {budget.get('tokens_charged_or_reserved', 0):,} tokens; "
            f"${budget.get('usd_charged_or_reserved', 0):.4f} conservative estimate/reservation; {budget.get('elapsed_seconds', 0):.1f}s"
        )
        lines.append(
            "Cost is a configured-rate estimate, not an invoice. Unknown usage retains its reservation."
        )
    for role, calls in manifest.get("model_calls", {}).items():
        lines.append(
            f"{role.title()}: {calls['ok']}/{calls['attempts']} calls OK; {calls['errors']} non-OK; "
            f"{calls['input_tokens']:,} input / {calls['output_tokens']:,} output tokens"
        )
    lines += ["", "WHAT THE RUN SHOWED"]
    if not closed:
        lines.append(
            "No game has completed yet. Each turn is saved immediately in games.jsonl."
        )
    else:
        lines.append(
            f"{len(closed)} recorded games; {sum(g['metrics']['success'] for g in closed)} mission completions; "
            f"{sum(g['metrics']['fallbacks'] for g in closed)} action fallbacks."
        )
        capped = sum(
            g["outcome"]["ending_reason"] == "max_episode_steps" for g in closed
        )
        if capped:
            lines.append(
                f"{capped} games ended at the external {config.get('max_episode_steps', '?')}-step limit. "
                "A cutoff is neither mission completion nor a survival failure."
            )
        for game in closed:
            state = game["metrics"].get("final_state")
            if state:
                cargo, delivery = state["rover"]["cargo"], fractions(state)
                lines.append(
                    f"- {_label(game)}: delivered ice {_quota(delivery['ice'])}, samples {_quota(delivery['samples'])}; "
                    f"undelivered cargo: ice {cargo['ice_kg']:g} kg, samples {cargo['samples_kg']:g} kg."
                )
            context = game["metrics"].get("learning_context_decisions")
            reduced = game["metrics"].get("history_budget_reductions", 0)
            if reduced:
                lines.append(
                    f"  Optional history fitted to the Actor input limit on {reduced} decisions; current telemetry and all learning statements retained. "
                    "Per-turn omission counts are recorded in games.jsonl."
                )
            if game["metrics"].get("context_profile") == "focused-v2":
                lines.append(
                    f"  Focused memory: {game['metrics'].get('goal_selections', 0)} goal selections; "
                    f"{game['metrics']['learning_statement_count']} retained learning statements; "
                    f"applicable advice supplied on {context} action decisions. "
                    "All compatible statements remain available at goal selection; exposure does not prove benefit."
                )
            elif context is not None:
                lines.append(
                    f"  Persistent learning context: {game['metrics']['learning_statement_count']} distinct statements "
                    f"on {context} decisions; condition-matched advice on "
                    f"{game['metrics'].get('advice_exposed_decisions', 0)} decisions. "
                    "Supplied context does not establish that advice was followed or helped."
                )
            elif game["phase"].startswith("candidate-"):
                exposed = game["metrics"].get("candidate_advice_exposed_decisions")
                if (
                    exposed is None
                    and game["metrics"].get("advice_exposed_decisions") == 0
                ):
                    exposed = 0
                if exposed == 0:
                    lines.append(
                        "  Candidate advice appeared in 0 decisions: its effect was not exercised. "
                        "This comparison cannot establish that the advice helped or harmed decisions."
                    )
                elif exposed is not None:
                    lines.append(
                        f"  Candidate advice appeared in {exposed} decisions; exposure alone is not proof of benefit."
                    )
    if mode in {"smoke", "sequential_diagnostic"}:
        lines.append(
            "Diagnostic runs verify the workflow and cannot promote a policy or establish general performance."
        )
    for note in manifest.get("review_notes", []):
        lines.append(f"Review: {note['finding']}")
    verification = manifest.get("verification")
    if verification:
        lines.append(
            f"Verification: {verification.get('status', '?')}; {verification.get('replayed_transitions', '?')} checksum-replayed transitions."
        )
    lines += ["", "LEARNER AND EVALUATION"]
    if manifest.get("experiment") == "sequential_diagnostic":
        lines.append(
            "Each game is followed by a Learner review before the next game. All earlier distinct statements carry forward in persistent prompt context, even when no condition matches. "
            "No policy is promoted: promotion is separate from within-run reuse. no_change retains earlier learnings."
        )
    elif manifest.get("experiment") == "sequential_two_game_diagnostic":
        proposals = [
            event
            for event in learning
            if event["event"] in {"learner_proposed", "learner_no_change"}
        ]
        if proposals and proposals[-1]["proposal"]["outcome"] == "no_change":
            lines.append(
                "Game 1 -> Learner no_change -> unchanged parent in game 2. No new advice was available to apply."
            )
        else:
            lines.append(
                "Game 1 -> Learner -> unpromoted candidate in game 2. Exactly two games; the selected parent policy is retained."
            )
    if not learning:
        lines.append(
            "Frozen evaluation does not adapt the policy."
            if mode in {"frozen_evaluation", "actor_ablation"}
            else "No Learner event recorded yet."
        )
    for event in learning:
        kind = event["event"]
        if kind == "learner_retry_started":
            lines.append(f"Learner retry: {event['reason']}")
        if kind == "learner_started":
            lines.append(
                f"Batch {event['batch']}: Learner context omitted {event.get('omitted_transitions', 0)} detailed "
                f"transition records and {event.get('omitted_timeline_transitions', 0)} compact timeline rows. Full games remain in replays."
            )
        if kind in {"learner_proposed", "learner_no_change"}:
            proposal = event["proposal"]
            lines += [
                f"Proposal: {proposal['outcome']} ({proposal.get('target') or 'parent retained'})",
                f"Learner interpretation (hypothesis): {proposal['reason']}",
            ]
            if proposal.get("text"):
                lines += [
                    f"Advice: {proposal['text']}",
                    f"Expected benefit (hypothesis): {proposal['expected_benefit']}",
                    f"Failure case: {proposal['failure_case']}",
                ]
            lines.append(
                "Evidence: "
                + "; ".join(_source_label(ref, games) for ref in proposal["support"])
            )
        if kind in {"candidate_rejected", "bundle_published"}:
            labels = {
                "research_mode": "smoke mode cannot publish",
                "at_least_30_paired_seeds": "fewer than 30 paired seeds",
                "positive_success_delta_lower_95_ci": "no established success improvement",
                "survival_failure_increase_at_most_5pp": "survival failures increased too much",
                "fallback_rate_at_most_1pct": "too many action fallbacks",
                "p95_latency_within_30_seconds": "decision latency exceeded the limit",
                "complete_episodes": "evaluation games incomplete",
            }
            failed = [
                labels.get(name, name.replace("_", " "))
                for name, passed in event.get("checks", {}).items()
                if not passed
            ]
            lines.append(
                f"Decision: {kind.replace('_', ' ')}. Failed gates: {', '.join(failed) or 'none'}."
            )
        if kind == "sequential_trial_completed":
            if "learning_context_decisions" in event:
                lines.append(
                    f"Game {event.get('batch', 0) + 2}: {event['learning_statement_count']} retained statements; "
                    f"learning context on {event['learning_context_decisions']} action decisions; "
                    f"{event['candidate_advice_exposed_decisions']} decisions matched tentative advice conditions. "
                    "Selected initial policy retained; earlier learnings remain available to subsequent games."
                )
            else:
                lines.append(
                    f"Game {event.get('batch', 0) + 2} received condition-matched tentative advice on "
                    f"{event['candidate_advice_exposed_decisions']} decisions. Selected initial policy retained."
                )
        if kind in {"learner_failed", "learner_rejected_invalid_output", "run_stopped"}:
            lines.append(
                f"{kind}: {event.get('reason') or event.get('error_code') or event.get('status')}"
            )
    lines += [
        "",
        "GAME PROGRESSION",
        "Turns are 1-based. Routine moves are sampled every 8 turns; cargo/delivery changes, fallbacks and the final turn are shown.",
    ]
    for game in games:
        lines += ["", _label(game)]
        metrics = game.get("metrics")
        if metrics:
            first = metrics["first_unload_step"]
            lines.append(
                f"Outcome: {metrics['ending_reason']} | {metrics['steps']} turns | return {metrics['episode_return']:.3f} | "
                f"visited {metrics['visited_cells']} cells | first unload turn: {first + 1 if first is not None else 'none'}"
            )
        turns = progress.get(game["episode_id"], [])
        for index, turn in enumerate(turns):
            if not (
                turn["effects"]
                or turn["fallback"]
                or index % 8 == 0
                or index == len(turns) - 1
            ):
                continue
            pos, cargo = turn["position"], turn["cargo"]
            detail = "; ".join(turn["effects"]) or "no cargo or delivery change"
            if turn["fallback"]:
                detail += f"; fallback: {turn['fallback']}"
            lines.append(
                f"  {turn['step_index'] + 1:4d} {turn['issued_action']:12s} ({pos['x']},{pos['y']}) "
                f"battery {turn['battery_pct']:.1f}% | ice {cargo['ice_kg']:g} kg samples {cargo['samples_kg']:g} kg | {detail}"
            )
        replay = next(
            (
                r["path"]
                for r in manifest.get("replays", [])
                if r["phase"] == game["phase"]
                and r["source_episode_id"] == game["source_episode_id"]
            ),
            None,
        )
        if replay:
            lines.append(f"UI replay: {replay}")
    lines += [
        "",
        "FILES",
        "summary.log — start here: results, Learner decisions and game progression.",
        "games.jsonl — every real turn as compact telemetry; unexecuted decisions and model errors.",
        "learning.jsonl — actual Learner evidence, proposals, evaluation and publication decisions (when learning runs).",
        "manifest.json — resolved setup, Learner prompt, cost/latency totals, policy versions and replay index.",
        "selected-bundle.json — portable selected policy; includes its exact Actor prompt.",
        "replays/*.json — one UI-importable file per game, updated atomically every tick and finalized in place. Import while running to view the recorded prefix; reimport later for newer turns.",
        "Exact per-call prompts, schemas and packets require debug_trace: true (debug.jsonl.gz).",
    ]
    return "\n".join(lines) + "\n"
