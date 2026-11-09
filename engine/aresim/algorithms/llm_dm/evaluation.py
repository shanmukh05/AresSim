"""Independent paired-episode metrics and frozen candidate-promotion gates.

These gates consume real simulator outcomes, never replayed counterfactual reward
or the Learner's own judgment. Small smoke comparisons cannot promote a candidate.
"""

from __future__ import annotations

import numpy as np

from .contracts import Json


def compare(
    parent: list[Json], candidate: list[Json], *, smoke: bool, bootstrap_seed: int
) -> Json:
    """Pair by scheduled seed order and bootstrap at the environment-seed level."""
    if len(parent) != len(candidate) or not parent:
        raise ValueError(
            "paired evaluation requires equal nonempty completed episode sets"
        )
    difference = np.array(
        [int(c["success"]) - int(p["success"]) for p, c in zip(parent, candidate)],
        dtype=float,
    )
    rng = np.random.default_rng(bootstrap_seed)
    resampled = difference[
        rng.integers(0, len(difference), size=(10000, len(difference)))
    ].mean(axis=1)
    ci = [float(v) for v in np.quantile(resampled, [0.025, 0.975])]
    parent_failures = np.mean([p["survival_failure"] for p in parent])
    candidate_failures = np.mean([c["survival_failure"] for c in candidate])
    steps = sum(c["steps"] for c in candidate)
    fallbacks = sum(c["fallbacks"] for c in candidate) / max(1, steps)
    latencies = [v for c in candidate for v in c["latencies"]]
    p95 = float(np.quantile(latencies, 0.95)) if latencies else None
    checks = {
        "research_mode": not smoke,
        "at_least_30_paired_seeds": len(parent) >= 30,
        "positive_success_delta_lower_95_ci": ci[0] > 0,
        "survival_failure_increase_at_most_5pp": candidate_failures - parent_failures
        <= 0.05 + 1e-12,
        "fallback_rate_at_most_1pct": fallbacks <= 0.01,
        "p95_latency_within_30_seconds": p95 is not None and p95 <= 30,
        "complete_episodes": all(c["steps"] > 0 for c in parent + candidate),
    }
    return {
        "schema_version": "aresim.llm.evaluation.v1",
        "paired_seeds": len(parent),
        "parent_success_rate": float(np.mean([p["success"] for p in parent])),
        "candidate_success_rate": float(np.mean([c["success"] for c in candidate])),
        "success_delta_95_ci": ci,
        "candidate_fallback_rate": fallbacks,
        "candidate_p95_seconds": p95,
        "parent_survival_failure_rate": float(parent_failures),
        "candidate_survival_failure_rate": float(candidate_failures),
        "checks": checks,
        "promote": all(checks.values()),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": 10000,
        "parent_episodes": parent,
        "candidate_episodes": candidate,
        "note": "All costs, truncations and infrastructure failures are retained in the run manifest. Smoke never promotes.",
    }
