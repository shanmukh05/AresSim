"""Inference-only rover policy that asks Jev for one legal Discrete(10) action.

Implements the shared ``Agent`` contract. Consumes ``aresim.obs.local.v1`` plus
the authoritative mask; never reads ``WorldState``. Timeouts, transport errors,
and exhausted call budgets still fall back to Wait. Legal extract/scan with free
cargo always collect this tick. Idle Wait, action loops, and a full bay on the
pad are overridden from Choice probabilities, with a 0.3 chance to take the
second-highest legal mass.

**Last updated:** September 20, 2026

**Registry name:** ``jev`` → ``policy_id`` ``aresim.agent.jev.v1``.
"""

from __future__ import annotations

import logging
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ...components.actions import ACTION_SCHEMA
from ...components.observations import OBSERVATION_SCHEMA
from ...config import EnvironmentConfig
from ...defaults import DEFAULT_ENVIRONMENT_CONFIG
from ..common.masks import require_mask
from .client import FakeJevClient, JevClient, JevDecision, JevUnavailableError, TypeSafeJevClient
from .config import JevAgentConfig
from .questions import build_action_question
from .state import ACTION_IDS, ACTION_NAMES, OpsScales, encode_ops_state


logger = logging.getLogger(__name__)

WAIT_ACTION = 0
UNLOAD_ACTION = ACTION_IDS["unload"]
PAD_IDLE = frozenset({"wait", "build"})
PRODUCTIVE_ORDER: tuple[str, ...] = (
    "unload",
    "extract",
    "scan",
    "service",
    "build",
    "move_north",
    "move_east",
    "move_south",
    "move_west",
)


class JevAgent:
    """Map one Jev Choice onto a rover action index with deterministic fallbacks."""

    policy_id = "aresim.agent.jev.v1"
    observation_schema = OBSERVATION_SCHEMA
    action_schema = ACTION_SCHEMA

    def __init__(
        self,
        config: JevAgentConfig | None = None,
        *,
        scales: OpsScales | None = None,
        client: JevClient | None = None,
    ) -> None:
        resolved = JevAgentConfig() if config is None else config
        resolved.validate()
        self.config = resolved
        self.scales = scales
        self._client = client
        self._reset = False
        self._calls = 0
        self._recent: list[str] = []
        self._remembered_pad: tuple[int, int] | None = None
        self._rng = random.Random(0)
        self._log_file: Path | None = None
        self.last_info: dict[str, object] | None = None

    @classmethod
    def from_environment(
        cls,
        environment_config: EnvironmentConfig,
        *,
        config: JevAgentConfig | None = None,
        client: JevClient | None = None,
    ) -> JevAgent:
        """Copy observation denormalization scales from the environment config."""
        resolved = JevAgentConfig() if config is None else config
        observation = environment_config.observation_config
        scales = OpsScales(
            payload_capacity_kg=environment_config.engine.payload.capacity_kg,
            power_scale=observation.power_scale,
            water_scale=observation.water_scale,
            oxygen_scale=observation.oxygen_scale,
            neighborhood_size=resolved.neighborhood_size,
        )
        return cls(resolved, scales=scales, client=client)

    def reset(self, seed: int) -> None:
        """Clear call budget, recent actions, remembered pad, and last Choice trace."""
        self._reset = True
        self._calls = 0
        self._recent = []
        self._remembered_pad = None
        self._rng = random.Random(seed)
        self._log_file = _run_log_file(self.config.log_path)
        self.last_info = None

    def _resolve_client(self) -> JevClient:
        if self._client is not None:
            return self._client
        try:
            self._client = TypeSafeJevClient(self.config.model, self.config.timeout_seconds)
        except JevUnavailableError:
            raise
        return self._client

    def _remember(self, action_name: str) -> None:
        self._recent.append(action_name)
        overflow = len(self._recent) - self.config.memory_length
        if overflow > 0:
            self._recent = self._recent[overflow:]

    def _remember_pad(self, cells: object) -> None:
        if not isinstance(cells, list):
            return
        for cell in cells:
            if isinstance(cell, dict) and cell.get("terrain") == "build_pad" and cell.get("in_bounds"):
                self._remembered_pad = (int(cell["dx"]), int(cell["dy"]))
                return

    def _commit(self, mask: np.ndarray, action_id: int, *, decision: JevDecision | None, reason: str | None, wait_recharges: bool) -> int:
        if mask[action_id] != 1:
            if mask[WAIT_ACTION] != 1:
                raise ValueError("Wait must remain legal")
            action_id = WAIT_ACTION
            reason = reason or "illegal selected action"
        name = ACTION_NAMES[action_id]
        self._remember(name)
        info = _trace(name, decision, reason, wait_recharges)
        self.last_info = info
        _append_log(self._log_file, info["summary"], self._calls, self._recent)
        logger.debug("%s", info["summary"])
        return action_id

    def _from_ranked(
        self,
        mask: np.ndarray,
        decision: JevDecision,
        wait_ok: bool,
        reason: str,
        exclude: set[str] | frozenset[str] = frozenset(),
    ) -> tuple[int, str]:
        ranked = _ranked_legal(decision.probabilities, mask, allow_wait=wait_ok, exclude=exclude)
        picked = _pick_top_or_second(ranked, self._rng, self.config.explore_second_prob)
        if picked is None:
            name = _productive_or_wait(mask, exclude)
            return ACTION_IDS[name], reason
        if picked != ranked[0]:
            reason = f"{reason}; explore second"
        return ACTION_IDS[picked], reason

    def _redirect_to_seek(
        self,
        decision: JevDecision,
        chosen: int | None,
        mask: np.ndarray,
        wait_ok: bool,
        seek: str | None,
    ) -> tuple[int, str] | None:
        if seek is None:
            return None
        if decision.choice in PAD_IDLE:
            return ACTION_IDS[seek], "seek resource"
        if chosen is None or mask[chosen] != 1:
            return ACTION_IDS[seek], "illegal or unknown choice"
        if decision.confidence < self.config.confidence_floor:
            return ACTION_IDS[seek], "low confidence"
        if chosen == WAIT_ACTION and not wait_ok and _has_productive(mask):
            return ACTION_IDS[seek], "idle wait skipped"
        return None

    def _ranked_fallback(
        self,
        mask: np.ndarray,
        decision: JevDecision,
        chosen: int | None,
        wait_ok: bool,
    ) -> tuple[int, str] | None:
        if chosen is None or mask[chosen] != 1:
            return self._from_ranked(mask, decision, wait_ok, "illegal or unknown choice")
        if decision.confidence < self.config.confidence_floor:
            return self._from_ranked(mask, decision, wait_ok, "low confidence")
        if chosen == WAIT_ACTION and not wait_ok and _has_productive(mask):
            return self._from_ranked(mask, decision, False, "idle wait skipped")
        if _continues_repeat(decision.choice, self._recent) and _has_productive(mask):
            blocked = {decision.choice, *self._recent[-2:]}
            return self._from_ranked(mask, decision, wait_ok, "repeat broken", exclude=blocked)
        return None

    def _forced_action(
        self,
        mask: np.ndarray,
        state: dict[str, object],
        decision: JevDecision,
    ) -> tuple[int, str] | None:
        collect = _collect_here(state, mask)
        if collect is not None:
            return ACTION_IDS[collect], "collect here"
        wait_ok = _wait_recharges(state)
        if _unload_ready(state, mask):
            return UNLOAD_ACTION, "unload cargo"
        chosen = ACTION_IDS.get(decision.choice)
        redirected = self._redirect_to_seek(decision, chosen, mask, wait_ok, _seek_resource(state, mask))
        if redirected is not None:
            return redirected
        return self._ranked_fallback(mask, decision, chosen, wait_ok)

    def _select(self, mask: np.ndarray, state: dict[str, object], decision: JevDecision) -> tuple[int, str | None]:
        forced = self._forced_action(mask, state, decision)
        if forced is not None:
            return forced
        return ACTION_IDS[decision.choice], None

    def act(self, observation: object, action_mask: np.ndarray) -> int:
        """Return one legal action index from Jev, or a productive fallback."""
        mask = require_mask(action_mask)
        if not self._reset:
            raise RuntimeError("agent has not been reset")
        if self.config.max_calls and self._calls >= self.config.max_calls:
            return self._commit(mask, WAIT_ACTION, decision=None, reason="max_calls exhausted", wait_recharges=False)
        if not isinstance(observation, dict):
            raise ValueError("jev agent requires a local observation dict")
        try:
            state, criteria = encode_ops_state(
                observation,
                mask,
                recent_actions=tuple(self._recent),
                remembered_pad=self._remembered_pad,
                scales=self.scales,
            )
        except ValueError as error:
            return self._commit(mask, WAIT_ACTION, decision=None, reason=str(error), wait_recharges=False)
        grid = state["grid"]
        if isinstance(grid, dict):
            self._remember_pad(grid.get("cells"))
        questions = build_action_question(criteria)
        self._calls += 1
        try:
            decision = self._resolve_client().system_one(state, questions)
        except Exception as error:
            return self._commit(
                mask,
                WAIT_ACTION,
                decision=None,
                reason=type(error).__name__,
                wait_recharges=_wait_recharges(state),
            )
        action_id, reason = self._select(mask, state, decision)
        return self._commit(
            mask,
            action_id,
            decision=decision,
            reason=reason,
            wait_recharges=_wait_recharges(state),
        )


def _wait_recharges(state: dict[str, object]) -> bool:
    rover = state.get("rover")
    if isinstance(rover, dict) and "wait_recharges" in rover:
        return bool(rover["wait_recharges"])
    colony = state.get("colony")
    if not isinstance(colony, dict):
        return False
    return float(colony.get("power_margin", 0) or 0) > 0


def _cargo_mass_kg(state: dict[str, object]) -> float:
    rover = state.get("rover")
    if not isinstance(rover, dict):
        return 0.0
    cargo = rover.get("cargo")
    if not isinstance(cargo, dict):
        return 0.0
    return float(cargo.get("ice_kg") or 0) + float(cargo.get("ore_kg") or 0) + float(cargo.get("samples_kg") or 0)


def _has_free_cargo(state: dict[str, object]) -> bool:
    rover = state.get("rover")
    if not isinstance(rover, dict):
        return False
    cargo = rover.get("cargo")
    return isinstance(cargo, dict) and float(cargo.get("free_kg") or 0) > 0


def _collect_here(state: dict[str, object], mask: np.ndarray) -> str | None:
    """Extract or scan the cell under the rover whenever that action is legal."""
    if not _has_free_cargo(state):
        return None
    if mask[ACTION_IDS["extract"]] == 1:
        return "extract"
    if mask[ACTION_IDS["scan"]] == 1:
        return "scan"
    return None


def _unload_ready(state: dict[str, object], mask: np.ndarray) -> bool:
    return mask[UNLOAD_ACTION] == 1 and _cargo_mass_kg(state) > 0


def _seek_target(state: dict[str, object]) -> dict[str, object] | None:
    if not _has_free_cargo(state):
        return None
    prospects = state.get("prospects")
    if not isinstance(prospects, dict):
        return None
    ice = prospects.get("best_ice")
    if isinstance(ice, dict):
        return ice
    ore = prospects.get("best_ore")
    return ore if isinstance(ore, dict) else None


def _seek_resource(state: dict[str, object], mask: np.ndarray) -> str | None:
    """Return the first step toward the named ice/ore cell when extract is not already legal."""
    target = _seek_target(state)
    if target is None:
        return None
    move = target.get("first_move")
    if isinstance(move, str) and move in ACTION_IDS and mask[ACTION_IDS[move]] == 1:
        return move
    return None


def _has_productive(mask: np.ndarray) -> bool:
    return any(mask[ACTION_IDS[name]] == 1 for name in PRODUCTIVE_ORDER)


def _productive_or_wait(mask: np.ndarray, exclude: set[str] | frozenset[str] = frozenset()) -> str:
    for name in PRODUCTIVE_ORDER:
        if name in exclude:
            continue
        if mask[ACTION_IDS[name]] == 1:
            return name
    return "wait"


def _continues_repeat(choice: str, recent: list[str]) -> bool:
    if len(recent) >= 3 and recent[-1] == recent[-2] == recent[-3]:
        return choice == recent[-1]
    if len(recent) >= 4 and recent[-4:-2] == recent[-2:]:
        return choice in {recent[-1], recent[-2]}
    return False


def _ranked_legal(
    probabilities: dict[str, float],
    mask: np.ndarray,
    *,
    allow_wait: bool,
    exclude: set[str] | frozenset[str] = frozenset(),
) -> list[str]:
    scored: list[tuple[float, str]] = []
    for name, action_id in ACTION_IDS.items():
        if mask[action_id] != 1 or name in exclude:
            continue
        if name == "wait" and not allow_wait:
            continue
        p = float(probabilities.get(name, 0.0))
        if p > 0:
            scored.append((p, name))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in scored]


def _pick_top_or_second(ranked: list[str], rng: random.Random, explore_prob: float) -> str | None:
    if not ranked:
        return None
    if len(ranked) < 2 or rng.random() >= explore_prob:
        return ranked[0]
    return ranked[1]


def _run_log_file(log_path: str, started: datetime | None = None) -> Path | None:
    """Return one file for this run, or None when logging is disabled.

    A path with a suffix is that file. A directory (the default ``results/jev``)
    gets ``jev-<UTC start>.log``.
    """
    if not log_path.strip():
        return None
    path = Path(log_path)
    if path.suffix:
        return path
    stamp = (started or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return path / f"jev-{stamp}.log"


def _append_log(target: Path | None, summary: str, calls: int, recent: list[str]) -> None:
    if target is None:
        return
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    recent_text = ",".join(recent) if recent else "-"
    line = f"{stamp} call={calls} {summary} recent={recent_text}\n"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError:
        logger.warning("jev log write failed")


def _trace(
    action: str,
    decision: JevDecision | None,
    reason: str | None,
    wait_recharges: bool,
) -> dict[str, object]:
    probabilities = dict(decision.probabilities) if decision is not None else {}
    parts = [f"Jev: {action}"]
    if decision is not None:
        parts.append(f"choice={decision.choice}")
        parts.append(f"conf={decision.confidence:.2f}")
    if reason:
        parts.append(f"fallback={reason}")
    ranked = sorted(probabilities.items(), key=lambda item: (-float(item[1]), item[0]))
    dist = " ".join(f"{name}={float(p):.2f}" for name, p in ranked if float(p) > 0)
    if dist:
        parts.append(dist)
    return {
        "choice": None if decision is None else decision.choice,
        "confidence": 0.0 if decision is None else float(decision.confidence),
        "probabilities": {name: float(value) for name, value in probabilities.items()},
        "action": action,
        "fallback": reason,
        "wait_recharges": wait_recharges,
        "summary": " ".join(parts),
    }


def build_jev_agent(context: object) -> JevAgent:
    """Registry factory: live client is resolved on first ``act``."""
    environment_config = getattr(context, "environment_config", DEFAULT_ENVIRONMENT_CONFIG)
    return JevAgent.from_environment(environment_config)


def make_test_jev_agent(
    client: JevClient | None = None,
    config: JevAgentConfig | None = None,
    environment_config: EnvironmentConfig = DEFAULT_ENVIRONMENT_CONFIG,
) -> JevAgent:
    """Construct an agent with an injected client for smoke rollouts."""
    return JevAgent.from_environment(
        environment_config,
        config=config,
        client=client if client is not None else FakeJevClient(),
    )


__all__ = ["JevAgent", "build_jev_agent", "make_test_jev_agent"]
