"""Conservative shared reservations for provider attempts, tokens, wall time and USD.

Unknown usage retains the worst-case reservation. Role subtotals never replace the
run-wide ceiling, so Learner and comparison costs cannot disappear from accounting.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ...algorithms.base import EpisodeStop
from ..common.config_decode import finite_number
from .config import BudgetConfig, RoleConfig
from .contracts import Json, strict_json

logger = logging.getLogger(__name__)


def estimate_input_tokens(system: str, user: str) -> int:
    """Use the same conservative byte reservation for input fitting and billing limits."""
    return len(system.encode()) + len(user.encode()) + 1024


@dataclass(frozen=True)
class Reservation:
    role: str
    tokens: int
    usd: float
    input_tokens: int
    output_tokens: int
    price: Json


class BudgetLedger:
    """Reserve before a call and reconcile only with usable reported usage."""

    def __init__(self, config: BudgetConfig, *, fake: bool = False) -> None:
        self.config, self.fake = config, fake
        self.started = time.monotonic()
        self.attempts, self.tokens, self.usd = 0, 0, 0.0
        self.roles: dict[str, Json] = {}
        self.prices = {}
        self.price_date = None
        self.input_budget_failure: Json | None = None
        if not fake:
            if not config.price_sheet:
                raise ValueError(
                    "live execution requires a dated price_sheet with explicit model/route rates"
                )
            sheet = strict_json(Path(config.price_sheet).read_text())
            if set(sheet) - {"as_of", "models", "source"} or not {
                "as_of",
                "models",
            } <= set(sheet):
                raise ValueError(
                    "price sheet requires as_of and models; source URL is optional"
                )
            self.price_date = date.fromisoformat(sheet["as_of"])
            self.prices = sheet["models"]
            if not isinstance(self.prices, dict):
                raise ValueError("price sheet models must be an object")

    def check(self) -> None:
        if (
            time.monotonic() - self.started >= self.config.max_wall_seconds
            or self.tokens > self.config.max_total_tokens
            or self.usd > self.config.max_usd
        ):
            raise EpisodeStop("budget_exhausted")

    def price(self, config: RoleConfig) -> Json:
        if self.fake:
            return {"input_per_million": 0.0, "output_per_million": 0.0}
        key = f"{config.provider}/{config.model}" + (
            f"@{config.route}" if config.route else ""
        )
        raw = self.prices.get(key)
        if not isinstance(raw, dict) or set(raw) != {
            "input_per_million",
            "output_per_million",
        }:
            raise ValueError(f"missing billing rates for {key}")
        values = {name: finite_number(value, name) for name, value in raw.items()}
        if min(values.values()) < 0:
            raise ValueError("billing rates cannot be negative")
        return values

    def reserve(
        self, role: str, config: RoleConfig, system: str, user: str
    ) -> Reservation:
        """UTF-8 byte count is a conservative token estimate for standalone JSON/text."""
        self.check()
        input_tokens = estimate_input_tokens(system, user)
        if input_tokens > config.max_input_tokens:
            self.input_budget_failure = {
                "role": role,
                "provider": config.provider,
                "model": config.model,
                "estimated_input_tokens": input_tokens,
                "max_input_tokens": config.max_input_tokens,
                "system_bytes": len(system.encode()),
                "packet_bytes": len(user.encode()),
                "note": "Conservative UTF-8-byte reservation plus 1024 overhead; not provider-reported token usage.",
            }
            logger.error(
                "%s input reservation %d exceeds max_input_tokens=%d (conservative UTF-8 bytes plus overhead, not provider token usage)",
                role,
                input_tokens,
                config.max_input_tokens,
            )
            raise EpisodeStop("input_budget_exceeded")
        tokens = input_tokens + config.max_output_tokens
        price = self.price(config)
        usd = (
            input_tokens * price["input_per_million"]
            + config.max_output_tokens * price["output_per_million"]
        ) / 1_000_000
        if (
            self.attempts + 1 > self.config.max_provider_attempts
            or self.tokens + tokens > self.config.max_total_tokens
            or self.usd + usd > self.config.max_usd
        ):
            raise EpisodeStop("budget_exhausted")
        self.attempts += 1
        self.tokens += tokens
        self.usd += usd
        record = self.roles.setdefault(
            role, {"attempts": 0, "tokens": 0, "usd": 0.0, "unknown_usage_attempts": 0}
        )
        record["attempts"] += 1
        record["tokens"] += tokens
        record["usd"] += usd
        return Reservation(
            role, tokens, usd, input_tokens, config.max_output_tokens, price
        )

    def reconcile(self, reservation: Reservation, usage: Json | None) -> None:
        role = self.roles[reservation.role]
        if usage is None or any(
            type(usage.get(k)) is not int or usage[k] < 0
            for k in ("input_tokens", "output_tokens")
        ):
            role["unknown_usage_attempts"] += 1
            return
        # Provider totals already include cached input/reasoning where documented;
        # do not double-count those subcategories. Undiscounted input is conservative.
        tokens = usage["input_tokens"] + usage["output_tokens"]
        usd = (
            usage["input_tokens"] * reservation.price["input_per_million"]
            + usage["output_tokens"] * reservation.price["output_per_million"]
        ) / 1_000_000
        self.tokens += tokens - reservation.tokens
        self.usd += usd - reservation.usd
        role["tokens"] += tokens - reservation.tokens
        role["usd"] += usd - reservation.usd

    def snapshot(self) -> Json:
        return {
            "attempts": self.attempts,
            "tokens_charged_or_reserved": self.tokens,
            "usd_charged_or_reserved": self.usd,
            "elapsed_seconds": time.monotonic() - self.started,
            "roles": self.roles,
            "cost_note": "Conservative cost estimate from configured rates and reported usage; not a provider invoice. Unknown usage retains worst-case reservation.",
            "price_as_of": self.price_date.isoformat() if self.price_date else None,
            **(
                {"input_budget_failure": self.input_budget_failure}
                if self.input_budget_failure
                else {}
            ),
        }
