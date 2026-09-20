"""TypeSafe client wrapper, fake client, and API-key resolution for Jev.

The live SDK is an optional extra. Smoke rollouts inject ``FakeJevClient``.
Never log API keys. Auth is ``JEV_API_KEY`` only.

**Last updated:** September 20, 2026

**Contains:** ``JevDecision``, ``JevUnavailableError``, ``FakeJevClient``,
``TypeSafeJevClient``, :func:`resolve_api_key`, :func:`jev_sdk_available`.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class JevUnavailableError(Exception):
    """Raised when the TypeSafe SDK or API key is missing for a live call."""


@dataclass(frozen=True)
class JevDecision:
    """Structured Choice result consumed by ``JevAgent``."""

    choice: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)


class JevClient(Protocol):
    """Ask one batch of typed questions about a JSON state."""

    def system_one(self, state: object, questions: dict[str, object]) -> JevDecision:
        """Return the ``action`` Choice, or raise on transport failure."""
        ...


class FakeJevClient:
    """Deterministic stand-in that never calls the network.

    ``queue`` is consumed first; then ``default_choice``. ``None`` in the queue
    raises ``TimeoutError``. Confidence applies to every returned decision.
    """

    def __init__(
        self,
        *,
        default_choice: str | None = None,
        queue: Sequence[str | None] = (),
        confidence: float = 1.0,
        probabilities: dict[str, float] | None = None,
    ) -> None:
        self.default_choice = default_choice
        self.confidence = confidence
        self.probabilities = probabilities
        self._queue = list(queue)
        self.calls: list[tuple[object, dict[str, object]]] = []

    def system_one(self, state: object, questions: dict[str, object]) -> JevDecision:
        """Return the next queued choice or the first legal criterion key."""
        self.calls.append((state, questions))
        if self._queue:
            next_choice = self._queue.pop(0)
            if next_choice is None:
                raise TimeoutError("fake Jev timeout")
            return self._decision(next_choice)
        if self.default_choice is not None:
            return self._decision(self.default_choice)
        action = questions.get("action")
        if not isinstance(action, dict):
            raise ValueError("fake client expected an action Choice")
        criteria = action.get("criteria")
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError("fake client expected non-empty criteria")
        return self._decision(str(next(iter(criteria))))

    def _decision(self, choice: str) -> JevDecision:
        dist = dict(self.probabilities) if self.probabilities is not None else {choice: 1.0}
        return JevDecision(choice=choice, confidence=self.confidence, probabilities=dist)


def jev_sdk_available() -> bool:
    """Return whether ``typesafe_sdk`` can be imported."""
    try:
        import typesafe_sdk  # noqa: F401
    except ImportError:
        return False
    return True


def _load_dotenv_files() -> None:
    here = Path.cwd()
    candidates = [here / ".env", here / ".env.local", here.parent / ".env", here.parent / ".env.local"]
    for path in candidates:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            name = key.strip()
            if name in os.environ and os.environ[name].strip():
                continue
            os.environ[name] = value.strip().strip("'").strip('"')


def resolve_api_key() -> str | None:
    """Return ``JEV_API_KEY``, loading ``.env.local`` if needed."""
    value = os.environ.get("JEV_API_KEY", "").strip()
    if value:
        return value
    _load_dotenv_files()
    value = os.environ.get("JEV_API_KEY", "").strip()
    return value or None


def jev_available() -> bool:
    """Return whether a live TypeSafe call can be attempted."""
    return jev_sdk_available() and resolve_api_key() is not None


class TypeSafeJevClient:
    """Synchronous TypeSafe System One client for one Choice named ``action``."""

    def __init__(self, model: str, timeout_seconds: float) -> None:
        try:
            from typesafe_sdk import TypeSafeClient
        except ImportError as error:
            raise JevUnavailableError("install aresim[jev] to use the live TypeSafe client") from error
        key = resolve_api_key()
        if key is None:
            raise JevUnavailableError("set JEV_API_KEY")
        self._timeout = timeout_seconds
        try:
            self._client = TypeSafeClient(api_key=key, model=model, timeout=timeout_seconds)
        except TypeError:
            self._client = TypeSafeClient(api_key=key, model=model)

    def system_one(self, state: object, questions: dict[str, object]) -> JevDecision:
        """Call System One and read the ``action`` Choice."""
        try:
            response = self._client.system_one(state=state, questions=questions, timeout=self._timeout)
        except TypeError:
            response = self._client.system_one(state=state, questions=questions)
        answers = getattr(response, "answers", None) or getattr(response, "choices", None)
        if isinstance(answers, dict) and "action" in answers:
            payload = answers["action"]
        else:
            choices = getattr(response, "choices", None)
            if not isinstance(choices, dict) or "action" not in choices:
                raise ValueError("TypeSafe response missing action Choice")
            payload = choices["action"]
        choice = getattr(payload, "choice", None)
        confidence = float(getattr(payload, "confidence", 0.0) or 0.0)
        probabilities = dict(getattr(payload, "probabilities", None) or {})
        if not isinstance(choice, str) or not choice:
            raise ValueError("TypeSafe action Choice was empty")
        return JevDecision(choice=choice, confidence=confidence, probabilities=probabilities)


__all__ = [
    "FakeJevClient",
    "JevClient",
    "JevDecision",
    "JevUnavailableError",
    "TypeSafeJevClient",
    "jev_available",
    "jev_sdk_available",
    "resolve_api_key",
]
