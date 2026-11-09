"""Provider-neutral structured requests and normalized transport results.

The explicit registry keeps orchestration independent of provider wire formats.
No provider gets an environment, filesystem handle, tool, or implicit chat state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol
from threading import Event

from ..config import RoleConfig
from ..contracts import Json


@dataclass(frozen=True)
class ProviderRequest:
    role: str
    request_id: str
    system: str
    user: str
    schema: Json


@dataclass(frozen=True)
class ProviderResult:
    status: str
    text: str = ""
    model: str | None = None
    request_id: str | None = None
    upstream: str | None = None
    finish_reason: str | None = None
    usage: Json | None = None
    error_code: str | None = None
    retryable: bool = False
    retry_after_seconds: float = 0
    latency_seconds: float = 0
    details: Json | None = None


class LLMProvider(Protocol):
    """Standalone structured generation with explicit deadline and cancellation."""

    def preflight(self) -> Json: ...

    def generate(
        self, request: ProviderRequest, timeout: float, cancellation: Event
    ) -> ProviderResult: ...

    def close(self) -> None: ...


ProviderFactory = Callable[[RoleConfig], LLMProvider]
_REGISTRY: dict[str, ProviderFactory] = {}
_BUILTINS_REGISTERED = False


def provider_known(name: str) -> bool:
    return (
        name
        in {
            "openai",
            "anthropic",
            "gemini",
            "openrouter",
            "deepseek",
            "decision_models",
            "fake",
        }
        or name in _REGISTRY
    )


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Explicit extension seam; duplicates are configuration errors."""
    if not name or name in _REGISTRY:
        raise ValueError(f"duplicate/invalid provider: {name}")
    _REGISTRY[name] = factory


def make_provider(config: RoleConfig, *, fake: bool = False) -> LLMProvider:
    """Import only the selected optional integration."""
    if fake:
        from .fake import FakeProvider

        return FakeProvider(config)
    global _BUILTINS_REGISTERED
    if not _BUILTINS_REGISTERED:
        from .http import register_builtin_providers

        register_builtin_providers()
        _BUILTINS_REGISTERED = True
    if config.provider == "fake":
        from .fake import FakeProvider

        return FakeProvider(config)
    try:
        factory = _REGISTRY[config.provider]
    except KeyError as error:
        raise ValueError(f"unregistered provider: {config.provider}") from error
    return factory(config)
