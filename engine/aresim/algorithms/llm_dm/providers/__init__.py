"""Optional LLM transports and an explicit provider factory.

The public protocol is in base.py; native REST adapters and the offline smoke
transport share the same structured request and normalized result contracts.
"""

from .base import (
    LLMProvider,
    ProviderRequest,
    ProviderResult,
    make_provider,
    register_provider,
)

__all__ = [
    "LLMProvider",
    "ProviderRequest",
    "ProviderResult",
    "make_provider",
    "register_provider",
]
