"""Lazy public exports for the inference-only Jev rover agent.

Does not import ``typesafe_sdk`` at package import. Training is not implemented;
Jev is a registered ``Agent`` like the scripted baseline.

**Last updated:** September 20, 2026
"""

from .config import JevAgentConfig

__all__ = [
    "FakeJevClient",
    "JevAgent",
    "JevAgentConfig",
    "JevUnavailableError",
    "build_jev_agent",
    "encode_ops_state",
    "jev_available",
    "make_test_jev_agent",
]


def __getattr__(name: str):
    if name in {"FakeJevClient", "JevUnavailableError", "jev_available"}:
        from .client import FakeJevClient, JevUnavailableError, jev_available

        return locals()[name]
    if name in {"JevAgent", "build_jev_agent", "make_test_jev_agent"}:
        from .agent import JevAgent, build_jev_agent, make_test_jev_agent

        return locals()[name]
    if name == "encode_ops_state":
        from .state import encode_ops_state

        return encode_ops_state
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
