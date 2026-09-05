"""Framework-neutral policy contract for rollouts, evaluation, and checkpoint inference.

Defines the structural ``Agent`` protocol every baseline and frozen learned policy
implements. Policies may only consume policy observations and authoritative action
masks; they must not read canonical ``WorldState`` or bypass environment validation.

**Last updated:** September 1, 2026

**Contains:** ``Agent`` protocol and type variables for observation/action typing.

**Consumers:** :mod:`aresim.factory`, :mod:`aresim.training.runner`,
:mod:`aresim.algorithms.ppo.checkpoint`, :mod:`aresim.training.evaluation`.

**Contract fields:** ``policy_id`` (versioned string), ``observation_schema``
(``None`` when ignored), ``action_schema`` (required, e.g. ``aresim.action.rover.v1``).
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

import numpy as np


ObservationT = TypeVar("ObservationT", contravariant=True)
ActionT = TypeVar("ActionT", covariant=True)


@runtime_checkable
class Agent(Protocol[ObservationT, ActionT]):
    """Choose one discrete action per environment step with resettable episode state.

    Implementations must call ``reset(seed)`` before the first ``act`` of each episode.
    Stochastic policies should derive RNG state only from the supplied seed at reset.
    """

    policy_id: str
    observation_schema: str | None
    action_schema: str

    def reset(self, seed: int) -> None:
        """Clear episode-local memory and re-seed any stochastic components."""
        ...

    def act(self, observation: ObservationT, action_mask: np.ndarray) -> ActionT:
        """Return one action index using only the policy input and private memory.

        The mask shape is ``(10,)`` for Phase 1 rover macros. Implementations may
        ignore the mask (uniform random) or require at least one legal action.
        """
        ...


__all__ = ["Agent"]
