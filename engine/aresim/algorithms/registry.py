"""Register learned-policy algorithms, models, and checkpoint loaders.

Owns the training-side extension registry separate from component registration in
:mod:`aresim.registry`. New algorithms register here with explicit names; there is
no automatic discovery.

**Last updated:** September 1, 2026

**Contains:** ``TrainingRegistry``, ``TrainingContext``, factory protocols
(``AlgorithmFactory``, ``ModelFactory``, ``CheckpointLoader``), and
:func:`create_training_registry`.

**Built-in registrations:** algorithm ``masked_ppo``, model
``local_cnn_actor_critic``, checkpoint loader ``rllib_masked_ppo``.

**Re-exported from:** :mod:`aresim.registry` (lazy) and :mod:`aresim.training`.

**See also:** :mod:`aresim.algorithms.ppo` for the masked PPO implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .base import Agent

if TYPE_CHECKING:
    from ..training.experiments import ExperimentSpec


@dataclass(frozen=True)
class TrainingContext:
    """Resolved experiment and optional factories passed to training builders.

    ``component_registry`` allows custom component compositions during RLlib env
    construction without changing global defaults.
    """

    experiment: ExperimentSpec
    model_factory: object | None = None
    component_registry: object | None = None


@runtime_checkable
class AlgorithmFactory(Protocol):
    """Build framework-specific training configuration for one algorithm name.

    ``trainable`` is the Ray Tune trainable (class name string or type).
    ``checkpoint_loader_id`` must match a registered :class:`CheckpointLoader`.
    """

    algorithm_id: str
    trainable: str | type
    checkpoint_loader_id: str
    observation_schema: str
    action_schema: str

    def decode_config(self, payload: object) -> object:
        """Decode untrusted YAML ``algorithm_config`` into a validated dataclass."""
        ...

    def build(self, context: TrainingContext) -> object:
        """Return a framework-native config ready for Ray Tune (e.g. ``PPOConfig``)."""
        ...


@runtime_checkable
class ModelFactory(Protocol):
    """Declare the neural module class and schema compatibility for one model name."""

    model_id: str
    observation_schema: str
    action_schema: str
    rl_module_class: type


@runtime_checkable
class CheckpointLoader(Protocol):
    """Load a frozen native checkpoint as a framework-neutral :class:`Agent`."""

    loader_id: str

    def load(self, path: str, *, deterministic: bool = True) -> Agent[Any, Any]:
        """Restore inference from a sidecar JSON path plus native framework files."""
        ...


class TrainingRegistry:
    """Store explicit algorithm, model, and checkpoint-loader factories by name.

    Rejects duplicate names and validates built implementations against their
    registered identifiers and schema strings at resolve time.
    """

    def __init__(self) -> None:
        self._algorithms: dict[str, Callable[[TrainingContext], AlgorithmFactory]] = {}
        self._algorithm_decoders: dict[str, Callable[[object], object]] = {}
        self._models: dict[str, Callable[[TrainingContext], ModelFactory]] = {}
        self._loaders: dict[str, Callable[[], CheckpointLoader]] = {}

    @staticmethod
    def _register(table: dict[str, object], category: str, name: str, factory: object) -> None:
        if not isinstance(name, str) or not name.strip() or not callable(factory):
            raise ValueError(f"{category} registration requires a name and callable factory")
        if name in table:
            raise ValueError(f"duplicate {category}: {name}")
        table[name] = factory

    @staticmethod
    def _resolve(table: dict[str, Any], category: str, name: str):
        try:
            return table[name]
        except KeyError as error:
            raise ValueError(f"unknown {category}: {name}") from error

    def register_algorithm(
        self,
        name: str,
        factory: Callable[[TrainingContext], AlgorithmFactory],
        *,
        config_decoder: Callable[[object], object],
    ) -> None:
        """Register one algorithm factory and its YAML config decoder under ``name``."""
        self._register(self._algorithms, "algorithm", name, factory)
        if not callable(config_decoder):
            raise ValueError("algorithm registration requires a config decoder")
        self._algorithm_decoders[name] = config_decoder

    def decode_algorithm_config(self, name: str, payload: object) -> object:
        """Decode and validate ``algorithm_config`` for a registered algorithm."""
        result = self._resolve(self._algorithm_decoders, "algorithm", name)(payload)
        validate = getattr(result, "validate", None)
        if not callable(validate):
            raise TypeError("algorithm config decoder returned an invalid typed configuration")
        validate()
        return result

    def register_model(self, name: str, factory: Callable[[TrainingContext], ModelFactory]) -> None:
        """Register one model factory under ``name``."""
        self._register(self._models, "model", name, factory)

    def register_checkpoint_loader(self, name: str, factory: Callable[[], CheckpointLoader]) -> None:
        """Register one checkpoint loader factory under ``name``."""
        self._register(self._loaders, "checkpoint loader", name, factory)

    def build_algorithm(self, name: str, context: TrainingContext) -> AlgorithmFactory:
        """Resolve and validate the algorithm factory registered as ``name``."""
        result = self._resolve(self._algorithms, "algorithm", name)(context)
        if not isinstance(result, AlgorithmFactory) or result.algorithm_id != name:
            raise TypeError("algorithm factory returned an invalid implementation")
        if not result.observation_schema.strip() or not result.action_schema.strip():
            raise ValueError("algorithm schemas cannot be empty")
        if not isinstance(result.trainable, (str, type)) or isinstance(result.trainable, str) and not result.trainable.strip():
            raise ValueError("algorithm trainable cannot be empty")
        if not result.checkpoint_loader_id.strip():
            raise ValueError("algorithm checkpoint loader cannot be empty")
        return result

    def build_model(self, name: str, context: TrainingContext) -> ModelFactory:
        """Resolve and validate the model factory registered as ``name``."""
        result = self._resolve(self._models, "model", name)(context)
        if not isinstance(result, ModelFactory) or result.model_id != name:
            raise TypeError("model factory returned an invalid implementation")
        if not result.observation_schema.strip() or not result.action_schema.strip():
            raise ValueError("model schemas cannot be empty")
        if not isinstance(result.rl_module_class, type):
            raise TypeError("model factory must declare an RLModule class")
        return result

    def build_checkpoint_loader(self, name: str) -> CheckpointLoader:
        """Instantiate and validate the checkpoint loader registered as ``name``."""
        result = self._resolve(self._loaders, "checkpoint loader", name)()
        if not isinstance(result, CheckpointLoader) or result.loader_id != name:
            raise TypeError("checkpoint loader factory returned an invalid implementation")
        return result


def create_training_registry() -> TrainingRegistry:
    """Return a fresh registry with built-in masked PPO, model, and loader entries."""
    from .ppo.checkpoint import BuiltinCheckpointLoader
    from .ppo.train import MaskedPPOFactory

    registry = TrainingRegistry()
    masked_ppo = MaskedPPOFactory()
    registry.register_algorithm("masked_ppo", lambda context: masked_ppo, config_decoder=masked_ppo.decode_config)
    registry.register_model("local_cnn_actor_critic", lambda context: _BuiltinModelFactory())
    registry.register_checkpoint_loader("rllib_masked_ppo", BuiltinCheckpointLoader)
    return registry


class _BuiltinModelFactory:
    """Built-in ``local_cnn_actor_critic`` model registration for masked PPO."""

    model_id = "local_cnn_actor_critic"
    observation_schema = "aresim.obs.local.v1"
    action_schema = "aresim.action.rover.v1"

    @property
    def rl_module_class(self) -> type:
        from .ppo.train import AresMaskedPPORLModule

        return AresMaskedPPORLModule


__all__ = [
    "AlgorithmFactory",
    "CheckpointLoader",
    "ModelFactory",
    "TrainingContext",
    "TrainingRegistry",
    "create_training_registry",
]
