"""Register learned-policy algorithms, models, checkpoint loaders, and lazy training runners.

Owns the training-side extension registry separate from component registration in
:mod:`aresim.registry`. New algorithms register here with explicit names; there is
no automatic discovery.

**Last updated:** October 16, 2026

**Contains:** ``TrainingRegistry``, ``TrainingContext``, factory protocols
(``AlgorithmFactory``, ``ModelFactory``, ``CheckpointLoader``), and
:func:`create_training_registry`.

**Built-ins:** PPO, feedforward/recurrent DQN, MuZero, DreamerV3 and Plan2Explore.
Each registration owns config decoding, model pairing, runner dispatch and dependency metadata.

**Re-exported from:** :mod:`aresim.registry` (lazy) and :mod:`aresim.training`.

**See also:** :mod:`aresim.algorithms.ppo` and :mod:`aresim.algorithms.dqn`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
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
    model_class: type


@runtime_checkable
class CheckpointLoader(Protocol):
    """Load a frozen native checkpoint as a framework-neutral :class:`Agent`."""

    loader_id: str

    def load(self, path: str, *, deterministic: bool = True) -> Agent[Any, Any]:
        """Restore inference from a sidecar JSON path plus native framework files."""
        ...




class TrainingRunner(Protocol):
    """Run one resolved experiment and return its artifact directory without implicit evaluation."""

    def __call__(self, spec: ExperimentSpec, registry=None, *, report: bool = False,
                 component_registry=None, resume_from: str | Path | None = None) -> Path:
        ...


@dataclass(frozen=True)
class AlgorithmRegistration:
    """Lazy framework entry points and strict configuration decoders for an algorithm."""

    config_decoder: Callable[[object], object]
    model_decoder: Callable[[object], object] | None = None
    resource_decoder: Callable[[object], object] | None = None
    runner: TrainingRunner | None = None
    model_id: str | None = None
    install_extra: str = "rllib"
    checkpoint_loader_id: str = ""


def _entrypoint(module: str, name: str) -> Callable:
    """Defer optional framework imports until the selected operation is invoked."""
    def invoke(*args, **kwargs):
        return getattr(import_module(module), name)(*args, **kwargs)
    return invoke


@dataclass
class BuiltinModelFactory:
    """Describe a model without importing its numerical framework during parsing."""

    model_id: str
    module: str
    class_name: str
    observation_schema: str = "aresim.obs.local.v1"
    action_schema: str = "aresim.action.rover.v1"

    @property
    def model_class(self) -> type:
        return getattr(import_module(self.module), self.class_name)

    @property
    def rl_module_class(self) -> type:
        """Compatibility name retained for existing RLlib factories."""
        return self.model_class



@dataclass
class LegacyModelAdapter:
    """Preserve older rl_module_class factories, including frozen custom implementations."""

    implementation: object
    model_id: str
    observation_schema: str
    action_schema: str
    model_class: type

    @property
    def rl_module_class(self) -> type:
        return self.model_class

    def __getattr__(self, name: str):
        return getattr(self.implementation, name)

class TrainingRegistry:
    """Store explicit algorithm, model, and checkpoint-loader factories by name.

    Rejects duplicate names and validates built implementations against their
    registered identifiers and schema strings at resolve time.
    """

    def __init__(self) -> None:
        self._algorithms: dict[str, Callable[[TrainingContext], AlgorithmFactory]] = {}
        self._registrations: dict[str, AlgorithmRegistration] = {}
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
        model_decoder: Callable[[object], object] | None = None,
        resource_decoder: Callable[[object], object] | None = None,
        runner: TrainingRunner | None = None,
        model_id: str | None = None,
        install_extra: str = "rllib",
        checkpoint_loader_id: str = "",
    ) -> None:
        """Register one algorithm factory and its YAML config decoder under ``name``."""
        self._register(self._algorithms, "algorithm", name, factory)
        if not callable(config_decoder):
            raise ValueError("algorithm registration requires a config decoder")
        self._registrations[name] = AlgorithmRegistration(
            config_decoder, model_decoder, resource_decoder, runner, model_id,
            install_extra, checkpoint_loader_id)

    def registration(self, name: str) -> AlgorithmRegistration:
        """Return metadata without importing the optional training backend."""
        return self._resolve(self._registrations, "algorithm", name)

    def decode_model_config(self, name: str, payload: object) -> object:
        from .common.model_config import ModelConfig
        from .common.config_decode import decode_dataclass
        decoder = self.registration(name).model_decoder
        return decoder(payload) if decoder else decode_dataclass(ModelConfig, payload, "model_config")

    def decode_resources(self, name: str, payload: object) -> object:
        from ..training.experiments import RLlibResourceConfig
        from .common.config_decode import decode_dataclass
        decoder = self.registration(name).resource_decoder
        return decoder(payload) if decoder else decode_dataclass(RLlibResourceConfig, payload, "resources")

    def run(self, spec, **kwargs):
        """Dispatch a resolved experiment using its registration-owned runner."""
        entry = self.registration(spec.algorithm)
        runner = entry.runner or _entrypoint("aresim.training.rllib.runner", "run_experiment")
        return runner(spec, registry=self, **kwargs)

    def checkpoint_extra(self, loader_id: str) -> str:
        for entry in self._registrations.values():
            if entry.checkpoint_loader_id == loader_id:
                return entry.install_extra
        raise ValueError(f"unknown checkpoint loader: {loader_id}")

    def decode_algorithm_config(self, name: str, payload: object) -> object:
        """Decode and validate ``algorithm_config`` for a registered algorithm."""
        result = self.registration(name).config_decoder(payload)
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
        entry = self.registration(name)
        if entry.model_id is not None and context.experiment.model != entry.model_id:
            raise ValueError(f"{name} requires model {entry.model_id}")
        result = self._resolve(self._algorithms, "algorithm", name)(context)
        if not isinstance(result, AlgorithmFactory) or result.algorithm_id != name:
            raise TypeError("algorithm factory returned an invalid implementation")
        _validate_algorithm_metadata(result)
        return result

    def build_model(self, name: str, context: TrainingContext) -> ModelFactory:
        """Resolve and validate the model factory registered as ``name``."""
        result = self._resolve(self._models, "model", name)(context)
        if not hasattr(result, "model_class") and hasattr(result, "rl_module_class"):
            result = LegacyModelAdapter(result, result.model_id, result.observation_schema, result.action_schema, result.rl_module_class)
        if not isinstance(result, ModelFactory) or result.model_id != name:
            raise TypeError("model factory returned an invalid implementation")
        if not result.observation_schema.strip() or not result.action_schema.strip():
            raise ValueError("model schemas cannot be empty")
        if not isinstance(result.model_class, type):
            raise TypeError("model factory must declare a model class")
        return result

    def build_checkpoint_loader(self, name: str) -> CheckpointLoader:
        """Instantiate and validate the checkpoint loader registered as ``name``."""
        result = self._resolve(self._loaders, "checkpoint loader", name)()
        if not isinstance(result, CheckpointLoader) or result.loader_id != name:
            raise TypeError("checkpoint loader factory returned an invalid implementation")
        return result


def _validate_algorithm_metadata(result: AlgorithmFactory) -> None:
    if not result.observation_schema.strip() or not result.action_schema.strip():
        raise ValueError("algorithm schemas cannot be empty")
    if not isinstance(result.trainable, (str, type)) or isinstance(result.trainable, str) and not result.trainable.strip():
        raise ValueError("algorithm trainable cannot be empty")
    if not result.checkpoint_loader_id.strip():
        raise ValueError("algorithm checkpoint loader cannot be empty")


def create_training_registry() -> TrainingRegistry:
    """Register built-ins explicitly, keeping Torch and Ray behind lazy entry points."""
    from .common.config_decode import decode_dataclass
    from .ppo.config import MaskedPPOConfig
    from .dqn.config import MaskedDQNConfig
    from .muzero.config import MuZeroConfig, MuZeroModelConfig, MuZeroResourceConfig
    from .recurrent_dqn.config import RecurrentDQNConfig, RecurrentModelConfig
    from .dreamerv3.config import DreamerConfig, DreamerModelConfig, WorldModelResources
    from .plan2explore.config import Plan2ExploreConfig

    registry = TrainingRegistry()
    def decoder(cls, label):
        return lambda raw: decode_dataclass(cls, raw, label)

    definitions = (
        ("masked_ppo", "ppo", "MaskedPPOFactory", MaskedPPOConfig, None, None, "local_cnn_actor_critic", "ppo.train", "AresMaskedPPORLModule", "rllib_masked_ppo", "rllib", None),
        ("masked_dqn", "dqn", "MaskedDQNFactory", MaskedDQNConfig, None, None, "local_cnn_q", "dqn.train", "AresMaskedDQNRLModule", "rllib_masked_dqn", "rllib", None),
        ("muzero", "muzero", "MuZeroFactory", MuZeroConfig, MuZeroModelConfig, MuZeroResourceConfig, "local_muzero", "muzero.model", "MuZeroNetwork", "torch_muzero", "muzero", "aresim.algorithms.muzero.train"),
        ("recurrent_dqn", "recurrent_dqn", "RecurrentDQNFactory", RecurrentDQNConfig, RecurrentModelConfig, None, "local_cnn_lstm_q", "recurrent_dqn.train", "AresRecurrentDQNRLModule", "rllib_recurrent_dqn", "rllib", None),
        ("dreamerv3", "dreamerv3", "DreamerFactory", DreamerConfig, DreamerModelConfig, WorldModelResources, "local_rssm", "dreamerv3.model", "WorldModel", "torch_dreamerv3", "world_models", "aresim.training.world_models.runner"),
        ("plan2explore", "plan2explore", "Plan2ExploreFactory", Plan2ExploreConfig, DreamerModelConfig, WorldModelResources, "local_rssm_disagreement", "dreamerv3.model", "WorldModel", "torch_plan2explore", "world_models", "aresim.training.world_models.runner"),
    )
    for name, package, factory, algo_cls, model_cls, resources_cls, model_id, model_module, model_class, loader, extra, runner in definitions:
        build = _entrypoint(f"aresim.algorithms.{package}.train", factory)
        def algorithm(context, build=build):
            return _with_observation(build(), context)
        registry.register_algorithm(name, algorithm,
            config_decoder=decoder(algo_cls, "algorithm_config"),
            model_decoder=decoder(model_cls, "model_config") if model_cls else None,
            resource_decoder=decoder(resources_cls, "resources") if resources_cls else None,
            runner=_entrypoint(runner, "run_experiment") if runner else None,
            # Legacy RLlib algorithms retain schema-compatible custom model selection.
            model_id=None if name in {"masked_ppo", "masked_dqn"} else model_id,
            install_extra=extra, checkpoint_loader_id=loader)
        def model(context, model_id=model_id, model_module=model_module, model_class=model_class):
            return _with_observation(BuiltinModelFactory(model_id, f"aresim.algorithms.{model_module}", model_class), context)
        registry.register_model(model_id, model)
        registry.register_checkpoint_loader(loader, _entrypoint(f"aresim.algorithms.{package}.checkpoint", "BuiltinCheckpointLoader"))
    return registry


def _with_observation(factory, context):
    factory.observation_schema = ("aresim.obs.local_memory.v2" if context.experiment.environment.observation == "local_memory" else "aresim.obs.local.v1")
    return factory


__all__ = ["AlgorithmFactory", "AlgorithmRegistration", "CheckpointLoader", "ModelFactory", "TrainingContext", "TrainingRunner", "TrainingRegistry", "create_training_registry"]
