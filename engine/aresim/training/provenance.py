"""Freeze numerical environment settings, seed contents and semantic source hashes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

from ..config import EnvironmentConfig, TaskConfig
from ..defaults import DEFAULT_ENVIRONMENT_CONFIG


def serialize(value):
    if is_dataclass(value):
        return {field.name: serialize(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {serialize(key): serialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serialize(item) for item in value]
    return value


def _decode(kind, value):
    if is_dataclass(kind):
        if not isinstance(value, dict) or set(value) - {field.name for field in fields(kind)}:
            raise ValueError(f"invalid frozen {kind.__name__} configuration")
        types = get_type_hints(kind)
        return kind(**{key: _decode(types[key], item) for key, item in value.items()})
    origin, arguments = get_origin(kind), get_args(kind)
    if origin is tuple:
        return tuple(_decode(arguments[0], item) for item in value)
    if origin in (dict, Mapping):
        return {_decode(arguments[0], key): _decode(arguments[1], item) for key, item in value.items()}
    if isinstance(kind, type) and issubclass(kind, Enum):
        return kind(value)
    return value


def environment_from_payload(payload):
    config = _decode(EnvironmentConfig, payload)
    config.validate()
    return config


def environment_config(spec):
    values = spec.environment
    if values.resolved_config is not None:
        config = environment_from_payload(values.resolved_config)
        for name in ("scenario_id", "observation", "action", "reward", "task"):
            if getattr(config, name) != getattr(values, name):
                raise ValueError(f"frozen environment {name} disagrees with experiment")
        if config.reward_config.gamma != spec.algorithm_config.gamma:
            raise ValueError("potential and learner discounts disagree")
        payload = serialize(config)
        for group, overrides in (("engine", values.engine_overrides),
                                 ("observation_config", values.observation_overrides),
                                 ("reward_config", values.reward_overrides)):
            expected = json.loads(json.dumps(payload[group]))
            _merge(expected, overrides)
            if expected != payload[group]:
                raise ValueError(f"{group} override disagrees with frozen snapshot; start a fresh experiment")
        if any(payload["task_config"].get(name) != value for name, value in values.task_config.items()):
            raise ValueError("task override disagrees with frozen snapshot; start a fresh experiment")
        return config
    config = replace(DEFAULT_ENVIRONMENT_CONFIG,
        **{name: getattr(values, name) for name in ("scenario_id", "observation", "action", "reward", "task")},
        reward_config=replace(DEFAULT_ENVIRONMENT_CONFIG.reward_config, gamma=spec.algorithm_config.gamma),
        task_config=TaskConfig(**values.task_config))
    payload = serialize(config)
    _merge(payload["engine"], values.engine_overrides)
    _merge(payload["observation_config"], values.observation_overrides)
    _merge(payload["reward_config"], values.reward_overrides)
    resolved = environment_from_payload(payload)
    if resolved.reward_config.gamma != spec.algorithm_config.gamma:
        raise ValueError("potential and learner discounts disagree")
    return resolved


def _merge(target, changes):
    if not isinstance(changes, dict):
        raise ValueError("environment overrides must be mappings")
    for key, value in changes.items():
        if key not in target:
            raise ValueError(f"unknown environment override: {key}")
        if isinstance(target[key], dict):
            _merge(target[key], value)
        else:
            target[key] = value


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    # Only environment semantics: unrelated report/document edits cannot invalidate a run.
    files = [root / "config.py", root / "defaults.py", root / "types.py", root / "registry.py"]
    for directory in ("core", "components", "envs", "algorithms/baselines"):
        files.extend((root / directory).glob("*.py"))
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}


def freeze_experiment(spec, *, component_registry=None):
    from .seeds import load_seed_manifest
    if spec.provenance is not None:
        verify_provenance(spec, training=True)
        return spec
    config = environment_config(spec)
    seeds = load_seed_manifest(spec.evaluation.seed_manifest)
    seed_payload = {"schema_version": "aresim.seed_split.v1", **serialize(seeds)}
    # Seed manifests define split families, not the new observation/reward contract.
    from ..factory import make_env
    identifiers = make_env(config, registry=component_registry, audit=False).reset(seed=spec.learner_seed).info
    seed_payload.update({key: identifiers[key] for key in ("observation_schema", "action_schema", "reward_profile")})
    provenance = {"schema_version": "aresim.provenance.v1", "environment_sources": source_hashes(),
                  "seed_manifest": seed_payload, "seed_sha256": digest(seed_payload)}
    root = Path(__file__).resolve().parents[1]
    provenance["runtime_sources"] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in sorted(root.rglob("*.py"))}
    import subprocess
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
    provenance.update(git_revision=revision.stdout.strip() if revision.returncode == 0 else None,
                      git_dirty=bool(status.stdout.strip()))
    return replace(spec, environment=replace(spec.environment, resolved_config=serialize(config)), provenance=provenance)


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def verify_provenance(spec, *, allow_environment_mismatch=False, training=False):
    if spec.provenance is None or spec.environment.resolved_config is None:
        if allow_environment_mismatch:
            return False
        raise ValueError("legacy checkpoint has no frozen environment; pass --allow-environment-mismatch for an explicitly incomparable diagnostic evaluation")
    metadata = spec.provenance
    if not isinstance(metadata, dict) or metadata.get("schema_version") != "aresim.provenance.v1":
        raise ValueError("invalid experiment provenance schema")
    if metadata.get("seed_sha256") != digest(metadata.get("seed_manifest")):
        raise ValueError("frozen seed manifest checksum mismatch")
    environment_config(spec)
    matches = metadata.get("environment_sources") == source_hashes()
    if not matches and not allow_environment_mismatch:
        raise ValueError("environment source differs from checkpoint; pass --allow-environment-mismatch for diagnostic evaluation")
    if training:
        root = Path(__file__).resolve().parents[1]
        actual = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(root.rglob("*.py"))}
        if metadata.get("runtime_sources") != actual:
            raise ValueError("runtime source differs from checkpoint; training resume requires unchanged code")
    return matches


def experiment_seeds(spec):
    from .seeds import load_seed_manifest, parse_seed_manifest
    return (parse_seed_manifest(spec.provenance["seed_manifest"]) if spec.provenance else
            load_seed_manifest(spec.evaluation.seed_manifest))

def validate_checkpoint_contract(payload):
    """Require metadata and the hashed experiment to describe the same model."""
    from .experiments import parse_experiment
    spec = parse_experiment(payload["experiment"])
    expected_schema = {"local": "aresim.obs.local.v1", "local_memory": "aresim.obs.local_memory.v2"}.get(spec.environment.observation)
    if (spec.algorithm != payload["algorithm_id"] or spec.model != payload["model_id"]
            or expected_schema != payload["observation_schema"]):
        raise ValueError("checkpoint metadata disagrees with its hashed experiment")
    return spec
