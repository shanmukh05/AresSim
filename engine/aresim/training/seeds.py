"""Versioned deterministic seed splits for training and evaluation.

Manifests keep train, validation, and test environment seeds disjoint. Checked-in
YAML lives in repository ``notebooks/``, not the installable package.

**Last updated:** September 1, 2026

**Contains:** ``SeedSplitManifest``, :func:`load_seed_manifest`,
:func:`resolve_checkout_file`.

**Schema:** ``aresim.seed_split.v1``.

**Used by:** PPO training env cycling and :mod:`aresim.training.evaluation`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .runner import EpisodeSpec


SEED_SPLIT_SCHEMA = "aresim.seed_split.v1"


@dataclass(frozen=True)
class SeedSplitManifest:
    """Disjoint environment seeds and deterministic evaluation identities."""

    manifest_id: str
    scenario_id: str
    task_id: str
    observation_schema: str
    action_schema: str
    reward_profile: str
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    agent_seed_offset: int

    def validate(self) -> None:
        identifiers = (self.manifest_id, self.scenario_id, self.task_id, self.observation_schema, self.action_schema, self.reward_profile)
        if not all(identifiers):
            raise ValueError("seed manifest identifiers cannot be empty")
        _validate_splits(self.train, self.validation, self.test)

    def episodes(self, split: str) -> tuple[EpisodeSpec, ...]:
        """Return stable rollout episode definitions for validation or test."""
        if split not in {"validation", "test"}:
            raise ValueError("evaluation split must be validation or test")
        seeds = getattr(self, split)
        return tuple(EpisodeSpec(f"{split}_{index:03d}", seed, self.agent_seed_offset + seed) for index, seed in enumerate(seeds))


def _validate_splits(*splits: tuple[int, ...]) -> None:
    if any(not split for split in splits):
        raise ValueError("seed manifest splits cannot be empty")
    groups = tuple(set(split) for split in splits)
    if _splits_overlap(groups):
        raise ValueError("seed manifest splits must be disjoint")
    if any(_invalid_seed(seed) for group in groups for seed in group):
        raise ValueError("seed manifest values must be non-negative integers")


def _splits_overlap(groups: tuple[set[int], ...]) -> bool:
    return any(not left.isdisjoint(right) for index, left in enumerate(groups) for right in groups[index + 1:])


def _invalid_seed(seed: object) -> bool:
    return isinstance(seed, bool) or not isinstance(seed, int) or seed < 0


def resolve_checkout_file(configured: str | Path, *, kind: str = "repository file") -> Path:
    """Resolve a checkout file against cwd, the engine tree, and the repository root."""
    path = Path(configured)
    engine_root = Path(__file__).resolve().parents[2]
    for candidate in (path, engine_root / path, engine_root.parent / path):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"{kind} does not exist: {configured}")


def load_seed_manifest(path: str | Path) -> SeedSplitManifest:
    """Load a safe YAML seed manifest and reject unknown fields."""
    import yaml

    payload = yaml.safe_load(resolve_checkout_file(path, kind="seed manifest").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SEED_SPLIT_SCHEMA:
        raise ValueError("unsupported seed split manifest")
    allowed = {"schema_version", "manifest_id", "scenario_id", "task_id", "observation_schema", "action_schema", "reward_profile", "train", "validation", "test", "agent_seed_offset"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown seed manifest fields: {', '.join(unknown)}")
    manifest = SeedSplitManifest(
        manifest_id=str(payload["manifest_id"]), scenario_id=str(payload["scenario_id"]), task_id=str(payload["task_id"]),
        observation_schema=str(payload["observation_schema"]), action_schema=str(payload["action_schema"]), reward_profile=str(payload["reward_profile"]),
        train=tuple(payload["train"]), validation=tuple(payload["validation"]), test=tuple(payload["test"]),
        agent_seed_offset=int(payload["agent_seed_offset"]),
    )
    manifest.validate()
    return manifest


__all__ = ["SEED_SPLIT_SCHEMA", "SeedSplitManifest", "load_seed_manifest", "resolve_checkout_file"]
