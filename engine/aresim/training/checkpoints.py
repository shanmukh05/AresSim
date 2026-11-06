"""Framework-neutral dispatch from portable checkpoint sidecars to registered agents."""
from __future__ import annotations

import json
from pathlib import Path


RLLIB_CHECKPOINT_SCHEMA = "aresim.checkpoint.rllib.v1"


def make_checkpoint_agent(path: str | Path, registry=None, *, deterministic: bool = True):
    """Load the sidecar's registered backend without importing unrelated frameworks."""
    from ..algorithms.registry import create_training_registry

    selected = create_training_registry() if registry is None else registry
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return selected.build_checkpoint_loader(str(payload.get("loader_id", ""))).load(
        str(path), deterministic=deterministic)
