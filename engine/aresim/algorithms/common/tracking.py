"""W&B run identity helpers shared across algorithm training entry points.

Each training invocation gets a fresh run id because W&B permanently rejects ids
from deleted runs. ``config_hash`` remains separate for reproducibility.
"""

from __future__ import annotations

import json
import secrets
from pathlib import Path

WANDB_RUN_ID_LENGTH = 24


def new_wandb_run_id() -> str:
    """Return a fresh W&B run id (24 lowercase hex characters)."""
    return secrets.token_hex(WANDB_RUN_ID_LENGTH // 2)


def load_wandb_run_id(run_directory: Path) -> str | None:
    """Read ``wandb_run_id`` from one run manifest, or ``None`` if absent or invalid."""
    manifest_path = run_directory / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    run_id = payload.get("wandb_run_id")
    if isinstance(run_id, str) and len(run_id) == WANDB_RUN_ID_LENGTH:
        return run_id
    return None


def resolve_wandb_run_id(run_directory: Path, wandb_run_id: str | None = None) -> str:
    """Return an explicit, persisted, or newly generated W&B run id."""
    if wandb_run_id:
        return wandb_run_id
    existing = load_wandb_run_id(run_directory)
    if existing:
        return existing
    return new_wandb_run_id()
