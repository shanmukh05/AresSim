"""Atomic metadata and checksum inventories shared by native trainers.

These helpers preserve each backend's sidecar format; native model formats remain backend owned.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def atomic_json(path: Path, payload: object) -> None:
    """Publish complete JSON only after writing a neighboring temporary file."""
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_inventory(directory: Path) -> list[dict[str, object]]:
    """Return relative file names, sizes, and SHA-256 digests in stable order."""
    return [{"path": item.relative_to(directory).as_posix(), "size": item.stat().st_size,
             "sha256": hashlib.sha256(item.read_bytes()).hexdigest()}
            for item in sorted(directory.rglob("*")) if item.is_file()]


def validate_inventory(directory: Path, inventory: object) -> None:
    """Reject malformed, escaping, missing, or modified native checkpoint files."""
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("checkpoint inventory is empty")
    names = set()
    for entry in inventory:
        name = _validate_file(directory, entry)
        if name in names:
            raise ValueError("duplicate checkpoint file")
        names.add(name)


def _validate_file(directory: Path, entry: object) -> str:
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise ValueError("invalid checkpoint inventory")
    relative = Path(entry["path"])
    item = (directory / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not item.is_relative_to(directory.resolve()):
        raise ValueError("unsafe checkpoint file")
    if not item.is_file() or item.stat().st_size != entry.get("size") or hashlib.sha256(item.read_bytes()).hexdigest() != entry.get("sha256"):
        raise ValueError(f"checkpoint checksum mismatch: {relative}")
    return relative.as_posix()


def numbered_checkpoints(directory: Path) -> list[Path]:
    """Sort complete numbered exports by real steps instead of lexicographic width."""
    return sorted((p for p in (directory / "checkpoints").glob("step_*")
                   if p.name[5:].isdigit() and (p / "checkpoint.json").is_file()),
                  key=lambda p: int(p.name[5:]))


def latest_checkpoint(path: str | Path) -> Path:
    """Resolve a native run directory, export directory, or sidecar path."""
    requested = Path(path).expanduser().resolve()
    if requested.is_file():
        return requested
    if (requested / "checkpoint.json").is_file():
        return requested / "checkpoint.json"
    candidates = [p / "checkpoint.json" for p in numbered_checkpoints(requested)]
    final = requested / "checkpoints/final/checkpoint.json"
    if final.is_file():
        candidates.append(final)
    if not candidates:
        raise FileNotFoundError(f"no checkpoint found at {requested}")
    return max(candidates, key=lambda p: (int(json.loads(p.read_text())["environment_steps"]), p.parent.name == "final"))
