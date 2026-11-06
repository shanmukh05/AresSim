"""Allowlisted runtime bundles shared by cloud providers.

Only Python source, project metadata, YAML configuration and named launch scripts
leave the checkout. Local credentials, cached data and training outputs are excluded.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


def stage_payload(root: Path, dest: Path, *, provider: str = "jarvislabs") -> None:
    """Stage an inspectable runtime payload without following source symlinks."""
    if provider not in {"jarvislabs", "vastai"}:
        raise ValueError("unsupported cloud provider")
    if dest.exists():
        raise FileExistsError(f"staging destination already exists: {dest}")
    files = [root / "engine/pyproject.toml", root / "scripts/cloud/remote_setup.sh"]
    files.extend((root / "engine/aresim").rglob("*.py"))
    files.extend((root / "configs").rglob("*.yaml"))
    files.extend((root / "configs").rglob("*.yml"))
    if provider == "jarvislabs":
        files.append(root / "scripts/jarvislabs/remote_setup.sh")
    else:
        files.append(root / "scripts/vastai/supervisor.py")
    # Old configurations can still name the two historical notebook manifests.
    for name in ("phase1_open_exploration_split_v1.yaml", "phase1_smoke_eval_v1.yaml"):
        path = root / "notebooks" / name
        if path.is_file():
            files.append(path)
    inventory = {}
    for source in sorted(set(files)):
        relative = source.relative_to(root)
        if any(part.startswith(".") or part in {"__pycache__", "node_modules"} for part in relative.parts):
            continue
        if source.is_symlink() or any(parent.is_symlink() for parent in source.parents if parent != root):
            raise ValueError(f"source symlink is not allowed: {relative}")
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        inventory[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    (dest / "source_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
