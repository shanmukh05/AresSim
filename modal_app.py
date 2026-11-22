"""Modal deployment for the AresSim FastAPI engine and RL/LLM-DM inference server.

Mounts a persistent Modal Volume (`/checkpoints`) synced from Hugging Face Hub,
serves `aresim.api:create_app()` on a T4 container in deployment mode (`ARESSIM_DEPLOYED=1`),
and keeps `max_containers=1` with concurrent request handling so multi-user in-memory
sessions stay pinned to the active container.

Deploy:
    modal run modal_app.py::sync_checkpoints
    modal deploy modal_app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = "aressim-inference"
CHECKPOINTS_VOLUME_NAME = "aressim-checkpoints"
CHECKPOINTS_MOUNT_PATH = "/checkpoints"
HF_REPO_ID = os.environ.get("ARESSIM_HF_CHECKPOINTS_REPO", "tensorfiend/aressim-checkpoints")
DEFAULT_GPU = os.environ.get("ARESSIM_MODAL_GPU", "T4")

app = modal.App(APP_NAME)
checkpoints_volume = modal.Volume.from_name(CHECKPOINTS_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("gcc", "g++", "libc6-dev")
    .uv_pip_install(
        "fastapi>=0.115,<1",
        "uvicorn>=0.34,<1",
        "numpy>=2,<3",
        "gymnasium==1.2.2",
        "PyYAML>=6.0.3,<7",
        "jsonschema>=4.23,<5",
        "torch==2.13.0",
        "ray[rllib]==2.58.0",
        "huggingface_hub>=0.28,<2",
    )
    .env({"PYTHONPATH": "/root"})
    .add_local_dir("engine/aresim", remote_path="/root/aresim")
)


def _ensure_checkpoints_present(repo_id: str) -> None:
    """Download curated checkpoints into the mounted Modal Volume if missing."""
    from huggingface_hub import snapshot_download

    target = Path(CHECKPOINTS_MOUNT_PATH)
    marker = target / "ppo" / "seed_37" / "checkpoints" / "final" / "checkpoint.json"
    if marker.is_file():
        return
    print(f"Downloading checkpoints from {repo_id} into {CHECKPOINTS_MOUNT_PATH}...", flush=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=CHECKPOINTS_MOUNT_PATH,
        token=os.environ.get("HF_TOKEN") or None,
    )
    checkpoints_volume.commit()
    print("Checkpoints synced and committed to Modal Volume.", flush=True)


@app.function(
    image=image,
    volumes={CHECKPOINTS_MOUNT_PATH: checkpoints_volume},
    timeout=1800,
)
def sync_checkpoints(repo_id: str = HF_REPO_ID, force: bool = False) -> dict[str, object]:
    """Explicitly pull/refresh checkpoints from Hugging Face Hub into the Modal Volume."""
    from huggingface_hub import snapshot_download

    if force or not (Path(CHECKPOINTS_MOUNT_PATH) / "ppo" / "seed_37" / "checkpoints" / "final" / "checkpoint.json").is_file():
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=CHECKPOINTS_MOUNT_PATH,
            token=os.environ.get("HF_TOKEN") or None,
        )
        checkpoints_volume.commit()
    from aresim.integrations.policy import DEFAULT_BEST_CHECKPOINTS

    present = {
        algo: (Path(CHECKPOINTS_MOUNT_PATH) / rel).is_file()
        for algo, rel in DEFAULT_BEST_CHECKPOINTS.items()
    }
    return {"repoId": repo_id, "present": present}


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    volumes={CHECKPOINTS_MOUNT_PATH: checkpoints_volume},
    env={
        "ARESSIM_DEPLOYED": "1",
        "ARESSIM_CHECKPOINTS_DIR": CHECKPOINTS_MOUNT_PATH,
        "ARESSIM_HF_CHECKPOINTS_REPO": HF_REPO_ID,
    },
    max_containers=1,
    scaledown_window=300,
    timeout=600,
)
@modal.concurrent(max_inputs=32)
@modal.asgi_app(requires_proxy_auth=True)
def web():
    """Serve the AresSim FastAPI app with multi-user sessions and best RL checkpoints."""
    _ensure_checkpoints_present(os.environ.get("ARESSIM_HF_CHECKPOINTS_REPO", HF_REPO_ID))
    from aresim.api import create_app

    return create_app()
