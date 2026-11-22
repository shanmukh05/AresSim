"""Upload the curated best AresSim RL checkpoints to Hugging Face Hub.

Reads `HF_TOKEN` from the process environment or `.env.local` (never prints credentials)
and uploads each checkpoint folder preserving the exact relative paths expected by
`aresim.integrations.policy.DEFAULT_BEST_CHECKPOINTS`.
"""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "tensorfiend/aressim-checkpoints"

CHECKPOINT_DIRS: dict[str, str] = {
    "masked_ppo": "ppo/seed_37/checkpoints/final",
    "masked_dqn": "double_duelling_dqn/seed_3_sample_credit_030/checkpoints/step_985600",
    "recurrent_dqn": "recurrent_dqn/seed_11/checkpoints/step_1966080",
    "muzero": "muzero/seed_7_sample_credit_030_parallel_lossnorm/checkpoints/step_2048000",
    "dreamerv3": "dreamerv3/seed_11/checkpoints/step_2097152",
    "plan2explore": "plan2explore/adapt/seed_11_loopfix/checkpoints/step_1048576",
}


def resolve_hf_token() -> str:
    """Resolve HF_TOKEN from environment or checkout .env.local without shell execution."""
    direct = os.environ.get("HF_TOKEN", "").strip()
    if direct:
        return direct
    for candidate in (REPO_ROOT / ".env.local", REPO_ROOT / ".env"):
        if not candidate.is_file():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip().removeprefix("export ")
            key, sep, value = line.partition("=")
            if sep and key.strip() == "HF_TOKEN":
                parts = shlex.split(value.strip(), comments=True, posix=True)
                if len(parts) == 1 and parts[0]:
                    return parts[0]
    raise RuntimeError("HF_TOKEN not found in environment or .env.local")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload best AresSim checkpoints to Hugging Face Hub.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help=f"Target HF model repo (default: {DEFAULT_REPO_ID})")
    parser.add_argument("--private", action="store_true", help="Create repo as private instead of public")
    args = parser.parse_args()

    token = resolve_hf_token()
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    print(f"Target Hugging Face repo ready: https://huggingface.co/{args.repo_id}")

    for algo, rel_dir in CHECKPOINT_DIRS.items():
        local_dir = (REPO_ROOT / "results" / rel_dir).resolve()
        if not (local_dir / "checkpoint.json").is_file():
            raise FileNotFoundError(f"Missing checkpoint sidecar for {algo}: {local_dir / 'checkpoint.json'}")
        print(f"Uploading {algo} ({rel_dir})...", flush=True)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(local_dir),
            path_in_repo=rel_dir,
            commit_message=f"Upload best {algo} checkpoint ({rel_dir})",
        )
        print(f"  done: {algo}", flush=True)

    print("All 6 best checkpoints uploaded cleanly.")


if __name__ == "__main__":
    main()
