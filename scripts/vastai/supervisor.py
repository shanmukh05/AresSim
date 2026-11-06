#!/usr/bin/env python3
"""Bound setup/training and self-stop one Vast.ai container after artifact recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

STOP_SIGNAL = 0


def request_stop(signum: int, _frame) -> None:
    global STOP_SIGNAL
    STOP_SIGNAL = signum


def write_status(root: Path, **values: object) -> None:
    path = root / "cloud_status.json"
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update(values)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(current, indent=2) + "\n")
    temporary.replace(path)


def scoped_environment(instance_id: str) -> dict[str, str]:
    """Read only the required startup credentials; never log their values."""
    values = {name: os.environ.get(name, "") for name in ("CONTAINER_ID", "CONTAINER_API_KEY")}
    if not all(values.values()):
        for entry in Path("/proc/1/environ").read_bytes().split(b"\0"):
            key, _, value = entry.partition(b"=")
            name = key.decode(errors="replace")
            if name in values and not values[name]:
                values[name] = value.decode()
    if values["CONTAINER_ID"] != instance_id or not values["CONTAINER_API_KEY"]:
        raise RuntimeError("instance-scoped credentials are missing or instance ID differs")
    return {**os.environ, "VAST_API_KEY": values["CONTAINER_API_KEY"]}


def stop_workers(process: subprocess.Popen, grace: int) -> None:
    until = time.monotonic() + grace
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # Descendants may outlive their parent; allow the whole group its grace period.
    delay = until - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def artifact_inventory(root: Path, deadline: float) -> dict[str, str]:
    files = [root / "training.log", root / "source_inventory.json", root / "job.json"]
    allowed = {".json", ".jsonl", ".yaml", ".yml", ".pt", ".pkl", ".msgpack", ".npz", ".npy", ".csv", ".png", ".txt"}
    files.extend(path for path in (root / "results").rglob("*") if path.suffix in allowed)
    inventory = {}
    for path in sorted(files):
        if not path.is_file() or path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                if time.monotonic() >= deadline or STOP_SIGNAL:
                    raise TimeoutError("artifact hashing interrupted or recovery deadline reached")
                digest.update(chunk)
        inventory[str(path.relative_to(root))] = digest.hexdigest()
    return inventory


def self_stop(root: Path, identifier: int, control_env: dict[str, str]) -> None:
    cli_path = root / ".vast-control/bin/vastai"
    cli = str(cli_path) if cli_path.is_file() else shutil.which("vastai")
    if cli:
        stopped = subprocess.run([cli, "stop", "instance", str(identifier), "--raw"], env=control_env, capture_output=True, text=True, timeout=45)
        payload = json.loads(stopped.stdout)
        if stopped.returncode or not isinstance(payload, dict) or payload.get("success") is not True:
            raise RuntimeError("Vast CLI did not acknowledge self-stop")
    else:
        # Setup may fail before installing the CLI. The documented instance API
        # still permits scoped self-stop using only Python's standard library.
        request = urllib.request.Request(
            f"https://console.vast.ai/api/v0/instances/{identifier}",
            data=b'{"state":"stopped"}', method="PUT",
            headers={"Authorization": f"Bearer {control_env['VAST_API_KEY']}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.load(response)
        if not isinstance(payload, dict) or payload.get("success") is not True:
            raise RuntimeError("Vast API did not acknowledge self-stop")


def run(root: Path) -> int:
    job = json.loads((root / "job.json").read_text())
    control_env = scoped_environment(str(job["instance_id"]))
    clean_env = {key: value for key, value in os.environ.items() if key not in {"VAST_API_KEY", "CONTAINER_API_KEY"}}
    deadline = time.monotonic() + job["seconds"]
    write_status(root, state="running", instance_id=job["instance_id"])
    process = None
    code = 1
    try:
        command = ["bash", "-c", 'set -euo pipefail; python3 -m venv .vast-control; .vast-control/bin/pip -q install vastai; nvidia-smi; df -h .; bash scripts/cloud/remote_setup.sh "$1"; shift; exec .venv/bin/aresim-rl train "$@"', "aresim-cloud", job["extra"], *job["train_args"]]
        with (root / "training.log").open("w") as log:
            process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, env=clean_env, start_new_session=True)
            state = "running"
            while process.poll() is None and not STOP_SIGNAL and time.monotonic() < deadline:
                try:
                    process.wait(timeout=min(1, max(.01, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    continue
            if STOP_SIGNAL:
                code, state = 128 + STOP_SIGNAL, "interrupted"
            elif process.poll() is None:
                code, state = 124, "timed_out"
            else:
                code = process.returncode
                state = "completed" if code == 0 else "failed"
            stop_workers(process, job["grace_seconds"])
        write_status(root, state=state, exit_code=code)
        recovery_deadline = time.monotonic() + job["recovery_seconds"]
        inventory = artifact_inventory(root, recovery_deadline)
        (root / "artifact_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
        write_status(root, artifacts_ready=True, recovery_seconds_remaining=max(0, recovery_deadline-time.monotonic()))
        while time.monotonic() < recovery_deadline and not (root / "recovered").exists() and not STOP_SIGNAL:
            write_status(root, recovery_seconds_remaining=max(0, recovery_deadline-time.monotonic()))
            time.sleep(min(1, max(0, recovery_deadline-time.monotonic())))
    finally:
        if process is not None and process.poll() is None:
            stop_workers(process, job["grace_seconds"])
        try:
            self_stop(root, int(job["instance_id"]), control_env)
            write_status(root, shutdown="requested")
        except Exception as error:
            write_status(root, shutdown="failed", shutdown_error_type=type(error).__name__)
            print(f"AUTOMATIC STOP FAILED: instance {job['instance_id']}; client must stop compute", file=sys.stderr)
            code = code or 1
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    try:
        return run(args.root.resolve())
    except Exception as error:
        write_status(args.root, state="failed", supervisor_error_type=type(error).__name__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
