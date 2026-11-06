#!/usr/bin/env python3
"""Run bounded AresSim training on an explicitly selected Vast.ai instance.

Account operations use the official local CLI; SSH carries allowlisted source and
recovers checksum-verified artifacts. See docs/rl/vastai.md for rental and recovery.
This launcher never creates or destroys an instance.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import math
import re
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloud_payload import stage_payload

ROOT = Path(__file__).resolve().parents[2]
OFFER_FIELDS = ("id", "gpu_name", "num_gpus", "gpu_ram", "cpu_cores_effective", "cpu_name", "cpu_ram", "cuda_max_good", "disk_bw", "inet_down", "inet_up", "inet_down_cost", "inet_up_cost", "reliability", "verified", "duration", "dph_total", "storage_cost")
INSTANCE_FIELDS = ("id", "label", "actual_status", "desired_status", "gpu_name", "num_gpus", "dph_total", "disk_space", "ssh_host", "ssh_port", "public_ipaddr")


def vast(args: list[str]):
    """Reject empty/error JSON even when the CLI exits successfully."""
    completed = subprocess.run(["vastai", *args, "--raw"], capture_output=True, text=True, timeout=90)
    try:
        value = json.loads(completed.stdout)
    except ValueError as error:
        raise RuntimeError("Vast CLI returned no valid JSON; check account authentication with `vastai show user --raw`") from error
    if completed.returncode or value is None or isinstance(value, dict) and (value.get("error") or value.get("success") is False):
        raise RuntimeError("Vast CLI operation failed; inspect the account/instance with the official CLI")
    return value


def instance(identifier: int) -> dict:
    value = vast(["show", "instance", str(identifier)])
    if isinstance(value, list):
        value = next((item for item in value if item.get("id") == identifier), None)
    if not isinstance(value, dict) or int(value.get("id", -1)) != identifier:
        raise RuntimeError("Vast returned an unexpected instance identity")
    return value


def public(value: dict, fields: tuple[str, ...]) -> dict:
    return {key: value[key] for key in fields if key in value}


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def connection(value: dict, identity: str | None) -> tuple[list[str], list[str], str]:
    host = str(value.get("public_ipaddr") or value.get("ssh_host") or "")
    ipaddress.ip_address(host)
    port = int(value.get("ssh_port", 0))
    if not 1 <= port <= 65535:
        raise ValueError("instance has no direct SSH port")
    options = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new"]
    if identity:
        options.extend(["-i", str(Path(identity).expanduser().resolve()), "-o", "IdentitiesOnly=yes"])
    target = f"root@{host}"
    return ["ssh", *options, "-p", str(port), target], ["scp", *options, "-P", str(port)], target


def remote(ssh: list[str], command: list[str], *, timeout: float = 45) -> str:
    return subprocess.run([*ssh, shlex.join(command)], check=True, capture_output=True, text=True, timeout=timeout).stdout


def prepare(args, dest: Path) -> dict:
    from aresim.algorithms.registry import create_training_registry
    from aresim.training import load_experiment_with_overrides
    from aresim.training.provenance import experiment_seeds, freeze_experiment
    config = Path(args.config)
    if config.is_absolute() or ".." in config.parts or config.parts[:1] != ("configs",):
        raise ValueError("config must be a repository-relative path under configs/")
    overrides = list(args.overrides)
    init_from = getattr(args, "init_from", None)
    initialization_path = "inputs/plan2explore/checkpoint.json"
    if init_from:
        overrides.append(f"algorithm_config.initialization_checkpoint={initialization_path}")
    spec = freeze_experiment(load_experiment_with_overrides(ROOT / config, tuple(overrides)))
    experiment_seeds(spec).validate()
    if spec.tracking.mode == "online":
        raise ValueError("cloud launcher supports tracking.mode=disabled or offline; select one with --set")
    if spec.artifacts.resume_from:
        raise ValueError("this source-only launcher starts fresh trials; remote resume needs the original recovered source/artifacts")
    if spec.artifacts.root != "results":
        raise ValueError("cloud recovery requires artifacts.root=results")
    if not isinstance(spec.algorithm_config.total_environment_steps, int) or spec.algorithm_config.total_environment_steps <= 0:
        raise ValueError("training requires a finite positive environment-step limit")
    if hasattr(spec.resources, "device"):
        if spec.resources.device != "cuda":
            raise ValueError("select resources.device=cuda for this GPU run")
    elif spec.resources.gpus_per_learner <= 0:
        raise ValueError("select resources.gpus_per_learner=1 for this GPU run")
    adaptation = spec.algorithm == "plan2explore" and spec.algorithm_config.phase == "adapt"
    if adaptation and not init_from:
        raise ValueError("Plan2Explore adaptation requires --init-from PATH to an explicit local pretraining checkpoint.json")
    if init_from and not adaptation:
        raise ValueError("--init-from is only available for Plan2Explore adaptation")
    native = sidecar = None
    if init_from:
        from aresim.algorithms.plan2explore.checkpoint import validate_transfer
        sidecar = Path(init_from).expanduser().resolve()
        if sidecar.name != "checkpoint.json":
            raise ValueError("--init-from must point to checkpoint.json")
        metadata, _, native = validate_transfer(sidecar, spec)
        if metadata["native_path"] != "native" or any((native / name).is_symlink() for name in ("model.pt", "trainer.pt")):
            raise ValueError("pretraining export requires a regular native/model.pt and native/trainer.pt")
    stage_payload(ROOT, dest, provider="vastai")
    if sidecar is not None:
        checkpoint_dest = dest / initialization_path
        checkpoint_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sidecar, checkpoint_dest)
        for name in ("model.pt", "trainer.pt"):
            output = checkpoint_dest.parent / "native" / name
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(native / name, output)
        validate_transfer(checkpoint_dest, spec)
        inventory_path = dest / "source_inventory.json"
        inventory = json.loads(inventory_path.read_text())
        for path in sorted(checkpoint_dest.parent.rglob("*")):
            if path.is_file():
                with path.open("rb") as handle:
                    inventory[str(path.relative_to(dest))] = hashlib.file_digest(handle, "sha256").hexdigest()
        save(inventory_path, inventory)
    return {"extra": create_training_registry().registration(spec.algorithm).install_extra,
            "resolved_config": spec.as_dict(), "train_args": [str(config), *[part for item in overrides for part in ("--set", item)]]}


def remaining(deadline: float, maximum: float = 120) -> float:
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("cloud deadline reached; unrecovered files remain on instance storage")
    return min(seconds, maximum)


def fetch(ssh: list[str], scp: list[str], target: str, remote_root: str, destination: Path, *, deadline: float) -> int:
    """Recover a frozen allowlist within one deadline and verify before acknowledgment."""
    if not re.fullmatch(r"/root/aresim-cloud/[A-Za-z0-9_-]+", remote_root):
        raise ValueError("unexpected remote job directory")
    inventory = json.loads(remote(ssh, ["cat", f"{remote_root}/artifact_inventory.json"], timeout=remaining(deadline, 45)))
    if not isinstance(inventory, dict) or not inventory:
        raise RuntimeError("remote artifact inventory is empty")
    destination = destination.resolve()
    for name, digest in inventory.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_./+-]+", name):
            raise ValueError("unsafe artifact inventory path")
        path = Path(name)
        if not path.parts or path.is_absolute() or ".." in path.parts or not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("unsafe artifact inventory")
        if path.parts[0] != "results" and name not in {"training.log", "source_inventory.json", "job.json"}:
            raise ValueError(f"unexpected artifact path: {name}")
        output = destination / path
        if output.is_symlink() or not output.resolve().is_relative_to(destination):
            raise ValueError("artifact destination follows a local symlink")
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([*scp, f"{target}:{remote_root}/{name}", str(output)], check=True, timeout=remaining(deadline))
        digest_state = hashlib.sha256()
        with output.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                remaining(deadline)
                digest_state.update(chunk)
        if digest_state.hexdigest() != digest:
            raise RuntimeError(f"artifact checksum mismatch: {name}")
    save(destination / "artifact_inventory.json", inventory)
    remote(ssh, ["touch", f"{remote_root}/recovered"], timeout=remaining(deadline, 45))
    return len(inventory)


def stop(identifier: int) -> None:
    value = vast(["stop", "instance", str(identifier)])
    if not isinstance(value, dict) or value.get("success") is not True:
        raise RuntimeError(f"stop was not acknowledged for instance {identifier}")


def run(args) -> int:
    if any(not math.isfinite(value) or value <= 0 for value in (args.hours, args.max_hourly, args.max_cost)):
        raise ValueError("hours and cost limits must be finite positive values")
    with tempfile.TemporaryDirectory(prefix="aresim-vast-") as temporary:
        payload = Path(temporary) / "payload"
        job = prepare(args, payload)
        job.update(instance_id=args.on, seconds=max(1, int(args.hours * 3600)), grace_seconds=args.grace_seconds, recovery_seconds=args.recovery_seconds)
        save(payload / "job.json", job)
        bundle = Path(temporary) / "payload.tar.gz"
        with tarfile.open(bundle, "w:gz") as archive:
            for path in sorted(payload.rglob("*")):
                if path.is_file():
                    archive.add(path, arcname=str(path.relative_to(payload)))
        value = instance(args.on)
        if value.get("actual_status") != "running":
            raise RuntimeError("selected instance must already be running")
        rate = float(value.get("dph_total", float("nan")))
        bounded_hours = args.hours + (args.grace_seconds + args.recovery_seconds + 90) / 3600
        if not math.isfinite(rate) or rate <= 0 or rate > args.max_hourly or rate * bounded_hours > args.max_cost:
            raise ValueError("live hourly rate or estimated running cost exceeds bounds (or price is unavailable)")
        ssh, scp, target = connection(value, args.identity)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        local = ROOT / "results/cloud/vastai" / f"{args.on}_{stamp}"
        remote_root = f"/root/aresim-cloud/{stamp}"
        record = {"instance": public(value, INSTANCE_FIELDS), "remote_root": remote_root, "hours": args.hours,
                  "max_cost": args.max_cost, "estimated_running_cost": rate * bounded_hours, "state": "preparing"}
        deadline = time.monotonic() + args.hours * 3600
        save(local / "lifecycle.json", record)
        try:
            remote(ssh, ["mkdir", "-p", remote_root], timeout=remaining(deadline, 45))
            subprocess.run([*scp, str(bundle), f"{target}:{remote_root}/payload.tar.gz"], check=True, timeout=remaining(deadline, args.hours * 3600))
            remote(ssh, ["tar", "-xzf", f"{remote_root}/payload.tar.gz", "-C", remote_root], timeout=remaining(deadline, 45))
            seconds = int(remaining(deadline, args.hours * 3600))
            if seconds < 1:
                raise TimeoutError("no setup/training time remains after upload")
            remote(ssh, ["python3", "-c", "import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); j=json.loads(p.read_text()); j['seconds']=int(sys.argv[2]); p.write_text(json.dumps(j))", f"{remote_root}/job.json", str(seconds)], timeout=remaining(deadline, 45))
            # Account API keys never enter SSH command arguments or the payload.
            shell = f"cd {shlex.quote(remote_root)} && nohup python3 scripts/vastai/supervisor.py {shlex.quote(remote_root)} > supervisor.log 2>&1 < /dev/null &"
            remote(ssh, ["bash", "-c", shell], timeout=remaining(deadline, 45))
            record["state"] = "running"
            save(local / "lifecycle.json", record)
            print(f"instance {args.on}; lifecycle {local / 'lifecycle.json'}", flush=True)
            recovery_deadline = deadline + args.grace_seconds + args.recovery_seconds
            status_seen = False
            while time.monotonic() < recovery_deadline:
                completed = subprocess.run([*ssh, shlex.join(["cat", f"{remote_root}/cloud_status.json"])], capture_output=True, text=True, timeout=remaining(recovery_deadline, 45))
                if completed.returncode:
                    if status_seen or completed.returncode == 255:
                        raise RuntimeError("remote status became unavailable; recover retained artifacts after inspecting the instance")
                    record["remote_status"] = {"state": "starting", "status_not_written_yet": True}
                else:
                    status = json.loads(completed.stdout)
                    if not isinstance(status, dict):
                        raise RuntimeError("supervisor returned an invalid status")
                    status_seen = True
                    record["remote_status"] = status
                    save(local / "lifecycle.json", record)
                    if status.get("artifacts_ready"):
                        remote_seconds = float(status.get("recovery_seconds_remaining", args.recovery_seconds))
                        fetch_deadline = min(recovery_deadline, time.monotonic() + max(0, remote_seconds - 2))
                        record["recovered_files"] = fetch(ssh, scp, target, remote_root, local / "artifacts", deadline=fetch_deadline)
                        record["state"] = status["state"]
                        return int(status["exit_code"])
                    if status.get("supervisor_error_type") or status.get("state") == "failed" and "exit_code" not in status:
                        raise RuntimeError("remote supervisor failed before publishing an artifact inventory; inspect supervisor.log via SSH")
                save(local / "lifecycle.json", record)
                time.sleep(min(10, remaining(recovery_deadline)))
            raise TimeoutError("client deadline reached; artifacts may remain on instance storage")
        except BaseException as error:
            record.update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error_type=type(error).__name__)
            raise
        finally:
            try:
                stop(args.on)
                record["shutdown"] = "stop_requested"
            except Exception as shutdown_error:
                record["shutdown"] = "FAILED"
                record["shutdown_error_type"] = type(shutdown_error).__name__
                print(f"AUTOMATIC STOP FAILED: run `vastai stop instance {args.on} --raw`", file=sys.stderr)
            save(local / "lifecycle.json", record)
            print(f"instance {args.on}: {record['state']}; {record['shutdown']}; storage remains billed", flush=True)
            if record["shutdown"] == "FAILED" and sys.exception() is None:
                raise RuntimeError(f"automatic stop failed for instance {args.on}; verified outputs are retained locally")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    offers = commands.add_parser("offers", help="query current on-demand offers without renting")
    offers.add_argument("--query", default="num_gpus=1 gpu_ram>=24 cpu_ram>=48 cpu_cores_effective>=8 reliability>=0.98 verified=true rentable=true direct_port_count>=1")
    offers.add_argument("--storage", type=int, default=80)
    offers.add_argument("--limit", type=int, default=20)
    for name in ("bundle", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--set", dest="overrides", action="append", default=[])
        command.add_argument("--init-from", type=Path, help="explicit local Plan2Explore pretraining checkpoint.json for adaptation")
        if name == "bundle":
            command.add_argument("--output", required=True, type=Path, help="new directory for inspecting upload contents")
        else:
            command.add_argument("--on", required=True, type=int)
            command.add_argument("--identity")
            command.add_argument("--hours", required=True, type=float, help="setup/training time limit")
            command.add_argument("--max-hourly", required=True, type=float, help="USD/hour ceiling")
            command.add_argument("--max-cost", required=True, type=float, help="estimated running USD ceiling; excludes transfer and subsequent storage")
            command.add_argument("--grace-seconds", type=int, default=30)
            command.add_argument("--recovery-seconds", type=int, default=300)
    recover = commands.add_parser("fetch", help="recover outputs after manually starting the same instance")
    recover.add_argument("lifecycle", type=Path)
    recover.add_argument("--identity")
    recover.add_argument("--seconds", type=int, default=300, help="aggregate recovery deadline")
    args = parser.parse_args()
    if args.command == "bundle":
        prepare(args, args.output)
        print(f"source bundle: {args.output.resolve()}")
        return 0
    if shutil.which("vastai") is None:
        raise RuntimeError("install the official CLI separately: uv tool install vastai")
    if args.command == "offers":
        value = vast(["search", "offers", args.query, "--type", "on-demand", "--storage", str(args.storage), "--order", "dph_total", "--limit", str(args.limit)])
        if not isinstance(value, list):
            raise RuntimeError("unexpected offer search response")
        print(json.dumps([public(item, OFFER_FIELDS) for item in value], indent=2))
        return 0
    if args.command == "fetch":
        record = json.loads(args.lifecycle.read_text())
        ssh, scp, target = connection(instance(int(record["instance"]["id"])), args.identity)
        if not 1 <= args.seconds <= 3600:
            raise ValueError("recovery deadline must be 1..3600 seconds")
        count = fetch(ssh, scp, target, record["remote_root"], args.lifecycle.parent / "artifacts", deadline=time.monotonic()+args.seconds)
        record.update(recovered_files=count)
        save(args.lifecycle, record)
        print(f"verified {count} artifacts; stop compute when recovery is complete")
        return 0
    if not 1 <= args.grace_seconds <= 300 or not 30 <= args.recovery_seconds <= 3600:
        raise ValueError("grace must be 1..300 seconds; recovery must be 30..3600 seconds")
    return run(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
