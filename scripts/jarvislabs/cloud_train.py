#!/usr/bin/env python3
"""Launch AresSim masked-PPO training on a JarvisLabs instance via the ``jl`` CLI.

Stages engine, configs, and seed manifests, creates or reuses an instance, installs
a Python 3.12 venv, runs ``aresim-rl train``, then downloads ``results/``. Simulator
rules stay in ``engine/``; this file only wraps JarvisLabs plumbing.

See ``docs/rl/jarvislabs.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REMOTE_DIRNAME = "aresim"
SESSION_NAME = ".jarvislabs_session.json"
SKIP_DIR_NAMES = frozenset(
    {".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git"}
)
REQUIRED_NOTEBOOKS = (
    "phase1_open_exploration_split_v1.yaml",
    "phase1_smoke_eval_v1.yaml",
)
DETACH_EXIT_CODES = {130, 143, -2, -15}


def repo_root() -> Path:
    """Return the checkout root (the directory that contains ``engine/`` and ``configs/``)."""
    here = Path(__file__).resolve()
    for candidate in (here.parents[2], Path.cwd()):
        if (candidate / "engine" / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return candidate
    raise FileNotFoundError("run this script from the AresSim repository root")


def load_local_env(root: Path) -> None:
    """Load unset keys from repo ``.env`` / ``.env.local``. Does not log values."""
    for name in (".env", ".env.local"):
        path = root / name
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = value.strip().strip("'").strip('"')


def session_path(root: Path) -> Path:
    return root / "results" / SESSION_NAME


def load_session(root: Path) -> dict[str, str]:
    path = session_path(root)
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(value) for key, value in payload.items()}


def save_session(root: Path, **fields: str) -> None:
    path = session_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    current = load_session(root)
    current.update({key: value for key, value in fields.items() if value})
    path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")


def _ignore_engine(_directory: str, names: list[str]) -> list[str]:
    return [name for name in names if name in SKIP_DIR_NAMES or name.endswith(".egg-info")]


def stage_payload(root: Path, dest: Path) -> None:
    """Copy the training payload into ``dest`` without local venvs, UI, or results."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    shutil.copytree(root / "engine", dest / "engine", ignore=_ignore_engine)
    shutil.copytree(root / "configs", dest / "configs")
    notebooks = dest / "notebooks"
    notebooks.mkdir()
    missing = [name for name in REQUIRED_NOTEBOOKS if not (root / "notebooks" / name).is_file()]
    if missing:
        paths = ", ".join(str(root / "notebooks" / name) for name in missing)
        raise FileNotFoundError(f"seed manifests are required for training and are missing locally: {paths}")
    for name in REQUIRED_NOTEBOOKS:
        shutil.copy2(root / "notebooks" / name, notebooks / name)
    script_dir = dest / "scripts" / "jarvislabs"
    script_dir.mkdir(parents=True)
    shutil.copy2(Path(__file__).with_name("remote_setup.sh"), script_dir / "remote_setup.sh")


def _decode_json(text: str) -> Any:
    text = text.strip()
    return json.loads(text) if text else None


def command_with_json(args: list[str]) -> list[str]:
    """Put ``--json`` before ``--`` so it stays a ``jl`` flag, not a remote argument."""
    command = ["jl", *args]
    separator = command.index("--") if "--" in command else len(command)
    command.insert(separator, "--json")
    return command


def jl(args: list[str], *, capture: bool = False, json_output: bool = False) -> subprocess.CompletedProcess[str]:
    """Run ``jl`` and raise a short error if it fails."""
    command = command_with_json(args) if json_output else ["jl", *args]
    completed = subprocess.run(command, check=False, capture_output=capture, text=True)
    if completed.returncode == 0:
        return completed
    detail = _jl_error_detail(completed) if capture else ""
    raise RuntimeError(detail or f"jl {' '.join(args)} failed with exit {completed.returncode}")


def _jl_error_detail(completed: subprocess.CompletedProcess[str]) -> str:
    payload = _decode_json(completed.stdout) if completed.stdout else None
    if isinstance(payload, dict) and payload.get("error"):
        return str(payload["error"])
    return (completed.stderr or completed.stdout or "").strip()


def jl_json(args: list[str]) -> Any:
    completed = jl(args, capture=True, json_output=True)
    payload = _decode_json(completed.stdout)
    if payload is None:
        raise RuntimeError(f"jl {' '.join(args)} returned empty JSON")
    return payload


def require_ssh_identity() -> None:
    """Fail before creating a billed instance if scp cannot offer an unlocked key."""
    candidates = [Path.home() / ".ssh" / name for name in ("id_ed25519", "id_rsa", "id_ecdsa")]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise RuntimeError("no ~/.ssh/id_ed25519 or id_rsa; generate a key and run jl ssh-key add <key>.pub --name my-laptop")
    listed = subprocess.run(["ssh-add", "-l"], capture_output=True, text=True)
    if listed.returncode == 0:
        return
    locked = [path for path in existing if b"ENCRYPTED" in path.read_bytes()[:120] or b"aes256" in path.read_bytes()[:120]]
    if locked:
        raise RuntimeError(
            "jl upload uses scp and cannot prompt for a key passphrase. Unlock it first:\n"
            f"  ssh-add {locked[0]}"
        )


def require_jl() -> None:
    if shutil.which("jl") is None:
        raise RuntimeError("jl is not on PATH; install with `uv tool install jarvislabs` then `jl setup`")


def machine_id_from(payload: Any) -> str:
    if isinstance(payload, dict) and payload.get("machine_id") is not None:
        return str(payload["machine_id"])
    raise RuntimeError(f"instance id missing from jl output: {payload!r}")


def instance_status(machine_id: str) -> str:
    payload = jl_json(["get", machine_id])
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected jl get payload: {payload!r}")
    return str(payload.get("status") or "")


def ensure_running(machine_id: str) -> str:
    """Resume a paused instance; resume may assign a new machine id."""
    status = instance_status(machine_id).lower()
    if status == "running":
        return machine_id
    if status != "paused":
        raise RuntimeError(f"instance {machine_id} is {status or 'unknown'}; resume or create a new one")
    payload = jl_json(["resume", machine_id, "--yes"])
    if isinstance(payload, dict) and payload.get("machine_id"):
        return str(payload["machine_id"])
    return machine_id


def create_instance(args: argparse.Namespace) -> str:
    create = ["create", "--storage", str(args.storage), "--name", args.name, "--yes"]
    if args.region:
        create.extend(["--region", args.region])
    if args.cpu:
        create.extend(["--vm", "--cpu", "--vcpus", str(args.vcpus), "--ram", str(args.ram)])
    else:
        create.extend(["--gpu", args.gpu, "--template", "pytorch"])
    return machine_id_from(jl_json(create))


def remote_home(machine_id: str) -> str:
    completed = jl(["exec", machine_id, "--", "sh", "-lc", 'printf %s "$HOME"'], capture=True)
    home = (completed.stdout or "").strip()
    if not home.startswith("/"):
        raise RuntimeError(f"could not read remote HOME for {machine_id}")
    return home


def upload_wandb_env(machine_id: str, remote_root: str, api_key: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", prefix="aresim-wandb-", suffix=".env", delete=False) as handle:
        handle.write(f"WANDB_API_KEY={shlex.quote(api_key)}\n")
        temp_path = Path(handle.name)
    try:
        temp_path.chmod(0o600)
        jl(["upload", machine_id, str(temp_path), f"{remote_root}/.wandb.env"])
    finally:
        temp_path.unlink(missing_ok=True)


def setup_remote(machine_id: str, remote_root: str) -> None:
    script = f"{remote_root}/scripts/jarvislabs/remote_setup.sh"
    jl(["exec", machine_id, "--", "sh", "-lc", f"chmod +x {shlex.quote(script)} && {shlex.quote(script)}"])


def train_overrides(args: argparse.Namespace) -> list[str]:
    overrides = list(args.overrides)
    gpu_requested = bool(args.gpu) and not args.cpu
    already_set = any(item.startswith("resources.gpus_per_learner=") for item in overrides)
    if gpu_requested and not already_set:
        overrides.append("resources.gpus_per_learner=1")
    return overrides


def remote_train_shell(remote_root: str, config: str, overrides: list[str], resume_from: str | None) -> str:
    command = [".venv/bin/aresim-rl", "train", config]
    for item in overrides:
        command.extend(["--set", item])
    if resume_from:
        command.extend(["--resume-from", resume_from])
    quoted = " ".join(shlex.quote(part) for part in command)
    return (
        f"set -euo pipefail; cd {shlex.quote(remote_root)}; "
        "if [ -f .wandb.env ]; then set -a; . ./.wandb.env; set +a; fi; "
        f"exec {quoted}"
    )


def start_train(machine_id: str, remote_root: str, args: argparse.Namespace) -> str:
    shell = remote_train_shell(remote_root, args.config, train_overrides(args), args.resume_from)
    payload = jl_json(["run", "--on", machine_id, "--no-follow", "--yes", "--", "bash", "-lc", shell])
    if not isinstance(payload, dict) or not payload.get("run_id"):
        raise RuntimeError(f"jl run did not return a run_id: {payload!r}")
    return str(payload["run_id"])


def follow_run(run_id: str) -> bool:
    """Stream logs until the run finishes. Return True if the user detached with Ctrl+C."""
    try:
        completed = subprocess.run(["jl", "run", "logs", run_id, "--follow"], check=False)
    except KeyboardInterrupt:
        return True
    if completed.returncode in DETACH_EXIT_CODES:
        return True
    if completed.returncode != 0:
        raise RuntimeError(f"jl run logs {run_id} failed with exit {completed.returncode}")
    return False


def run_state(run_id: str) -> str:
    payload = jl_json(["run", "status", run_id])
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("state") or payload.get("status") or "").lower()


def run_is_finished(state: str) -> bool:
    return state in {"succeeded", "failed"}


def fetch_results(machine_id: str, remote_root: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    jl(["download", machine_id, f"{remote_root}/results", str(dest), "-r"])
    nested = dest / "results"
    if not nested.is_dir():
        return
    for child in nested.iterdir():
        target = dest / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        elif not target.exists():
            shutil.copy2(child, target)
    shutil.rmtree(nested)


def stop_instance(machine_id: str, *, destroy: bool) -> None:
    if destroy:
        jl_json(["destroy", machine_id, "--yes"])
        return
    jl_json(["pause", machine_id, "--yes"])


def resolve_machine_id(args: argparse.Namespace, root: Path) -> str:
    if args.machine_id:
        return args.machine_id
    session = load_session(root)
    if session.get("machine_id"):
        return session["machine_id"]
    raise RuntimeError("pass --on MACHINE_ID or run `cloud_train.py run` first")


def tracking_needs_wandb(config_path: Path, overrides: list[str]) -> bool:
    for item in overrides:
        if item.startswith("tracking.mode="):
            return item.split("=", 1)[1].strip() == "online"
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("mode:"):
            return stripped.split(":", 1)[1].strip() == "online"
    return True


def _require_files(paths: list[Path], *, error: str) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"{error}: " + ", ".join(missing))


def _self_check_helpers(root: Path) -> None:
    smoke = root / "configs" / "masked_ppo" / "smoke.yaml"
    reference = root / "configs" / "masked_ppo" / "reference.yaml"
    if tracking_needs_wandb(smoke, []) or not tracking_needs_wandb(reference, []):
        raise RuntimeError("W&B mode detection is wrong for smoke/reference YAML")
    if tracking_needs_wandb(reference, ["tracking.mode=disabled"]):
        raise RuntimeError("tracking.mode override should disable W&B")
    gpu = argparse.Namespace(overrides=[], gpu="L4", cpu=False)
    cpu = argparse.Namespace(overrides=[], gpu=None, cpu=True)
    if train_overrides(gpu) != ["resources.gpus_per_learner=1"] or train_overrides(cpu) != []:
        raise RuntimeError("GPU resource overrides are wrong")
    json_command = command_with_json(["run", "--on", "1", "--", "bash", "-lc", "true"])
    if json_command.index("--json") > json_command.index("--"):
        raise RuntimeError("--json must be inserted before -- for jl run")


def cmd_self_check(_args: argparse.Namespace) -> int:
    root = repo_root()
    _self_check_helpers(root)
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / REMOTE_DIRNAME
        stage_payload(root, dest)
        _require_files(
            [
                dest / "engine" / "pyproject.toml",
                dest / "configs" / "masked_ppo" / "reference.yaml",
                dest / "notebooks" / "phase1_open_exploration_split_v1.yaml",
                dest / "scripts" / "jarvislabs" / "remote_setup.sh",
            ],
            error="staged payload missing",
        )
        leaked = (dest / "engine" / ".venv").exists() or (dest / "web").exists() or (dest / "results").exists()
        if leaked:
            raise RuntimeError("staged payload included local venv, UI, or results")
    print("self-check ok")
    return 0


def provision_machine(args: argparse.Namespace, root: Path) -> str:
    if args.machine_id:
        machine_id = ensure_running(args.machine_id)
    else:
        machine_id = create_instance(args)
    print(f"instance {machine_id}")
    save_session(root, machine_id=machine_id)
    return machine_id


def sync_code(machine_id: str, root: Path, args: argparse.Namespace) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / REMOTE_DIRNAME
        stage_payload(root, staging)
        jl(["upload", machine_id, str(staging)])
    remote_root = f"{remote_home(machine_id)}/{REMOTE_DIRNAME}"
    save_session(root, machine_id=machine_id, remote_root=remote_root)
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        upload_wandb_env(machine_id, remote_root, api_key)
    if not args.skip_setup:
        setup_remote(machine_id, remote_root)
    return remote_root


def validate_run(args: argparse.Namespace, root: Path) -> None:
    if not args.machine_id and not args.gpu and not args.cpu:
        raise RuntimeError("pass --gpu TYPE, --cpu, or --on MACHINE_ID")
    if Path(args.config).is_absolute():
        raise RuntimeError("pass a repo-relative config path, for example configs/masked_ppo/reference.yaml")
    config_path = root / args.config
    if not config_path.is_file():
        raise FileNotFoundError(f"experiment YAML not found: {args.config}")
    if tracking_needs_wandb(config_path, args.overrides) and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("set WANDB_API_KEY for online W&B, or pass --set tracking.mode=disabled")


def cmd_run(args: argparse.Namespace) -> int:
    require_jl()
    require_ssh_identity()
    root = repo_root()
    load_local_env(root)
    validate_run(args, root)
    machine_id = provision_machine(args, root)
    leave_running = False
    try:
        remote_root = sync_code(machine_id, root, args)
        run_id = start_train(machine_id, remote_root, args)
        save_session(root, machine_id=machine_id, remote_root=remote_root, run_id=run_id)
        print(f"run {run_id}")
        if args.detach:
            leave_running = True
            _print_detach_help(machine_id, run_id)
            return 0
        if follow_run(run_id) and not run_is_finished(run_state(run_id)):
            leave_running = True
            print(f"\ndetached; training still running on {machine_id} as {run_id}")
            print("results were NOT downloaded. Do not destroy the instance.")
            _print_detach_help(machine_id, run_id)
            return 0
        fetch_results(machine_id, remote_root, root / "results")
        print(f"downloaded results to {root / 'results'}")
    finally:
        if not args.keep and not leave_running:
            stop_instance(machine_id, destroy=args.destroy)
            print(f"instance {machine_id} {'destroyed' if args.destroy else 'paused'}")
    return 0


def _print_detach_help(machine_id: str, run_id: str) -> None:
    print(f"follow with: jl run logs {run_id} --follow")
    print(f"fetch later with: python3 scripts/jarvislabs/cloud_train.py fetch --on {machine_id}")


def cmd_fetch(args: argparse.Namespace) -> int:
    require_jl()
    root = repo_root()
    machine_id = ensure_running(resolve_machine_id(args, root))
    session = load_session(root)
    remote_root = args.remote_root or session.get("remote_root") or f"{remote_home(machine_id)}/{REMOTE_DIRNAME}"
    save_session(root, machine_id=machine_id, remote_root=remote_root)
    fetch_results(machine_id, remote_root, root / "results")
    print(f"downloaded results to {root / 'results'}")
    return 0


def cmd_down(args: argparse.Namespace) -> int:
    require_jl()
    root = repo_root()
    machine_id = resolve_machine_id(args, root)
    stop_instance(machine_id, destroy=args.destroy)
    print(f"instance {machine_id} {'destroyed' if args.destroy else 'paused'}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    require_jl()
    root = repo_root()
    machine_id = resolve_machine_id(args, root)
    session = load_session(root)
    print(json.dumps({"machine_id": machine_id, **session, "status": instance_status(machine_id)}, indent=2))
    if session.get("run_id"):
        jl(["run", "status", session["run_id"]])
    return 0


def _add_on_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--on", dest="machine_id", help="existing JarvisLabs machine id")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train AresSim on a JarvisLabs GPU or CPU instance")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="create or reuse an instance, train, download results/")
    _add_on_flag(run)
    run.add_argument("--gpu", help="GPU type from `jl gpus` (for example L4 or A100)")
    run.add_argument("--cpu", action="store_true", help="create a CPU VM instead of a GPU container")
    run.add_argument("--vcpus", type=int, default=8, help="CPU VM vCPU count")
    run.add_argument("--ram", type=int, default=32, help="CPU VM RAM in GB")
    run.add_argument("--storage", type=int, default=100, help="disk in GB")
    run.add_argument("--name", default="aresim-train", help="instance name")
    run.add_argument("--region", help="IN1, IN2, or EU1")
    run.add_argument("--config", default="configs/masked_ppo/reference.yaml")
    run.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE")
    run.add_argument("--resume-from", dest="resume_from", help="forwarded to aresim-rl train")
    run.add_argument("--detach", action="store_true", help="start training and return without waiting")
    run.add_argument("--keep", action="store_true", help="leave the instance running after the run")
    run.add_argument("--destroy", action="store_true", help="destroy the instance instead of pausing")
    run.add_argument("--skip-setup", action="store_true", help="skip venv install on a reused instance")

    fetch = commands.add_parser("fetch", help="download remote results/ into local results/")
    _add_on_flag(fetch)
    fetch.add_argument("--remote-root", help="remote project path (default: $HOME/aresim)")

    down = commands.add_parser("down", help="pause (default) or destroy the instance")
    _add_on_flag(down)
    down.add_argument("--destroy", action="store_true", help="destroy the instance instead of pausing")

    status = commands.add_parser("status", help="print the saved session and instance status")
    _add_on_flag(status)

    commands.add_parser("self-check", help="stage a payload locally and verify required files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {
        "self-check": cmd_self_check,
        "run": cmd_run,
        "fetch": cmd_fetch,
        "down": cmd_down,
        "status": cmd_status,
    }
    try:
        return commands[args.command](args)
    except (FileNotFoundError, RuntimeError, json.JSONDecodeError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
