"""Launch the Decision Model / Learner matrix through the resumable rollout CLI."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen

import yaml

from aresim.algorithms.llm_dm.config import (
    ROOT,
    RunConfig,
    load_llm_dm_run,
    validate_experiment_name,
)

GAME_FINISHED = re.compile(rb"^Game (\d+)/\d+: ")
TURN_STARTED = re.compile(rb"^[A-Za-z0-9_-]+ turn (\d+): calling ")
LEARNER_STARTED = re.compile(rb"^Learner batch (\d+): calling ")
SUCCESS = {"completed", "already completed"}


def bar(value: float, total: int, width: int = 16) -> str:
    filled = min(width, int(width * value / total))
    return f"[{'#' * filled}{'-' * (width - filled)}]"


class ExperimentProgress:
    """Seed progress from the resume checkpoint and follow only this launch's log."""

    def __init__(self, config: RunConfig) -> None:
        self.name = config.experiment_name
        self.total = len(config.episodes)
        self.steps = config.max_episode_steps
        self.log = config.output_directory / ".logs" / f"{self.name}.log"
        try:
            self.offset = self.log.stat().st_size
        except OSError:
            self.offset = 0
        self.games = self.turn = 0
        self.phase = "Starting"
        checkpoint_path = config.output_directory / self.name / "resume.json"
        try:
            checkpoint = json.loads(checkpoint_path.read_text())
        except (OSError, ValueError):
            return
        if not isinstance(checkpoint, dict) or not isinstance(
            checkpoint.get("game_ids"), list
        ):
            return
        self.games = len(checkpoint["game_ids"])
        self.phase = "Checking saved result" if self.games == self.total else "Resuming"
        if checkpoint.get("pending_evidence") is not None and self.games < self.total:
            self.phase = f"Learner after game {self.games}"

    def line(self) -> str:
        try:
            stream = self.log.open("rb")
        except OSError:
            pass
        else:
            with stream:
                stream.seek(self.offset)
                while row := stream.readline():
                    if not row.endswith(b"\n"):
                        break
                    self.offset = stream.tell()
                    if match := GAME_FINISHED.match(row):
                        self.games = int(match[1])
                        self.turn = 0
                        self.phase = "Saving game"
                    elif match := LEARNER_STARTED.match(row):
                        self.phase = f"Learner after game {int(match[1])}"
                    elif row.startswith(b"Starting "):
                        self.phase = f"Starting game {self.games + 1}"
                    elif match := TURN_STARTED.match(row):
                        self.turn = int(match[1])
                        self.phase = (
                            f"Game {self.games + 1}, turn {self.turn}/{self.steps}"
                        )
        game_fraction = max(0, self.turn - 1) / self.steps
        return (
            f"{bar(self.games + game_fraction, self.total)} {self.games}/{self.total} games"
            f" | {self.phase} | {self.name}"
        )


class ProgressDisplay:
    """Refresh the active rows on a terminal and write periodic snapshots to pipes."""

    def __init__(self) -> None:
        self.interactive = sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
        self.previous: list[str] = []
        self.last_printed = float("-inf")

    def show(self, lines: list[str], messages: list[str]) -> None:
        now = monotonic()
        if self.interactive:
            size = shutil.get_terminal_size()
            lines = [
                line[: max(1, size.columns - 1)]
                for line in lines[: max(1, size.lines - 1)]
            ]
            if lines == self.previous and not messages:
                return
            if self.previous:
                sys.stdout.write(f"\x1b[{len(self.previous)}A\r\x1b[J")
        elif not messages and (lines == self.previous or now - self.last_printed < 10):
            return
        print("\n".join([*messages, *lines]), flush=True)
        self.previous = lines
        self.last_printed = now


def openrouter_routes(model: str) -> list:
    url = f"https://openrouter.ai/api/v1/models/{quote(model, safe='/')}/endpoints"
    try:
        with urlopen(url, timeout=15) as response:
            return json.load(response)["data"]["endpoints"]
    except HTTPError as error:
        if error.code == 404:
            return []
        raise


def run_job(path: Path, env_file: Path, stopping: Event) -> tuple[str, str]:
    if stopping.is_set():
        return "cancelled", "Resume with the same command"
    config = load_llm_dm_run(path)
    name = config.experiment_name
    manifest = config.output_directory / name / "manifest.json"
    completed = (
        manifest.is_file() and json.loads(manifest.read_text())["status"] == "completed"
    )
    if (
        not completed
        and config.transport == "live"
        and config.actor.provider == "openrouter_decisions"
    ):
        if not openrouter_routes(config.actor.model):
            return (
                "unavailable",
                f"OpenRouter lists no active route for {config.actor.model}",
            )

    logs = config.output_directory / ".logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{name}.log"
    command = [
        sys.executable,
        "-m",
        "aresim.training.cli",
        "rollout",
        str(path),
        "--resume",
    ]
    if config.transport == "live" and env_file.exists() and not completed:
        command += ["--env-file", str(env_file)]
    if stopping.is_set():
        return "cancelled", "Resume with the same command"
    with log.open("a") as stream:
        result = subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT
        )
    if result.returncode:
        return "incomplete", str(log)
    return ("already completed" if completed else "completed"), str(manifest)


def load_matrix(source: Path) -> tuple[dict, Path]:
    """Read a matrix or a second block that reuses its policy and budget settings."""
    matrix = yaml.safe_load(source.read_text(encoding="utf-8"))
    schema = "aresim.llm_dm.ablation.v1"
    if not isinstance(matrix, dict) or matrix.get("schema_version") != schema:
        raise ValueError("unsupported ablation config schema")
    if "base_config" not in matrix:
        return matrix, source

    if set(matrix) != {"schema_version", "base_config", "run", "experiments"}:
        raise ValueError(
            "a second block requires a base config, run settings and experiment names"
        )
    overrides = matrix["run"]
    if not isinstance(overrides, dict) or set(overrides) != {
        "output_directory",
        "episodes",
    }:
        raise ValueError(
            "a second block can change only the output directory and episodes"
        )
    base_path = (source.parent / matrix["base_config"]).resolve()
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if (
        not isinstance(base, dict)
        or base.get("schema_version") != schema
        or "base_config" in base
    ):
        raise ValueError("base_config must point to a full ablation matrix")
    old_run = base["run"]
    if (ROOT / overrides["output_directory"]).resolve() == (
        ROOT / old_run["output_directory"]
    ).resolve():
        raise ValueError("a second block requires its own output directory")
    old_seeds = {episode["environment_seed"] for episode in old_run["episodes"]}
    if any(
        episode["environment_seed"] in old_seeds for episode in overrides["episodes"]
    ):
        raise ValueError("a second block must use fresh world seeds")
    return {
        **base,
        "run": {**old_run, **overrides},
        "experiments": matrix["experiments"],
    }, base_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/llm_dm/ablation.yaml"
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.local")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--only", action="append", help="experiment name, repeat to select more"
    )
    parser.add_argument("--output", type=Path, help="override the result directory")
    parser.add_argument(
        "--list", action="store_true", help="show the selected experiments"
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="use offline providers and a separate result directory",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        help="override the turn limit for offline checks",
    )
    args = parser.parse_args()
    if args.jobs < 1 or (args.max_episode_steps is not None and not args.fake):
        parser.error("--jobs must be positive and --max-episode-steps requires --fake")

    source = args.config.resolve()
    try:
        matrix, profile_source = load_matrix(source)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    run = matrix["run"]
    if (
        len(run["episodes"]) != 5
        or len({e["environment_seed"] for e in run["episodes"]}) != 5
    ):
        parser.error("declare exactly five distinct worlds")
    actors, learners = matrix["actors"], matrix["learners"]
    fixed_actor = matrix["design"]["fixed_actor"]
    fixed_learner = matrix["design"]["fixed_learner"]
    pairs = [(fixed_actor, name) for name in learners] + [
        (name, fixed_learner) for name in actors if name != fixed_actor
    ]
    experiments = {
        validate_experiment_name(f"{actor}--{learner}"): (
            actors[actor],
            learners[learner],
        )
        for actor, learner in pairs
    }
    if "experiments" in matrix:
        selected = matrix["experiments"]
        if (
            not isinstance(selected, list)
            or not selected
            or not all(
                isinstance(name, str) and name in experiments for name in selected
            )
            or len(set(selected)) != len(selected)
        ):
            parser.error(
                "experiments must list distinct experiment names from the base matrix"
            )
        experiments = {name: experiments[name] for name in selected}
    if args.only:
        unknown = set(args.only) - experiments.keys()
        if unknown:
            parser.error(f"unknown experiment: {', '.join(sorted(unknown))}")
        experiments = {name: experiments[name] for name in dict.fromkeys(args.only)}

    output = args.output or ROOT / run["output_directory"]
    if args.fake and args.output is None:
        output = output.with_name(f"{output.name}-fake")
    output = output.resolve()
    budgets = dict(run["budgets"])
    budgets["price_sheet"] = str(
        (profile_source.parent / budgets["price_sheet"]).resolve()
    )
    with TemporaryDirectory(prefix="aresim-ablation-") as stage:
        configs = {}
        progress = {}
        for name, (actor, learner) in experiments.items():
            payload = {
                **run,
                "experiment_name": name,
                "output_directory": str(output),
                "budgets": budgets,
                "actor": {**matrix["actor_defaults"], **actor},
                "learner": {**matrix["learner_defaults"], **learner},
            }
            if args.fake:
                payload["transport"] = "fake"
            if args.max_episode_steps is not None:
                payload["max_episode_steps"] = args.max_episode_steps
            path = Path(stage) / f"{name}.yaml"
            path.write_text(yaml.safe_dump(payload, sort_keys=False))
            config = load_llm_dm_run(path)
            configs[name] = path
            if args.list:
                print(f"{name}: {actor['model']} + {learner['model']}")
            else:
                progress[name] = ExperimentProgress(config)
        if args.list:
            return 0

        completed = stopped = 0
        display = ProgressDisplay()
        stopping = Event()
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            jobs = {
                pool.submit(run_job, path, args.env_file.resolve(), stopping): name
                for name, path in configs.items()
            }

            def refresh(messages: list[str]) -> None:
                active = [jobs[job] for job in jobs if job in pending and job.running()]
                queued = len(pending) - len(active)
                display.show(
                    [
                        f"Batch {bar(completed, len(jobs))} {completed}/{len(jobs)} complete"
                        f" | {len(active)} active | {queued} queued | {stopped} stopped",
                        *(progress[name].line() for name in active),
                    ],
                    messages,
                )

            pending = set(jobs)
            try:
                refresh([])
                while pending:
                    finished, pending = wait(
                        pending, timeout=0.5, return_when=FIRST_COMPLETED
                    )
                    messages = []
                    for job in finished:
                        try:
                            status, detail = job.result()
                        except Exception as error:
                            status, detail = "failed", str(error)
                        if status in SUCCESS:
                            completed += 1
                        else:
                            stopped += 1
                        messages.append(f"{jobs[job]}: {status} — {detail}")
                    refresh(messages)
            except KeyboardInterrupt:
                stopping.set()
                for job in jobs:
                    job.cancel()
                return 130
        return 2 if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
