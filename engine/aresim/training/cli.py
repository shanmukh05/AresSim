"""``aresim-rl`` CLI for training, evaluation, inspection, and W&B reports.

Heavy optional dependencies load only for the subcommand that needs them.

**Last updated:** September 1, 2026

**Commands:** ``train``, ``evaluate``, ``report``, ``inspect``.

**Entry point:** ``aresim-rl`` (console script).

**See also:** :mod:`aresim.training.experiments`, :mod:`aresim.algorithms.ppo.train`,
:mod:`aresim.training.evaluation`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aresim-rl", description="Train and evaluate AresSim RLlib policies")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="run a validated experiment YAML")
    train.add_argument("config", type=Path)
    train.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE")
    evaluate = commands.add_parser("evaluate", help="evaluate a frozen checkpoint on fixed seeds")
    evaluate.add_argument("run", type=Path)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    commands.add_parser("report", help="generate W&B metric plots under <run>/reports/").add_argument("run", type=Path)
    inspect = commands.add_parser("inspect", help="print a run manifest or checkpoint sidecar")
    inspect.add_argument("path", type=Path)
    return parser


def _checkpoint_path(run: Path, value: str) -> Path:
    requested = Path(value)
    if requested.is_file():
        return requested
    candidate = run / "checkpoints" / value / "checkpoint.json"
    if not candidate.is_file():
        raise FileNotFoundError(f"checkpoint not found: {value}")
    return candidate


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a command, keeping Ray imports out of inspect/report startup."""
    arguments = _parser().parse_args(argv)
    if arguments.command == "train":
        from .experiments import load_experiment_with_overrides
        from ..algorithms.ppo.train import run_experiment

        spec = load_experiment_with_overrides(arguments.config, tuple(arguments.overrides))
        print(run_experiment(spec))
        return 0
    if arguments.command == "evaluate":
        from .evaluation import evaluate_checkpoint

        checkpoint = _checkpoint_path(arguments.run, arguments.checkpoint)
        output = arguments.run / "evaluation" / f"manual-{arguments.split}"
        print(evaluate_checkpoint(checkpoint, split=arguments.split, output_directory=output))
        return 0
    if arguments.command == "report":
        from .reports import generate_report

        print(generate_report(arguments.run))
        return 0
    path = arguments.path
    if path.is_dir():
        manifest = path / "manifest.json"
        path = manifest if manifest.is_file() else path / "checkpoint.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
