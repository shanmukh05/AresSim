"""``aresim-rl`` CLI for training, evaluation, inspection, and W&B reports.

Heavy optional dependencies load only for the subcommand that needs them.

**Last updated:** September 12, 2026

**Commands:** ``train``, ``evaluate``, ``compare``, ``report``, ``inspect``, ``rollout``.

**Entry point:** ``aresim-rl`` (console script).

**See also:** :mod:`aresim.training.experiments`, :mod:`aresim.algorithms.ppo.train`,
:mod:`aresim.algorithms.dqn.train`, :mod:`aresim.training.evaluation`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aresim-rl", description="Train and evaluate AresSim learned policies")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="run a validated experiment YAML")
    train.add_argument("config", type=Path)
    train.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE")
    train.add_argument(
        "--resume-from",
        dest="resume_from",
        default=None,
        help="continue a prior run from its Ray state or checkpoints/<label>/checkpoint.json",
    )
    evaluate = commands.add_parser("evaluate", help="evaluate a frozen checkpoint on fixed seeds")
    evaluate.add_argument("run", type=Path)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", type=Path, help="new evaluation output directory")
    evaluate.add_argument("--action-mode", choices=("deterministic", "stochastic", "both"))
    evaluate.add_argument("--allow-environment-mismatch", action="store_true")
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    compare = commands.add_parser("compare", help="score several checkpoints on one shared world list")
    compare.add_argument("config", type=Path)
    commands.add_parser("report", help="generate W&B metric plots under <run>/reports/").add_argument("run", type=Path)
    inspect = commands.add_parser("inspect", help="print a run manifest or checkpoint sidecar")
    inspect.add_argument("path", type=Path)
    rollout = commands.add_parser("rollout", help="run the llm_dm Actor/Memory/LLM-Learner workflow")
    rollout.add_argument("config", type=Path, nargs="?")
    rollout.add_argument("--config", dest="config_option", type=Path)
    rollout.add_argument("--env-file", type=Path, help="load configured LLM credential names from a dotenv file")
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
        from . import run_experiment

        spec = load_experiment_with_overrides(arguments.config, tuple(arguments.overrides))
        print(run_experiment(spec, resume_from=arguments.resume_from))
        return 0
    if arguments.command == "evaluate":
        from .evaluation import evaluate_checkpoint

        checkpoint = _checkpoint_path(arguments.run, arguments.checkpoint)
        output = arguments.output or arguments.run / "evaluation" / f"manual-{arguments.split}"
        print(evaluate_checkpoint(checkpoint, split=arguments.split, output_directory=output, action_mode=arguments.action_mode,
                                  allow_environment_mismatch=arguments.allow_environment_mismatch))
        return 0
    if arguments.command == "compare":
        from .panel import run_panel

        print(run_panel(arguments.config))
        return 0
    if arguments.command == "report":
        from .reports import generate_report

        print(generate_report(arguments.run))
        return 0
    if arguments.command == "rollout":
        import logging
        import yaml
        from .seeds import resolve_checkout_file
        source = arguments.config_option or arguments.config
        if source is None or (arguments.config_option and arguments.config):
            raise ValueError("supply exactly one rollout config path")
        payload = yaml.safe_load(resolve_checkout_file(source, kind="rollout config").read_text())
        if not isinstance(payload, dict):
            raise ValueError("rollout config must be an object")
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        if payload.get("agent") != "llm_dm":
            raise ValueError("rollout agent must be llm_dm")
        from ..algorithms.llm_dm.config import load_llm_dm_run, load_role_credentials
        from ..algorithms.llm_dm.rollout import run_llm_dm
        config = load_llm_dm_run(source)
        if arguments.env_file:
            load_role_credentials(config, arguments.env_file)
        manifest = run_llm_dm(config)
        print(manifest)
        return 0 if json.loads(manifest.read_text())["status"] == "completed" else 2
    path = arguments.path
    if path.is_dir():
        manifest = path / "manifest.json"
        path = manifest if manifest.is_file() else path / "checkpoint.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
