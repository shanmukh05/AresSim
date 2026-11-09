#!/usr/bin/env python3
"""Benchmark a bounded real training pipeline on the current CPU or NVIDIA GPU.

Runs the normal trainer, collectors, replay, checkpoint export and frozen inference.
Tracking is disabled; output stays under the explicitly selected artifact directory.
"""
from __future__ import annotations

import argparse
import cProfile
from dataclasses import replace
import json
from pathlib import Path
import platform
import pstats
import sys
import time


def main() -> int:
    """Record end-to-end wall time, phase metrics, driver profile and legal inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE")
    parser.add_argument("--profile", action="store_true", help="record cProfile data for the driver (adds overhead)")
    args = parser.parse_args()

    import torch
    from aresim.factory import make_env
    from aresim.training import load_experiment_with_overrides, make_checkpoint_agent, run_experiment

    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    spec = load_experiment_with_overrides(args.config, tuple(args.overrides))
    if hasattr(spec.resources, "device"):
        resources = replace(spec.resources, device=args.device)
    else:
        resources = replace(spec.resources, gpus_per_learner=1.0 if args.device == "cuda" else 0.0)
    output = args.output.expanduser().resolve()
    spec = replace(spec, resources=resources,
        algorithm_config=replace(spec.algorithm_config, total_environment_steps=args.steps),
        artifacts=replace(spec.artifacts, root=str(output)), tracking=replace(spec.tracking, mode="disabled"))
    spec.validate()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / f"{spec.experiment_id}_{spec.trial_id}_benchmark.json"
    if report_path.exists():
        parser.error(f"benchmark already exists: {report_path}")
    profiler = cProfile.Profile() if args.profile else None
    if profiler is not None:
        profiler.enable()
    started = time.perf_counter()
    try:
        directory = run_experiment(spec)
        if args.device == "cuda":
            torch.cuda.synchronize()
    finally:
        elapsed = time.perf_counter() - started
        if profiler is not None:
            profiler.disable()
            profiler.dump_stats(str(output / f"{spec.experiment_id}_{spec.trial_id}.pstats"))
            with (output / f"{spec.experiment_id}_{spec.trial_id}_profile.txt").open("w") as handle:
                pstats.Stats(profiler, stream=handle).sort_stats("cumulative").print_stats(50)

    agent = make_checkpoint_agent(directory / "checkpoints/final/checkpoint.json")
    environment = make_env(agent.environment_config, max_episode_steps=8, audit=False)
    result = environment.reset(seed=33)
    agent.reset(33)
    actions = []
    for _ in range(8):
        action = agent.act(result.observation, result.action_mask)
        if not result.action_mask[action]:
            raise RuntimeError("exported checkpoint selected an illegal action")
        actions.append(action)
        result = environment.step(action)
        if result.terminated or result.truncated:
            break
    metrics_path = directory / "last_metrics.json"
    manifest = json.loads((directory / "manifest.json").read_text())
    steps = manifest["environment_steps"]
    report = {"algorithm": spec.algorithm, "device": args.device, "platform": platform.platform(),
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
              "wall_seconds_including_startup_and_checkpoints": elapsed,
              "environment_steps": steps, "end_to_end_steps_per_second": steps / elapsed,
              "checkpoint_inference_actions": actions, "run_directory": str(directory),
              "metrics": json.loads(metrics_path.read_text()) if metrics_path.is_file() else {},
              "driver_profile_enabled": args.profile}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "metrics"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
