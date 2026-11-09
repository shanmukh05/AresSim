#!/usr/bin/env python3
"""Checkpoint shared-instance world-model jobs before their parent run stops compute."""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path


def main() -> None:
    import torch
    from aresim.training import load_experiment_with_overrides
    from aresim.training.world_models.runner import run_experiment

    root = Path(sys.argv[1]).resolve()
    job = json.loads((root / "job.json").read_text())
    arguments = job["train_args"]
    assert all(arguments[index] == "--set" for index in range(1, len(arguments), 2))
    spec = load_experiment_with_overrides(arguments[0], tuple(arguments[2::2]))
    if spec.algorithm not in {"dreamerv3", "plan2explore"} or not job.get("shared_instance"):
        raise ValueError("shared worker requires a world-model job")
    parent = Path(job["stop_with_manifest"])
    if not parent.is_absolute() or not parent.resolve().is_relative_to(Path("/root/aresim-cloud")):
        raise ValueError("parent manifest must belong to a cloud job")
    # Leave the supervisor time to save full replay/optimizer/collector state.
    deadline = time.monotonic() + max(0, job["seconds"] - 180)
    requested = False
    reason = None

    def request_stop(_signum, _frame):
        nonlocal requested
        requested = True

    def stop_requested():
        nonlocal reason
        if requested or time.monotonic() >= deadline:
            reason = "signal" if requested else "shared_job_deadline"
        else:
            try:
                state = json.loads(parent.read_text())
                if state.get("status") in {"completed", "failed", "stopped", "interrupted"}:
                    reason = "parent_finished"
                elif int(state.get("environment_steps", 0)) >= job["stop_with_steps"]:
                    reason = "parent_nearing_completion"
            except (OSError, ValueError):
                reason = "parent_status_unavailable"
        return reason is not None

    (root / "shared_worker_pid.json").write_text(json.dumps({"pid": os.getpid()}))
    # Independent local Ray runtimes must not each reserve most of the host RAM.
    import ray
    ray.init(address="local", include_dashboard=False, log_to_driver=False,
        num_cpus=spec.resources.num_collectors * spec.resources.collector_torch_threads + 1,
        num_gpus=0, object_store_memory=512 * 2**20, _memory=8 * 2**30,
        _temp_dir="/tmp/ar-" + root.name[-12:])
    if spec.resources.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(.30)
    if job.get("cuda_graph_prefix"):
        os.environ["ARESIM_CUDA_GRAPH_PREFIX"] = "1"
    # ray.init installs its own SIGTERM handler; the trainer must own it afterward.
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        directory = run_experiment(spec, stop_requested=stop_requested)
        print(json.dumps({"run_directory": str(directory), "stop_reason": reason,
            "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
            "peak_cuda_reserved_mb": torch.cuda.max_memory_reserved() / 2**20}), flush=True)
    finally:
        ray.shutdown()
        if reason:
            (root / "shared_stop_reason.json").write_text(json.dumps({"reason": reason}))


if __name__ == "__main__":
    main()
