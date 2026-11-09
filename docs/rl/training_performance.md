# Training performance

Measured November 7, 2026 against commit `c84dbe5`, using the same model widths, search counts, replay ratios, sequence lengths, burn-in and update budgets. Optimizations target NVIDIA CUDA training on Vast.ai. Local measurements below use an Apple Silicon CPU, PyTorch 2.13, one Torch thread and real simulator histories; they are **component timings, not GPU or full-run speedup claims**.

## Measured changes

Reference-sized models consumed four scripted simulator episodes from seeds 3, 33, 35 and 7, capped at 256 steps (actual lengths 204, 253, 256 and 216). Learner and collector measurements use one warmup and the median of three measured calls; replay uses ten calls and warm caches. Both runs used the same driver profiler. Setup/data generation is excluded from these component measurements.

| Component | Before | After | Speedup |
|---|---:|---:|---:|
| Dreamer replay batch packing | 21.34 ms | 0.78 ms | 27.4× |
| Dreamer collector history reconstruction | 528.20 ms | 189.32 ms | 2.8× |
| Dreamer inference snapshot creation | 5.02 ms | 2.23 ms | 2.3× |
| Dreamer learner update | 968.99 ms | 770.14 ms | 1.26× |
| Plan2Explore pretrain learner update | 1,054.45 ms | 902.20 ms | 1.17× |
| Plan2Explore adapt learner update | 886.60 ms | 812.26 ms | 1.09× |
| MuZero learner update | 55.15 ms | 48.05 ms | 1.15× |
| Recurrent DQN forward/backward/optimizer | 110.46 ms | 98.16 ms | 1.13× |

Plan2Explore replay packing improved from roughly 21–22 ms to 0.78–0.84 ms. World-model policy payloads fell from 53,283,176 to 19,839,176 bytes (63% fewer bytes). MuZero replay sampling changed little in this small population, and no material search/reanalysis acceleration is claimed from the timing variation.

Raw measurements were saved as `performance_cpu_2026-11-07.json`. This local artifact is not included in the checkout. Profiling artifacts for this session are `/tmp/ares-speed-reference-before/` and `/tmp/ares-speed-packed-after/`. These measurements do not establish long-budget mission learning or GPU throughput.

## What changed

- **Dreamer and both Plan2Explore phases:** replay caches raw observation columns, gathers windows with NumPy indexing, and sends only actual prefix frames to the encoder. It retains full recurrent prefixes, the original batch-row layout and latent RNG consumption. Raw caches are invalidated on append/eviction and excluded from checkpoints. They trade additional host RAM for less packing work; model embeddings are never cached across weight updates.
- **World-model collection:** encode history frames in bounded chunks, then reconstruct the posterior in the original time order. Posterior-only execution skips unused prior heads and discarded latent samples. Collectors receive encoder/RSSM/actor parameters, omitting decoder and critic weights; historical full snapshots still load through strict inference submodules.
- **CUDA hot paths:** masked reductions keep fixed shapes rather than calling boolean nonzero/indexing and host validity branches. Legality diagnostics reuse existing logits and vector reductions; scalar metrics cross to CPU together. Nonfinite loss/gradient rejection remains before optimizer steps. Internal actor masks are already validated on CPU or constructed with legal Wait; public mask validation remains enabled by default.
- **MuZero:** reuse crop embeddings within each immutable collection wave, clearing history on episode reset and every weight publication. Cache exact per-episode priority totals, normalize each selected episode once per sample batch, and compute two-bin scalar losses without allocating dense targets. Priority errors share the diagnostic device transfer. Checkpoint export copies detached CPU tensors without moving the live CUDA model or replacing its parameter storage.
- **Recurrent DQN:** encode burn-in without retaining an unused encoder graph, retain the contiguous online/target unroll and detached recurrent burn-in, and use the existing `ARESIM_VALIDATE_MODEL_OUTPUTS=1` diagnostic switch consistently instead of forcing GPU mask-content synchronization.
- **CUDA optimizers:** native trainers and recurrent RLlib DQN use fused Adam/AdamW on CUDA, preserving RLlib's learning-rate scheduler and native checkpoint state. CPU paths retain ordinary optimizers. Fused CUDA execution has not been exercised on this local machine.

## Pipeline verification

Manual comparisons used real observations, rewards and legal actions. Packed replay preserved window, previous-action and ending semantics; replay serialization/reload was checked. Dreamer and both Plan2Explore phases preserved latent RNG state after a learner update; largest post-update parameter differences were below `1e-6`. MuZero's sparse scalar loss/gradients matched the dense calculation exactly, cached collector actions/root values matched, and sampled positions/importance weights matched before and after priority updates. Historical checkpoint formats are retained. No unit tests were created.

All five 256-step training smoke runs passed with the normal Ray collectors, replay, optimizer updates, checkpoint export, generic checkpoint reload and eight legal real-world inference steps. MuZero retained 25 updates; Dreamer and each Plan2Explore phase retained 40. Recurrent DQN completed its native four reporting iterations. Small floating-point changes can produce different later stochastic training trajectories; bitwise-identical final policies are not claimed. The new benchmark CLI also completed a real 64-step Dreamer run and wrote its driver profile and hardware report. These short runs include substantial framework startup time, so they are correctness checks rather than predictions of sustained training throughput.

The optimized Dreamer run also resumed from 64 to 128 steps, restoring collector policy state and raw replay, and reloaded its final checkpoint successfully.

## Measure on the actual Vast GPU

The benchmark script runs the normal complete pipeline with tracking disabled and a finite step budget. It is included in cloud source bundles. Run from the repository root after dependency setup:

```bash
.venv/bin/python scripts/benchmark_training.py configs/dreamerv3/smoke.yaml \
  --output results/performance/dreamer --steps 512 --device cuda

.venv/bin/python scripts/benchmark_training.py configs/muzero/smoke.yaml \
  --output results/performance/muzero --steps 512 --device cuda

.venv/bin/python scripts/benchmark_training.py configs/recurrent_dqn/smoke.yaml \
  --output results/performance/recurrent --steps 512 --device cuda

.venv/bin/python scripts/benchmark_training.py configs/plan2explore/pretrain/smoke.yaml \
  --output results/performance/p2e-pretrain --steps 512 --device cuda

.venv/bin/python scripts/benchmark_training.py configs/plan2explore/adapt/smoke.yaml \
  --output results/performance/p2e-adapt --steps 512 --device cuda \
  --set algorithm_config.initialization_checkpoint=results/performance/p2e-pretrain/plan2explore_pretrain_smoke_v1/seed_7/checkpoints/final/checkpoint.json
```

Use `engine/.venv/bin/python` for the local checkout's environment. Each invocation writes a benchmark JSON with hardware identity, complete wall time, environment steps/second, phase metrics and checkpoint inference actions. `--profile` also writes driver `.pstats` and a readable cumulative-time profile; it adds overhead and does not profile separate Ray actors or CUDA kernels. Compare unprofiled runs with identical configs on the same host for throughput.

Smoke presets establish execution, not GPU saturation. For reference-sized measurements, choose a budget that reaches warmup and completed episodes, retain matching model settings for Plan2Explore transfer, and keep collector CPU/RAM demand within the selected instance allocation. Search simulations, full-prefix reconstruction and imagined horizon remain real computational costs. No paid GPU instance was created for this optimization pass.
