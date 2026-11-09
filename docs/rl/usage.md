# AresSim RL Usage Guide

Updated October 7, 2026: read [pipeline corrections](pipeline_fixes_2026-10-06.md) before starting another experiment. New presets use observed navigation memory, reward/task v2 and frozen environment snapshots. Training saves periodic and final checkpoints without evaluation rollouts, validation selection or success-gate checks. Use `configs/masked_ppo/reference.yaml` directly for a fresh full-mission PPO run; no smaller delivery preset is required. Historical checkpoints require an explicit `--allow-environment-mismatch` for diagnostic evaluation and cannot resume under changed semantics.

MuZero is available through `aresim-rl train configs/muzero/smoke.yaml` after installing `./engine[muzero]`. It uses a dedicated PyTorch trainer with the same experiment, checkpoint, and evaluation commands; see the [MuZero workflow](../../engine/aresim/algorithms/muzero/DETAILS.md). The `dev.yaml` and CUDA `reference.yaml` presets live beside the smoke config.

Practical guide for running environments, training PPO, DQN, Recurrent DQN, MuZero, DreamerV3 and Plan2Explore, configuring W&B, recording trajectories, and extending components. For network architecture, PPO/DQN internals, and metric definitions, see [RL Algorithms, Training, and Evaluation](rl_quickstart.md).

## Contents

| Section | What you will find |
|---|---|
| [Setup](#setup) | Virtual environment and optional extras |
| [Train from the CLI](#train-from-the-cli) | `aresim-rl train`, configs, overrides, run layout |
| [Train on JarvisLabs](#train-on-jarvislabs) | Cloud GPU/CPU instances, upload, download `results/` |
| [Weights & Biases (W&B)](#weights--biases-wb) | YAML `tracking` block, auth, modes, logged metrics |
| [Experiment YAML](#experiment-yaml) | Checked-in configs and key fields |
| [Train from Python](#train-from-python) | `load_experiment`, `run_experiment`, checkpoints |
| [After training](#after-training) | Evaluate, compare checkpoints, inspect, and report |
| [UI policy inference](#ui-policy-inference) | Attach baselines, Decision Models, or learned-policy checkpoints in Algorithm mode |
| [Create an environment](#create-an-environment) | Framework-neutral, Gymnasium, PettingZoo factories |
| [Run baseline rollouts](#run-baseline-rollouts) | Deterministic episodes with built-in agents |
| [Record trajectories](#record-trajectories) | `aresim.trajectory.v1` datasets and validation |
| [Extend environment components](#extend-environment-components) | Observations, actions, rewards, tasks |
| [Extend rollout agents](#extend-rollout-agents) | Custom `Agent` implementations |
| [Extend learned policies](#extend-learned-policies) | `TrainingRegistry` for new algorithms |

**Notebooks**

- [Masked PPO training](../../notebooks/masked_ppo_training.ipynb) — end-to-end learned-policy training in Jupyter
- [Algorithm pipeline](../../notebooks/rl_algorithm_pipeline.ipynb) — baseline rollout, trajectory, and determinism smoke test
- [Evaluation report template](../../notebooks/evaluation_report_template.ipynb) — optional manual W&B analysis notebook (not executed by the CLI)

---

## Setup

The canonical dependency manifest is [`engine/pyproject.toml`](../../engine/pyproject.toml). From the repository root:

```bash
python3 -m venv engine/.venv
engine/.venv/bin/python -m pip install -e './engine[dev,rllib,notebook]'
engine/.venv/bin/python -m ipykernel install --user --name aresim --display-name "AresSim (.venv)"
```

| Extra | Installs | Use when |
|---|---|---|
| `dev` | httpx | Local API client work |
| `env` | NumPy, Gymnasium, PettingZoo | Baselines and rollouts only |
| `rllib` | Ray, RLlib, PyTorch, W&B | Masked PPO, feedforward/recurrent DQN training and evaluation |
| `world_models` | PyTorch, Ray Core, environment/tracking/report dependencies | DreamerV3 and Plan2Explore; no RLlib/Tune |
| `muzero` | PyTorch, W&B, environment packages, Ray Core 2.58 | MuZero Ray collection and frozen evaluation; no RLlib/Tune required |
| `llm_dm` | Native HTTPS + JSON contracts | Decision Model or LLM Actor, shared Memory and required LLM Learner |
| `notebook` | Jupyter, papermill | Running notebooks locally |

Select the `AresSim (.venv)` kernel in Jupyter. The base UI/backend install does not require NumPy, Gymnasium, Ray, PyTorch, or W&B.

Seed splits for evaluation live in [`configs/seeds/phase1_open_exploration_split_v1.yaml`](../../configs/seeds/phase1_open_exploration_split_v1.yaml). Experiment YAML lives under [`configs/`](../../configs/); training outputs go to [`results/`](../../results/) (gitignored).

Reference results are consolidated by algorithm under `results/ppo/`, `results/double_duelling_dqn/` and `results/muzero/`. Former `_v2` runs now live alongside older runs with the same trial names. Reference configs target these folders; use a new `trial_id` for a fresh run. Historical frozen experiment IDs and W&B identities are preserved. See the [results layout](../../results/README.md).


---

## Train from the CLI

Masked PPO, mask-aware DQN, and MuZero are the implemented learned algorithms. Training is driven by validated YAML under `configs/<algorithm>/` and the `aresim-rl` console script.

### Checked-in configs

| File | Environment steps | W&B | Purpose |
|---|---:|---|---|
| [`configs/masked_ppo/smoke.yaml`](../../configs/masked_ppo/smoke.yaml) | 8,192 | disabled | Fast pipeline smoke test (2-seed eval, no trajectory export) |
| [`configs/masked_ppo/dev.yaml`](../../configs/masked_ppo/dev.yaml) | 102,400 | online | CPU development run (4 env runners, 2 PPO epochs, summary-only eval) |
| [`configs/masked_ppo/reference.yaml`](../../configs/masked_ppo/reference.yaml) | 2,097,152 | online | Longer reference training (28 env runners + CUDA learner; needs a GPU host such as JarvisLabs L4) |
| [`configs/masked_dqn/smoke.yaml`](../../configs/masked_dqn/smoke.yaml) | 4,096 | disabled | Masked DQN pipeline smoke (uniform replay, n-step=1) |
| [`configs/masked_dqn/dev.yaml`](../../configs/masked_dqn/dev.yaml) | 102,400 | online | Local DQN development (2 env runners, summary-only eval) |
| [`configs/masked_dqn/reference.yaml`](../../configs/masked_dqn/reference.yaml) | 2,097,152 | online | Double Dueling DQN sample-credit trial (`seed_3_sample_credit_030`, 28 env runners + CUDA learner; n-step=3, time cost −0.025, capped sample bonus +0.30) |
| [`configs/muzero/reference.yaml`](../../configs/muzero/reference.yaml) | 2,097,152 | online | MuZero learning-path sample-credit trial (`seed_7_sample_credit_030_learning_v3`, eight CPU collectors × four environments, one CUDA learner, 50 simulations/action; time cost −0.025, capped sample bonus +0.30) |
| [`configs/recurrent_dqn/smoke.yaml`](../../configs/recurrent_dqn/smoke.yaml) | 512 | disabled | LSTM sequence-replay DQN smoke |
| [`configs/dreamerv3/smoke.yaml`](../../configs/dreamerv3/smoke.yaml) | 256 | disabled | Structured RSSM and imagined behavior smoke |
| [`configs/plan2explore/pretrain/smoke.yaml`](../../configs/plan2explore/pretrain/smoke.yaml) | 256 | disabled | Reward-free disagreement exploration smoke |
| [`configs/plan2explore/adapt/smoke.yaml`](../../configs/plan2explore/adapt/smoke.yaml) | 256 | disabled | Separate task adaptation; requires initializer |
| [`configs/llm_dm/fake_smoke.yaml`](../../configs/llm_dm/fake_smoke.yaml) | 32 (rollout) | n/a | Offline llm_dm Actor/Memory/Learner smoke |
| [`configs/llm_dm/10-games.yaml`](../../configs/llm_dm/10-games.yaml) | 128 per game | n/a | Hybrid ten-game learning run (Actor and Learner credentials required) |

The llm_dm Learner improves prompt advice between games; model weights are unchanged. See the [llm_dm workflow](../llm_dm/implementation.md). Roll it out with:

```bash
engine/.venv/bin/aresim-rl rollout configs/llm_dm/fake_smoke.yaml
```

Run from the **repository root** so seed manifests and artifact paths resolve correctly:

```bash
# PPO smoke test (no W&B, ~minutes on CPU)
engine/.venv/bin/aresim-rl train configs/masked_ppo/smoke.yaml

# DQN smoke test
engine/.venv/bin/aresim-rl train configs/masked_dqn/smoke.yaml

# Development run (requires W&B login — see below)
engine/.venv/bin/aresim-rl train configs/masked_ppo/dev.yaml
engine/.venv/bin/aresim-rl train configs/masked_dqn/dev.yaml

# Override hyperparameters without editing YAML
engine/.venv/bin/aresim-rl train configs/masked_ppo/dev.yaml \
  --set algorithm_config.learning_rate=0.0001 \
  --set resources.num_env_runners=2
```

`--set path=value` accepts dotted paths into the resolved experiment dict. Unknown paths, type mismatches, unsafe YAML, incompatible batch sizes, and missing seed manifests fail **before** Ray or W&B start.

The built-in PPO/DQN collectors use a compact pre-step reward snapshot and skip audit checksums during training. Recorded trajectories and the normal simulator/API remain fully audited. Evaluation settings apply to later manual evaluation; add `--set evaluation.record_trajectories=true` to save that setting for UI replay export. Training never launches those rollouts. PPO dev uses four CPU EnvRunners and two update epochs, while the 8,192-step smoke preset is only a wiring check, not a learning-quality benchmark. Set `ARESIM_VALIDATE_MODEL_OUTPUTS=1` for stricter per-forward model checks during diagnosis; this may slow GPU training.

**DQN reference update, October 8:** run `seed_3_sample_credit_030` with learner seed 3, 28 CPU environment runners and one CUDA learner. It preserves the seed_3 budget of 2,097,152 steps, epsilon schedule, mission, maps, and DQN hyperparameters. The single reward change is `environment.reward_overrides.samples_collected: 0.30`: +0.30 per 0.5 kg acquired, capped at the 4 kg sample quota (maximum 2.4). Acquisition is delivered plus onboard sample mass, so Unload cannot reset the bonus. Partial quota increments receive proportional credit. `new_scan` remains zero; scanning excess rocks earns no acquisition bonus. The new shaped profile/breakdown are `aresim.reward.shaped_train.v3` / `aresim.reward.mission.v3`; zero-default and sparse rewards retain v2. PPO presets retain zero acquisition credit; the October 10 MuZero reference now enables the same bonus. DQN’s sample-credit run completed 9/32 greedy validation missions; MuZero’s learning quality remains unverified. See [seed_3 evidence and next experiment](../../results/double_duelling_dqn/seed_3/learnings.md).

**MuZero reference update, October 11:** `configs/muzero/reference.yaml` uses the sole Ray collection path with eight CPU collectors × four environments and one CUDA learner in fresh trial `seed_7_sample_credit_030_parallel`. Batched MCTS/reanalysis preserve the architecture, gamma 0.995, 50 simulations/action, 0.1 updates/step, 2,097,152-step budget and corrected reward. Default batch size remains 32. Checkpoints save at quiescent collection-wave boundaries every 81,920 requested steps, retaining three intermediate snapshots plus final and preserving unfinished trajectories for resume. A bounded same-A30 batch-256/eight-thread comparison measured 13.52 → 111.26 steps/s, about 8.2× faster. CPU/Ray partial resume and CUDA checkpoint/RNG restore passed; mission-learning acceptance remains pending. See [implementation, verification and commands](muzero_parallel_2026-10-11.md). Training performs no automatic evaluation.

**MuZero checkpoint/loss corrections, October 13:** that trial was `seed_7_sample_credit_030_parallel_lossnorm`. Numeric checkpoint ordering now preserves the newest steps across digit widths and resumes the newest snapshot even when final is older. The learner divides weighted losses by total importance weight and reports unweighted losses, weight statistics and effective sample size. Manual reports export/plot these fields. The model, reward, search and replay sampling settings remain unchanged. See [math and verification](muzero_learning_fixes_2026-10-13.md). Previous loss magnitudes are not directly comparable.

**MuZero learning-path follow-up, October 13:** the current reference is `seed_7_sample_credit_030_learning_v3` with the unchanged full 2,097,152-step budget. It refreshes all real policy positions in selected minibatch unrolls, balances positive Scan/Extract/Unload legality labels and adds temporal consistency against stopped recorded-history targets. Reports include target age/coverage, consistency and legality recall/false-positive counts. Existing checks and tiny Ray resume pass; no CUDA training or benchmark was launched, and the temporary instance was destroyed at the user's request. See [settings, verification and ablations](muzero_learning_v3_2026-10-13.md).

The −0.025 cost still applies to every nonterminal action; `5 × (1 − 0.995)` removes the failure/time-cost incentive for postponing an otherwise identical failure. Wait/Service legality, mission requirements, and engine/UI scores are unchanged. The new trial rejects an existing output directory, preserving the original seed_3 artifacts. Launch on the CUDA host:

```bash
engine/.venv/bin/aresim-rl train configs/masked_dqn/reference.yaml

# After training: 32 greedy episodes and 96 sampled episodes (three action seeds/map).
engine/.venv/bin/aresim-rl evaluate \
  results/double_duelling_dqn/seed_3_sample_credit_030 \
  --checkpoint final --split validation --action-mode both
```

Training only saves checkpoints; the evaluation command is a separate manual step. Compare mission success and sample-quota completion, ice delivery, and post-ice camping. Raw shaped returns differ across rewards. Changed source provenance prevents resuming old training; use a fresh trial. Historical checkpoint diagnostics require `--allow-environment-mismatch` and are labelled incomparable by the strict evaluator, even though zero-default v2 reward behavior is retained.

With RLlib 2.58, 28 runners, fragments of 32, minibatches of 256, and `training_intensity: 4.0`, the scheduler chooses one collection round followed by 14 replay updates: 896 new steps and 3,584 replayed samples, preserving a replay-to-collection ratio of 4. A requested minimum of 4,096 steps per reporting iteration is rounded by collection batches to approximately 4,480; the 20-iteration checkpoint frequency therefore corresponds to roughly 89,600 actual steps. Labels and stopping use observed step counters. More runners improve collection capacity, but learner throughput still limits overall speed. Launch with `--set artifacts.reject_existing=true` to prevent reusing an existing output directory.

### Run directory layout

Each trial writes to:

```text
results/<experiment_id>/<trial_id>/
├── resolved_config.yaml    # fully resolved experiment (including overrides)
├── manifest.json           # status, config hash, W&B run id, artifact inventory
├── status.json             # running | completed | failed
├── checkpoints/
│   ├── step_081920/        # mid-training exports (UI-loadable while training runs)
│   │   ├── checkpoint.json
│   │   └── native/
│   └── final/
│       ├── checkpoint.json # aresim.checkpoint.rllib.v1 sidecar
│       └── native/         # RLlib checkpoint files
└── evaluation/             # created later by explicit manual evaluation
```

`experiment_id` and `trial_id` come from the YAML (`rllib_masked_ppo_smoke`, `seed_7`, etc.). Set `artifacts.reject_existing: false` in YAML or change `trial_id` to reuse a directory.

---

## Train on JarvisLabs

The CLI above is the same on a cloud box. To create a JarvisLabs GPU or CPU instance, upload the training payload, install Python 3.12, run `aresim-rl train`, and download the entire `results/` folder, see [Train on JarvisLabs](jarvislabs.md).

From the repository root, after `jl setup` and `export WANDB_API_KEY=...`:

```bash
python3 scripts/jarvislabs/cloud_train.py run --gpu L4 \
  --config configs/masked_ppo/reference.yaml
python3 scripts/jarvislabs/cloud_train.py run --gpu L4 \
  --config configs/masked_dqn/reference.yaml
```

That helper pauses the instance when the run finishes. Use `--detach` to leave training running, then `fetch` and `down` later. Do not `jl run .` from the repo root: it uploads local `results/` and the pytorch template's Python 3.10 cannot install AresSim.

---

## Weights & Biases (W&B)

W&B is the sole application-level training log. Configure it in the YAML `tracking` block — never put API keys in config files, manifests, or command arguments.

```yaml
tracking:
  mode: online          # online | offline | disabled
  project: aresim       # W&B project name
  entity: my-team       # optional; defaults to your W&B default entity
  group: reference      # optional; groups related runs in the UI
  tags: [dev, cpu]      # optional list of strings
```

| `mode` | Behavior |
|---|---|
| `online` | Live sync to wandb.ai. Requires authentication before training starts. Used by `dev.yaml` and `reference.yaml`. |
| `offline` | Writes local W&B files; sync later with `wandb sync`. Useful on air-gapped machines. |
| `disabled` | No W&B callbacks or evaluation logging. Used by `smoke.yaml` for fast local/CI runs. |

### Authenticate

For `mode: online`, log in once per machine:

```bash
engine/.venv/bin/wandb login
```

Training calls `wandb.login()` before Ray workers start. If login fails, the run is marked failed in `status.json` and no partial metrics are silently dropped.

### Run identity

- **W&B run name:** `{experiment_id}-{trial_id}` (e.g. `rllib_masked_ppo_dev-seed_7`)
- **W&B run id:** random 24-character hex string assigned at training start (`wandb_run_id` in `manifest.json`; separate from `config_hash`)

Each `aresim-rl train` invocation gets a fresh W&B id so a deleted remote run cannot block future uploads for the same YAML.

### What gets logged

During training, [`AresMetricsCallback`](../../engine/aresim/training/rllib/metrics.py) maps RLlib metrics to stable names. The learning-curve x-axis is `train/environment_steps`. Logged groups include shaped return, episode length, action counts, invalid actions, policy/value losses (PPO), TD error and Q stats (DQN), entropy, KL, clip fraction, explained variance, and throughput.

Training logs learner and collection metrics to W&B. It does not append frozen validation metrics or run post-training evaluation. Use the explicit evaluation and report commands for later analysis.

Override tracking from the CLI without editing YAML:

```bash
engine/.venv/bin/aresim-rl train configs/masked_ppo/smoke.yaml \
  --set tracking.mode=online \
  --set tracking.project=my-sandbox \
  --set tracking.tags=[notebook,debug]
```

### Offline runs

```yaml
tracking:
  mode: offline
  project: aresim
```

Train normally, then sync when network is available:

```bash
engine/.venv/bin/wandb sync results/<experiment_id>/<trial_id>/wandb/
```

---

## Experiment YAML

Configs use schema `aresim.experiment.v1`. Each file selects a registered algorithm, model, environment components, hyperparameters, resources, a manual evaluation protocol, checkpoint policy, tracking, and artifact root. Legacy `evaluation.interval_environment_steps` is accepted when reading older configs/snapshots but is ignored; only `checkpoint.interval_environment_steps` schedules training-time saves.

Minimal skeleton:

```yaml
schema_version: aresim.experiment.v1
experiment_id: my_masked_ppo_trial
trial_id: seed_0
algorithm: masked_ppo
model: local_cnn_actor_critic
learner_seed: 0

environment:
  observation: local
  action: discrete
  reward: shaped_train
  task: resource_mission
  max_episode_steps: 250

algorithm_config:
  total_environment_steps: 102400
  rollout_batch_size: 4096
  learning_rate: 0.0001

evaluation:
  seed_manifest: configs/seeds/phase1_open_exploration_split_v1.yaml
  split: validation
  record_trajectories: true

checkpoint:
  interval_environment_steps: 20480
  keep: 5

tracking:
  mode: online
  project: aresim

artifacts:
  root: results
  reject_existing: true
```

Add new algorithms under `configs/<algorithm>/`. The `algorithm` field must match a name registered in `TrainingRegistry` (`masked_ppo` or `masked_dqn` today).

---

## Train from Python

The same path as the CLI, for notebooks and scripts:

```python
from aresim.training import load_experiment, make_checkpoint_agent, run_experiment

spec = load_experiment("configs/masked_ppo/smoke.yaml")
run_directory = run_experiment(spec)  # starts Ray, trains, saves checkpoints, shuts down Ray

agent = make_checkpoint_agent(run_directory / "checkpoints/final/checkpoint.json")
```

`run_experiment` accepts a YAML path or a resolved `ExperimentSpec`. Keyword arguments:

| Argument | Default | Purpose |
|---|---|---|
| `report` | `False` | Generate plots from existing W&B history/artifacts (requires `tracking.mode: online`; does not run rollouts) |
| `registry` | built-in | Custom `TrainingRegistry` for extended algorithms |
| `component_registry` | built-in | Custom environment component registry |

For a notebook-friendly flow with explicit control, see [Masked PPO training](../../notebooks/masked_ppo_training.ipynb).

The former `evaluate` training argument has been removed. Training does not generate `checkpoints/best` or enforce `evaluation.minimum_success_rate`; those are manual analysis operations. A completed training run means the training budget finished, not that its policy passed a validation gate. The usual `checkpoint.keep` rotation still applies, so increase it before training if later analysis needs more saved snapshots.

---

## After training

All commands take the **run directory** (`results/<experiment_id>/<trial_id>/`), not the repository root.

```bash
# Print manifest.json or checkpoint sidecar
engine/.venv/bin/aresim-rl inspect results/rllib_masked_ppo_smoke_v2/seed_7

# Evaluate a checkpoint on the fixed validation or test split
engine/.venv/bin/aresim-rl evaluate results/rllib_masked_ppo_dev_v2/seed_7 \
  --checkpoint final \
  --split validation

# Generate local training plots from W&B history (tracked runs only)
engine/.venv/bin/aresim-rl report results/rllib_masked_ppo_dev_v2/seed_7
```

`--checkpoint` accepts a path to `checkpoint.json` or a checkpoint folder name under `checkpoints/` (e.g. `final`).

### Compare checkpoints on the same worlds

`aresim-rl compare` generates one list of worlds, runs every configured checkpoint on that list, then writes raw rows and bar charts. Each policy still uses its own frozen environment, so shaped return follows that checkpoint's reward profile. Ice, samples, engine return, success, length, and ending reasons come from the shared mission.

```bash
engine/.venv/bin/aresim-rl compare configs/evaluation/reference_panel.yaml
```

The reference panel plays the first 20 validation seeds (2000–2019), one deterministic episode each. Output goes to `results/reports/reference_panel_loopfix/`:

| File | Contents |
|---|---|
| `environments.json` | The generated world and agent seeds, written before any policy runs |
| `episodes.jsonl` | One raw row per policy per game |
| `summary.json` / `summary.csv` | Means, population standard deviations, reward weights, ending reasons |
| `figures/*.png` | One bar chart per metric, `figures/all_metrics.png`, and `figures/action_paths.png` |

`action_paths.png` is the same normalized action-mix view as `scripts/plot_reference_action_paths.py`, with every panel algorithm on one figure. Each raw row's `actions` list is the chosen action id at every step, so later plots can redraw paths without another rollout.

A new panel is another YAML file with `schema_version: aresim.evaluation.panel.v1`. Set `episode_count`, `split` (`validation` or `test`), `action_mode` (`deterministic` or `stochastic`), `workers`, and a `checkpoints` list of `name` / `path` entries pointing at `checkpoint.json`. `output` is a repository-relative directory and is not overwritten. Set `allow_environment_mismatch: true` only for a diagnostic run against checkpoints frozen before the current environment sources. The summary then records `environment_comparable: false`. Observation, action, task, and reward schemas are still checked. A stored `shaped_train.v1` or `phase1_resource_mission_v1` name is accepted when the current code reports the v2 name of that same built-in profile. PPO `seed_33` `final` is the reference checkpoint: it delivered the most ice in `configs/evaluation/ppo_last_checkpoints.yaml`, and no PPO trial succeeded. Its frozen reward has no sample-credit override, so its shaped return is not on the same scale as the sample-credit runs. Plan2Explore is `adapt/seed_11_loopfix` step `1048576`. The reference panel sets the diagnostic flag because those six checkpoints predate the current environment sources.

Evaluation writes `evaluation/<label>/summary.json` and optional trajectory shards only when requested through `aresim-rl evaluate` or the Python evaluation API. PPO, DQN and MuZero training never invoke the evaluator, including at completion. The checkpoint evaluator runs only the learned policy and contains no baseline scheduling or opt-in option. Baseline comparisons belong to separate analysis using the existing [rollout tools](#run-baseline-rollouts); new checkpoint summaries contain no `baseline_comparisons` field. Historical reports can still read that field from old artifacts.

For manual evaluation, `record_trajectories: false` and `evaluation.workers > 1` use a CPU process pool. The default `workers: 1` and trajectory recording use the sequential path. PPO, DQN and MuZero retain their configured deterministic/stochastic evaluation modes and repeats for these explicit runs. Logs print `Frozen evaluation N/M` on the pool path. `aresim-rl report` reads W&B history plus local evaluation artifacts without restarting Ray or collecting episodes.

This October 7 change applies when the updated package is loaded. An already-running training process retains its loaded code. Runtime source verification continues to reject resuming a checkpoint produced by different trainer/evaluator code.

### Resume training

Continue a prior run from its Ray experiment state or an exported checkpoint sidecar. The resumed run reuses the same run directory, manifest `wandb_run_id`, and requires the same resolved experiment (`config_hash` must match).

```bash
# Resume from the default run directory for experiment_id/trial_id in the YAML
engine/.venv/bin/aresim-rl train configs/masked_ppo/reference.yaml --resume-from .

# Resume from a specific exported checkpoint under checkpoints/
engine/.venv/bin/aresim-rl train configs/masked_ppo/reference.yaml \
  --resume-from results/ppo/seed_35/checkpoints/final/checkpoint.json
```

Set `artifacts.reject_existing: false` (or use `--resume-from`) so the existing run directory is reused. Increase `algorithm_config.total_environment_steps` in YAML when extending a completed run.

---

## UI policy inference

In Algorithm mode, choose the agent, checkpoint, and Action selection in the compact **Agent settings** panel floating over the world above the bottom controls. Close the panel after configuring it using X, Escape, or the footer's **Agent settings** toggle; use the toggle to reopen it. Your selections and attached policy are preserved. The footer contains mode switching, run controls, and world setup; the settings fields stack on narrow screens.

Algorithm mode in the React UI runs policies **server-side** through the Python API. The browser never loads Ray, PyTorch, or checkpoint files.

PPO, DQN and MuZero default to **Sampling** in the UI. PPO samples masked policy probabilities; MuZero samples temperature-adjusted MCTS visit counts without training root noise. DQN uses seeded epsilon-greedy selection: with probability `epsilon_end` from the checkpoint's experiment it samples a uniformly random legal action, otherwise it chooses the highest legal Q-value. This is an evaluation policy with a fixed final epsilon, even for early checkpoints; it does not restart the training epsilon schedule or apply softmax to Q-values. The selector offers **Highest Q-value** for DQN and **Highest probability** for PPO/MuZero comparisons. Changing selection mode preserves the current world, discovered-cell memory, and MuZero history. Sampling is seeded from the world seed when the policy is first attached; resetting a run with the same seed and controls reproduces its action sequence.

The attach-policy API accepts optional `actionMode: "stochastic" | "deterministic"` and returns the effective mode. Omitting it samples for all six learned algorithms. Agent-step responses include `policyMeta.actionMode` for checkpoints. Explicit CLI evaluation retains the mode from the experiment or `--action-mode`; DQN now supports stochastic and both modes too. Restart a running API process to load these changes.

### Action-path plots and training exploration

`scripts/plot_reference_action_paths.py` defaults to seeded sampling for all learned policies. `--action-mode deterministic` requests a greedy comparison. It stores v3 caches separately under `action_path_cache_v3/stochastic` and `action_path_cache_v3/deterministic`, records the sampler, epsilon/temperature, checkpoint hash, and seeds, and rejects outdated or incompatible caches. Plot metadata is written to `action_paths_metadata.json`. Scripted rollouts remain deterministic when explicitly included in this post-training analysis.

```bash
PYTHONPATH=engine engine/.venv/bin/python scripts/plot_reference_action_paths.py --trials seed_33 seed_35 --no-scripted
PYTHONPATH=engine engine/.venv/bin/python scripts/plot_reference_action_paths.py --trials seed_33 seed_35 --no-scripted --action-mode deterministic --output results/reports/greedy
# dqn_seed_N and muzero_seed_N route to their respective experiment directories.
# Use --dqn-experiment / --muzero-experiment for a different experiment root,
# and --checkpoint-name step_NNNNNN for a saved checkpoint instead of final.
```

Training already uses stochastic exploration: PPO samples masked categorical actions; DQN uses its configured epsilon schedule with uniformly sampled legal exploration actions; MuZero samples tempered visit counts and adds Dirichlet root noise. These training paths are unchanged. Sampling during evaluation can expose capabilities hidden by greedy action selection, but does not establish that learning improved. Compare held-out mission success and delivery across repeated agent seeds. Evaluation remains a separate manual action, with no new training-time evaluations.

### Requirements

| Policy type | API install |
|---|---|
| Baselines (`random`, `random_valid`, `wait`, `scripted`) | `aresim[env]` (default dev install) |
| llm_dm | `aresim[llm_dm]`, frozen bundle config and selected Actor credentials |
| Masked PPO or DQN checkpoint | `aresim[rllib]` on the API host |

Start the API with `engine/.venv/bin/python -m aresim.api` (from any working directory).

```bash
engine/.venv/bin/python -m aresim.api
```

`GET /api/health` returns `rllibAvailable`, `muzeroAvailable`, `worldModelsAvailable`, and `llmDmAvailable` when those optional stacks can actually run.

### Attach and step

```bash
# List built-in policies
curl http://127.0.0.1:8000/api/policies

# Start a session, then attach a baseline
curl -X POST http://127.0.0.1:8000/api/sessions -H 'Content-Type: application/json' -d '{"seed":1447}'
curl -X POST http://127.0.0.1:8000/api/sessions/<sessionId>/attach-policy \
  -H 'Content-Type: application/json' \
  -d '{"algorithmId":"wait"}'

# One server-side policy step (observation → act → decode → engine step)
curl -X POST http://127.0.0.1:8000/api/sessions/<sessionId>/agent-step
```

For masked PPO or DQN during development, pass the **checkpoint sidecar** path in the UI or API:

```json
{
  "algorithmId": "masked_ppo",
  "checkpointPath": "/absolute/path/to/results/rllib_masked_ppo_smoke_v2/seed_7/checkpoints/final/checkpoint.json"
}
```

```json
{
  "algorithmId": "masked_dqn",
  "checkpointPath": "/absolute/path/to/results/rllib_masked_dqn_smoke_v2/seed_7/checkpoints/final/checkpoint.json"
}
```

While a run is still training, use the latest mid-training export instead, for example:

```text
/absolute/path/to/results/ppo/seed_35/checkpoints/final/checkpoint.json
```

`aresim-rl evaluate <run_dir> --checkpoint step_081920` accepts the same folder name under `checkpoints/`.

Checkpoint paths must be **absolute** and point to an existing `checkpoint.json`. Loaded learned-policy models are cached in-process (LRU, max 4) so swapping checkpoints during development does not reload unchanged files on every step.

Validate a path without attaching:

```bash
curl -X POST http://127.0.0.1:8000/api/policies/validate-checkpoint \
  -H 'Content-Type: application/json' \
  -d '{"checkpointPath":"/absolute/path/to/results/rllib_masked_ppo_smoke_v2/seed_7/checkpoints/final/checkpoint.json"}'
```

Saved trajectories record `algorithmId` and, for learned policies, `checkpointPath` in replay metadata.

---

## Create an environment

```python
from aresim.factory import make_env, make_gym_env, make_parallel_env

environment = make_env()
reset = environment.reset(seed=1447)
step = environment.step(0)

gym_environment = make_gym_env(max_episode_steps=1200)
parallel_environment = make_parallel_env(max_episode_steps=1200)
```

The external step limit produces truncation; it is not a simulator failure or task deadline. An authoritative engine termination takes precedence when both occur on the same transition.

---

## Run baseline rollouts

Built-in agent registry names (see [baseline workflow](../../engine/aresim/algorithms/baselines/DETAILS.md)):

| Name | Behavior |
|---|---|
| `random` | Uniform over all 10 actions; may select illegal actions |
| `random_valid` | Uniform over legal actions only |
| `wait` | Always action 0 (Wait) |
| `scripted` | Deterministic partial-observation heuristic |
| `llm_dm` | Decision Model or LLM Actor with shared observed Memory and a required LLM Learner; honors legal choices, carries prior advice forward, and writes essential logs/replays under `results/llm_dm/`. See [llm_dm workflow](../llm_dm/implementation.md) |

```python
from aresim.training import EpisodeSpec, RolloutConfig, RolloutRunner

plan = RolloutConfig(
    episodes=(
        EpisodeSpec("random-valid-000", environment_seed=1447, agent_seed=9001),
        EpisodeSpec("random-valid-001", environment_seed=2468, agent_seed=9002),
    ),
    max_episode_steps=1200,
)
result = RolloutRunner(plan, "random_valid").run()

for summary in result.summaries:
    print(summary.episode_id, summary.length, summary.episode_return)
```

The runner collects complete episodes with observations, masks, rewards, events, and checksums. It is not an online RL replay buffer.

---

## Record trajectories

`aresim.trajectory.v1` is the manifest-backed dataset schema. Each standalone episode file uses `aresim.trajectory.episode.v1` with both policy tensors and a UI replay projection.

```python
from aresim.training import (
    EpisodeSpec,
    RolloutConfig,
    RolloutRunner,
    TrajectoryWriter,
    validate_trajectory_dataset,
)

plan = RolloutConfig(
    episodes=(EpisodeSpec("random-valid-000", environment_seed=1447, agent_seed=9001),),
    max_episode_steps=1200,
)
writer = TrajectoryWriter(
    "datasets/random-valid-v1",
    dataset_id="random-valid-v1",
    compression="gzip",
)
RolloutRunner(plan, "random_valid").run(writer)
validate_trajectory_dataset("datasets/random-valid-v1")
```

Output layout:

```text
random-valid-v1/
├── manifest.json
├── episodes-00000.jsonl
└── episodes/
    └── episode-000000.json
```

Set `include_episode_artifacts=False` only for compact shard-only datasets. The output directory must not already exist; shards are written atomically and `manifest.json` is published last.

Read and validate:

```python
from aresim.training import iter_trajectory_episodes, validate_trajectory_episode

for episode in iter_trajectory_episodes("datasets/random-valid-v1"):
    print(episode.episode_id, episode.length, episode.episode_return)

standalone = validate_trajectory_episode("datasets/random-valid-v1/episodes/episode-000000.json")
```

Treat published datasets as immutable; create a new `dataset_id` instead of editing an existing artifact. Legacy `aresim.gameplay.v1` imports are normalized to `aresim.trajectory.replay.v1` in memory.

---

## Extend environment components

Observations, action adapters, rewards, and tasks are structural protocols in `aresim.components`:

- `ObservationBuilder` — `schema`, `space`, `reset`, `build`
- `ActionAdapter` — `schema`, `space`, `mask_space`, `decode`, `mask`
- `RewardFunction` — `profile`, `calculate`
- `TaskEvaluator` — `task_id`, `reset`, `evaluate`

Register a factory on `create_default_registry()` and select the new name in `EnvironmentConfig`:

```python
from dataclasses import dataclass, replace

from aresim.components.rewards import RewardBreakdown, RewardTerm
from aresim.defaults import DEFAULT_ENVIRONMENT_CONFIG
from aresim.factory import make_gym_env
from aresim.registry import create_default_registry


@dataclass(frozen=True)
class MyRewardConfig:
    scale: float


class MyReward:
    profile = "my_project.reward.fixed.v1"

    def __init__(self, config: MyRewardConfig) -> None:
        self.config = config

    def calculate(self, before, transition, outcome) -> RewardBreakdown:
        term = RewardTerm(raw=1, weight=self.config.scale, value=self.config.scale)
        return RewardBreakdown(
            schema_version="my_project.reward.breakdown.v1",
            profile=self.profile,
            terms={"fixed": term},
            total_unclipped=term.value,
            total=term.value,
        )


registry = create_default_registry()
registry.register_reward("my_reward", lambda context: MyReward(MyRewardConfig(scale=0.5)))
environment = make_gym_env(replace(DEFAULT_ENVIRONMENT_CONFIG, reward="my_reward"), registry=registry)
```

Registration is programmatic; package entry points remain deferred.

---

## Extend rollout agents

Agents implement `aresim.algorithms.Agent`:

```python
from aresim.factory import make_agent
from aresim.registry import create_default_registry


class FirstLegalAgent:
    policy_id = "my_project.agent.first_legal.v1"
    observation_schema = None
    action_schema = "aresim.action.rover.v1"

    def reset(self, seed: int) -> None:
        pass

    def act(self, observation, action_mask) -> int:
        return int(action_mask.nonzero()[0][0])


registry = create_default_registry()
registry.register_agent("first_legal", lambda context: FirstLegalAgent())
agent = make_agent("first_legal", registry=registry)
```

Checkpoint-backed policies implement the same contract for framework-neutral evaluation.

---

## Extend learned policies

`TrainingRegistry` exposes `AlgorithmFactory`, `ModelFactory`, `CheckpointLoader`, and the callable `TrainingRunner` contract. Built-ins are `masked_ppo`, `masked_dqn`, `recurrent_dqn`, `muzero`, `dreamerv3`, and `plan2explore`. Registrations own compatible model/loader names, typed model/resource decoders, lazy runner dispatch, and install metadata; [the implementation guide](../../engine/aresim/algorithms/README.md#shared-ownership) lists the IDs and owners.

```python
from aresim.training import create_training_registry, load_experiment, run_experiment

registry = create_training_registry()
custom = MyAlgorithmFactory()
registry.register_algorithm(
    "my_masked_algorithm",
    lambda context: custom,
    config_decoder=custom.decode_config,
)
spec = load_experiment("configs/my_algorithm/dev.yaml", registry=registry)
run_directory = run_experiment(spec, registry=registry)
```

An algorithm factory decodes typed configuration and builds its framework-native configuration. A model factory declares `model_class`; existing `rl_module_class` factories remain compatible, including frozen custom factories. `register_algorithm` defaults to the current RLlib model/resources/runner. Supply `model_decoder`, `resource_decoder`, `runner`, and compatible `model_id` when registering a different backend. A runner accepts the resolved spec, registry, optional component registry, report flag and resume path, and returns the run directory. Add semantic registrations rather than branching on algorithm IDs in the public runner or parser.

RLlib owns online batches, learners, and native checkpoint state. AresSim owns simulator semantics, canonical W&B names, seed splits, frozen evaluation, and trajectories.


## Recurrent and world-model training

Recurrent DQN uses `aresim[rllib]`; DreamerV3 and Plan2Explore use `aresim[world_models]`. Their permanent [implementation guide](../../engine/aresim/algorithms/README.md) covers network/loss details, all smoke/dev/reference presets, CPU/Ray versus CUDA resources, transfer, checkpoint formats, memory and masks, local verification and known limits.

```bash
engine/.venv/bin/aresim-rl train configs/recurrent_dqn/smoke.yaml
engine/.venv/bin/aresim-rl train configs/dreamerv3/smoke.yaml
engine/.venv/bin/aresim-rl train configs/plan2explore/pretrain/smoke.yaml
engine/.venv/bin/aresim-rl train configs/plan2explore/adapt/smoke.yaml \
  --set algorithm_config.initialization_checkpoint=results/plan2explore_pretrain_smoke_v1/seed_7/checkpoints/final/checkpoint.json
```

Plan2Explore initialization starts a new adaptation run; `--resume-from` restores the same phase without repeating transfer. Keep the original initializer field when resuming, even though the parent file is no longer read. Reference pretrain+adapt budgets total the same 2,097,152 interactions as Dreamer from scratch. Pretraining discards task rewards and exports the exploration actor; adaptation exports the task actor.

World-model checkpoint inference needs no Ray/RLlib runtime. Training uses bounded CPU Ray collector waves and one learner; `resources.device` selects `cpu` or `cuda`. Its replay ratio counts valid learned real timesteps per collected real transition, excluding padding/prefix/imagination. Exact same-source CPU partial-episode resume is verified; CUDA/long-budget quality remains unverified. Recurrent DQN's native checkpoints include seeded replay, with bitwise partial-episode EnvRunner resume unclaimed.

Algorithm mode exposes all six learned policies. Recurrent DQN uses Q argmax or seeded epsilon-greedy; world models use actor sampling or argmax, with modal latents for deterministic evaluation. Mode toggles retain state/RNG; manual intervention resets neural memory before the next action. Training remains checkpoint-only; run evaluation and reporting explicitly.
