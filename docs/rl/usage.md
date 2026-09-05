# AresSim RL Usage Guide

Practical guide for running environments, training masked PPO, configuring W&B, recording trajectories, and extending components. For network architecture, PPO internals, and metric definitions, see [RL Algorithms, Training, and Evaluation](rl_quickstart.md).

## Contents

| Section | What you will find |
|---|---|
| [Setup](#setup) | Virtual environment and optional extras |
| [Train masked PPO from the CLI](#train-masked-ppo-from-the-cli) | `aresim-rl train`, configs, overrides, run layout |
| [Weights & Biases (W&B)](#weights--biases-wb) | YAML `tracking` block, auth, modes, logged metrics |
| [Experiment YAML](#experiment-yaml) | Checked-in configs and key fields |
| [Train from Python](#train-from-python) | `load_experiment`, `run_experiment`, checkpoints |
| [After training](#after-training) | Evaluate, inspect, and report |
| [UI policy inference](#ui-policy-inference) | Attach baselines or PPO checkpoints in Algorithm mode |
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
| `dev` | pytest, coverage | Running tests |
| `env` | NumPy, Gymnasium, PettingZoo | Baselines and rollouts only |
| `rllib` | Ray, RLlib, PyTorch, W&B | Masked PPO training and evaluation |
| `notebook` | Jupyter, papermill | Running notebooks locally |

Select the `AresSim (.venv)` kernel in Jupyter. The base UI/backend install does not require NumPy, Gymnasium, Ray, PyTorch, or W&B.

Seed splits for evaluation live in [`notebooks/phase1_open_exploration_split_v1.yaml`](../../notebooks/phase1_open_exploration_split_v1.yaml). Experiment YAML lives under [`configs/`](../../configs/); training outputs go to [`results/`](../../results/) (gitignored).

---

## Train masked PPO from the CLI

Masked PPO is the only implemented learned algorithm. Training is driven by validated YAML under `configs/<algorithm>/` and the `aresim-rl` console script.

### Checked-in configs

| File | Environment steps | W&B | Purpose |
|---|---:|---|---|
| [`configs/masked_ppo/smoke.yaml`](../../configs/masked_ppo/smoke.yaml) | 4,096 | disabled | Fast pipeline smoke test (2-seed eval, no trajectory export) |
| [`configs/masked_ppo/dev.yaml`](../../configs/masked_ppo/dev.yaml) | 102,400 | online | Local development run |
| [`configs/masked_ppo/reference.yaml`](../../configs/masked_ppo/reference.yaml) | 1,048,576 | online | Longer reference training |

Run from the **repository root** so seed manifests and artifact paths resolve correctly:

```bash
# Smoke test (no W&B, ~minutes on CPU)
engine/.venv/bin/aresim-rl train configs/masked_ppo/smoke.yaml

# Development run (requires W&B login — see below)
engine/.venv/bin/aresim-rl train configs/masked_ppo/dev.yaml

# Override hyperparameters without editing YAML
engine/.venv/bin/aresim-rl train configs/masked_ppo/dev.yaml \
  --set algorithm_config.learning_rate=0.0001 \
  --set resources.num_env_runners=2
```

`--set path=value` accepts dotted paths into the resolved experiment dict. Unknown paths, type mismatches, unsafe YAML, incompatible batch sizes, and missing seed manifests fail **before** Ray or W&B start.

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
└── evaluation/             # post-training frozen evaluation (when enabled)
```

`experiment_id` and `trial_id` come from the YAML (`rllib_masked_ppo_smoke`, `seed_7`, etc.). Set `artifacts.reject_existing: false` in YAML or change `trial_id` to reuse a directory.

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

During training, [`AresMetricsCallback`](../../engine/aresim/algorithms/ppo/train.py) maps RLlib metrics to stable names. The learning-curve x-axis is `train/environment_steps`. Logged groups include shaped return, episode length, action counts, invalid actions, policy/value losses, entropy, KL, clip fraction, explained variance, and throughput.

After training, frozen validation evaluation metrics are appended to the same W&B run (unless `mode: disabled`).

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

Configs use schema `aresim.experiment.v1`. Each file selects a registered algorithm, model, environment components, hyperparameters, resources, evaluation schedule, checkpoint policy, tracking, and artifact root.

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
  task: open_exploration
  max_episode_steps: 1200

algorithm_config:
  total_environment_steps: 102400
  rollout_batch_size: 4096
  learning_rate: 0.0003

evaluation:
  seed_manifest: notebooks/phase1_open_exploration_split_v1.yaml
  interval_environment_steps: 20480
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

Add new algorithms under `configs/<algorithm>/`. The `algorithm` field must match a name registered in `TrainingRegistry` (`masked_ppo` today).

---

## Train from Python

The same path as the CLI, for notebooks and scripts:

```python
from aresim.training import load_experiment, make_checkpoint_agent, run_experiment

spec = load_experiment("configs/masked_ppo/smoke.yaml")
run_directory = run_experiment(spec)  # starts Ray, trains, evaluates, shuts down Ray

agent = make_checkpoint_agent(run_directory / "checkpoints/final/checkpoint.json")
```

`run_experiment` accepts a YAML path or a resolved `ExperimentSpec`. Keyword arguments:

| Argument | Default | Purpose |
|---|---|---|
| `evaluate` | `True` | Run frozen validation evaluation after training |
| `report` | `True` | Execute the evaluation notebook (requires `tracking.mode: online`) |
| `registry` | built-in | Custom `TrainingRegistry` for extended algorithms |
| `component_registry` | built-in | Custom environment component registry |

For a notebook-friendly flow with explicit control, see [Masked PPO training](../../notebooks/masked_ppo_training.ipynb).

---

## After training

All commands take the **run directory** (`results/<experiment_id>/<trial_id>/`), not the repository root.

```bash
# Print manifest.json or checkpoint sidecar
engine/.venv/bin/aresim-rl inspect results/rllib_masked_ppo_smoke/seed_7

# Evaluate a checkpoint on the fixed validation or test split
engine/.venv/bin/aresim-rl evaluate results/rllib_masked_ppo_dev/seed_7 \
  --checkpoint final \
  --split validation

# Generate local training plots from W&B history (tracked runs only)
engine/.venv/bin/aresim-rl report results/rllib_masked_ppo_dev/seed_7
```

`--checkpoint` accepts a path to `checkpoint.json` or a checkpoint folder name under `checkpoints/` (e.g. `final`).

Evaluation writes `evaluation/<label>/summary.json` and optional trajectory shards. `aresim-rl report` reads W&B history plus local evaluation artifacts without restarting Ray.

---

## UI policy inference

Algorithm mode in the React UI runs policies **server-side** through the Python API. The browser never loads Ray, PyTorch, or checkpoint files.

### Requirements

| Policy type | API install |
|---|---|
| Baselines (`random`, `random_valid`, `wait`, `scripted`) | `aresim[env]` (default dev install) |
| Masked PPO checkpoint | `aresim[rllib]` on the API host |

Start the API with `engine/.venv/bin/python -m aresim.api` (from any working directory).

```bash
engine/.venv/bin/python -m aresim.api
```

`GET /api/health` returns `rllibAvailable: true` when the optional RLlib stack is importable.

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

For masked PPO during development, pass the **checkpoint sidecar** path in the UI or API:

```json
{
  "algorithmId": "masked_ppo",
  "checkpointPath": "/absolute/path/to/results/rllib_masked_ppo_smoke/seed_7/checkpoints/final/checkpoint.json"
}
```

While a run is still training, use the latest mid-training export instead, for example:

```text
/absolute/path/to/results/rllib_masked_ppo_reference/seed_7/checkpoints/step_081920/checkpoint.json
```

`aresim-rl evaluate <run_dir> --checkpoint step_081920` accepts the same folder name under `checkpoints/`.

Checkpoint paths must be **absolute** and point to an existing `checkpoint.json`. Loaded RLModules are cached in-process (LRU, max 4) so swapping checkpoints during development does not reload unchanged files on every step.

Validate a path without attaching:

```bash
curl -X POST http://127.0.0.1:8000/api/policies/validate-checkpoint \
  -H 'Content-Type: application/json' \
  -d '{"checkpointPath":"/absolute/path/to/results/rllib_masked_ppo_smoke/seed_7/checkpoints/final/checkpoint.json"}'
```

Saved trajectories record `algorithmId` and, for masked PPO, `checkpointPath` in replay metadata.

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

Built-in agent registry names:

| Name | Behavior |
|---|---|
| `random` | Uniform over all 10 actions; may select illegal actions |
| `random_valid` | Uniform over legal actions only |
| `wait` | Always action 0 (Wait) |
| `scripted` | Deterministic partial-observation heuristic |

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

Use `engine/tests/test_extensibility.py` patterns for contract tests. Registration is programmatic; package entry points remain deferred.

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

`TrainingRegistry` exposes three extension seams: `AlgorithmFactory`, `ModelFactory`, and `CheckpointLoader`. Built-in names are `masked_ppo`, `local_cnn_actor_critic`, and `rllib_masked_ppo`.

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

An algorithm factory decodes typed configuration and builds an RLlib `AlgorithmConfig`. A model factory declares the RLModule class. Add a new semantic name instead of branching on algorithm IDs in the runner.

RLlib owns online batches, learners, and native checkpoint state. AresSim owns simulator semantics, canonical W&B names, seed splits, frozen evaluation, and trajectories.
