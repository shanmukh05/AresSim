# AresSim Engine

This package contains the deterministic Phase 1 gameplay engine, the local REST API used by the React UI, portable trajectory export/replay support, and optional environment/training layers. The base install intentionally has no numerical, environment-framework, distributed-training, tracking, plotting, or model-inference dependencies.

## Development

`pyproject.toml` is the canonical backend dependency manifest. The deterministic core remains usable without numerical frameworks. The API policy bridge uses the `env` extra; `dev` adds development checks, `rllib`/`muzero`/`world_models` add the selected training backend, `llm_dm` adds the hybrid agent pipeline, and `notebook` adds the Jupyter kernel.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,env]'
python -m aresim.api
```

In a second terminal, start the web application from the repository root:

```bash
npm run dev
```

The API listens on `127.0.0.1:8000`; Vite proxies `/api` to it.

## Python API

```python
from dataclasses import replace

from aresim import AresEngine, DEFAULT_ENGINE_CONFIG

config = replace(DEFAULT_ENGINE_CONFIG, world=replace(DEFAULT_ENGINE_CONFIG.world, size=32))
engine = AresEngine(config)
state = engine.reset(seed=1447)
```

All simulator defaults live in `aresim/defaults.py`. Typed configuration shapes and validation live in `aresim/config.py`.

Source files start with a module docstring; public classes and functions (`AresEngine`, `apply_action`, `AresService`, and so on) are documented for a first-time reader.

## RL environment API

Install the optional numerical environment and its test dependencies:

```bash
pip install -e '.[dev,env]'
```

To run the repository notebooks from this virtual environment, install and register its kernel. The notebooks and seed-split YAML live in `notebooks/` at the checkout root, not inside the `aresim` package:

```bash
pip install -e '.[dev,env,notebook]'
python -m ipykernel install --user --name aresim --display-name "AresSim (.venv)"
```

```python
from aresim.factory import make_env, make_gym_env, make_parallel_env

environment = make_env()
initial = environment.reset(seed=1447)
transition = environment.step(0)  # Wait

gym_environment = make_gym_env()
parallel_environment = make_parallel_env()
```

`AresEnvironment` composes the canonical engine with registered observation, action, reward, and task implementations. The defaults remain the local numerical observation, masked ten-action adapter, open-exploration task, and shaped RL reward. `AresGymEnv` unwraps exactly `rover_0`; `AresParallelEnv` exposes the same results through PettingZoo dictionaries. RL rewards are separate from the existing engine/UI reward retained in history and exposed as `engine_reward` in step info.

The supported extension surface is the typed protocols in `aresim.components`, the context-aware `ComponentRegistry`, and the public factories. Custom components may use their own typed config captured by a registry factory; no source edits, inheritance, automatic plugin discovery, or YAML loading are required. See [Extend environment components](../docs/rl/usage.md#extend-environment-components) for a complete example and verification requirements.

## Baseline rollouts

The environment extra includes random, random-valid, Wait and scripted agents. The unified `llm_dm` algorithm uses a Decision Model or LLM Actor, focused episode Memory and a mandatory LLM Learner. The Actor selects a persistent goal and each primitive action. Action requests receive only the current working context and applicable advice. Install `aresim[llm_dm]`; use native HTTPS providers with locally configured credentials. See [the implementation guide](../docs/llm_dm/implementation.md) for complete configs and commands.

```python
from aresim.training import EpisodeSpec, RolloutConfig, RolloutRunner, TrajectoryWriter

plan = RolloutConfig(
    episodes=(EpisodeSpec("sample-000", environment_seed=1447, agent_seed=7),),
    max_episode_steps=1200,
)
writer = TrajectoryWriter("datasets/sample-v1", "sample-v1", compression="gzip")
result = RolloutRunner(plan, "random_valid").run(writer)
```

The 1,200-step limit is an external truncation, not simulator failure. `aresim.trajectory.v1` is the transition-aligned JSONL dataset schema (optionally gzip-compressed). By default, each recorded episode also produces a standalone `aresim.trajectory.episode.v1` file under `episodes/`. It combines the complete policy trace with the replay projection and loads directly in the UI. The UI Export action emits the same episode schema. See [Record trajectories](../docs/rl/usage.md#record-trajectories) for its manifest, validation, and reader APIs.

For a maintained folder tree and a description of every engine file, see [`docs/product/engine_code_reference.md`](../docs/product/engine_code_reference.md).
For project-wide completed, next, and future work, see [`docs/project/implementation_checklist.md`](../docs/project/implementation_checklist.md).

## Learned-policy training

Install the complete learned-policy stack and run the checked-in smoke experiment:

```bash
pip install -e '.[dev,rllib,world_models,notebook]'
aresim-rl train configs/masked_ppo/smoke.yaml
aresim-rl train configs/masked_dqn/smoke.yaml
aresim-rl train configs/recurrent_dqn/smoke.yaml
aresim-rl train configs/dreamerv3/smoke.yaml
```

The optional `aresim.training` package provides strict experiment YAML, explicit algorithm/model/loader/runner registration, RLlib PPO and feedforward/recurrent DQN, native PyTorch/Ray MuZero and DreamerV3/Plan2Explore, fixed seed splits, canonical W&B metrics, native checkpoint sidecars, framework-neutral evaluation, UI-loadable trajectories, and executed Jupyter reports. Experiment configs live in repository `configs/`; training run artifacts in `results/`. It deliberately avoids parallel JSONL, CSV, and TensorBoard logging. See [RL Algorithms, Training, and Evaluation](../docs/rl/rl_quickstart.md) and [RL Usage Guide](../docs/rl/usage.md#train-from-the-cli).


Recurrent DQN adds LSTM sequence replay and burn-in. DreamerV3 learns a structured categorical RSSM and actor-critic in imagination. Plan2Explore shares that runtime for reward-free disagreement exploration followed by a separately initialized task-adaptation run. Each has smoke/dev/reference YAML; Plan2Explore keeps a set for each phase. See [Recurrent DQN, DreamerV3, and Plan2Explore](aresim/algorithms/README.md) for commands, exact ownership, checkpoint/resume semantics, local evidence and pending reference-learning validation.

Shared RLlib infrastructure lives in `aresim/training/rllib/`; world-model collection/replay/lifecycle in `aresim/training/world_models/`. Algorithms own their networks/objectives. Existing PPO public driver/config aliases and existing checkpoint schemas remain available.
