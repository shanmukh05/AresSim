# AresSim Engine

This package contains the deterministic Phase 1 gameplay engine, the local REST API used by the React UI, portable trajectory export/replay support, and optional environment/training layers. The base install intentionally has no numerical, environment-framework, distributed-training, tracking, plotting, or model-inference dependencies.

## Development

`pyproject.toml` is the canonical backend dependency manifest. The base dependencies run the API, while the `dev`, `env`, and `notebook` extras add tests, RL environment frameworks, and the Jupyter kernel respectively.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
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

The supported extension surface is the typed protocols in `aresim.components`, the context-aware `ComponentRegistry`, and the public factories. Custom components may use their own typed config captured by a registry factory; no source edits, inheritance, automatic plugin discovery, or YAML loading are required. See [Extend environment components](../docs/rl/usage.md#extend-environment-components) for a complete example and contract-test requirements.

## Baseline rollouts

The optional environment extra also includes deterministic random, random-valid, Wait, and local-observation scripted agents:

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
pip install -e '.[dev,rllib,notebook]'
aresim-rl train configs/masked_ppo/smoke.yaml
```

The optional `aresim.training` package provides strict experiment YAML, a compact algorithm/model/checkpoint registry, RLlib action-masked PPO, fixed seed splits, canonical W&B metrics, native checkpoint sidecars, framework-neutral evaluation, UI-loadable trajectories, and executed Jupyter reports. Experiment configs live in repository `configs/`; training run artifacts in `results/`. It deliberately avoids parallel JSONL, CSV, and TensorBoard logging. See [RL Algorithms, Training, and Evaluation](../docs/rl/rl_quickstart.md) and [RL Usage Guide](../docs/rl/usage.md#train-and-extend-learned-policies).
