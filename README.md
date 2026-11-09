<p align="center">
  <img src="docs/assets/logo.png" alt="AresSim: Mars Survival Simulation" width="420" />
</p>

<h1 align="center">AresSim</h1>

<p align="center">
  A deterministic Mars simulation for human play, reinforcement learning, and model agents.
</p>

AresSim has one rover, one landing pad, and a seeded `32×32` world. Python controls all game rules. The React UI displays the game and sends actions. Gymnasium, PettingZoo, and RLlib use the same engine.

The same seed and commands produce the same state checksums.

Walkthrough of the environment: [AresSim on TensorWrites](https://www.tensorwrites.com/posts/aressim-env).

<p align="center">
  <a href="https://youtu.be/bGrpud_uAj8">
    <img src="https://img.youtube.com/vi/bGrpud_uAj8/maxresdefault.jpg" alt="AresSim demo" width="720" />
  </a>
</p>

## Capabilities

One simulator supports human play, scripted baselines, and learned policies:

- **Play:** Explore terrain, scan rocks for samples, extract ice, deliver cargo, service the pad, and build. Monitor battery, health, power, and colony livability.
- **Inspect** the same episode as a player, an analyst, or a policy: Manual, Algorithm, and Replay modes share snapshots from the Python API.
- **Train:** Use PPO, DQN, Recurrent DQN, MuZero, DreamerV3, or Plan2Explore. YAML files set the run parameters. W&B records training metrics.
- **Run model agents:** Use `llm_dm` with a Decision Model or LLM Actor and a required LLM Learner. The Actor selects a goal and each action. The Learner reviews completed games and proposes advice.
- **Record** portable trajectories (`aresim.trajectory.episode.v1`) from the UI or from `RolloutRunner`, then reload them in Replay.
- **Attach** a baseline or any learned-policy checkpoint in Algorithm mode; inference stays on the API host, and Python still validates every action.

The open-exploration task has no mission-complete victory. The resource-mission task adds explicit delivery/service success; evaluation reports both mission outcomes and survival/coverage diagnostics.

## Features

- **Deterministic world** — seeded terrain (height, roughness, ice, ore, dust), weather, landing pad, and rover start; SHA-256 state checksums after every reset and step
- **Survival rules** — 10 discrete actions (Wait, N/E/S/W, Scan, Extract, Build, Service, Unload), 12 kg cargo, battery/power/health/livability, and terminal failure conditions
- **Local policy crop** — `aresim.obs.local.v1` is a rover-centered window (`window_size` in `engine/aresim/defaults.py`) with an authoritative legal-action mask; Wait is always legal
- **3D shell** — orthographic Survey, north-up Top, and Rover POV; zoom, follow, mini-map, visibility flashlight, layers, and Martian day/night
- **HUD and analytics** — grouped status, mission/warning chips, inspector, guide, and run charts without duplicating engine math
- **Training stack** — RLlib PPO and feedforward/recurrent DQN; native PyTorch/Ray MuZero, DreamerV3 and Plan2Explore; shared symbolic observations, UI-loadable checkpoints and W&B metrics
- **Baselines** — Wait, uniform random, random-valid, and a scripted local-observation heuristic, all on the same `Agent` contract as PPO and DQN

See the [algorithm guide](engine/aresim/algorithms/README.md) for all policies. See [LLM-DM details](engine/aresim/algorithms/llm_dm/DETAILS.md) for model prompts, Memory, learning, and replay workflows.

## Flexibility

The engine is the source of truth. New behavior plugs in at registries; adapters do not reimplement rules.

| Seam | How you extend it |
|---|---|
| Observation, action, reward, task | Implement the protocol, register a name, select it in `EnvironmentConfig` |
| Rollout agent | Same `Agent` contract as Wait / PPO checkpoints |
| Learned algorithm | `TrainingRegistry` algorithm / model / loader / runner registration + typed YAML |
| Framework | `make_env`, `make_gym_env`, `make_parallel_env` — one transition, three APIs |
| Install | Base extra is FastAPI only; `env`, `rllib`, `muzero`, `world_models`, `llm_dm`, and `notebook` stay optional |

Unknown YAML fields, unsafe tags, and incompatible batch sizes fail before Ray workers start. `--set path=value` overrides a config without editing the file. Custom components do not require source edits, inheritance, or plugin discovery.

Future multi-rover and hybrid controllers must use the same engine. See [Implementation Checklist](docs/project/implementation_checklist.md).

## Repository layout

| Path | Owns |
|---|---|
| [`engine/`](engine/) | Deterministic core, REST API, environment adapters, training |
| [`web/`](web/) | React + R3F presentation (no gameplay rules) |
| [`configs/`](configs/) | Checked-in experiment YAML |
| [`docs/`](docs/) | Canonical product, RL, and project docs |

Generated artifacts (`results/`, `datasets/`), local notebooks, editor config, and historical notes (`design_docs/`, `papers/`) are gitignored.

## Quick start — play

Requires Python 3.12+ and Node.js.

```bash
python3 -m venv engine/.venv
source engine/.venv/bin/activate
pip install -e './engine[dev,env]'
python -m aresim.api
```

In a second terminal, from the repository root:

```bash
npm install
npm run dev
```

The API listens on `127.0.0.1:8000`. Vite proxies `/api` to it. Open the URL Vite prints (typically `http://127.0.0.1:5173`).

## Quick start — train

```bash
python3 -m venv engine/.venv
engine/.venv/bin/pip install -e './engine[dev,rllib,notebook]'
engine/.venv/bin/aresim-rl train configs/masked_ppo/smoke.yaml
engine/.venv/bin/aresim-rl train configs/masked_dqn/smoke.yaml
```

| Config | Steps | Purpose |
|---|---:|---|
| [`configs/masked_ppo/smoke.yaml`](configs/masked_ppo/smoke.yaml) | 102,400 | Pipeline smoke test |
| [`configs/masked_ppo/dev.yaml`](configs/masked_ppo/dev.yaml) | 102,400 | Local development |
| [`configs/masked_ppo/reference.yaml`](configs/masked_ppo/reference.yaml) | 2,097,152 | Longer reference run (28 env runners, CUDA learner) |
| [`configs/masked_dqn/smoke.yaml`](configs/masked_dqn/smoke.yaml) | 4,096 | Masked DQN pipeline smoke |
| [`configs/masked_dqn/dev.yaml`](configs/masked_dqn/dev.yaml) | 102,400 | Local DQN development |
| [`configs/masked_dqn/reference.yaml`](configs/masked_dqn/reference.yaml) | 2,097,152 | DQN reference (4 env runners, CUDA learner, n-step=3) |

See the [RL usage guide](docs/rl/usage.md) for W&B, evaluation, trajectories, and extension contracts. For cloud GPUs, use the [Vast.ai scripts and run guide](docs/rl/vastai.md). The [illustrated algorithm guide](engine/aresim/algorithms/README.md) maps each policy’s inputs, outputs and training steps.

## Quick start — model agents

Install the model-agent dependencies. Run the offline check first:

```bash
engine/.venv/bin/pip install -e './engine[llm_dm]'
engine/.venv/bin/aresim-rl rollout configs/llm_dm/fake_smoke.yaml
```

For a live run, set `JEV_API_KEY` and `OPENAI_API_KEY` in `.env.local`. The main profile uses Jev as Actor and GPT-6 Luna as Learner:

```bash
engine/.venv/bin/aresim-rl rollout configs/llm_dm/10-games.yaml --env-file .env.local
```

Edit `experiment_name`, `episodes`, and `max_episode_steps` in the config. The episode list sets the game count. The filename does not set it. Use `configs/llm_dm/llm-actor.yaml` to select an LLM Actor.

Focused Memory supplies the current state, the active goal route, recent outcomes, and applicable advice. All compatible lessons remain available for goal selection. The Learner updates advice between games. It does not change model weights. A live run is needed to measure game performance.

Open `results/llm_dm/<experiment_name>/summary.log` first. The run also saves `manifest.json`, `games.jsonl`, `learning.jsonl`, and `selected-bundle.json`. Each game has one JSON file in `replays/`. The file updates during play and can be imported into the UI. Repeated experiment names get a numeric suffix.

## Documentation

Start at [`docs/README.md`](docs/README.md). Suggested order:

1. New to the product: [Environment Rules](docs/product/environment_rules.md) → [UI Design](docs/product/ui_design.md)
2. Working on the backend: [Engine Code Reference](docs/product/engine_code_reference.md)
3. Starting RL work: [Usage](docs/rl/usage.md) → [Algorithms](docs/rl/rl_quickstart.md)
4. Model agents: [LLM-DM details](engine/aresim/algorithms/llm_dm/DETAILS.md) → [Run guide](docs/llm_dm/implementation.md)
5. Status: [Implementation Checklist](docs/project/implementation_checklist.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security reports go through [SECURITY.md](SECURITY.md).

## License

MIT — see [LICENSE](LICENSE).
