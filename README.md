<p align="center">
  <img src="docs/assets/logo.png" alt="AresSim: Mars Survival Simulation" width="420" />
</p>

<h1 align="center">AresSim</h1>

<p align="center">
  A deterministic Mars survival simulation — playable in the browser, trainable as an RL environment.
</p>

AresSim is a Phase 1 Mars civilization-building environment: one unmanned rover, one landing build pad, and a seeded `32×32` world. Python owns every gameplay rule. The React UI only displays and dispatches; Gymnasium, PettingZoo, and RLlib adapters wrap the same engine without duplicating it.

Same seed + same commands → same checksums.

## Capabilities

One simulator supports human play, scripted baselines, and learned policies:

- **Play** a survival loop: scout terrain, scan, extract ice and ore, haul cargo, service the pad, and build while battery, health, power, weather, and livability decay.
- **Inspect** the same episode as a player, an analyst, or a policy: Manual, Algorithm, and Replay modes share snapshots from the Python API.
- **Train** action-masked PPO with checked-in YAML, W&B metrics, native checkpoints, and frozen evaluation on fixed seed splits.
- **Record** portable trajectories (`aresim.trajectory.episode.v1`) from the UI or from `RolloutRunner`, then reload them in Replay.
- **Attach** a baseline or a PPO checkpoint in Algorithm mode; inference stays on the API host, and Python still validates every action.

Phase 1 is open exploration: there is no mission-complete victory. Reward, survival, and coverage are diagnostics, not a score to “win.”

## Features

- **Deterministic world** — seeded terrain (height, roughness, ice, ore, dust), weather, landing pad, and rover start; SHA-256 state checksums after every reset and step
- **Survival rules** — 10 discrete actions (Wait, N/E/S/W, Scan, Extract, Build, Service, Unload), 12 kg cargo, battery/power/health/livability, and terminal failure conditions
- **Local policy crop** — `aresim.obs.local.v1` is a fixed `8×8` window with an authoritative legal-action mask; Wait is always legal
- **3D shell** — orthographic Survey, north-up Top, and Rover POV; zoom, follow, mini-map, visibility flashlight, layers, and Martian day/night
- **HUD and analytics** — grouped status, mission/warning chips, inspector, guide, and run charts without duplicating engine math
- **Training stack** — RLlib masked PPO, CNN actor-critic, mid-run UI-loadable checkpoints, W&B as the sole app-level log
- **Baselines** — Wait, uniform random, random-valid, and a scripted local-observation heuristic, all on the same `Agent` contract as PPO

## Flexibility

The engine is the source of truth. New behavior plugs in at registries; adapters do not reimplement rules.

| Seam | How you extend it |
|---|---|
| Observation, action, reward, task | Implement the protocol, register a name, select it in `EnvironmentConfig` |
| Rollout agent | Same `Agent` contract as Wait / PPO checkpoints |
| Learned algorithm | `TrainingRegistry` algorithm / model / checkpoint factories + YAML |
| Framework | `make_env`, `make_gym_env`, `make_parallel_env` — one transition, three APIs |
| Install | Base extra is FastAPI only; `env`, `rllib`, and `notebook` stay optional |

Unknown YAML fields, unsafe tags, and incompatible batch sizes fail before Ray workers start. `--set path=value` overrides a config without editing the file. Custom components do not require source edits, inheritance, or plugin discovery.

Later milestones (multi-rover, DQN, recurrence, LLM agents) are designed to reuse this core rather than fork it. See [Implementation Checklist](docs/project/implementation_checklist.md).

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
pip install -e './engine[dev]'
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
```

| Config | Steps | Purpose |
|---|---:|---|
| [`configs/masked_ppo/smoke.yaml`](configs/masked_ppo/smoke.yaml) | 4,096 | Pipeline smoke test |
| [`configs/masked_ppo/dev.yaml`](configs/masked_ppo/dev.yaml) | 102,400 | Local development |
| [`configs/masked_ppo/reference.yaml`](configs/masked_ppo/reference.yaml) | 1,048,576 | Longer reference run |

See the [RL usage guide](docs/rl/usage.md) for W&B, evaluation, trajectories, and extension contracts.

## Tests

```bash
engine/.venv/bin/pytest engine/tests
npm test
```

## Documentation

Start at [`docs/README.md`](docs/README.md). Suggested order:

1. New to the product: [Environment Rules](docs/product/environment_rules.md) → [UI Design](docs/product/ui_design.md)
2. Working on the backend: [Engine Code Reference](docs/product/engine_code_reference.md)
3. Starting RL work: [Usage](docs/rl/usage.md) → [Algorithms](docs/rl/rl_quickstart.md)
4. Status: [Implementation Checklist](docs/project/implementation_checklist.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security reports go through [SECURITY.md](SECURITY.md).

## License

MIT — see [LICENSE](LICENSE).
