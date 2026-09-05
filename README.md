<p align="center">
  <img src="docs/assets/logo.png" alt="AresSim: Mars Survival Simulation" width="420" />
</p>

<h1 align="center">AresSim</h1>

<p align="center">
  A deterministic Mars survival simulation — playable in the browser, trainable as an RL environment.
</p>

AresSim is a Phase 1 Mars civilization-building environment: one unmanned rover, one landing build pad, and a seeded `32×32` world. Python owns every gameplay rule. The React UI only displays and dispatches; Gymnasium, PettingZoo, and RLlib adapters wrap the same engine without duplicating it.

Same seed + same commands → same checksums.

## Features

- **Playable simulator** — Move, Scan, Extract, Build, Service, Unload, and Wait on a deterministic Mars grid
- **3D interface** — orthographic Survey, Top, and Rover POV views with HUD, replay, and analytics
- **RL-ready** — local `8×8` observation, 10-action mask, Gymnasium and PettingZoo adapters
- **Baselines + masked PPO** — Wait, random, random-valid, scripted, and action-masked PPO via RLlib
- **Portable trajectories** — UI export and training rollouts share `aresim.trajectory.episode.v1`

## Repository layout

| Path | Owns |
|---|---|
| [`engine/`](engine/) | Deterministic core, REST API, environment adapters, training |
| [`web/`](web/) | React + R3F presentation (no gameplay rules) |
| [`configs/`](configs/) | Checked-in experiment YAML |
| [`docs/`](docs/) | Canonical product, RL, and project docs |
| [`notebooks/`](notebooks/) | Training, pipeline, and evaluation notebooks |

Generated artifacts (`results/`, `datasets/`) and historical notes (`design_docs/`, `papers/`) are gitignored.

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
