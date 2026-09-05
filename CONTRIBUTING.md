# Contributing to AresSim

Thanks for helping. Keep changes small, match nearby style, and leave gameplay rules in Python.

## Ground rules

- **Python owns simulation truth.** Rules, validation, rewards, saves, and replay reconstruction live in `engine/`. The UI must not invent gameplay semantics.
- **Keep the deterministic core pure.** `engine/aresim/core/` must not import NumPy, Gymnasium, PettingZoo, Ray, Torch, or wall-clock/unseeded randomness. Same seed + commands → same checksums.
- Change only what the task needs. No drive-by refactors or extra dependencies.
- When behavior or ownership changes, update the matching doc under `docs/` and [`docs/project/implementation_checklist.md`](docs/project/implementation_checklist.md).

Full coding standards: [`.cursor/rules/coding-standards.mdc`](.cursor/rules/coding-standards.mdc).
Architecture map: [`docs/product/engine_code_reference.md`](docs/product/engine_code_reference.md).

## Setup

```bash
python3 -m venv engine/.venv
source engine/.venv/bin/activate
pip install -e './engine[dev]'
npm install
```

For RL work, install `./engine[dev,rllib,notebook]` instead. See [`docs/rl/usage.md`](docs/rl/usage.md).

## Run

```bash
# API (from engine/.venv)
python -m aresim.api

# UI (repository root, second terminal)
npm run dev
```

## Tests

```bash
engine/.venv/bin/pytest engine/tests
npm test
```

Add or update tests when behavior changes. Do not weaken tests to make them pass.

## Pull requests

1. Branch from `main`.
2. Keep the diff focused. Do not commit `results/`, `datasets/`, `design_docs/`, secrets, or virtualenvs.
3. Document surprising contracts (copy vs mutate, invalid actions still tick the clock, UI helpers that only display engine numbers).
4. New major extension (observation, action, reward, task, agent, algorithm) must implement the existing contract, register explicitly, and include contract tests. See [Extend environment components](docs/rl/usage.md#extend-environment-components).

## Where to put code

| Kind of change | Put it here |
|---|---|
| Gameplay rules | `engine/aresim/core/rules.py` |
| World generation | `engine/aresim/core/generation.py` |
| Tunables | `engine/aresim/defaults.py` / `config.py` |
| Save / replay | `engine/aresim/gameplay.py` |
| HTTP | `engine/aresim/api.py` |
| UI presentation | `web/src/` |
| Experiment YAML | `configs/` |
