# AresSim Engine Code Reference

**Last updated:** August 31, 2026  
**Status:** Living document

This document explains the purpose of every maintained file in `engine/`, how the files work together, and where future backend code should be added. Update it in the same change whenever an engine file is added, removed, renamed, or given a different responsibility.

Generated files such as `__pycache__/`, `.pyc` files, virtual environments, coverage output, and test caches are intentionally omitted.

## 1. Current folder structure

```text
engine/
├── pyproject.toml
├── README.md
├── aresim/
│   ├── __init__.py
│   ├── types.py
│   ├── config.py
│   ├── defaults.py
│   ├── registry.py
│   ├── factory.py
│   ├── gameplay.py
│   ├── service.py
│   ├── api.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── generation.py
│   │   └── rules.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── observations.py
│   │   ├── actions.py
│   │   ├── rewards.py
│   │   └── tasks.py
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── baselines/
│   │   │   ├── __init__.py
│   │   │   ├── random.py
│   │   │   ├── random_valid.py
│   │   │   ├── scripted.py
│   │   │   └── wait.py
│   │   ├── common/
│   │   │   └── masks.py
│   │   └── ppo/
│   │       ├── config.py
│   │       ├── train.py
│   │       └── checkpoint.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── environment.py
│   │   ├── gymnasium.py
│   │   └── pettingzoo.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   ├── evaluation.py
│   │   ├── experiments.py
│   │   ├── reports.py
│   │   ├── runner.py
│   │   ├── seeds.py
│   │   └── trajectories.py
│   └── integrations/
│       ├── __init__.py
│       └── ui.py
└── tests/
    ├── test_agents.py
    ├── test_components.py
    ├── test_extensibility.py
    ├── test_envs.py
    ├── test_engine.py
    ├── test_gameplay.py
    ├── test_rollouts.py
    ├── test_trajectories.py
    ├── test_api.py
    └── test_rllib_pipeline.py
```

The engine is the deterministic gameplay backend used by the UI and the core composed by the optional RL environment. Local observations, masks, rewards, adapters, baselines, truncation, rollouts, trajectories, fixed seed splits, RLlib masked PPO, checkpoint evaluation, metrics, tracking, and reports are implemented. Jupyter notebooks, seed-split YAML, and experiment configs live in repository `notebooks/` and `configs/`, not in the installable package. Training run artifacts live in `results/`.

## 2. How the engine is organized

```text
React UI
   │ REST requests and camelCase JSON snapshots
   ▼
api.py
   │ validates HTTP data and maps errors
   ▼
service.py
   │ owns the active session and replay
   ├───────────────► gameplay.py
   │                  records saves and reconstructs replays
   ▼
core/engine.py
   │ coordinates deterministic reset and step
   ├───────────────► core/generation.py
   │                  creates a seeded world
   └───────────────► core/rules.py
                      validates and applies gameplay rules

types.py       shared state and command types
config.py      configuration shapes and validation
defaults.py    canonical simulator values
core/engine.py deterministic coordination and state fingerprints
integrations/ui.py internal state → UI snapshot
components/       state/transition → observation, mask, reward, task outcome
algorithms/       policy observation + mask → baseline and learned policies
envs/             framework-neutral composition plus Gymnasium/PettingZoo shapes
training/         rollouts, trajectories, evaluation harness, CLI, experiment YAML
factory.py        explicit registry resolution and public constructors
```

The dependency direction should remain inward: transport and UI integration can depend on the engine, but the deterministic core must not depend on FastAPI, the React application, replay transport, or future RL libraries.

## 3. Root files

### `engine/pyproject.toml`

Defines the installable `aresim` Python package.

- Declares Python 3.12 as the minimum version.
- Lists the runtime dependencies: FastAPI and Uvicorn.
- Defines the optional `env` extra for NumPy, Gymnasium, and PettingZoo.
- Defines the self-contained `rllib` extra and the `aresim-rl` command.
- Lists the development dependencies: pytest and HTTPX.
- Configures the `engine/tests/` directory as the pytest test path.

Change this file when the package metadata, supported Python version, dependencies, or test configuration changes. Dependencies should not be added for functionality that can be implemented clearly with the standard library.

### `engine/README.md`

Provides the short entry point for backend developers.

- Explains what is currently implemented.
- Shows how to create a virtual environment, install the package, and run the API.
- Shows the minimal public Python API for creating and resetting an engine.
- Points developers to the default configuration.

Keep it concise. Detailed architecture and per-file responsibilities belong in this document.

## 4. Package entry point and shared definitions

### `engine/aresim/__init__.py`

Defines the small public Python import surface for consumers of the package.

It currently exports:

- `AresEngine`
- `EngineConfig`
- `DEFAULT_ENGINE_CONFIG`
- `ActionCommand`
- `ActionType`
- `Actor`

Only stable, generally useful types should be re-exported here. Internal helpers should be imported from their owning modules when needed inside the package.

### `engine/aresim/types.py`

Contains the canonical types shared by the engine layers.

- Enums define terrain, weather, game status, actions, actors, structures, rule status, and build-pad status.
- Immutable command types define a grid position and an action request.
- Mutable state dataclasses represent terrain cells, the rover, structures, colony resources, mission data, objectives, the build pad, rules, history, and the complete world.
- Transition types describe action validation and the result of one engine step, including checksums, reward terms, events, and the resulting state.

This module describes data, not behavior. Do not put rule calculations, HTTP models, or frontend formatting in it.

### `engine/aresim/config.py`

Defines the frozen, grouped configuration dataclasses used by the engine.

Configuration is separated by responsibility:

- `WorldConfig`: map size, seed bounds, clock, Sol length, and weather cycle.
- `GenerationConfig`: procedural terrain feature counts and dimensions.
- `LandingConfig`: safe landing-area search and sanitization limits.
- `InitialConfig`: initial rover, infrastructure, colony, and dust values.
- `PayloadConfig`: cargo capacity, item masses, extraction, and unloading conversions.
- `PowerConfig`: generation, loads, charging, dust, and deficit behavior.
- `ServiceConfig`: service thresholds, degradation, dust accumulation, and repair values.
- `LifeSupportConfig`: water, oxygen, livability, and rover-health changes.
- `ActionConfig`: action costs, terrain/weather stress, and building values.
- `WarningConfig`: thresholds used to surface resource and health warnings.
- `RewardConfig`: every reward contribution and invalid-action penalty.
- `ReplayConfig`: gameplay schema version, application version, checkpoint frequency, and upload limit.
- `EngineConfig`: combines all groups and validates cross-cutting invariants.
- `ObservationConfig`: local crop dimensions and policy normalization scales.
- `RewardProfileConfig`: RL-facing shaped/sparse weights and clipping bounds.
- `EnvironmentConfig`: selected scenario identifier, engine, observation, action, reward, and task components.

Add a field here when a new configurable backend value is introduced. Add its canonical value in `defaults.py`, validate it where appropriate, and use the field from the owning rule rather than embedding a new number in gameplay code.

### `engine/aresim/defaults.py`

Creates the immutable `DEFAULT_ENGINE_CONFIG` and `DEFAULT_ENVIRONMENT_CONFIG` used by callers that do not provide overrides.

This is the single source of truth for simulator and replay defaults, including:

- the 32×32 world and time progression;
- terrain and landing-pad generation;
- initial rover, colony, infrastructure, and weather values;
- the 12 kg rover capacity, 0.5 kg sample mass, and 2 kg extracted-ice mass;
- action energy, load, stress, and construction values;
- power, service, life-support, and livability behavior;
- warning thresholds and reward coefficients;
- gameplay schema, checkpoint interval, and upload limit.

The module validates the completed default configuration during import so invalid defaults fail immediately. Callers can create isolated variations with `dataclasses.replace`; they should never mutate the global default.

### `engine/aresim/registry.py`

Owns the explicit repository-local component and agent registry and the uniform `ComponentBuildContext`. Every entry is a typed factory over that context. The registry includes the built-in environment components plus `random`, `random_valid`, `wait`, and `scripted` policies, validates factory results against public contracts and spaces, and rejects duplicate or unknown names. It does not use entry points or automatic import discovery.

### `engine/aresim/factory.py`

Provides `make_engine`, `make_agent`, `make_env`, `make_gym_env`, and `make_parallel_env`. It resolves configuration through one registry and optionally applies the same external step limit before wiring framework adapters. Import this module only when the optional `env` dependencies are installed.

## 5. Deterministic core

The `core/` package owns authoritative world generation and state transitions. It must remain usable without starting a web server and without importing frontend or learning-framework code.

### `engine/aresim/core/__init__.py`

Defines the public surface of the core package. It re-exports `AresEngine` and `state_checksum` so callers can use the engine and its deterministic fingerprint helper without knowing their file location.

### `engine/aresim/core/engine.py`

Provides `AresEngine`, the main Python interface to deterministic simulation.

- `state_checksum(state)` produces a stable SHA-256 fingerprint of a canonical serialized `WorldState`.
- Validates the selected configuration at construction time.
- `reset(seed)` creates and initializes a new world.
- `step(command, actor)` applies exactly one command and returns an `EngineTransition`.
- Computes state checksums immediately before and after a step.
- `pause()` and `resume()` change session status without advancing simulation time.
- Returns deep copies of state so external callers cannot mutate authoritative engine state.

Checksums prove that identical seeds and command sequences produce identical state transitions. Keep their serialization stable and include all authoritative gameplay state. `AresEngine` otherwise coordinates generation and rules; it should not contain the detailed calculations owned by those modules.

### `engine/aresim/core/generation.py`

Creates a complete deterministic world from a seed and `EngineConfig`.

- Implements the seeded pseudo-random generator used by terrain generation.
- Produces smooth terrain features for regolith, ice, ore, rocks, dunes, ridges, and craters.
- Searches for a safe landing location.
- Sanitizes and marks the contiguous 5×5 landing/build-pad area.
- Creates the initial rover, structures, resources, mission, objectives, rules, weather, and world metadata.

The same seed and configuration must always create the same world. New generation logic must not read wall-clock time or use an unseeded random source.

### `engine/aresim/core/rules.py`

Owns the gameplay behavior for all public actions: Move, Scan, Extract, Build, Service, Unload, and Wait.

Its responsibilities include:

- resolving omitted action targets when a supported contextual target can be inferred;
- checking bounds, terrain, range, service, build, extraction, and payload constraints;
- applying valid actions and recording invalid actions without mutating protected gameplay state;
- enforcing atomic cargo capacity for 0.5 kg samples and 2 kg ice extraction;
- unloading rover cargo at the build pad;
- applying battery use, power generation/consumption, charging, weather stress, dust, and degradation;
- applying water, oxygen, rover health, habitat health, and livability changes;
- calculating individual reward terms and cumulative reward totals;
- updating build-pad service state, warnings, mission rules, events, history, time, Sol, victory, and failure state.

Behavioral rule changes belong here. Numerical tuning belongs in `config.py` and `defaults.py` unless the value is structural rather than configurable.

## 6. Trajectory recording and replay

### `engine/aresim/gameplay.py`

Owns the replay projection nested in `aresim.trajectory.episode.v1`, legacy `aresim.gameplay.v1` import, and exact replay reconstruction.

- Computes compact step deltas by comparing consecutive UI snapshots.
- Applies deltas to rebuild later snapshots without re-running simulator rules.
- Creates initial, interval, significant-event, and final checkpoints.
- Wraps canonical Manual or Algorithm exports in the unified trajectory episode envelope.
- Sanitizes downloaded filenames.
- Validates canonical gameplay files.
- Normalizes supported legacy wrappers and raw snapshots into the canonical format, including older cargo fields.
- Rejects unsupported schema versions through `UnsupportedGameplaySchema`.
- Uses `ReplayCursor` to step, jump, and reset a replay from checkpoints and deltas.

Replay is intentionally state reconstruction rather than simulation. Changing current rules must not change the outcome of an already saved canonical replay.

## 7. Session service and REST API

### `engine/aresim/service.py`

Provides the application service between transport code and the engine.

- Owns one in-memory live session and one in-memory loaded replay.
- Serializes mutations with a reentrant lock.
- Creates random or explicitly seeded sessions.
- Applies player or agent commands through `AresEngine` and records completed steps.
- Pauses, resumes, and exports the current session.
- Enforces replay upload size, parses JSON, accepts trajectory and legacy gameplay imports, and manages replay navigation.
- Raises `ServiceError` with stable machine-readable codes and HTTP status information.

Session ownership, replay ownership, and use-case coordination belong here. HTTP request parsing belongs in `api.py`, while gameplay rules belong in `core/rules.py`.

### `engine/aresim/api.py`

Defines the FastAPI application used by the React UI.

- Declares strict Pydantic request models; unknown fields are rejected.
- Builds the app through `create_app(config)` so tests and local callers can supply configuration.
- Converts service and validation failures into the common `{ "error": { "code", "message" } }` response shape.
- Exposes health, session creation/read/action, pause/resume, save, replay load, replay step, replay jump, and replay reset endpoints.
- Converts public JSON action and position values into canonical engine command types.
- Exposes a module-level `app` for ASGI tools and starts Uvicorn when run with `python -m aresim.api`.

Keep this module thin. It should translate HTTP requests and responses, not duplicate session or simulation behavior.

## 8. RL composition and environment adapters

### `engine/aresim/components/`

- `base.py` defines the public generic `ObservationBuilder`, `ActionAdapter`, `RewardFunction`, and `TaskEvaluator` protocols and their lifecycle contracts.
- `observations.py` builds the bounded local crop, self/colony vectors, categorical telemetry, and empty targetless objective tensors.
- `actions.py` maps the ten stable action IDs to explicit current/adjacent canonical commands and derives an `int8[10]` mask from core validation.
- `rewards.py` calculates shaped-training and sparse-evaluation breakdowns from immutable before/after transitions without changing engine/UI history.
- `tasks.py` delegates termination to the engine's game status and failed rules; open exploration has no success condition or deadline.
- `__init__.py` re-exports the public protocols and built-in component classes.

### `engine/aresim/envs/`

- `environment.py` owns framework-neutral reset/step composition, its shared protocol, and `AresTimeLimit`. The external wrapper changes only truncation/lifecycle metadata and never simulator state.
- `gymnasium.py` unwraps exactly `rover_0` into Gymnasium's scalar API and declared spaces.
- `pettingzoo.py` exposes the same transition through one-rover Parallel dictionaries and stable agent identity.
- `__init__.py` re-exports the three environment boundaries and their result types.

Neither adapter implements gameplay validation, state mutation, observation features, masking, or reward formulas independently.

## 9. Baseline agents, algorithms, and rollout data

### `engine/aresim/algorithms/`

- `base.py` defines the public generic `Agent` contract, schema compatibility, reset seeding, and action interface.
- `baselines/` groups the four built-in baseline policies (`random.py`, `random_valid.py`, `wait.py`, `scripted.py`). The scripted policy cannot access canonical `WorldState`.
- `common/masks.py` shares mask validation across baselines.
- `common/config_decode.py` shares YAML-to-dataclass decoding and scalar validation for algorithm and experiment configs.
- `registry.py` owns the algorithm, model, and checkpoint-loader training registry.
- `ppo/config.py` owns masked PPO hyperparameters and model architecture dataclasses.
- `ppo/workflow.md` documents the masked PPO data flow, model inputs, and training/checkpoint path.
- `ppo/train.py` owns the actor-critic, RLModule, W&B callbacks, and Ray Tune training path.
- `ppo/checkpoint.py` validates native checkpoint provenance and adapts frozen inference to `Agent`.
- `__init__.py` re-exports the contract and built-in policies.

### `engine/aresim/training/`

- `runner.py` owns explicit environment/agent episode seeds, the 1,200-step default cutoff, complete episode collection, summaries, and optional writer integration. It is not an online training collector or replay buffer.
- `trajectories.py` owns `aresim.trajectory.v1`, standalone `aresim.trajectory.episode.v1` artifacts, recursive Gymnasium space descriptors, plain/gzip JSONL shards, SHA-256 manifests, atomic publication, typed reading, and structural/reward/alignment/replay validation.
- `experiments.py` owns immutable experiment settings, safe YAML, strict typed overrides, and configuration hashes.
- `seeds.py` and `evaluation.py` own fixed splits and framework-neutral frozen-policy evaluation. The checked-in split is `notebooks/phase1_open_exploration_split_v1.yaml`.
- `reports.py` pulls W&B training history and writes matplotlib plots plus exported metrics under `<run>/reports/` without starting Ray or the simulator.
- `cli.py` provides train, evaluate, report, and inspect commands.
- `__init__.py` re-exports rollout, trajectory, and lazy learned-policy APIs.

RLlib uses its own collectors for online learning. Checkpoint-backed policies implement `Agent` and reuse the runner for matched evaluation and trajectory export.

## 10. Integration adapters

### `engine/aresim/integrations/__init__.py`

Defines the public surface of the integrations package. It currently re-exports `snapshot_from_state`.

### `engine/aresim/integrations/ui.py`

Converts the internal Python `WorldState` into the existing frontend `SimSnapshot` contract.

- Converts snake_case dataclass fields to the camelCase keys expected by TypeScript.
- Serializes nested terrain, rover, structure, resource, objective, reward, mission, rule, and history data.
- Preserves UI compatibility without forcing frontend naming conventions into the deterministic core.

When a field is added to authoritative state, decide explicitly whether the UI needs it. If it does, update this adapter and the matching frontend type together.

## 11. Tests

### `engine/tests/test_agents.py`

Tests agent registration/contracts, deterministic RNG reset, random legality differences, Wait, and scripted priorities without exposing engine state.

### `engine/tests/test_components.py`

Tests observation schemas, configurable crops, edge padding, hidden-cell exclusion, action decoding/mask parity, targetless reward profiles, and all terminal reasons.

### `engine/tests/test_envs.py`

Runs Gymnasium and PettingZoo compliance/seed tests, direct-engine parity, adapter parity, lifecycle validation, reward auditing, and all three engine failure paths.

### `engine/tests/test_extensibility.py`

Provides reusable protocol contract checks and deliberately different custom observation, action, reward, and task implementations. It verifies reset-aware state, typed context factories, invalid factory rejection, custom configuration capture, non-`Discrete.n` mask spaces, and direct/Gymnasium/PettingZoo parity.

### `engine/tests/test_rollouts.py`

Tests exact external limits, natural termination precedence, explicit seed behavior, deterministic rollout results, reward auditing, and configuration validation.

### `engine/tests/test_trajectories.py`

Tests plain/gzip JSONL round trips, supported spaces, dtype restoration, deterministic compression, sharding, and corruption rejection.

### `engine/tests/test_rllib_pipeline.py`

Tests checkout experiment YAML, fixed seed splits, model masking, registry name rejection, canonical W&B metric mapping, W&B-backed report inputs, optional-dependency isolation, and the opt-in real PPO/checkpoint smoke path.

### `engine/tests/test_engine.py`

Tests deterministic generation and simulator rules.

Coverage currently includes:

- default validation and isolated configuration overrides;
- same-seed determinism, different generated worlds, and safe landing pads;
- identical final checksums for identical command sequences;
- every public action and invalid transitions;
- atomic payload-capacity enforcement and unloading;
- latched build-pad service state;
- livability failure behavior;
- pause and resume without time advancement.

Add tests here for changes to `types.py`, configuration, generation, checksums, engine coordination, or gameplay rules.

### `engine/tests/test_gameplay.py`

Tests gameplay recording, validation, compatibility, and reconstruction.

Coverage currently includes:

- step deltas and interval, event, and final checkpoints;
- exact replay step, jump, and reset reconstruction;
- checkpoint priority when multiple checkpoint reasons share a step;
- canonical files, legacy LLM metadata, legacy wrappers, and raw snapshots;
- rejection of structurally invalid gameplay.

Add tests here whenever the save schema, delta format, checkpoint policy, legacy normalization, or replay navigation changes.

### `engine/tests/test_api.py`

Tests the REST boundary through an in-process FastAPI client.

Coverage currently includes:

- session creation, action, pause, save, replay load, and replay navigation routes;
- consistent typed error responses;
- raw snapshot upload normalization;
- oversized files, unsupported schemas, and malformed gameplay.

Add tests here for route contracts, request validation, error mapping, service/session behavior visible over HTTP, or application construction.

## 12. Where to make common changes

| Change | Primary file(s) | Also verify |
|---|---|---|
| Tune an existing gameplay value | `aresim/defaults.py` | Relevant rule test and behavior documentation |
| Add a configurable value | `aresim/config.py`, `aresim/defaults.py` | Validation and tests |
| Add or change authoritative state | `aresim/types.py` | Generation, rules, UI adapter, save/replay compatibility, tests |
| Change terrain or initial-world generation | `aresim/core/generation.py` | Determinism and landing-pad tests |
| Change an action or simulator rule | `aresim/core/rules.py` | Engine tests, UI action contract, environment rules |
| Change reset, step, pause, or resume coordination | `aresim/core/engine.py` | Engine and API tests |
| Change deterministic fingerprinting | `aresim/core/engine.py` | Determinism tests and stored-manifest expectations |
| Change the frontend snapshot | `aresim/integrations/ui.py` | Frontend types/store, replay format, API tests |
| Change saves, checkpoints, or replay | `aresim/gameplay.py` | Gameplay format documentation and replay tests |
| Change session ownership or a use case | `aresim/service.py` | API and service-facing tests |
| Add or change an HTTP endpoint | `aresim/api.py` | REST client, API tests, backend documentation |
| Add a stable public Python import | `aresim/__init__.py` | README example and import tests if needed |
| Add or change a dependency | `pyproject.toml` | README setup and dependency rationale |
| Change actor observation or normalization | `components/observations.py`, environment config/defaults | Space, leakage, and adapter-parity tests |
| Change the RL action mapping or mask | `components/actions.py` | Core validator agreement and PettingZoo sampling |
| Change RL reward/task projection | `components/rewards.py`, `components/tasks.py` | Engine reward separation and parity tests |
| Change environment adapter shapes | `envs/` | Gymnasium/PettingZoo compliance and direct parity |
| Add or change a baseline policy | `algorithms/` | Agent contracts, mask behavior, fixed-seed rollouts |
| Add or change a learned algorithm | `algorithms/ppo/` (or future `algorithms/<name>/`) | Registry, experiment YAML, RLlib smoke tests |
| Change rollout or trajectory semantics | `training/` | Alignment, integrity, round-trip, and gameplay-format separation |

## 13. Learned-policy training

The optional learned-policy code is split between `aresim/algorithms/` (policy implementations and training) and `aresim/training/` (rollouts, trajectories, evaluation harness, CLI). RLlib is an implementation detail inside `algorithms/ppo/train.py`, not the public package name:

```text
algorithms/ppo/config.py     masked PPO hyperparameters and model architecture
algorithms/ppo/train.py      actor-critic, RLModule, RLlib/Ray Tune, W&B metrics
algorithms/ppo/checkpoint.py native checkpoint sidecar and Agent adapter
algorithms/registry.py       algorithms, models, and checkpoint loaders
training/experiments.py      immutable experiment envelope, safe YAML, hashes
training/seeds.py            fixed train/validation/test manifest
training/evaluation.py       frozen RolloutRunner evaluation and trajectories
training/reports.py          W&B metric plots and exported tables under <run>/reports/
training/cli.py              train/evaluate/report/inspect commands
```

Single-rover PPO trains through `AresGymEnv`. `AresParallelEnv` remains the canonical future multi-rover boundary. Neither training nor evaluation reimplements simulator rules, observations, rewards, or legality.

The public boundary is imported from `aresim.training`:

```python
from aresim.training import evaluate_checkpoint, generate_report, load_experiment, make_checkpoint_agent, run_experiment

spec = load_experiment("configs/masked_ppo/smoke.yaml")
run_directory = run_experiment(spec)
agent = make_checkpoint_agent(run_directory / "checkpoints/final/checkpoint.json")
evaluate_checkpoint(run_directory / "checkpoints/final/checkpoint.json")
generate_report(run_directory)
```

RLlib checkpoints remain native and restore through `Agent` for shared evaluation. New algorithms, models, and checkpoint loaders register under new semantic names rather than adding framework conditionals. W&B is the sole application-level training log; `aresim-rl report` writes local plots from that history. See [RL Algorithms, Training, and Evaluation](../rl/rl_quickstart.md).

## 14. Maintenance checklist

Whenever `engine/` changes:

1. Update the **Last updated** date at the top of this document.
2. Update the current folder tree.
3. Add, remove, or revise the affected per-file description.
4. Update the data-flow or common-change table if ownership moved.
5. Keep planned code clearly separated from implemented code.
6. Update the relevant tests and domain/API documents.
7. Confirm generated caches and local environment files are not documented as source files.
