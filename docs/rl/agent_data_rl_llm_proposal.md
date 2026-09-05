# AresSim Agent Data, RL, and LLM Architecture Proposal

Last updated: 2026-08-25

Status: The framework-neutral environments, baselines, truncation, deterministic rollouts, unified trajectories, fixed seed splits, RLlib masked PPO pipeline, checkpoint evaluation, canonical metrics, and reports are implemented. Goal-bearing tasks, DQN/recurrent policies, advanced representation learning, and multiple rovers remain proposed. Existing simulation semantics remain governed by [Environment Rules Reference](../product/environment_rules.md), trajectory usage by [RL Usage Guide](usage.md), and learned-policy implementation by [RL Algorithms, Training, and Evaluation](rl_quickstart.md).

> New to RL or implementing an agent? Start with [RL Algorithms, Training, and Evaluation](rl_quickstart.md). It explains the active contracts, implemented policies, online sampling, learner updates, and evaluation workflow.

## 1. Decision summary

AresSim should not expose one large JSON snapshot to every consumer. It should maintain one authoritative, typed world state and derive purpose-specific views from it:

```text
Scenario + seed
      |
      v
Canonical WorldState -----> UI snapshot / gameplay replay
      |
      +-----> local 8 x 8 RL observation     (active actor contract)
      +-----> discovered-map memory          (optional research setting)
      +-----> global critic state            (future multi-agent CTDE)
      +-----> compact LLM context + tools    (strategic control)
      |
Validated structured action
      |
      v
Core transition + events
      |
      v
AresEnvironment -> next observation + reward -> trajectory dataset
```

The recommended first implementation is:

1. Keep the implemented deterministic, headless Python engine with ordinary typed dataclass/list state; introduce NumPy only for numerical policy observations in the PettingZoo phase.
2. A rover-centered `8 x 8` symbolic observation for all acting policies, with privileged global state kept outside the actor contract.
3. Seven canonical verbs represented by 10 fixed policy actions: Wait, four movement directions, Scan, Extract, Build, Service, and Unload.
4. Configurable training and evaluation reward profiles built from the same named reward terms.
5. PettingZoo Parallel as the canonical multi-agent RL API from the first one-rover adapter, preserving the same dictionary contract when multiple rovers arrive.
6. A thin `AresGymEnv` for exactly one rover, unwrapping PettingZoo-shaped results into Gymnasium's scalar API without becoming another source of behavior.
7. RLlib as the single supported learned-policy framework. Masked PPO is the reference; DQN, recurrence, JEPA, world models, and hybrids use explicit RLlib module/learner extensions when implemented.
8. LLMs first as low-frequency mission planners that issue structured subgoals to an RL or deterministic executor—not as unvalidated, per-tick simulator controllers.
9. RLlib-owned online collection/training through EnvRunners and learners, with AresSim-owned configuration, evaluation, metrics, artifacts, and optional unified trajectory persistence.

The most important architectural rule is:

> Canonical state, agent observation, legal actions, rewards, trajectories, and presentation are separate versioned contracts.

## 2. Goals and non-goals

### 2.1 Goals

- Make every input feature explicit, typed, bounded, normalized, and versioned.
- Support scripted, RL, planning, LLM, and hybrid agents without changing simulation logic.
- Treat partial observability as the baseline and make memory requirements explicit in algorithm comparisons.
- Preserve deterministic replay and make every reward auditable.
- Allow future variable entity counts and multiple rovers without redesigning the entire API.
- Keep the hot simulation/training path compact and avoid serializing JSON on every environment step.
- Make benchmark comparisons reproducible across algorithms, seeds, scenarios, and reward profiles.

### 2.2 Non-goals for the first engine milestone

- Pixel-only learning from the rendered UI.
- Continuous-control driving physics.
- Free-form LLM mutation of world state.
- Learned reward models replacing the deterministic reward engine.
- Simultaneous human crew, rover fleets, and colony-scale economics.
- Changing Phase 1 terrain and action semantics merely to fit a particular library.

## 3. Contract families and versioning

Every run must pin the versions below in its manifest. A schema changes only when a field's meaning, shape, bounds, ordering, or legality changes.

| Contract | Initial identifier | Purpose |
|---|---|---|
| Canonical state | `aresim.state.v1` | Authoritative simulator truth |
| Scenario | `aresim.scenario.v1` | Seed, map, limits, capacities, objectives, randomization |
| Task | `aresim.task.v1` | Success, failure, truncation, objectives, reward profile |
| Local rover observation | `aresim.obs.local.v1` | Active fixed `8 x 8` actor input without global memory |
| Partial-map observation | `aresim.obs.partial_map.v1` | Optional discovered global memory derived only from past local observations |
| Global critic state | `aresim.state.critic_global.v1` | Privileged centralized value input |
| Canonical action | `aresim.action.command.v1` | Stable verb, movement direction, and resolved audit command |
| Rover RL action | `aresim.action.rover.v1` | Ten fixed movement/current-cell actions with legality mask |
| Reward terms | `aresim.reward.mission.v1` | Named raw terms and configurable weights |
| LLM context/tools | `aresim.llm.ops.v1` | Compact context and strict tool schemas |
| Trajectory dataset | `aresim.trajectory.v1` | Manifest-backed transition-aligned data |
| Standalone trajectory | `aresim.trajectory.episode.v1` | Policy trace plus complete UI replay projection |

Each saved model must record at least:

```json
{
  "environment_version": "0.1.0",
  "git_commit": "<commit>",
  "environment_api": "pettingzoo_parallel",
  "framework": "rllib",
  "framework_version": "<pinned-version>",
  "possible_agents": ["rover_0"],
  "policy_mapping": "shared_rover_policy",
  "state_schema": "aresim.state.v1",
  "observation_schema": "aresim.obs.local.v1",
  "action_schema": "aresim.action.rover.v1",
  "reward_profile": "aresim.reward.shaped_train.v1",
  "task_id": "phase1_integrated_survival_v1",
  "scenario_set": "phase1_train_v1",
  "algorithm": "ppo",
  "agent_count": 1,
  "algorithm_config_hash": "<sha256>",
  "seed": 1234
}
```

Do not silently load a model against a different schema. An explicit migration or compatibility adapter is required.

`environment_api` records the actual adapter used by the run: `gymnasium` or `pettingzoo_parallel`. A Gymnasium run still records `possible_agents = ["rover_0"]` and `agent_count = 1` so evaluation and trajectory metadata retain stable identity. Changing adapter must not change state, observation, action, reward, or task signatures.

## 4. Canonical authoritative state

### 4.1 What belongs in canonical state

`WorldState` contains physical and mission truth required to advance the simulation. It is not directly returned to a policy.

```python
@dataclass
class WorldStateV1:
    schema_version: Literal["aresim.state.v1"]
    session_id: str
    seed: int
    step: int
    sim_time_seconds: float
    sol: int
    local_time_fraction: float       # [0, 1)
    status: GameStatus

    terrain: TerrainStateV1
    rovers: dict[int, RoverStateV1]
    structures: dict[int, StructureStateV1]
    colony: ColonyStateV1
    weather: WeatherStateV1
    task: TaskRuntimeStateV1

    rng_state: bytes
    next_entity_id: int
```

Use stable unsigned integer IDs internally. Display names and UI string IDs are presentation metadata. Never use an entity's array row as its canonical identity.

### 4.2 Dense terrain storage

The world grid is dense and fixed for an environment instance. Store it as structure-of-arrays rather than `TerrainCell[][]` objects in the hot path.

| Field | dtype and shape | Bounds/meaning |
|---|---|---|
| `terrain_type` | `uint8[H,W]` | Enum in Section 5.2 |
| `height` | `float32[H,W]` | `[0,1]` |
| `roughness` | `float32[H,W]` | `[0,1]` |
| `ice` | `float32[H,W]` | `[0,1]` |
| `ore` | `float32[H,W]` | `[0,1]` |
| `dust` | `float32[H,W]` | `[0,1]` |
| `scanned` | `bool[H,W]` | Successful scan has occurred |
| `extracted` | `bool[H,W]` | Successful extraction marker |

Coordinates are integer `(x, y)`, but arrays are indexed `[y, x]`. This distinction must be asserted in tests and stated in every external schema.

For the default map, `H = W = 32`. A PettingZoo environment instance has fixed Gymnasium spaces; changing dimensions creates a different configured environment and therefore different fixed spaces.

Phase 1 keeps no dense occupant grid in canonical terrain state. The rover's coordinates live in `RoverStateV1`, and integrated Build Pad roles live in their infrastructure/Colony records. If later multi-rover collision or independently placed blocking entities make a spatial index useful, derive or add it with the corresponding state and observation schema version rather than exposing a redundant Phase 1 tensor.

### 4.3 Entity records

Entities remain typed records in canonical state for simulation, replay, and future extensions. Phase 1 does not convert them into actor-observation tables: the only rover is already described by `self`, and infrastructure roles are integrated into the Build Pad rather than independently navigable policy objects. A future multi-rover or independently placed-structure observation may add versioned entity tables.

```python
@dataclass
class RoverStateV1:
    id: int
    x: int
    y: int
    battery: float          # [0, battery_capacity]
    health: float           # [0, health_capacity]
    cargo_ice: float
    cargo_ore: float
    cargo_samples: float
    cargo_capacity_kg: float
    current_task_id: int | None
    alive: bool

@dataclass
class StructureStateV1:
    id: int
    structure_type: StructureType
    x: int
    y: int
    health: float
    powered: bool
    status: StructureStatus
    build_progress: float   # [0, 1]
```

Do not store missing type-specific values as `NaN`. Use explicit optional canonical fields, and use a presence mask when converting heterogeneous entities to tensors.

### 4.4 Colony, weather, task, and time

Canonical state should retain physical values, not already normalized model inputs:

- Power generated, consumed, stored, and capacity.
- Water, oxygen, and livability with scenario-defined capacities.
- Build-pad service state and habitat progress.
- Weather enum, severity/dust intensity, start time, and remaining duration.
- Objective counters and completion predicates.
- Episode success/failure status and reason.
- Simulation time and environment step, separately.

The scenario declares capacities and legal bounds. Observation builders use those values to normalize inputs; reward code uses physical deltas and task configuration.

The current Phase 1 mock uses a small passive livability delta of `-0.03` per valid action, then applies additional power-deficit and empty-reserve penalties. Its build-pad maintenance state is intentionally infrequent and sticky: it changes to `needs_service` only when dust exceeds `0.78`, infrastructure health falls below `62%`, or power margin is below `-8` while dust is above `0.55`; it remains set until Service succeeds. These values belong in scenario/environment configuration when the mock rules move into the package engine, rather than being hard-coded in an RL policy.

### 4.5 Events as the mutation audit trail

Every accepted or rejected command emits typed events. State mutation, rewards, UI history, and trajectory diagnostics consume the same events.

```json
{
  "event_id": 1842,
  "step": 71,
  "type": "ice_extracted",
  "actor_id": 1,
  "target": {"kind": "cell", "x": 13, "y": 9},
  "values": {"ice_removed": 0.35, "battery_used": 2.1}
}
```

Events should use enum codes internally and readable names at serialization boundaries. Reward terms should reference contributing `event_id` values so a surprising reward can be traced back to world mutations.

## 5. Observation inputs

### 5.1 General rules

All agent inputs must obey these rules:

- Use a `Dict` of semantically typed arrays rather than one unexplained flat vector.
- Use `float32` for continuous policy inputs, `uint8`/integer for categories and masks.
- Normalize continuous values using scenario-declared constants, not batch statistics that change between runs.
- Map an unknown spatial category to ID `0`; never pretend an unseen cell is regolith.
- Use masks for padded or unavailable entries. After masking, padded numeric values are zero.
- Sort padded entity rows deterministically: self first, then increasing Manhattan distance, then stable entity ID.
- Keep stable IDs out of neural features. Return row-to-ID mappings in `info` for debugging and pointer actions.
- Include legal-action information through a dedicated mask, not as an implicit feature.
- Do not expose hidden simulator truth to a partially observable policy.
- Do not include UI camera, hover, selection, audio, overlays, formatted strings, or rendered colors.

### 5.2 Stable category mappings

Spatial/world categories reserve `0` for unknown or padding. `pad_proximity` is actor-relative and uses `0` for its explicit outside-range state.

```text
terrain_type:
  0 unknown/out_of_bounds
  1 regolith
  2 rock
  3 ice
  4 crater
  5 dune
  6 build_pad
  7 ridge

weather_type:
  0 unknown
  1 clear
  2 dusty
  3 dust_front
  4 severe_storm
  5 cold_night

pad_proximity:
  0 outside_service_range
  1 within_service_range
  2 on_build_pad
```

Enums are append-only within a schema version. Reassigning a number requires a new schema.

### 5.3 Active actor input: `aresim.obs.local.v1`

All acting policies receive a fixed rover-centered `8 x 8` crop. The authoritative `32 x 32` map remains available to the engine, replay reconstruction, oracle evaluation, and a future centralized critic, but never to the decentralized actor at inference.

```python
spaces.Dict({
    "terrain_type":     spaces.Box(0, 7, shape=(8, 8), dtype=np.uint8),
    "spatial":          spaces.Box(0.0, 1.0, shape=(5, 8, 8), dtype=np.float32),
    "cell_flags":       spaces.Box(0, 1, shape=(4, 8, 8), dtype=np.uint8),
    "self":             spaces.Box(-1.0, 1.0, shape=(10,), dtype=np.float32),
    "pad_proximity":    spaces.Discrete(3),
    "colony":           spaces.Box(-1.0, 1.0, shape=(14,), dtype=np.float32),
    "weather_type":     spaces.Discrete(6),
    "objective_type":   spaces.Box(0, 8, shape=(8,), dtype=np.uint8),
    "objectives":       spaces.Box(0.0, 1.0, shape=(8, 4), dtype=np.float32),
    "objective_mask":   spaces.Box(0, 1, shape=(8,), dtype=np.uint8),
})
```

The rover is anchored at local `[3,3]`. For world rover coordinate `(rx, ry)`, local `[0,0]` maps to `(rx - 3, ry - 3)`, so each axis covers offsets `-3..+4`. This asymmetry is the deterministic consequence of an even-sized window. Out-of-bounds slots are unknown padding and remain at their local index; the crop never shifts at an edge. The TypeScript preview/builder implements the same mapping in `web/src/lib/roverObservation.ts`.

Legal actions accompany the observation as `action_mask[10]`. Gymnasium and PettingZoo expose `{"observation": observation, "action_mask": mask}` once, and the RLlib RLModule applies that mask to final logits. No adapter may infer legality or create another mask.

The declared maximum of eight objectives is a scenario-family limit, not a promise that all rows exist. A scenario exceeding it must fail at reset rather than silently truncate objectives.

#### Spatial channels

`spatial[5,8,8]` uses this fixed order:

| Channel | Value |
|---:|---|
| 0 | Height |
| 1 | Roughness |
| 2 | Ice signal |
| 3 | Ore signal |
| 4 | Dust accumulation |

`cell_flags[4,8,8]` uses:

| Channel | Value |
|---:|---|
| 0 | Known/discovered |
| 1 | Currently visible |
| 2 | Scanned |
| 3 | Extracted |

For `local`, known and visible are `1` for every in-bounds slot in the current crop and `0` for world-edge padding. No prior global map is included. Terrain and resource fields for padding are zero. Phase 1 has no actor-visible occupancy grid or entity tables.

Categorical grids should be embedded or one-hot encoded inside the model. Store categories compactly in observations and replay; do not save expanded one-hot maps.

#### Self vector

`self[10]` uses:

| Index | Feature | Normalization |
|---:|---|---|
| 0 | `x` | `x / (W-1)` |
| 1 | `y` | `y / (H-1)` |
| 2 | Battery | current/capacity |
| 3 | Health | current/capacity |
| 4 | Carried ice mass | kg/shared payload capacity |
| 5 | Carried ore mass | kg/shared payload capacity |
| 6 | Carried geological-sample mass | kg/shared payload capacity |
| 7 | Remaining payload capacity | `max(0, capacity - ice - ore - samples) / capacity` |
| 8 | Local time sine | `sin(2πt)` |
| 9 | Local time cosine | `cos(2πt)` |

The shared capacity is fixed by scenario configuration (`12 kg` in Phase 1), so it is not repeated as a per-step feature. The three carried masses remain separate; there is no opaque aggregate cargo category. Ore remains zero until an ore-collection mechanic exists. If a scenario does not use a cargo type, its normalized value remains zero.

#### Pad proximity category

`pad_proximity` replaces the redundant `on_build_pad` and `within_service_range` booleans:

```text
0 outside_service_range
1 within_service_range, but not on build-pad terrain
2 on_build_pad
```

Compute `2` first by checking the rover's current terrain cell. Otherwise return `1` if the Euclidean distance from the rover to any build-pad cell is at most the configured service radius (`2` cells in Phase 1); return `0` otherwise. Treat this as a category through one-hot encoding or an embedding, not as a normalized scalar distance.

This category describes rover location and action reach. It is distinct from `colony[12]`, which describes whether the infrastructure actually needs maintenance. Service range may be available while no service warning exists, and the warning may be active while the rover is far away. The legal-action mask remains authoritative for all additional preconditions.

#### Colony vector

`colony[14]` uses:

| Index | Feature |
|---:|---|
| 0 | Generated power / configured scale |
| 1 | Consumed power / configured scale |
| 2 | Signed power margin / configured scale, clipped to `[-1,1]` |
| 3 | Stored base battery / capacity |
| 4 | Water / capacity |
| 5 | Oxygen / capacity |
| 6 | Livability / capacity |
| 7 | Dust intensity |
| 8 | Habitat build progress |
| 9 | Ice-delivery objective progress |
| 10 | Scan objective progress |
| 11 | Service objective progress |
| 12 | Build pad needs service (`0`/`1`) |
| 13 | Aggregate required-objective progress |

Values with no active objective remain zero; the objective table tells the policy which objectives matter.

#### Objective table

Each of at most eight rows represents one declared objective.

`objective_type` initial values:

```text
0 padding, 1 navigate, 2 scan, 3 extract_ice, 4 deliver_ice,
5 build, 6 service, 7 survive, 8 maintain_resource
```

`objectives[row]` contains:

```text
[current_progress, target_progress, remaining_fraction, required_flag]
```

All values are normalized to `[0,1]`. `objective_mask[row] = 1` marks a valid row.

#### Why Phase 1 omits occupancy and entity tables

The single active rover is always at local `[3,3]`, and its global position and state are already present in `self`. Habitat, solar, battery, storage, and extractor roles belong to the integrated Build Pad; they are not independently navigable obstacles or separate action targets for the Phase 1 policy. Their useful aggregate effects are already represented by `terrain_type = build_pad`, Colony telemetry, habitat progress, and the service-needed bit.

Consequently, `aresim.obs.local.v1` contains no `occupancy_type`, rover table, or structure table. Internal engine coordinates may still support simulation and presentation, but they are not actor features. Introduce occupancy only through a new observation version if other rovers, independently placed structures, blocking entities, or construction sites make it decision-relevant.

### 5.4 Optional observation-memory profile

Partial observability is the active baseline. A discovered-map profile may later test explicit memory against recurrent policies, but it must be built only from the history of valid local observations.

#### `aresim.obs.partial_map.v1`

Use full-map memory shapes, but update them solely from `local` crops:

- Unknown cells have `terrain_type = 0`, numeric channels set to zero, and known/visible flags set to zero.
- Discovered but not currently visible cells retain the last observed static terrain and resource signals.
- `visible = 1` only inside the current 8 x 8 crop.
- Scan reveals scenario-configured resource information; ordinary visibility does not automatically reveal hidden resource signals.

This profile tests exploration and belief under fog-of-war while preserving a spatial memory map.

`local` omits a discovered global map and therefore makes memory valuable. Compare feed-forward and recurrent/state-space policies on the same crop. Legal-action data still arrives separately. A heading is not included because Phase 1 simulator movement is world-aligned; UI POV yaw is presentation-only.

### 5.5 Global critic state

`aresim.state.critic_global.v1` exposes the complete state-derived tensors to a centralized critic during multi-agent training. Individual actors still receive their own partial observation.

It may include:

- Complete map and all entity tables.
- All rover batteries, cargo, health, tasks, and positions.
- Complete weather and colony state.
- Joint objective progress.

It must not be returned to decentralized actors at inference. Actor and critic input schemas must be logged separately.

### 5.6 Optional pixel profile

Rendered RGB/depth/segmentation should be an optional benchmark profile, never the only source of truth. If later added, use fixed camera calibration, deterministic rendering, explicit frame stacking, and symbolic ground truth for evaluation. Pixel policies should still act through the same canonical command validator.

## 6. Actions and legal-action data

### 6.1 Canonical action command

All humans, scripted policies, RL adapters, planners, and LLM tools normalize into one command:

```json
{
  "schema_version": "aresim.action.command.v1",
  "command_id": "01J...",
  "actor_id": 1,
  "issued_at_step": 71,
  "verb": "extract",
  "target": null,
  "parameters": {}
}
```

Initial verbs are `move`, `scan`, `extract`, `build`, `service`, `unload`, and `wait`. Scan, Extract, and Build operate on the rover's current cell. Service uses the rover's current position and validates build-pad proximity. Unload is valid only while the rover is on the build pad with a non-empty payload and transfers the entire payload atomically. Move carries one of four cardinal directions in `parameters`; the engine resolves the adjacent world cell. Resolved coordinates belong in events and history for auditability, not in the policy action. `invalid` and `event` remain outcome/history categories, not actions an agent can choose.

Validation order must be deterministic:

1. Schema and actor validity.
2. Stale-step check.
3. Required/forbidden direction and parameter form.
4. Bounds and target existence.
5. Actor state and resource preconditions.
6. Terrain/range/action-specific rules.
7. Conflict resolution in multi-agent mode.
8. State transition and event emission.

Invalid commands consume a step only if the task profile says so. Training and evaluation must use the same setting.

### 6.2 Fixed rover adapter: `aresim.action.rover.v1`

Use this adapter first because PPO, DQN, behavior cloning, and most standard libraries handle a fixed `Discrete` space easily.

The `8 x 8` crop is observation context for navigation and planning; it is not a list of 64 possible targets for every verb. The current mechanics are local: the rover moves one adjacent cell, while Scan, Extract, Build, Service, and Unload act from its current position.

Use `action_space = Discrete(10)`:

| ID | Meaning | Resolution |
|---:|---|---|
| `0` | Wait | Current position |
| `1` | Move north | Adjacent northern cell |
| `2` | Move east | Adjacent eastern cell |
| `3` | Move south | Adjacent southern cell |
| `4` | Move west | Adjacent western cell |
| `5` | Scan | Current cell |
| `6` | Extract | Current cell |
| `7` | Build | Current cell |
| `8` | Service | Current position; validator checks pad proximity |
| `9` | Unload | Current build-pad cell |

`aresim.action.rover.v1` uses one flat categorical head. Do not factor it into separate verb and direction heads: the four Move directions are already four stable logits, while every other logit resolves directly to its current-cell command. This keeps masked PPO, DQN, behavior cloning, replay, and model compatibility straightforward.

`action_mask[10]` contains `1` exactly for selectable actions. Movement is masked for an observed out-of-bounds or blocked adjacent cell. Scan requires an unscanned rock under the rover and at least 0.5 kg free payload. Extract requires extractable ice under the rover and at least 2 kg free payload. Build requires the current cell to be build pad. Service requires the rover to be on or near the pad. Unload requires the rover to be on the pad with non-empty cargo. Wait remains legal so at least one action is always available.

The policy still needs the whole local crop. For example, if ice is two cells east, it should learn `Move east`, `Move east`, then `Extract`. Observation answers where to go; the action space answers what to do here.

Advantages:

- Stable meaning for every output logit.
- Straightforward masked categorical PPO and Q-learning.
- Only ten outputs instead of hundreds of mostly invalid cell/verb combinations.
- No local-target decoder or crop-origin conversion in the action path.
- Actions match the physical rover mechanics and produce simpler trajectory logs.

The limitation is intentional: current actions cannot affect a distant cell. If a future rover gains a ranged scanner, robotic arm, multi-cell construction tool, or entity-target command, add a new versioned targeted-action adapter rather than expanding this schema preemptively.

### 6.3 Deferred targeted-action extension

Do not implement this for the first environment. A future targeted adapter could expose:

```text
verb
target kind
local cell or entity pointer

verb mask
target mask
```

Such an adapter needs explicit range, line-of-sight, target identity, and masking rules. It should be introduced only with mechanics that require targeting. The existing `rover` schema must keep its meanings unchanged for model and trajectory compatibility.

Entity-target actions added in later phases should use stable entity IDs in canonical commands and masked row pointers in policies. Never reinterpret a changing padded row as permanent identity.

### 6.4 Meaning of legality masks

The canonical `LegalActionSet` is generated once from the agent's permitted information and then adapted through PettingZoo for RLlib, or another framework consumer.

```python
@dataclass
class LegalActionSetV1:
    flat_mask: np.ndarray              # int8[10], required by Gymnasium masked sampling
    rejection_reasons: dict[int, int]  # debug only, not policy input
```

Masks prevent structurally impossible actions and reduce wasted exploration. They must not leak hidden facts. Adjacent movement cells are within the current observation, so their visible boundary and terrain facts may be used. Privileged cells outside the crop must never affect the mask. The environment still validates every selected action.

## 7. Step and environment APIs

### 7.1 Canonical core transition

```python
@dataclass(frozen=True)
class EngineTransitionV1:
    command: CommandV1
    events: list[EventV1]
    before_checksum: str
    after_checksum: str
    after_state: WorldStateV1
```

The deterministic core accepts a validated canonical command and returns transition evidence. It does not build policy observations, calculate training rewards, enforce external episode length, or import environment/training libraries.

`AresEnvironment` composes the core with the selected scenario, observation, action, reward, and task components. Its internal result can therefore include learning-facing fields:

```python
@dataclass
class EnvironmentStepV1:
    observation: ObservationV1
    reward: float
    terminated: bool
    info: dict
    transition: EngineTransitionV1
    reward_breakdown: RewardBreakdownV1
```

Use `terminated` for success or environment failure. An outer PettingZoo wrapper adds `truncated` for an external training/evaluation limit. Do not collapse the flags into one `done` value in stored datasets. The Phase 1 simulator itself has no step or Sol deadline.

### 7.2 Canonical PettingZoo Parallel API

The first RL adapter is `AresParallelEnv(ParallelEnv)`. It begins with `possible_agents = ["rover_0"]`, but uses the final multi-agent shape immediately:

```python
observations, infos = env.reset(seed=seed, options=scenario_options)
observations, rewards, terminations, truncations, infos = env.step({
    "rover_0": action_id,
})
```

Each returned mapping is keyed by every live stable agent ID. Individual spaces are Gymnasium spaces exposed through `observation_space(agent_id)` and `action_space(agent_id)`. `info[agent_id]` includes reward terms, event codes, terminal reason, unnormalized metrics, scenario ID, and state checksum. Large full snapshots are opt-in debug data, not per-step output.

Apply `max_episode_steps` in an outer PettingZoo training/evaluation wrapper, not in WorldState or the actor observation. At the limit, set each live agent's `truncated = True` and preserve bootstrapping semantics; do not emit mission failure. Record the limit in the experiment manifest. PPO rollout fragment length remains an independent batching parameter, not an episode deadline.

PettingZoo's `parallel_api_test` is required for one rover and later multi-rover scenarios. Single-rover RLlib training uses `AresGymEnv`; future multi-rover RLlib adapters will wrap `AresParallelEnv`. AresSim does not copy rules, observations, rewards, or masks.

### 7.3 Single-rover Gymnasium API

`AresGymEnv(gymnasium.Env)` is an optional convenience adapter over the same composed `AresEnvironment`. It supports Gymnasium-native tools and checkers without making Gymnasium a second gameplay implementation:

```python
observation, info = env.reset(seed=seed, options=scenario_options)
observation, reward, terminated, truncated, info = env.step(action_id)
```

The adapter unwraps only `rover_0`: it removes the outer agent-ID dictionary and returns scalar reward/end flags. Observation and action spaces, action masks, reward terms, events, scenario IDs, and checksums retain exactly the same meanings as the one-rover PettingZoo environment. The adapter must fail clearly when the configured scenario has zero or more than one rover; it never selects an agent implicitly and is not a multi-agent compatibility layer.

Training trajectories produced through Gymnasium still store the stable agent ID `rover_0`; the convenience API does not erase identity from persistent artifacts.

Apply the same external `max_episode_steps` semantics in the adapters. A fixed seed and action sequence must produce identical transitions through direct composition, Gymnasium, and one-rover PettingZoo. Gymnasium's environment checker is required alongside PettingZoo's `parallel_api_test`.

### 7.4 Multi-rover extension on the same API

Multiple rovers are a later mechanics milestone, but not a later environment API. When multiple rovers are active, use one action per live rover per world tick:

```python
observations, rewards, terminations, truncations, infos = env.step({
    "rover_1": action_1,
    "rover_2": action_2,
})
```

Simultaneous conflicts must be resolved by deterministic rules independent of dictionary iteration order. For example: reject swaps if unsupported; reject all same-cell movement conflicts or resolve them by a scenario-declared rule; decide conflicts against the same pre-tick state; sort committed mutations by stable actor ID only after conflict decisions are complete.

RLlib maps stable agent IDs to one or more policy/module IDs through an explicit `policy_mapping_fn`. Start multi-rover learning with one shared rover policy, then compare role-specific policies and centralized-critic methods without changing PettingZoo observations or canonical commands.

## 8. Tasks and rewards

### 8.1 Task specification

Tasks should be data, not hard-coded algorithm branches.

```yaml
schema_version: aresim.task.v1
task_id: phase1_open_exploration_v1
success: null  # no checklist victory in open exploration

failure:
  any:
    - {metric: rover_health, op: le, value: 0.0}
    - {metric: rover_battery, op: le, value: 0.0}
    - {metric: livability, op: le, value: 0.0}

reward_profile: aresim.reward.shaped_train.v1
```

Evaluation can load the same task with `aresim.reward.sparse_eval.v1`. Objective counters and metrics remain identical. If a future task has a real deadline, define exactly one task-visible limit—steps or Sols—and introduce one explicit `time_remaining_fraction` observation through a versioned task/observation contract. Do not expose an external training cutoff as mission state.

### 8.2 Reward breakdown format

Never return only a scalar internally.

```json
{
  "schema_version": "aresim.reward.mission.v1",
  "step": 71,
  "terms": {
    "mission_success": {"raw": 0.0, "weight": 10.0, "value": 0.0},
    "objective_progress": {"raw": 0.05, "weight": 2.0, "value": 0.10},
    "energy_used": {"raw": 0.021, "weight": -0.05, "value": -0.00105},
    "invalid_action": {"raw": 0.0, "weight": -0.10, "value": 0.0}
  },
  "total_unclipped": 0.09895,
  "total": 0.09895,
  "source_event_ids": [1842]
}
```

Store raw measurements, weights, weighted contributions, unclipped total, final total, and source events.

### 8.3 Recommended reward profiles

#### Sparse evaluation profile

Use for final comparisons:

| Term | Initial weight |
|---|---:|
| Mission success | `+10.0` |
| Terminal rover/colony failure | `-5.0` |
| Invalid action | `-0.10` |
| All other shaping | `0.0` |

Report success rate and physical metrics separately from return. This profile prevents a policy from winning a benchmark by exploiting shaping without completing the mission.

#### Shaped training profile

Recommended initial values, to be tuned only on training/validation scenarios:

| Term | Raw definition | Initial weight |
|---|---|---:|
| `mission_success` | `1` on successful terminal step | `+10.0` |
| `terminal_failure` | `1` on failed terminal step | `-5.0` |
| `objective_progress` | Increase in normalized required-objective progress | `+2.0` |
| `new_scan` | `1` for first valid scan of a required site | `+0.10` |
| `ice_delivered` | Delivered ice / task target | `+0.50` |
| `samples_delivered` | Delivered geological sample mass / task target | `+0.20` |
| `build_progress` | Positive normalized build delta | `+0.50` |
| `service_recovery` | Normalized restored service/health | `+0.25` |
| `hazard_damage` | Normalized health loss caused this step | `-1.0` |
| `energy_used` | Battery used / capacity | `-0.05` |
| `invalid_action` | `1` for rejected command | `-0.10` |
| `time_cost` | `1` per nonterminal step | `-0.001` |

Clip only the final nonterminal shaped reward to `[-2, 2]`; do not clip terminal success/failure. Log the unclipped value. Reward normalization may be applied by a learner wrapper, but the stored environment reward stays canonical.

Where possible, use potential-based progress shaping:

```text
F(s, a, s') = gamma * Phi(s') - Phi(s)
```

This is preferable to repeatedly paying for being near a target or holding inventory, which invites loops. One-time achievements require explicit event guards.

### 8.4 Rewards are not metrics or constraints

Always log these independently of the scalar reward:

- Mission success and terminal reason.
- Objective completion times.
- Rover survival, final battery, final health, and minimum safety margin.
- Energy consumed, distance moved, hazard exposure, and idle steps.
- Invalid-action count and mask violations.
- Ice extracted versus delivered, samples collected versus delivered, unload count, payload utilization, and capacity-blocked actions.
- Habitat progress and service recoveries.
- Per-term cumulative reward.

Safety limits can later become constrained-RL costs, for example `battery_critical_steps` or `hazard_damage`, without hiding them inside a single shaped reward.

## 9. Trajectory and dataset format

### 9.1 Keep policy and replay views separate inside one trajectory

Bulk training datasets use compact, transition-aligned `aresim.trajectory.v1` shards. A default standalone `aresim.trajectory.episode.v1` contains that episode's policy view plus a readable, checkpointed replay projection in the same file. The views remain semantically separate—policies never consume privileged replay state—but users no longer need a second exported artifact. Legacy standalone `aresim.gameplay.v1` remains importable.

Recommended storage by use:

| Use | Format |
|---|---|
| In-process state and observation | NumPy arrays / Python records |
| Web control and UI snapshots | Existing typed JSON; later delta/WebSocket transport |
| Deterministic UI replay/export | `aresim.trajectory.episode.v1` JSON |
| Training/offline trajectories | Implemented JSONL shards, optionally gzip-compressed |
| Large event/reward analytics | Parquet tables |
| Checkpoints/models | Framework-native weights plus JSON manifest |

Do not encode/decode full JSON snapshots in the hot training loop.

### 9.2 Gameplay generation and collection

Training gameplay is generated by policies interacting with live, headless `AresParallelEnv` instances. It is not a library of hand-authored coordinate paths and it is not reconstructed from UI replays. A minimal one-rover loop, useful for tests and scripted collection, is:

```python
observations, infos = env.reset(seed=seed)
observation = observations["rover_0"]
mask = observation["action_mask"]

while not stopped:
    action, agent_state = agent.act(observation, mask, agent_state)
    next_observations, rewards, terminateds, truncateds, next_infos = env.step({"rover_0": action})
    next_observation = next_observations["rover_0"]
    transition = make_transition(
        observation=observation,
        action_mask=mask,
        action=action,
        reward=reward,
        next_observation=next_observation,
        next_action_mask=next_observation["action_mask"],
        terminated=terminateds["rover_0"],
        truncated=truncateds["rover_0"],
        info=next_infos["rover_0"],
    )
    sink.add(transition)
    observation, mask = next_observation, next_observation["action_mask"]

    if terminateds["rover_0"] or truncateds["rover_0"]:
        observations, infos = env.reset(seed=next_seed())
        observation = observations["rover_0"]
        mask = observation["action_mask"]
        agent_state = agent.initial_state()
```

The implemented `training/runner.py` collects complete baseline/evaluation episodes from explicit environment and agent seeds. RLlib EnvRunners, learners, and Ray Tune own online training. Evaluation, scripted/LLM collection, and dataset export use `Agent.act`; all paths consume the same observations, masks, rewards, and end flags.

The runner has two output modes independent of the selected framework:

1. **Online/in-memory:** let RLlib send transitions through its native collector path to the algorithm's rollout or replay storage. This is the default for PPO, DQN, and recurrent policies.
2. **Recorded episodes:** additionally send complete episodes to `training/trajectories.py` for validation and shard writing. Use this for demonstrations, offline RL, behavior cloning, JEPA, world models, reproducible evaluation, and selected debug runs.

`training/trajectories.py` is deliberately a thin optional sink. It does not choose actions, step the environment, calculate rewards, or own a second replay buffer. UI/manual sessions export the same standalone trajectory envelope with `policy: null`; they cannot invent transition-aligned local observations or masks after the fact. The privileged replay projection is never consumed by a learner.

#### Algorithm-specific use

| Algorithm family | Collection and storage behavior |
|---|---|
| RLlib PPO | EnvRunners collect episode fragments with the current action-masked RLModule; learners update the module and synchronized collection continues. Persistent recording is optional. |
| RLlib PPO | EnvRunners collect episodes; learners calculate advantages/losses and update policy/value modules. Persistent recording is optional. |
| Future mask-aware RLlib DQN | Store current/next masks and exclude illegal actions from selection and target calculation. |
| Recurrent RLlib policy | Preserve temporal order, episode-start markers, and recurrent state. Sequence chunks and burn-in windows must never cross an episode boundary. |
| Behavior cloning and offline RL | Train from recorded episodes produced by random-valid, scripted, human, LLM, or checkpoint policies. Dataset manifests identify every policy source. |
| JEPA | Sample ordered observation/action windows from recorded episodes. The collector is unchanged; only the dataset sampler and JEPA loss are new. |
| World model | Sample ordered transitions from the same recorded episodes. Predicted transitions may guide a planner but cannot be written back as authoritative simulator experience. |
| LLM and hybrid agents | Use the same `Agent.act` and validated environment step. Store prompts, tool calls, subgoals, fallbacks, latency, tokens, and cost as optional audit side data. |
| Multi-agent algorithms | Reuse the same PettingZoo Parallel contract, recording per-agent observations/actions/rewards plus the shared tick, joint conflict outcomes, and policy/module IDs. |

Each framework remains responsible for its native buffers, optimizer/learner loop, collectors/workers, and checkpoints. AresSim owns deterministic transitions, observations, legality masks, rewards, end signals, seeds, metric meanings, evaluation, and reproducibility metadata. Both frameworks receive masks from AresSim rather than infer game rules. Their checkpoints are deliberately framework-native and are not cross-loadable; thin checkpoint-backed agents expose the shared `Agent.act` interface for evaluation.

A practical first target is one local EnvRunner/learner. Batches may contain partial and completed episodes; collection boundaries are not episode boundaries. Scale only after determinism, worker seed independence, module shape checks, and adapter parity pass.

Training, validation, and test use separate saved seed lists. Training workers may cycle or sample only from the training list. Evaluation uses frozen checkpoints with learning disabled and records complete episodes. External time-limit transitions retain `truncated=True` and permit bootstrapping; mission completion/failure uses `terminated=True` and does not.

The Python package now provides the shared environments, numerical actor inputs, external truncation, registered baseline agents, deterministic complete-episode rollouts, and optional `aresim.trajectory.v1` JSONL shards. It does not yet provide RLlib integration, metric bridges, formal train/validation/test scenario manifests, aggregate evaluation, or online rollout/replay training. The old TypeScript simulator is retained only for test fixtures and historical parity, not as a production authority.

### 9.3 Transition alignment

For an episode of `T` actions:

```text
observations:     [0 .. T]      length T+1
action_masks:    [0 .. T]      length T+1
actions:          [0 .. T-1]   length T
rewards:          [0 .. T-1]   length T
reward_terms:     [0 .. T-1]   length T
terminated:       [0 .. T-1]   length T
truncated:        [0 .. T-1]   length T
events:           [0 .. T-1]   variable-length side table
```

The action at index `t` was selected from observation and action mask `t`; it produced reward `t` and observation `t+1`. Enforce these lengths in the writer.

### 9.4 Dataset manifest

```json
{
  "schema_version": "aresim.trajectory.v1",
  "dataset_id": "phase1_scripted_2026_07_16",
  "created_at": "2026-07-16T00:00:00Z",
  "application_version": "0.1.0",
  "observation_schemas": ["aresim.obs.local.v1"],
  "action_schemas": ["aresim.action.rover.v1"],
  "reward_profiles": ["aresim.reward.shaped_train.v1"],
  "task_ids": ["phase1_open_exploration_v1"],
  "scenario_ids": ["phase1_default_v1"],
  "policy_sources": ["aresim.agent.random_valid.v1", "aresim.agent.scripted.v1"],
  "compression": "gzip",
  "episode_count": 10000,
  "transition_count": 4100000,
  "shards": [{"path": "episodes-00000.jsonl.gz", "sha256": "<hash>"}]
}
```

The implemented schema and reader/writer API are documented in [Record trajectories](usage.md#record-trajectories). HDF5/Minari and Parquet exports remain optional later optimizations rather than the canonical first storage format.

Per-episode metadata should include scenario ID, seed, policy/model ID, return under every logged reward profile, success, terminal reason, wall-clock duration, and optional human tags. LLM episodes additionally record model/provider identifier, prompt template hash, tool calls, token usage, latency, retries, and cost when available.

For a multi-rover episode, keep one shared tick table plus one agent-transition table keyed by `(episode_id, tick, agent_id)`. The shared table stores the before/after checksum, joint conflict outcomes, team reward terms, world termination/truncation, and active-agent set. Each agent row stores its observation/mask, submitted action, effective action, individual reward terms, end flags, policy/module ID, and next observation/mask. A shared team reward is stored once in the shared table rather than duplicated and later summed across rover rows. The manifest additionally records `possible_agents`, agent count, policy mapping, parameter-sharing mode, and centralized-critic state schema when used.

The `T+1/T` observation/action invariant applies independently to every continuously active agent. If future scenarios can add or remove rovers, record an explicit presence interval and never fill a missing rover with a fabricated Wait transition.

## 10. Algorithms to implement and test

### 10.1 Baselines before learning

Implement these first; they catch environment bugs and define a performance floor:

1. **Uniform random:** samples all 10 actions, useful only for mask and invalid-action tests.
2. **Random valid:** samples only `action_mask == 1`; minimum meaningful baseline.
3. **No-op/Wait:** exposes passive survival and reward leakage.
4. **Greedy scripted mission agent:** shortest-path movement plus rule-based scan, extract, return, build, service, and wait.
5. **Oracle planner:** has full state and uses A*/task search; used as an approximate upper bound, not a fair partial-observation competitor.
6. **Tiny-grid tabular Q-learning:** only on `8 x 8` reduced tasks; useful for verifying Markov observations and reward learnability.

Every learned policy must beat random-valid and be compared with the scripted baseline.

### 10.2 Recommended algorithm sequence

| Priority | Algorithm | Input/action profile | What it tests |
|---:|---|---|---|
| P0 | RLlib PPO with action-masking RLModule | Local `8 x 8` + `Discrete(10)` | Scalable reference baseline and distributed collection path |
| P0 | RLlib PPO with masked categorical policy | Same local input, action space, and seed manifests | Explicit collector/advantage/loss loop and cross-adapter parity |
| P1 | Mask-aware RLlib DQN | Local `8 x 8` + `Discrete(10)` | Off-policy sample efficiency and correct masking in selection/targets |
| P1 | Recurrent RLlib policy | Local `8 x 8` + mask metadata | Explicit memory and sequence handling under partial observability |
| P1 | PPO partial-map | Discovered global map + mask | Exploration without recurrent-memory dependence |
| P1 | Behavior cloning | Scripted/human trajectories | Demonstration pipeline and policy warm-start |
| P2 | QR-DQN or Rainbow-style DQN | Local/discovered map | Distributional value learning and stronger DQN comparison |
| P2 | JEPA representation learning | Transition windows from local observations | Reusable latent encoder for RL and model-based experiments |
| P2 | DreamerV3-style world model | Compact symbolic input, fixed discrete action | Model-based data efficiency after core baselines are stable |
| P2 | IQL or CQL | Fixed offline dataset | Learning from accumulated scripted/human/LLM runs |
| P3 | IPPO and MAPPO | PettingZoo actors + global critic for MAPPO | Cooperative multi-rover control |
| P3 | QMIX | Cooperative discrete rover actions | Value decomposition and credit assignment |
| P3 | Hierarchical PPO/options | LLM or learned manager + low-level executor | Long-horizon colony tasks |

### 10.3 What not to prioritize

- **SAC:** the current AresSim macro action space is discrete. Standard SAC is designed for continuous `Box` actions and is not the right first comparison. Revisit it only if AresSim gains continuous throttle, steering, arm, or power-allocation controls.
- **Pixel-only PPO/DQN:** expensive and harder to debug while symbolic state is already available.
- **Multi-agent algorithms before single-agent correctness:** they multiply credit-assignment and synchronization problems.
- **World models before deterministic baselines:** model error makes basic simulator/reward bugs harder to isolate.
- **LLM-only per-tick control as the default:** it is slow, costly, nondeterministic, and weak at long repetitive navigation.

RLlib supports custom RLModules and learner extensions. PPO masks logits during exploration, inference, and training. Future DQN masks selection and target values; recurrence applies masks per sequence step and resets memory at episode boundaries. Manifests pin exact installed versions.

Folder names follow algorithm meaning. Later `dqn/` and `recurrent_ppo/` implementations use RLlib extensions. JEPA, world models, reusable encoders, and hybrids use PyTorch components hosted by RLlib without replacing authoritative state or reward logic.

### 10.4 Model architecture for the hybrid observation

Recommended first encoder:

```text
terrain embedding + continuous spatial channels + cell flags
                 |
             small CNN
                 |
        spatial representation --------+
                                       |
self/colony/objectives -> MLP ----------+--> fusion MLP --> policy/value heads
pad proximity embedding ---------------+
```

Use small category embeddings, two to four compact residual CNN blocks, and a small MLP for non-spatial values. The `8 x 8` crop needs a compact encoder, not an image-scale vision backbone or entity-attention stack.

The first policy head has only 10 logits. A future targeted mechanic may add a separate versioned pointer head, but the current model does not need one. For recurrent PPO, apply the recurrent layer after feature fusion and before policy/value heads.

### 10.5 Initial experiment matrix

Run these in order:

1. Local-8 shaped training: random-valid, scripted, and action-masked PPO in both RLlib.
2. Evaluate every checkpoint with both shaped and sparse profiles without further learning.
3. Local-8 sparse-only training to quantify dependence on shaping.
4. Discovered-map PPO versus memoryless local-8 PPO.
5. RLlib recurrent policy versus feed-forward local PPO.
6. Behavior-cloning warm-start versus training from scratch.
7. Scenario randomization and held-out seed/generalization evaluation.
8. Only then add RLlib-first DQN, world-model, and offline algorithms.
9. After multi-rover mechanics exist: IPPO, MAPPO, and QMIX on matched scenarios.

## 11. Curriculum and scenario design

Long integrated missions should be decomposed into task families while retaining one common state/action contract.

| Stage | Task | Main concept |
|---:|---|---|
| C0 | Move to a visible target/build pad | Movement and mask correctness |
| C1 | Avoid crater/ridge and minimize energy | Terrain costs and planning |
| C2 | Scan one rock site | Targeted information action |
| C3 | Extract ice and return/deliver | Multi-step resource loop |
| C4 | Build habitat progress | Resource prerequisites and construction |
| C5 | Detect and service build pad | Maintenance and recovery |
| C6 | Integrated survival mission | Long-horizon sequencing |
| C7 | Fog-of-war and weather randomization | Partial observability and robustness |
| C8 | Multiple rovers and shared resources | Coordination and credit assignment |

Curriculum promotion should be metric-based—for example, at least 80% success over 100 validation episodes—not based on training return alone. Always retain a mixture of earlier tasks to reduce forgetting.

Scenario randomization should be declared rather than hidden in generator code:

```yaml
schema_version: aresim.scenario.v1
scenario_family: phase1_train_v1
map: {width: 32, height: 32}
rover: {battery_capacity: 100, health_capacity: 100, payload_capacity_kg: 12}
randomization:
  terrain_seed: {distribution: uniform_int, low: 0, high: 2147483647}
  dust_intensity: {distribution: uniform, low: 0.0, high: 0.6}
  initial_battery: {distribution: uniform, low: 0.8, high: 1.0}
```

Keep the external cutoff in the experiment/adapter configuration instead:

```yaml
environment_wrapper:
  max_episode_steps: 1200
  limit_result: truncated
```

Maintain disjoint train, validation, and test scenario manifests. A seed appearing in test must never be used for hyperparameter selection. Use the same explicit evaluation cutoff when comparing algorithms, while reporting that the cutoff is not a gameplay deadline.

## 12. Evaluation protocol

This section defines the scientific comparison rules. Canonical W&B metric names, behavioral diagnostics, checkpoint interpretation, and reproducibility artifacts are specified in [RL Algorithms, Training, and Evaluation](rl_quickstart.md).

### 12.1 Required comparisons

For every algorithm/configuration:

- Train with at least five learner seeds for research-quality comparisons; three is acceptable during development.
- Evaluate deterministically where the policy supports it.
- Use at least 100 episodes per scenario split for final Phase 1 reports.
- Report mean, median, standard deviation, and bootstrap 95% confidence intervals.
- Report environment transitions and wall-clock time to threshold, not only final return.
- Use the same test scenario list and sparse evaluation reward for all policies.
- Preserve checkpoints and manifests for the best validation policy, not the best test outcome.

### 12.2 Core metrics

| Category | Metrics |
|---|---|
| Task | Success rate, objective completion rate, completion step |
| Safety | Failure rate, minimum battery/health/livability, hazard damage |
| Efficiency | Energy per success, distance, idle steps, wall-clock and sample efficiency |
| Control quality | Invalid-action rate, mask violations, action entropy, repeated-action loops |
| Generalization | Seen seeds, held-out seeds, altered weather, altered resource layout |
| Reward health | Sparse return, shaped return, each cumulative reward term |
| System | Environment steps/s, inference latency, observation bytes, memory use |
| Multi-agent | Team and per-agent return, contribution, conflicts, idle/utilization, workload fairness, policy mapping |
| LLM/hybrid | Tool calls, tokens, latency, retries, cost, planner interventions |

### 12.3 Required ablations

- With versus without action masks.
- Full grid versus partial map.
- Shaped versus sparse reward.
- Spatial CNN versus flattened MLP.
- With versus without the explicit `pad_proximity` category (the crop and action mask remain unchanged).
- Feed-forward versus recurrent policy under local observations.
- RL-only versus LLM-only versus LLM planner + RL executor on long tasks.

## 13. How to use LLMs

### 13.1 Recommended role: strategic planner

The best initial hybrid is:

```text
LLM mission planner (every 10–50 steps or on important event)
       |
       v
structured subgoal: navigate / scan / collect / build / service / conserve
       |
       v
RL or deterministic executor (every environment step)
       |
       v
validated canonical actions
```

This uses the LLM for semantic decomposition and the RL policy for fast spatial control. Replan on subgoal completion, failure, major weather change, low-resource alert, or a fixed horizon—not on every cell movement.

Recommended LLM roles, in order:

1. **Planner:** choose and sequence subgoals from mission state.
2. **Direct macro-agent benchmark:** choose one validated high-level command at a low frequency for comparison.
3. **Failure analyst:** summarize trajectories and classify failure modes offline.
4. **Scenario/curriculum assistant:** propose scenarios that are then schema-validated and human-reviewed.
5. **Demonstration generator:** produce candidate plans executed and filtered by the simulator.
6. **Multi-agent coordinator:** assign rover roles while low-level rover policies execute locally.

Do not initially use an LLM as the numeric physics model or as the authoritative reward function.

### 13.2 Compact LLM context: `aresim.llm.ops.v1`

LLMs should not receive the full `32 x 32 x channels` JSON unless they explicitly request a region. Send a compact operational state:

```json
{
  "schema_version": "aresim.llm.ops.v1",
  "session_id": "sim-123",
  "step": 71,
  "goal": {
    "task_id": "phase1_integrated_survival_v1",
    "required": [
      {"type": "deliver_ice", "current": 1.0, "target": 2.0},
      {"type": "build", "current": 0.4, "target": 1.0}
    ]
  },
  "self": {
    "rover_id": 1,
    "position": [11, 9],
    "battery_pct": 64.2,
    "health_pct": 100.0,
    "cargo": {"ice_kg": 2.0, "ore_kg": 0.0, "samples_kg": 0.5, "used_kg": 2.5, "capacity_kg": 12.0}
  },
  "colony": {
    "power_margin": 12.0,
    "water_pct": 72.0,
    "oxygen_pct": 81.0,
    "livability_pct": 88.0,
    "service_needed": false
  },
  "weather": {"type": "dusty", "severity": 0.31},
  "known_targets": [
    {"kind": "ice", "position": [13, 9], "signal": 0.82, "distance": 2},
    {"kind": "build_pad", "position": [6, 6], "distance": 8}
  ],
  "alerts": ["dust front expected"],
  "recent_events": [
    {"step": 70, "type": "move_succeeded", "summary": "Moved east"}
  ],
  "active_subgoal": null,
  "available_tools": ["set_subgoal", "inspect_region", "get_status", "wait"]
}
```

Keep numerical ground truth machine-readable and avoid verbose prose. Summarize old events into memory instead of repeatedly resending the full history.

### 13.3 LLM tool interface

Expose strict, validated tools rather than asking the model to emit arbitrary simulator JSON.

Read tools:

```text
get_status()
list_known_targets(kind?, max_results?)
inspect_region(center_x, center_y, radius)
estimate_route(target_x, target_y, objective="energy"|"distance"|"safety")
get_recent_events(since_step, max_results)
```

Control tools:

```text
set_subgoal(type, target?, quantity?, horizon_steps?)
cancel_subgoal(reason)
wait(steps=1)
```

For a direct macro-agent benchmark, additionally expose the seven canonical verbs, with Move requiring a cardinal direction and all work verbs using current-position semantics. Every tool call is converted into `aresim.action.command.v1` or a typed manager subgoal and passes deterministic validation.

Example planner result:

```json
{
  "decision_id": "plan-71-01",
  "issued_at_step": 71,
  "tool": "set_subgoal",
  "arguments": {
    "type": "extract_and_deliver_ice",
    "target": {"x": 13, "y": 9},
    "quantity": 1.0,
    "horizon_steps": 40
  },
  "confidence": 0.82
}
```

Free-form rationale may be logged for analysis but must never be parsed as an executable command.

### 13.4 LLM memory

Maintain three distinct stores:

- **Working memory:** last few events, current alerts, current subgoal, and immediate tool results.
- **Episodic memory:** summaries of prior attempts, failures, and successful plans for the current run.
- **Semantic memory:** stable environment rules and tool documentation, versioned with the engine.

Memory is advisory. It must not override canonical state. Any recalled location or resource claim is revalidated through read tools before execution.

### 13.5 Reliability and reproducibility

Every LLM call must have:

- Model/provider and model-version identifier.
- Prompt template and tool-schema hashes.
- Temperature and sampling settings.
- Exact input context or its content hash.
- Tool call/result, latency, token usage, retries, and timeout.
- `issued_at_step` and an idempotency/decision ID.
- A deterministic fallback when the model times out or emits invalid output.

Recommended fallbacks are: continue the current valid subgoal, invoke the scripted safety controller, or Wait. Reject stale decisions if the world has advanced beyond their declared step unless the command is explicitly revalidated.

For evaluation, cap calls, tokens, and wall-clock latency per episode. Compare success both with and without charging LLM cost into a reported utility metric; do not silently mix monetary cost into environment reward.

### 13.6 LLM evaluation matrix

Compare at least:

| Agent | Planning frequency | Executor |
|---|---:|---|
| Scripted baseline | Event-driven | Scripted |
| RL-only | None | PPO/DQN |
| LLM direct macro | Every step or macro step | LLM |
| LLM + scripted | Every 10–50 steps/events | Scripted controller |
| LLM + RL | Every 10–50 steps/events | PPO/recurrent PPO |

Report task success, environment steps, invalid outputs, replans, tokens, latency, cost, and robustness to held-out scenarios. The LLM evaluator should not see hidden state or reference solutions unavailable to the acting agent.

## 14. Suggested engine module boundaries

The implemented UI and RL-ready slice currently contains:

```text
engine/aresim/
  types.py
  config.py
  defaults.py
  registry.py
  factory.py
  core/                    # engine/checksum, generation, rules
  components/              # local observation, actions/mask, rewards, task
  algorithms/              # public Agent contract, baseline policies, and learned PPO
  envs/                    # composition, time limit, Gymnasium, PettingZoo
  training/                # rollouts, trajectories, RLlib training, evaluation, reports
  gameplay.py
  service.py
  integrations/ui.py
  api.py
```

Later phases expand this structure in place instead of replacing the working UI/RL backend. New algorithms remain modules within the training domain until their design genuinely requires another package:

```text
engine/
  pyproject.toml
  README.md
  aresim/
    __init__.py
    types.py
    config.py
    defaults.py
    registry.py
    factory.py
    gameplay.py
    service.py
    api.py
    core/                    # state, engine/checksum, rules, events, generation
    components/              # scenarios, observations, actions, rewards, tasks
    envs/                    # composed environment, Gymnasium, PettingZoo Parallel, wrappers
    algorithms/              # Agent protocol, baseline policies, and learned training
    training/                # rollouts, trajectories, experiments, RLlib modules, W&B, evaluation
    configs/                 # checked-in experiment YAML per algorithm
    results/                 # training run artifacts (generated)
    integrations/            # UI and future multi-agent framework adapters
  tests/                     # core, components, envs, algorithms, integration
```

This is an explicit programmatic extension layout rather than an automatic plugin ecosystem. `algorithms/base.py` owns the implemented direct-agent/evaluation protocol, and `registry.py` owns built-in component and agent registration. Add algorithm/framework contracts only when their first real implementations arrive.

`config.py` defines typed dataclasses for engine and experiment configuration. YAML is only an input format: load it into those dataclasses, reject unknown keys, resolve registry names once, and save the fully resolved configuration in every run artifact.

The core accepts canonical commands and emits immutable transitions/events. `AresEnvironment` composes registered observation, action, reward, and task components. PettingZoo, Ray/RLlib, PyTorch, JEPA/world-model code, and LLM integrations stay outside the core.

Public construction remains small:

```python
engine = make_engine(config)
env = make_env(config)
parallel_env = make_parallel_env(config)
gym_env = make_gym_env(config)
rllib_config = make_rllib_config("ppo", config)
rllib_env = make_rllib_env(config)
run_experiment(framework="rllib", algorithm="ppo", config=config)
run_experiment(framework="rllib", algorithm="ppo", config=config)
```

Keep runtime dependencies optional: `aresim[env]` installs NumPy, Gymnasium, and PettingZoo; `aresim[rllib]` adds Ray/RLlib, PyTorch, tracking, plotting, and reporting. The base UI/backend installation contains no training dependency.

Fork authors add a component or algorithm module and one explicit local registry entry. Registration rejects duplicates; configuration rejects unknown names and fields. No entry points or automatic plugin discovery are used.

Source filenames remain semantic (`observations.py`, `actions.py`, `rewards.py`) rather than embedding `8`, `10`, or `v1`. Dimensions belong in configuration and declared spaces. Compatibility remains explicit in artifacts through semantic component name, integer revision, and signature hash; a second source module is required only when incompatible revisions must coexist.

```yaml
observation: local
observation_config:
  window_size: 8
action: discrete
action_config:
  action_count: 10
reward: frontend_mock
task: open_exploration
```

The TypeScript UI contract remains in `web/`. `integrations/ui.py` converts engine snapshots/events into existing `SimSnapshot` and gameplay shapes; the UI never imports learner tensors.

## 15. Implementation sequence

### Phase A — UI-playable deterministic engine — implemented

- Implemented enums/dataclasses, canonical state, commands, history/events, and `EngineTransition`.
- Ported Phase 1 generation and all seven UI actions behind deterministic `reset/step`.
- Added grouped frozen configuration plus one immutable `DEFAULT_ENGINE_CONFIG`; simulator constants are not embedded in rule functions.
- Added stable checksums, UI snapshot conversion, gameplay delta/checkpoint export, legacy normalization, and reconstruction without re-running rules.
- Added a serialized one-live-session/one-replay service and local FastAPI routes used by the React UI.
- Kept Algorithm action selection temporarily in the UI while Python remains the validator and transition authority.

Exit criterion achieved: identical seeds produce identical state/snapshots/checksums, rule and replay tests pass, and the UI no longer imports the TypeScript simulator in production.

### Phase B — Local-observation RL environment — core environment implemented

- Implemented `aresim.obs.local.v1` exactly as specified, including `[3,3]` anchor and unknown world-edge padding.
- Implemented the 10-action rover mapping and leakage-safe mask.
- Added one-rover `AresParallelEnv`, `AresGymEnv`, PettingZoo Parallel/Gymnasium API tests, and exact direct-adapter parity.
- Added random, random-valid, Wait, and scripted baselines.
- Added the RLlib masked-PPO smoke path, fixed evaluation splits, checkpoint agents, and validated trajectory datasets.

Exit criterion: no mask/validator disagreements across exhaustive small-grid tests; scripted policy completes every solvable tutorial scenario.

### Phase C — PPO parity and analytics

- Scale collection through measured RLlib EnvRunner actors and verify that worker order does not change per-seed results.
- Measure EnvRunner/vector-environment scaling and verify collection boundaries and seed independence.
- Implement hybrid CNN/table encoder.
- Train RLlib PPO with a tested action-masking RLModule.
- Establish canonical W&B metric names and W&B-backed Jupyter evaluation reports.
- Tune reward weights only on train/validation; freeze sparse evaluation.

Exit criterion: direct/Gymnasium/PettingZoo transitions agree; live/restored module inference agrees; evaluation is reproduced from manifests and native checkpoints. Open exploration has no reward-improvement gate.

### Phase D — RLlib research algorithms, memory, and representation learning

- Add the optional discovered partial-map profile derived only from observation history.
- Add mask-aware RLlib DQN, covering both action selection and next-state target masks.
- Add a recurrent RLlib policy with explicit mask, sequence, and state-reset tests.
- Add RLlib/PyTorch JEPA training over transition windows and save a reusable encoder artifact.
- Add behavior cloning and one offline RL baseline.
- Export selected UI/manual/scripted runs into trajectory shards.

### Phase E — World models, LLMs, and hybrid agents

- Add a RLlib/PyTorch world model and planner without allowing predictions to mutate canonical state.
- Implement compact LLM context, read tools, manager subgoals, audit records, budgets, and fallbacks.
- Compare RL-only, JEPA-encoded RL, world-model-assisted RL, LLM direct, LLM + scripted, and LLM + RL.
- Use offline trajectory analysis before attempting online self-reflection loops.

### Phase F — Multi-rover training

- Introduce multiple rover state and deterministic simultaneous conflict rules behind the existing PettingZoo Parallel API.
- Use RLlib policy mapping for future shared-policy IPPO, role-specific, and centralized-critic/MAPPO-style experiments.
- Benchmark IPPO, MAPPO, and QMIX only after the multi-rover transition contract is stable.
- Add team, per-agent, coordination, utilization, fairness, and rover-count scaling reports.
- Add LLM commander as an optional high-level role allocator.

Ray Tune owns RLlib trial lifecycle. JAX and remote training/inference services remain optional future integrations.

## 16. Validation and performance gates

### 16.1 Contract tests

- Observation arrays always match declared dtype, shape, bounds, and masks.
- `AresGymEnv` passes Gymnasium's checker, rejects non-single-rover scenarios, and matches one-rover PettingZoo transitions exactly.
- Unknown cells never contain hidden terrain/resource values.
- Padded objective rows are zero and masked.
- Stable category IDs match the schema registry.
- Swappable observation, action, reward, and task selections preserve canonical transitions.
- Duplicate registry names, unknown component names, incompatible revisions, and unknown configuration keys fail clearly.
- Importing `aresim.core` does not import or require environment/training/tracking/LLM dependencies.
- Every flat action decodes to exactly one canonical command.
- Masked actions pass validation; known-invalid actions are masked when legality is observable.
- At least one action is always legal.
- Reward total equals the sum of weighted terms after documented clipping.
- One-time events cannot pay repeatedly.
- `terminated` and `truncated` are never ambiguous.
- Trajectory array lengths satisfy the `T+1/T` invariant.
- Online collection runs with persistent recording disabled and performs no UI rendering or gameplay JSON serialization.
- RLlib EnvRunners use independent deterministic seeds, reset independently, and preserve episode and agent identifiers.
- RLlib EnvRunners use independent deterministic seeds and preserve episode/agent identities.
- PPO sampled-step counts agree with each framework's resolved configuration; DQN entries contain both current and next masks; recurrent and representation-learning windows never cross episode boundaries.
- Saved manifests reconstruct component names, revisions, signature hashes, resolved configuration, and seed sets.
- RLlib PPO completes a focused one-rover smoke test and masks logits in exploration, inference, and training; future DQN masks selection and targets.
- Recurrent PPO tests explicitly record whether masks are supported.
- JEPA windows/encoder artifacts and world-model inputs/outputs match declared signatures.
- Fake-provider LLM and hybrid tests cover malformed output, timeout, staleness, fallback, and canonical validation.

### 16.2 Determinism tests

- Same seed + commands yields identical checksums across repeated runs.
- Replay from a checkpoint yields the same final checksum as uninterrupted execution.
- Vectorized environment order does not affect individual results.
- Captured parity fixtures and canonical documentation agree with Python transitions, named reward terms, and totals; documented values win over old mock differences.
- In the future multi-rover extension, dictionary order does not affect conflict resolution.

### 16.3 Leakage tests

- Presentation state never changes observations or dynamics.
- Partial observations contain no unseen resources, weather future, internal Build Pad component locations, or hidden terminal data.
- Action masks do not reveal unobserved hazards.
- Evaluation agents cannot query privileged critic state or oracle tools.
- LLM context and read tools apply the same visibility rules as the corresponding policy profile.

### 16.4 Initial engineering budgets

Treat these as benchmark gates to measure and revise, not correctness requirements:

- Raw local-8 observation under 16 KiB per rover before batching.
- No JSON serialization in the training step loop.
- Establish a measured EnvRunner scaling curve on a development machine rather than a fixed unverified worker-count target.
- Report one-EnvRunner and scaled EnvRunner steps/second in CI performance runs.
- Keep reward/action/observation builders deterministic and free of model-framework dependencies.
- LLM calls never block the simulator indefinitely; every call has a timeout and fallback.

## 17. Explicit choices and alternatives

| Decision | Recommended first choice | Alternative and when to use it |
|---|---|---|
| Extension model | Fork-local explicit registry | Published plugin entry points only if AresSim later becomes a plugin ecosystem |
| Current UI API | Direct `AresEngine` plus local FastAPI REST | Add WebSocket only for a demonstrated remote/live-streaming need |
| Environment APIs | PettingZoo Parallel is canonical for multi-agent growth; `AresGymEnv` is supported for exactly one rover | Never extend the Gymnasium adapter by silently selecting or flattening multiple agents |
| Learning framework | RLlib for reference runs and explicit research extensions | Add another framework only with demonstrated need |
| Actor visibility | Fixed local `8 x 8` crop | Discovered-map memory derived from prior local crops |
| Spatial storage | Mixed typed arrays | Packed float tensor only inside model preprocessing |
| Variable entities | None in the Phase 1 actor input | Add versioned padded tables or graph/set encoders only when multi-rover or independent entities exist |
| First RL action | Fixed masked `Discrete(10)` with current-cell operations | Add a versioned targeted adapter only when ranged/entity mechanics exist |
| First learned agent | Masked PPO in RLlib | Mask-aware RLlib DQN as a later off-policy comparison |
| Partial-observation agent | Recurrent PPO | Transformer/state-space memory after baseline |
| Continuous algorithm | None initially | SAC only after continuous controls exist |
| Multi-agent API | PettingZoo Parallel from the first one-rover adapter | Multiple rover mechanics remain deferred, not the API |
| LLM control | Low-frequency planner | Direct macro-agent as benchmark, not default |
| Training data | Implemented JSONL/gzip trajectory shards | HDF5/Minari or RLDS only if later workloads justify conversion |
| Analytics | Parquet | JSONL only for small debug logs |

## 18. Final recommendation

The next end-to-end RL slice should reuse the implemented deterministic integrated-survival core and add:

- `aresim.state.v1`
- `aresim.obs.local.v1`
- `aresim.action.rover.v1`
- `aresim.reward.shaped_train.v1` for learning
- `aresim.reward.sparse_eval.v1` for comparison
- Gymnasium single-rover + PettingZoo Parallel + the RLlib adapter + random-valid + scripted + action-masked PPO in RLlib
- `aresim.trajectory.v1` output on every evaluation run
- canonical W&B metrics and a reproducible Jupyter evaluation report

Once this slice is reproducible, add mask-aware DQN, discovered-map memory, recurrent control, and behavior cloning. Add the LLM planner after subgoals and visibility-safe tools are deterministic. Add multiple rovers and MARL only when simultaneous conflict rules exist.

This order gives AresSim a stable scientific core: algorithms can change, local observations can gain explicit memory, and LLMs can be added without allowing any of them to redefine the world, rewards, or replay semantics.

## 19. References

- [AresSim Agent and Environment Representation Survey](../../design_docs/AresSim_Agent_Environment_Representation_Survey.md)
- [AresSim Environment Rules Reference](../product/environment_rules.md)
- [AresSim Gameplay Save Format](../product/gameplay_save_format.md)
- [AresSim RL Algorithms, Training, and Evaluation](rl_quickstart.md)
- [Gymnasium environment API](https://gymnasium.farama.org/api/env/)
- [Gymnasium environment checker](https://gymnasium.farama.org/api/utils/#gymnasium.utils.env_checker.check_env)
- [Gymnasium vector environment tutorial](https://gymnasium.farama.org/main/tutorials/vector_envs_tutorial/)
- [PettingZoo environment creation and Parallel API](https://pettingzoo.farama.org/content/environment_creation/)
- [RLlib environments](https://docs.ray.io/en/latest/rllib/rllib-env.html)
- [RLlib RLModules](https://docs.ray.io/en/latest/rllib/rl-modules.html)
- [RLlib MetricsLogger](https://docs.ray.io/en/latest/rllib/metrics-logger.html)
- [Ray Tune result analysis](https://docs.ray.io/en/latest/tune/api/result_grid.html)
- [RLlib multi-agent environments](https://docs.ray.io/en/latest/rllib/multi-agent-envs.html)
- [RLlib documentation](https://docs.ray.io/en/latest/rllib/index.html)
- [RLlib multi-agent environments](https://docs.ray.io/en/latest/rllib/multi-agent-envs.html)
- [RLlib logging and metrics](https://docs.ray.io/en/latest/rllib/metrics-logger.html)
- [Minari dataset standard](https://minari.farama.org/)
- [Minari dataset collection](https://minari.farama.org/content/basic_usage/)
- [RLDS dataset format](https://github.com/google-research/rlds)
- [DreamerV3 reference implementation](https://github.com/danijar/dreamerv3)
