# Jev rover agent workflow

Jev is TypeSafe's System One model. AresSim uses it as an inference-only rover policy: one Choice per environment step over already-legal Discrete(10) actions. It is not trained and does not replace masked PPO/DQN. Simulator rules stay in `aresim.core`. Jev is not an RLlib trainable.

See [TypeSafe introduction](https://docs.typesafe.ai/introduction). For the broader RL guide, see [RL Algorithms, Training, and Evaluation](../../../../docs/rl/rl_quickstart.md).

**Source ownership**

| File | Responsibility |
|---|---|
| `config.py` | `JevAgentConfig` |
| `state.py` | `aresim.llm.ops.v1` 5x5 encoder |
| `questions.py` | Masked Choice question |
| `client.py` | TypeSafe client, fake client, `JEV_API_KEY` |
| `agent.py` | `JevAgent` implementing `Agent` |
| `rollout.py` | `aresim.rollout.v1` YAML for `aresim-rl rollout` |
| `../../registry.py` | Registers agent name `jev` |

## Why the state looks like this

Jev accepts a JSON object, not images. It is weak at unnamed tensors, counting, and arithmetic. The encoder therefore:

- Crops the local `8x8` observation to a rover-centered **5x5**
- Names every cell with exact `dx`, `dy` from the rover (`(0,0)` is under the rover; negative `dy` is north)
- Converts ice, ore, dust, height, and roughness to 0–100 percents
- Lists only legal actions in the Choice criteria

The leftover outer ring of the 8x8 is omitted. A Move cannot reach those cells this tick. `prospects.best_ice` / `best_ore` name the richest in-bounds deposit so Jev does not have to scan all 25 cells; `first_move` is the adjacent step toward that cell. If extract or scan is legal and the bay has room, the agent collects this tick instead of walking toward a richer neighbor. Otherwise, if Jev waits or builds while a deposit is visible, the agent takes that first step.

## Install and keys

```bash
engine/.venv/bin/pip install -e './engine[dev,env,jev]'
```

Set `JEV_API_KEY` (never commit the value). Repo-root `.env.local` is loaded if the process environment is unset. Copy [`.env.example`](../../../../.env.example). Never log the key.

## Run

Fake-client smoke (no network):

```bash
engine/.venv/bin/aresim-rl rollout configs/jev/smoke.yaml
```

Live short episodes (requires `JEV_API_KEY` and `aresim[jev]`):

```bash
engine/.venv/bin/aresim-rl rollout configs/jev/dev.yaml
```

In the UI, Algorithm mode → **Jev**. Autoplay waits on each `/agent-step` for the TypeSafe round trip, then the existing 700 ms interval. The option is disabled when `capabilities.jev` is false.

## Contract

- Registry name: `jev`
- `policy_id`: `aresim.agent.jev.v1`
- Observation: `aresim.obs.local.v1` only (no canonical `WorldState`)
- Action: `aresim.action.rover.v1`
- Fallback: Wait only when the TypeSafe call cannot run (timeout, transport, exhausted `max_calls`). `max_calls: 0` is unlimited (the UI default). Low confidence, illegal names, Wait with non-positive colony `power_margin`, and repeating recent actions use Choice probabilities instead of idling. With probability `explore_second_prob` (default 0.3) that pick is the **second**-highest legal mass, not the top. If extract or scan is legal and `free_kg > 0`, the agent collects this tick instead of walking off the cell. Unload fires when it is legal and the bay is full or the rover is already on the pad with cargo.

The engine warning `Wait could not recharge because power margin is not positive` means colony generation is not above consumption. Waiting then does not charge; it only burns time. The encoder sets `rover.wait_recharges` from `colony.power_margin`, `rover.should_unload` when the bay is full or the rover is on the pad with cargo, and `recent_repeating` from the last 10 actions.

Planner + scripted/RL executor remains the better long-horizon design; this agent is the first per-tick LLM-class baseline.

## Payload

```text
local 8x8 obs + mask
        |
        v
  5x5 named cells (dx, dy, ice/ore percents)
        |
        v
  TypeSafe system_one(state, questions)
        |
        v
  Choice name  ->  action index 0-9
```

Each tick the live client calls `system_one(state=..., questions=...)`. The state is rover-relative JSON (`aresim.llm.ops.v1`); there are no world coordinates and no canonical `WorldState`. Questions are a separate argument: one Choice named `action` whose criteria contain only mask-legal names.

## State (`aresim.llm.ops.v1`)

Built by `encode_ops_state` from `aresim.obs.local.v1`. Percents are 0–100 integers. `dx`/`dy` are rover-relative (negative `dy` is north, positive `dx` is east). Optional keys (`?`) are omitted when unknown.

```text
state
├── schema_version: "aresim.llm.ops.v1"
├── observation_schema: "aresim.obs.local.v1"
├── mission.priorities[]          # 7 fixed strings (survive, unload, service, build, collect, seek, anti-loop)
├── rover
│   ├── relative_origin           # {dx: 0, dy: 0, note}
│   ├── here                      # cell under the rover (terrain, ice_pct, ore_pct, scanned, extracted)
│   ├── battery_pct, health_pct
│   ├── cargo                     # ice_kg, ore_kg, samples_kg, free_kg, capacity_kg
│   ├── pad                       # outside_service_range | within_service_range | on_build_pad
│   ├── wait_recharges            # true iff colony.power_margin > 0
│   ├── should_unload             # true iff cargo > 0 and (on pad or free_kg <= 0)
│   └── remembered_pad?           # {dx, dy} after the pad has been in the 5x5
├── colony
│   ├── power_generated, power_consumed, power_margin
│   ├── battery_pct, water_pct, oxygen_pct, livability_pct
│   ├── dust_intensity_pct, habitat_progress_pct
│   ├── pad_needs_service
│   └── weather
├── objectives[]                  # {type, current_pct, target_pct, remaining_pct, required}
├── grid
│   ├── size: 5
│   ├── rover: {dx: 0, dy: 0}
│   └── cells[25]                 # dy, then dx, from (-2,-2) to (2,2)
├── adjacent                      # north/east/south/west: {dx, dy, legal}
├── recent_actions[]              # last 10 action names, oldest first
├── recent_repeating              # true on same-move x3 or A,B,A,B
└── prospects
    ├── best_ice?                 # richest in-bounds ice cell
    └── best_ore?                 # richest in-bounds ore cell
```

### Grid cell (`grid.cells[]`)

```json
{
  "dx": 1,
  "dy": 0,
  "steps": 1,
  "in_bounds": true,
  "terrain": "ice",
  "ice_pct": 72,
  "ore_pct": 8,
  "dust_pct": 11,
  "height_pct": 40,
  "roughness_pct": 22,
  "scanned": false,
  "extracted": false,
  "is_rover": false,
  "move_direction": "east"
}
```

Out-of-bounds cells stay `terrain: "unknown"` with percents 0. `move_direction` is only on the four adjacent cells (`north` / `east` / `south` / `west`).

### Prospect (`prospects.best_ice` / `best_ore`)

Omitted when no matching in-bounds, unextracted deposit is visible. `first_move` is the adjacent step that reduces Manhattan distance; `extract_here` is true only when that cell is `(0, 0)`.

```json
{
  "dx": 1,
  "dy": -1,
  "ice_pct": 72,
  "ore_pct": 8,
  "terrain": "ice",
  "steps": 2,
  "first_move": "move_east",
  "extract_here": false
}
```

## Questions

Built by `build_action_question`. Criteria keys are Discrete(10) names that are legal this tick: `wait`, `move_north`, `move_east`, `move_south`, `move_west`, `scan`, `extract`, `build`, `service`, `unload`. Instructions are the survival/resource rules in `questions.py`.

```json
{
  "action": {
    "type": "choice",
    "instructions": { "question": "Which one legal rover action should be taken right now?", "rules": ["..."] },
    "criteria": {
      "move_east": {
        "effect": "Move one cell east.",
        "destination": { "...cell..." },
        "toward_best_ice": true,
        "toward_best_ore": false
      },
      "extract": { "effect": "Extract ice from the cell under the rover.", "here": { "...rover cell..." } },
      "wait": { "effect": "...", "here": { "..." }, "recharges": false }
    }
  }
}
```

Move criteria cite the destination 5x5 cell and flag whether that step is `first_move` toward `best_ice` / `best_ore`. Work actions (`scan`, `extract`, `build`, `service`, `unload`) cite `here`. Wait includes `recharges` from `rover.wait_recharges`.

## Fallbacks and traces

Timeouts, missing answers, and exhausted `max_calls` still Wait. Legal extract/scan with free cargo always collect this tick. Low confidence, illegal/unknown Choice, idle Wait, and repeating `recent_actions` pick from Choice probabilities, with probability 0.3 of taking the second-highest legal mass. Legal unload fires when the bay is full or the rover is already on the pad with cargo.

Each step appends one line under `results/jev/` (not stdout). The file is named from the run start, for example `results/jev/jev-20260920T015200Z.log`:

```text
2026-09-20T01:52:00+00:00 call=12 Jev: move_east choice=wait conf=0.41 fallback=idle wait skipped wait=0.55 move_east=0.22 extract=0.10 recent=move_north,move_south,move_north,move_south
```

Set `algorithm_config.log_path` to an empty string to disable the file, or to a `.log` path to write that single file. The directory default is `results/jev`. The same Choice summary is stored on `last_info`, returned as `policyMeta.jev` from `/agent-step` (`choice`, `confidence`, `probabilities`, `action`, `fallback`, `waitRecharges`), appended to that step's history events (so Save JSON includes it), and written into rollout `events`. Do not print traces to stdout.
