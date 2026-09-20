# Baseline rover policies

Inference-only `Agent` implementations used for rollouts, frozen evaluation, and UI Algorithm mode. They share the Discrete(10) action space and the same mask contract as PPO, DQN, and Jev. Simulator rules stay in `aresim.core`. None of these policies train.

For the broader RL guide, see [RL Algorithms, Training, and Evaluation](../../../../docs/rl/rl_quickstart.md).

**Source ownership**

| File | Registry name | `policy_id` |
|---|---|---|
| `wait.py` | `wait` | `aresim.agent.wait.v1` |
| `random.py` | `random` | `aresim.agent.random.v1` |
| `random_valid.py` | `random_valid` | `aresim.agent.random_valid.v1` |
| `scripted.py` | `scripted` | `aresim.agent.scripted.v1` |
| `../registry.py` (package) | registers the four names above | |

Action IDs: `0` Wait, `1–4` N/E/S/W, `5` Scan, `6` Extract, `7` Build, `8` Service, `9` Unload.

```text
local obs + mask
        |
        v
  baseline Agent.act
        |
        v
  action index 0-9  ->  engine step
```

## Wait (`wait`)

Always returns action `0` after asserting `mask[0] == 1`. The episode seed is accepted for the `Agent` API and ignored. Use it to measure passive resource decay, survival, reward time cost, and external truncation with no navigation.

## Uniform random (`random`)

Samples all ten action IDs uniformly. It checks mask *shape* but ignores legality bits, so it can pick illegal actions. Use it to stress invalid-action handling and to quantify how much legality alone improves returns.

## Random-valid (`random_valid`)

Samples uniformly among indices whose mask entry is `1`. Observation contents are ignored (`observation_schema` is `None`). Raises if the mask is empty. This is the untrained legal-action comparison for masked PPO/DQN: it never knowingly violates the mask and has no strategy.

## Scripted (`scripted`)

A deterministic survival/resource heuristic from `aresim.obs.local.v1` plus the mask. It never reads canonical `WorldState`. Private memory keeps the spawn-pad location, an exploration heading, and the last move (to avoid immediate backtracking). The `reset` seed is ignored; given the same observations the policy is deterministic.

Priority, high to low:

1. Unload carried cargo when unload is legal (`mask[9]` and payload used).
2. Service when the colony needs service and service is legal.
3. Build when habitat progress is incomplete and build is legal.
4. Wait on the pad when battery is below 80%.
5. Return toward the remembered pad when battery is below 35%, payload is at least 75% used, or service is needed.
6. Extract ice on the current cell, else scan rock on the current cell.
7. Move toward the nearest visible ice or unscanned rock in the 8x8 crop.
8. Explore in heading order (straight, then turn, then reverse), skipping an immediate backtrack when another move is legal.
9. Wait as the last fallback.

It is a fair partial-observation heuristic, not an oracle and not a learned policy.

## Run

```python
from aresim.training import EpisodeSpec, RolloutConfig, RolloutRunner

plan = RolloutConfig(
    episodes=(EpisodeSpec("random-valid-000", environment_seed=1447, agent_seed=9001),),
    max_episode_steps=1200,
)
result = RolloutRunner(plan, "random_valid").run()
```

In the UI, Algorithm mode attaches the same registry names through `POST /api/sessions/{id}/attach-policy`. Requires `aresim[env]`. Frozen evaluation after PPO/DQN training also runs `random_valid` and `scripted` on the validation split.
