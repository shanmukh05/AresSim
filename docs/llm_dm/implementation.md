# Run llm_dm

Updated: 2026-11-09.

`llm_dm` combines an Actor, episode Memory, and an LLM Learner. It replaces the separate `llm` and `decision_models` algorithms.

Use [algorithm details](../../engine/aresim/algorithms/llm_dm/DETAILS.md) for the full contracts, source map, and seven editable workflow diagrams.

## Workflow

1. Read the permitted observation and legal-action mask.
2. Select a goal when a new goal or a review is needed. The Actor selects an observed target. This call does not advance the game.
3. Supply focused Memory to the Actor. Include the current state, active route, two recent outcomes, and applicable advice.
4. Validate one primitive action. Execute one game tick.
5. Update Memory from the actual result. Update the same replay file.
6. Review the completed game with the Learner when the schedule requires a review.

The Actor can be TypeSafe Jev or an LLM. Both use the same inputs and Memory rules. The Learner must be an LLM. Model weights stay fixed.

## Providers and prompts

| Role | Provider setting | Output |
|---|---|---|
| Jev Actor | `decision_models` | One typed `goal` or `action` Choice |
| LLM Actor | `openai`, `deepseek`, `anthropic`, `gemini`, `openrouter` | Strict goal or action JSON |
| LLM Learner | The same LLM providers | A proposal or `no_change` |

Configure each role separately. Set its model, credential variable, generation settings, and limits. Keep credentials in local environment variables or `.env.local`. The run does not save credential values.

The action and goal prompts include the configured game rules. Use `actor.prompt_path` and `actor.goal_prompt_path` for custom templates. The bundle saves the rendered prompts.

Focused Jev requests put option descriptions in Choice criteria and supporting facts in State. The Agent validates the returned Choice. Every legal primitive action remains available.

## Memory and learning

`history.mode: focused` is the default. The runtime keeps the observed map but sends only the required working context. Routes use observed passable cells. A route does not guarantee sufficient battery.

| Setting | Default | Use |
|---|---:|---|
| `history.goal_horizon` | 8 | Review a goal after this many ticks |
| `history.stall_steps` | 6 | Review a goal after a lack of progress |
| `history.target_limit` | 6 | Limit the goal options |
| `history.advice_limit` | 3 | Limit applicable lessons in an action request |

Goal selection receives all compatible learning statements. Actions receive a matching subset. Deferred advice stays in the policy. Conditions can refer to `decision.goal_kind` and `decision.goal_steps_without_progress`.

The Learner receives a compact timeline and up to eight key transitions. Detailed records include the actual goal and action context. Advice must cite supplied evidence. Its conditions must match a cited pre-action state. One correction is allowed for invalid output.

In `sequential_diagnostic`, N games use N−1 reviews. Prior advice remains available in later games. `no_change` retains it. This mode does not promote a validated policy.

In `development`, a valid change requires fresh paired parent and candidate games before promotion. `smoke` checks this workflow without promotion. Frozen evaluation and UI attachment use a saved policy without an online Learner.

Set `learnings_file` to import a previous `learning.jsonl` or curated statements. The loader extracts advice, removes duplicates, and saves source references. Evidence, failed output, and `no_change` explanations do not become advice.

Saved legacy bundles retain their `none`, `recent`, or `facts_recent` profile and `aresim.llm.packet.v1` input. They do not acquire goal requests. Focused mode uses `aresim.llm_dm.goal_packet.v1` and `aresim.llm_dm.packet.v2`. Primitive action JSON remains `aresim.llm.action.v1`.

## Configure and run

From the repository root:

```bash
engine/.venv/bin/pip install -e './engine[llm_dm]'
engine/.venv/bin/aresim-rl rollout configs/llm_dm/fake_smoke.yaml
engine/.venv/bin/aresim-rl rollout configs/llm_dm/10-games.yaml --env-file .env.local
```

| Config | Use |
|---|---|
| `configs/llm_dm/10-games.yaml` | Jev Actor and GPT-6 Luna Learner |
| `configs/llm_dm/llm-actor.yaml` | LLM Actor and LLM Learner |
| `configs/llm_dm/fake_smoke.yaml` | Offline pipeline check |
| `configs/llm_dm/frozen.yaml` | Saved-policy evaluation or UI attachment |
| `configs/llm_dm/prices.json` | Dated rates for budget estimates |

The Jev profile reads `JEV_API_KEY`. The OpenAI profile reads `OPENAI_API_KEY`. Configure other credentials in the selected provider profile.

Set `experiment_name` to a readable name. Set `episodes` to the required games. Set `max_episode_steps` to the turn limit. The config filename does not set either limit. Repeated names receive suffixes such as `-002`.

Budgets cover both roles, goal calls, retries, failures, and comparisons. Goal and action calls share one tick deadline. Invalid goal output stops before a tick. Recoverable action failures can use the legal Wait fallback. Fatal provider errors and failed Learner reviews stop the run.

## Read the output

Outputs are in `results/llm_dm/<experiment_name>/`.

| File | Use |
|---|---|
| `summary.log` | Start here. Read outcomes, deliveries, and learning decisions. |
| `manifest.json` | Check config, prompts, usage, budgets, policy versions, and replay paths. |
| `games.jsonl` | Inspect goals, actions, actual outcomes, and advice exposure. |
| `learning.jsonl` | Inspect supplied evidence, proposals, and evaluation decisions. |
| `selected-bundle.json` | Load the selected policy with its prompts and Memory settings. |
| `replays/*.json` | Import a game into the UI. Each file updates during play. |

Frozen runs without learning records can omit `learning.jsonl`. Set `debug_trace: true` only when exact call content is needed. It adds one compressed `debug.jsonl.gz`.

For UI Algorithm mode, set backend `ARESIM_LLM_DM_CONFIG` to a compatible frozen config. Set `ARESIM_LLM_DM_BUNDLE` to its saved bundle. Reimport a running replay to see later ticks.

Use manual simulator runs for verification. Fake providers check the pipeline, not model quality. Use bounded live runs to measure performance and learning benefit.
