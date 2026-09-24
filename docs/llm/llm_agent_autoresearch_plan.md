# LLM Agent and Autoresearch Plan

Last updated: 2026-09-22

Status: Proposed. This document defines the implementation boundary and staged plan; it does not claim that the generic LLM agent is implemented.

Related references:

- [Jev rover agent workflow](../../engine/aresim/algorithms/jev/workflow.md)
- [RL algorithms, training, and evaluation](../rl/rl_quickstart.md)
- [Agent data, RL, and LLM architecture proposal](../rl/agent_data_rl_llm_proposal.md)
- [Environment rules](../product/environment_rules.md)
- [Dream-RSI project page](https://dream-rsi.com/) and [technical report](https://dream-rsi.com/assets/dream-rsi.pdf)

## 1. Decision summary

AresSim should add one provider-neutral, inference-only `llm` agent family. OpenAI, Anthropic, Gemini, and OpenRouter are transports selected by configuration, not separate simulator algorithms. A policy experiment pins a provider, model, state view, prompt, memory strategy, decision protocol, budgets, and seed split in an immutable manifest.

The design has two deliberately separate loops:

1. **Gameplay loop:** a frozen policy observes only the versioned policy input, reads its own bounded memory, and returns one structured action. It cannot edit code, train weights, inspect canonical `WorldState`, or call arbitrary tools.
2. **Research loop:** after a batch of episodes, a researcher analyzes artifacts and proposes the next versioned policy specification. Most proposals are declarative changes. A proposal that truly needs new behavior may create an isolated Git worktree/branch, but automated edits are restricted to `engine/aresim/algorithms/llm/` and must pass review and tests before evaluation.

This is autoresearch over **inference strategies**, not online model training. The optimization target is a robust policy across held-out environment seeds, subject to safety, latency, and cost constraints.

The outer research loop should optionally adopt the core idea from Dream-RSI: preserve completed experiment exploration as a tree, replay that tree cheaply to improve the **research exploration policy**, and spend new API/simulation budget only on the most promising next research policy. This does not replace fixed-seed evaluation and does not let the rover policy rewrite itself.

The first implementation should compare four modes using the same provider interface:

- `reactive`: one model call per environment step;
- `deliberative`: bounded read-only tool calls, then one action;
- `planner_executor`: a lower-frequency LLM plan plus a deterministic executor;
- `swarm`: several specialized analysts and one coordinator, added only after the single-agent baselines are reliable.

Jev stays as an independent baseline. The generic LLM package may reuse concepts from Jev, but should not import Jev-specific state encoding, client types, prompts, or fallback heuristics.

## 2. Goals and non-goals

### Goals

- Compare actual LLM policies fairly across OpenAI, Anthropic, Gemini, and OpenRouter.
- Let a model decide how to inspect permitted state, select relevant history, maintain memory, and reason about the next action.
- Preserve authoritative action masks and simulator validation.
- Make every decision reproducible enough to audit: exact input view, tool results, output, selected action, usage, latency, fallback, and policy version.
- Discover whether providers/models converge on similar policies or develop materially different behavior.
- Support systematic autoresearch without contaminating held-out test seeds.
- Keep all LLM-agent implementation and research machinery under `engine/aresim/algorithms/llm/`, apart from small registration, API, CLI, configuration, and UI integration points.
- Leave a clean path to multi-agent deliberation without baking swarm assumptions into the first agent.

### Non-goals

- Fine-tuning or updating provider model weights during gameplay.
- Letting a gameplay agent edit files, execute shell commands, install packages, access Git, or make arbitrary network requests.
- Giving the model canonical `WorldState`, hidden map information, RNG state, future events, or UI-only data.
- Replacing deterministic action validation, reward calculation, task termination, or simulator rules with model judgment.
- Treating prompt changes chosen from test performance as valid research.
- Requiring a Git branch for every prompt or configuration trial.

## 3. Core boundary: policy freedom inside an immutable envelope

The model should have freedom over **attention and decision-making**, not over authority.

```text
Versioned local observation + authoritative action mask
                         |
                         v
             Immutable decision context
                         |
          +--------------+--------------+
          |                             |
          v                             v
  State/history tools            Bounded policy memory
   (read-only, typed)          (episode-scoped by default)
          |                             |
          +--------------+--------------+
                         v
               Provider-neutral LLM
                         |
                         v
             Structured decision object
                         |
                         v
       Schema check -> mask check -> stale-state check
                         |
                +--------+---------+
                |                  |
              valid              invalid/error
                |                  |
                v                  v
        simulator action      deterministic fallback
```

The model may choose which available state/history views to request, whether to revise a plan, and what legal action to recommend. A trusted host owns tool execution, budgets, schema validation, legality, staleness detection, retries, and fallback.

## 4. What “learning” means here

There are three different mechanisms and they must not be mixed in reports.

| Mechanism | When it changes | Persisted across episodes? | Allowed in gameplay? |
|---|---|---:|---:|
| Context adaptation | Within one decision/episode through history and memory | Normally no | Yes |
| Policy-spec iteration | Between experiment batches: prompt, state view, memory, cadence, tool policy | Yes, as a new version | No; frozen during a run |
| Model-weight training | Fine-tuning/RL of provider weights | Separate future project | No |

An LLM run is therefore an evaluation of a frozen `(policy_spec, provider, model)` tuple. If the tuple changes, it is a new trial, even if the user-facing model name is unchanged.

## 5. Proposed package ownership

All substantive LLM workflow code belongs below `engine/aresim/algorithms/llm/`:

```text
engine/aresim/algorithms/llm/
├── __init__.py
├── agent.py                  # Agent contract and decision orchestration
├── config.py                 # Strict typed policy/run configuration
├── protocol.py               # Versioned request, tool, decision schemas
├── context.py                # Immutable context assembly
├── state_views.py            # Permitted observation projections
├── history.py                # History query/filter implementations
├── memory.py                 # Episode memory backends and limits
├── guardrails.py             # Budgets, validation, staleness, fallbacks
├── trace.py                  # Redacted decision/event traces
├── rollout.py                # Fixed-seed LLM experiment runner
├── metrics.py                # Quality, behavior, cost, latency metrics
├── providers/
│   ├── base.py               # Provider protocol and normalized response
│   ├── openai.py
│   ├── anthropic.py
│   ├── gemini.py
│   └── openrouter.py
├── strategies/
│   ├── reactive.py
│   ├── deliberative.py
│   ├── planner_executor.py
│   └── swarm.py
├── research/
│   ├── spec.py               # PolicySpec and ExperimentSpec schemas
│   ├── propose.py            # Next-candidate proposal interface
│   ├── compare.py            # Paired fixed-seed comparisons
│   ├── promote.py            # Promotion/stopping rules
│   └── sandbox.py            # Optional code-experiment allowlist checks
└── workflow.md               # Implemented behavior once it exists
```

Provider SDK calls must remain in `providers/`. Strategy code consumes the provider-neutral protocol. Provider/model quirks belong in capability metadata or the relevant adapter, never in simulator code.

Small integration edits will necessarily exist outside this folder:

- `engine/aresim/registry.py`: register `llm`;
- `engine/aresim/integrations/policy.py`: resolve the LLM policy and expose generic metadata;
- `engine/aresim/api.py` and `engine/aresim/service.py`: validate policy configuration and report capabilities;
- `engine/aresim/training/cli.py`: dispatch generic LLM rollouts/research commands rather than hard-coding Jev;
- `engine/pyproject.toml`: optional provider extras;
- `web/src/`: provider/model/policy controls and decision inspection;
- `configs/llm/`: checked-in, non-secret experiment specifications;
- `results/llm/`: ignored run artifacts.

Those files integrate the feature; they must not contain prompt logic, memory algorithms, provider request construction, or research behavior.

### Model-specific code

Do not create an OpenAI-, Anthropic-, Gemini-, or OpenRouter-specific strategy folder just because a different model is selected. Use a model-specific directory only when an experiment introduces actual executable behavior that cannot be represented by `PolicySpec`. Prefer:

```text
engine/aresim/algorithms/llm/strategies/experimental/<experiment_slug>/
```

The experiment manifest must record why code was required and the Git commit. Successful general mechanisms should be promoted into a provider-neutral strategy; abandoned directories should not accumulate on the main branch.

## 6. Provider-neutral interface

### Supported providers

| Provider ID | Authentication | Adapter responsibility |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | Native structured output/tool calling, usage and finish-reason normalization |
| `anthropic` | `ANTHROPIC_API_KEY` | Tool blocks, usage, stop-reason, and error normalization |
| `gemini` | `GEMINI_API_KEY` | Function calling/schema and usage normalization |
| `openrouter` | `OPENROUTER_API_KEY` | OpenRouter endpoint, routed model ID, upstream/provider metadata when available |

API keys are read at runtime, never stored in YAML, traces, trajectory JSON, prompts, W&B, or UI responses. A provider is available only when its optional dependency and credential are present. OpenRouter is a distinct transport even when it routes to a model family also available natively; reports must retain both the requested model ID and returned routing metadata.

### Provider protocol

The common adapter should expose one async-capable interface conceptually equivalent to:

```python
class LlmProvider(Protocol):
    provider_id: str

    def capabilities(self) -> ProviderCapabilities: ...
    def complete(self, request: ModelRequest) -> ModelResponse: ...
```

`ModelRequest` contains messages, allowed tool schemas, response schema, timeout, sampling parameters, and provider-neutral budgets. `ModelResponse` contains the parsed content/tool calls, model-reported identifier, usage, finish reason, latency, provider request ID, and a normalized error category.

The agent consumes no provider SDK object directly. This makes fake-provider contract tests possible and prevents strategy behavior from silently diverging by SDK.

### Structured decision contract

The model's final output should be a strict object, not free-form text parsing:

```json
{
  "schema_version": "aresim.llm.decision.v1",
  "state_revision": "session-id:step",
  "action": "move_east",
  "intent": "Reach the nearest visible unextracted ice deposit",
  "plan": ["move east", "extract if legal", "return before battery reserve"],
  "memory_write": {
    "kind": "episode_note",
    "content": "Build pad is west of the current explored region",
    "ttl_steps": 40
  },
  "confidence": 0.78
}
```

Only `action` causes a world mutation. `intent`, `plan`, confidence, and memory writes are diagnostic/private policy data. The host rejects unknown fields, oversized strings, invalid memory types, stale `state_revision`, unknown actions, and mask-illegal actions.

Natural-language chain-of-thought should not be required, requested, stored, or used as the audit mechanism. Auditability comes from structured intent/plan summaries, tool calls, inputs, outputs, and state/action validation.

## 7. State, history, and memory

### State visibility

The generic LLM agent consumes the same `aresim.obs.local.v1` actor observation and authoritative `aresim.action.rover.v1` mask used by other policies. It must not receive the UI snapshot or canonical `WorldState`.

Unlike Jev's fixed 5x5 encoder, the generic agent should support versioned state views selected in `PolicySpec`, for example:

- `local_full_v1`: lossless named rendering of the entire 8x8 local observation;
- `local_compact_v1`: current cell, adjacent cells, resources, colony, objectives, and ranked visible points of interest;
- `local_delta_v1`: current compact view plus changes since the previous decision;
- `local_query_v1`: small status header, with read-only tools for cells and history.

Every view must be a deterministic projection of the policy observation. A state view can summarize but cannot add privileged information. Its schema/version and serialized hash belong in the decision trace.

### Read-only tools

The deliberative agent may receive a small, typed tool set:

| Tool | Purpose | Bound |
|---|---|---|
| `get_status` | Rover, colony, weather, objective status | One current snapshot |
| `inspect_local_cells` | Filter/sort cells already present in the local observation | Maximum returned cells and calls |
| `get_legal_actions` | Legal action names and local effects | Always authoritative |
| `get_recent_transitions` | Actions, rewards, events, resource deltas | Episode only; capped window |
| `search_episode_history` | Typed filters over this episode | Capped matches; no arbitrary query language |
| `read_memory` | Read allowed memory entries | Scoped and capped |
| `write_memory` | Propose a bounded memory entry | Validated; no world mutation |
| `submit_action` | Return one action recommendation | Exactly once per decision |

Tools operate against an immutable decision snapshot. They cannot advance the environment. If the live step changes before submission, the host rejects the result as stale and creates a new decision context.

### Explicitly forbidden gameplay tools

- Shell, Python, filesystem, Git, package manager, browser, arbitrary HTTP, database, email, or messaging access.
- Source-code reads or edits.
- API-key or environment-variable reads.
- Direct simulator mutation, reward modification, seed selection, reset, or hidden-state inspection.
- Creating new tool schemas at runtime.
- Delegating to unregistered external agents.

### Memory scopes

| Scope | Default content | Lifetime | Research rule |
|---|---|---|---|
| Decision scratch | Tool results and current deliberation | One action | Never persisted as hidden reasoning |
| Working history | Recent transitions and summaries | One episode | Reset with the episode |
| Spatial memory | Facts derived from past visible observations | One episode | Provenance and last-seen step required |
| Policy notebook | General tactics, never seed-specific coordinates/events | Across development trials | Versioned as part of `PolicySpec`; frozen for validation/test |
| Research memory | Trial hypotheses and aggregate results | Across experiments | Cannot be read by gameplay agents |

Cross-episode automatic memory is off by default because it can leak validation/test experience and makes comparisons order-dependent. If studied, it is a separate named condition with deterministic episode ordering and an explicit memory snapshot hash.

## 8. Decision strategies

### 8.1 Reactive baseline

One request receives one chosen state view, a small recent-history window, objectives, and legal actions. It returns the decision object. This is the simplest comparison with Jev and should be implemented first.

### 8.2 Deliberative tool-use agent

The model starts with a compact header and chooses which read-only state/history tools to call. The host enforces a small maximum round count, tool-call count, input tokens, output tokens, wall-clock timeout, and estimated cost. This directly tests the user's idea that the model should decide what information matters.

### 8.3 Planner/executor

The LLM produces a typed short-horizon plan or subgoal every `N` steps or when invalidated. A deterministic, partial-observation executor chooses individual legal actions between planning calls. Replanning triggers include goal completion, newly observed hazard/resource, low reserve, unexpected reward/event, repeated action, or plan infeasibility.

This should be prioritized after the reactive baseline: per-tick LLM calls are expensive and may be myopic, while the planner/executor split tests long-horizon reasoning without giving the LLM extra authority.

The executor must live in `engine/aresim/algorithms/llm/`; it may not import or silently fall back to PPO/DQN unless the experiment is explicitly labeled hybrid.

### 8.4 Swarm

A swarm is one policy decision with multiple bounded analysis roles, not multiple simulator actors. Recommended initial roles:

- `safety`: power, battery, health, weather, survival reserve;
- `navigator`: local geometry, exploration, loop detection, return path;
- `mission`: objectives, extraction/build/service/unload priorities;
- `critic`: identify conflicts, unsupported assumptions, and risky recommendations;
- `coordinator`: receives structured role reports and alone submits the final action.

Role reports use schemas such as `{recommendation, evidence_refs, risk, confidence}`. Agents do not chat freely or call one another recursively. They inspect the same immutable state revision, have individual call/token limits, and cannot submit actions except through the coordinator. The host validates the coordinator's result exactly like a single-agent result.

Compare swarms with a single model under both:

1. equal environment episodes; and
2. approximately equal token/cost budgets.

Without both views, a swarm improvement may merely reflect spending several times more inference.

## 9. Safety, robustness, and fallback rules

### Host-enforced invariants

- The simulator remains deterministic for a fixed seed and accepted action sequence.
- Legal actions come only from the engine mask.
- Wait remains the final safe fallback; strategies may define a deterministic mask-aware fallback before Wait.
- A response applies only to the exact session and step encoded by `state_revision`.
- At most one environment action is accepted per decision.
- Timeouts, rate limits, malformed output, refusal, empty output, unknown tools, recursion, and exhausted budgets cannot crash or stall an episode.
- Retry count is bounded. A retry sees the same immutable decision context.
- Prompts and tool results are size bounded before provider submission.
- All provider errors use stable categories; provider prose is not control flow.
- Secrets are redacted before logging and UI projection.

### Prompt injection

Current simulator observations are structured numeric/enum data, but future task or event text should still be treated as untrusted data. The host must frame it as data, never concatenate it into system instructions. The model cannot expand its permissions based on text found in state, history, memory, or provider output.

### Determinism claims

Provider inference may remain nondeterministic even with temperature zero. Runs should record sampling parameters, provider/model IDs, request IDs, timestamps, and raw-response hashes. “Reproducible” means replayable simulator actions and fully specified inference conditions, not guaranteed identical future API responses.

For debugging, the provider boundary should support record/replay fixtures so an episode can be rerun without live API calls.

### Live-service concurrency

The current live service calls `agent.act(...)` and applies the returned command while holding its session lock. That is tolerable for local policies but a poor fit for multi-round remote inference: a long provider call would also block pause, snapshot, and other session requests.

Refactor an LLM step into `prepare -> decide -> commit`:

1. Under the session lock, build the observation/mask and an immutable `state_revision`.
2. Release the lock before all provider/tool deliberation. Tools read only the captured decision context.
3. Reacquire the lock, confirm that the same session/revision is still current, validate the action, and apply it.
4. If the revision changed or the session was replaced, discard the decision as stale; never apply it to a newer state.

Only one decision may be in flight for a live session. Batch rollouts can keep the simpler synchronous loop because each worker exclusively owns its environment.

## 10. Policy and experiment specifications

### PolicySpec

A versioned policy specification should contain at least:

```yaml
schema_version: aresim.llm.policy.v1
policy_id: reactive-compact-history-v001
strategy: reactive
provider: openai
model: <explicit-provider-model-id>
state_view: local_compact_v1
system_prompt_file: prompts/rover_v003.md
history:
  mode: recent_plus_salient
  recent_steps: 12
  max_events: 24
memory:
  working: summary_v1
  spatial: discovered_cells_v1
  cross_episode: false
decision:
  temperature: 0.0
  max_output_tokens: 400
  timeout_seconds: 15
  max_retries: 1
  max_tool_rounds: 0
  max_tool_calls: 0
fallback: safest_legal_v1
```

Provider-specific pass-through parameters should be avoided. When unavoidable, they belong in a namespaced `provider_options` object and must be included in the policy hash.

### ExperimentSpec

```yaml
schema_version: aresim.llm.experiment.v1
experiment_id: llm-reactive-provider-comparison-v001
hypothesis: A compact semantic view improves mission return per token.
parent_experiment_id: null
policy_spec: configs/llm/policies/reactive-compact-v001.yaml
providers:
  - provider: openai
    model: <model-id>
  - provider: anthropic
    model: <model-id>
  - provider: gemini
    model: <model-id>
  - provider: openrouter
    model: <model-id>
seed_manifest: notebooks/phase1_open_exploration_split_v1.yaml
development_split: train
selection_split: validation
max_episode_steps: 400
replicates_per_seed: 1
budgets:
  max_requests_per_episode: 400
  max_input_tokens_per_episode: 300000
  max_output_tokens_per_episode: 50000
  max_cost_usd_per_episode: 5.00
promotion:
  primary_metric: success_rate
  minimum_seeds: 32
  require_no_safety_regression: true
```

Model IDs and prices change over time, so checked-in examples should use explicit placeholders until implementation chooses verified current IDs. A run manifest records the concrete ID and cost table/version actually used.

## 11. Autoresearch workflow

### 11.1 Research cycle

```text
Hypothesis
   |
   v
New immutable PolicySpec candidate
   |
   v
Static validation + fake-provider tests
   |
   v
Small train-seed smoke/cost gate
   |
   v
Paired train-seed batch
   |
   v
Aggregate analysis and failure taxonomy
   |
   +---- reject/archive
   |
   v
Paired validation batch
   |
   +---- reject/archive
   |
   v
Promote candidate and freeze
   |
   v
One final test evaluation (not fed back into search)
```

Each proposal must state one main hypothesis and a small diff from its parent. The researcher receives aggregate metrics plus sampled redacted failures, not unrestricted test artifacts.

### 11.2 What the researcher may change without code

- State view and its existing parameters.
- Prompt text and few-shot examples derived only from development data.
- History retrieval/filter configuration.
- Memory strategy and limits.
- Reactive versus tool-use versus planner/executor strategy.
- Planning/replanning cadence.
- Role definitions for an existing swarm strategy.
- Sampling, request, token, latency, and cost budgets within global limits.
- Deterministic fallback selection from registered fallbacks.

PolicySpec is schema validated. It cannot name arbitrary Python imports, commands, URLs, filesystem paths outside approved prompt/spec roots, or executable snippets.

### 11.3 When code changes are justified

Code is required only for a new general mechanism: a new state projection, history algorithm, memory backend, tool, fallback, decision strategy, or provider adapter. Prompt/model/budget changes do not justify code branches.

For code experiments:

1. Start from the recorded baseline commit in a dedicated Git worktree and `codex/llm-exp-<slug>` branch.
2. Permit automated writes only under `engine/aresim/algorithms/llm/`.
3. Deny simulator, reward, task, observation source, registry, API, UI, test-seed manifest, and existing baseline edits to the research agent.
4. Provide no production credentials to the code-writing process.
5. Run formatting, type/static checks, unit tests, provider contract tests, and fake-provider rollouts.
6. Inspect the diff size and allowlist before any live evaluation.
7. Require human approval to merge a mechanism into the main branch.

Integration changes outside the LLM package are made deliberately by the project implementation workflow, not by an autoresearch candidate.

### 11.4 Branch strategy

A branch per API trial is wasteful and obscures comparisons. Use:

- immutable config + run directory for prompt/model/state-view/budget trials;
- one branch/worktree only for experiments that change executable mechanisms;
- one commit per candidate mechanism where practical;
- a run manifest containing both Git commit and PolicySpec hash.

This gives isolation when needed without turning hundreds of inexpensive prompt trials into hundreds of branches.

### 11.5 Promotion rules

Use paired seeds and predeclared thresholds. A candidate is promoted only if it:

- completes the minimum seed count;
- improves the primary metric or meets a declared Pareto objective;
- does not regress hard safety measures beyond tolerance;
- stays within per-episode and aggregate cost/latency budgets;
- has an acceptable invalid-output/fallback rate;
- passes reproducibility and artifact validation;
- was selected without viewing test outcomes.

Do not optimize only mean reward. A high mean can hide catastrophic tails or a policy that works on a narrow seed family.

### 11.6 Dream-RSI-inspired meta-exploration

[Dream-RSI](https://dream-rsi.com/) proposes treating accumulated discovery trees as exact replay simulators over the search space that was actually explored. Alternative exploration policies can traverse the stored tree in different orders, select different recorded branches, allocate parallel work differently, and stop earlier without repeating the underlying expensive executions. The selected exploration policy is then deployed online to expand the tree, creating a recursive loop.

This maps well to AresSim's **autoresearch controller**, not directly to its rover controller:

| Dream-RSI concept | AresSim mapping |
|---|---|
| Discovery task | Improve an LLM rover PolicySpec or general LLM mechanism |
| Tree node | Immutable candidate spec/code commit plus hypothesis, parent, evaluation result, diagnostics, and cost |
| `CONTINUE(node)` | Ask a research agent to refine or branch from that candidate, then run the approved evaluation |
| Node score | Predeclared quality/safety/cost objective over development seeds |
| Exploration policy | Choose which candidates to extend, how many branches to run, concurrency, and when to stop |
| Replay world | One completed research discovery tree with all recorded node outcomes |
| Simulator pool | Trees from different research campaigns, tasks, providers, or starting policies |
| Online deployment | Execute the chosen research exploration policy to generate genuinely new candidate results |

The discovery tree should be append-only and record for every node:

- node and parent IDs;
- hypothesis and normalized change categories;
- PolicySpec hash and, when applicable, code commit/diff hash;
- exact development seed set and evaluation budget;
- aggregate metrics, failure taxonomy, latency, token usage, and cost;
- whether the node was proposed, approved, executed, rejected, promoted, or invalid;
- researcher inputs and a normalized proposal, with secrets and hidden reasoning excluded.

A replay evaluator can then simulate questions such as: Would another controller have refined the best-so-far branch sooner? Would it have stopped an unproductive family earlier? Would a different concurrency or cost threshold have reached the same promoted candidate with fewer real LLM rollouts?

#### Important limitation

Replay is exact only over outcomes already present in a discovery tree. It cannot reveal a candidate experiment that was never run. Likewise, a normal rover trajectory records only the outcome of the action actually taken; it cannot establish the episode return of a different unseen action or a different LLM policy. Therefore:

- use Dream-RSI replay to improve **which research experiments are selected and scheduled**;
- use fixed recorded decision contexts only for cheap action-agreement, schema, latency-free, and consistency screening;
- use real AresSim fixed-seed rollouts to evaluate a new rover policy's episode outcomes;
- periodically deploy the improved research policy online so it adds new branches and avoids becoming trapped in the support of old history.

AresSim's deterministic engine is already inexpensive to rerun; remote LLM inference is the costly and nondeterministic part. Replaying simulator transitions without fresh model decisions therefore does not evaluate a new LLM rover policy. The likely Dream-RSI benefit is reducing wasted **research candidate evaluations and provider calls**, not avoiding ordinary engine compute.

#### Adoption plan

1. First store the experiment lineage as a general tree even while using a fixed breadth/depth research scheduler.
2. Define a deterministic replay interface that reveals only the portions of a tree selected by the replayed controller; unrevealed outcomes must remain hidden.
3. Score research controllers on candidate quality, real evaluation cost, wall-clock rounds/concurrency, safety violations, and coverage/diversity.
4. Include the currently deployed research controller in every candidate set so selection cannot score worse on the accumulated replay pool, while recognizing this guarantee applies only to that replay score—not unseen future experiments.
5. Compare against fixed breadth-first, best-first, successive-halving, and simple bandit schedulers before claiming value.
6. Deploy a replay-selected controller only after validation and budget checks; keep test seeds entirely outside both tree construction and replay optimization.

As of this plan's date, the [official repository](https://github.com/zhengkid/Dream-RSI) says the full codebase and reproduction scripts are still being prepared. AresSim should therefore implement the minimal tree/replay contracts from its own experiment artifacts first, treat Dream-RSI as a design influence rather than a dependency, and reassess direct reuse after code and licensing details are available.

## 12. Evaluation design

### Metrics

| Category | Required metrics |
|---|---|
| Task quality | return, success rate, objective completion, episode length |
| Safety | terminal/failure reasons, minimum battery/health/livability, stranded rate |
| Behavior | action histogram, unique cells, loops/reversals, extraction/delivery/build/service counts |
| Reliability | malformed response, illegal recommendation, retry, timeout, provider error, fallback rates |
| Efficiency | requests, input/output/cached/reasoning tokens when reported, latency percentiles, estimated/actual cost |
| Planning | plan completion, replans, invalidations, action-to-plan consistency |
| Memory | reads/writes, stale facts, unsupported facts, memory size |
| Swarm | role agreement, coordinator overrides, marginal value and cost per role |

Reports should include mean, median, dispersion, bootstrap confidence intervals, lower-tail results, and paired per-seed deltas against baselines.

### Fair comparisons

Run at least these comparisons:

- same seeds, task, observation contract, step limit, and fallback policy;
- provider/model comparison with the same PolicySpec where capabilities permit;
- policy comparison on the same provider/model;
- deterministic scripted, random-valid, Jev, PPO, and DQN references where applicable;
- equal-episode and equal-cost views;
- optional repeated inference on identical recorded contexts to measure model decision variance.

Provider tool/schema capabilities are not identical. The normalized contract should target their common subset for the main leaderboard. Provider-specific enhanced runs must be labeled separately.

### Behavioral policy comparison

To answer whether models discover the same policy, compare more than final reward:

- action distributions conditioned on coarse state bins;
- route and coverage heatmaps;
- priority under conflicts such as low battery versus visible resource;
- time-to-unload, reserve thresholds, and replanning triggers;
- state/action disagreement rate on a shared corpus of recorded decision contexts;
- clustered structured intents/plans;
- counterfactual consistency on controlled state perturbations.

Store a normalized `decision_signature` per step containing state-view ID, coarse observable features, legal set, selected action, intent category, strategy mode, and fallback. This enables cross-model comparison without storing private chain-of-thought.

## 13. Artifacts and trace schema

Each run should be stored under a unique immutable directory such as:

```text
results/llm/<experiment_id>/<trial_id>/
├── manifest.json
├── policy_spec.yaml
├── summary.json
├── discovery_node.json       # Parent, hypothesis, spec/commit hashes, score and status
├── episodes.jsonl
├── decisions.jsonl
├── provider_usage.jsonl
├── failures.jsonl
├── trajectories/             # optional validated AresSim trajectories
└── reports/
```

`manifest.json` should record:

- experiment/trial/parent IDs;
- Git commit and dirty-state flag;
- policy and prompt hashes;
- provider, requested model, returned model, endpoint class, and SDK version;
- environment/task/observation/action/reward schema IDs;
- seed-manifest ID and selected split;
- all budgets, sampling settings, fallback, and strategy settings;
- dependency versions and start/end timestamps;
- aggregate request/token/cost totals.

`decisions.jsonl` should record one line per attempted action: episode/step IDs, immutable state revision, hashes or redacted snapshots of presented inputs, tool calls/results, structured decision, validation outcome, applied action, fallback, usage, latency, provider IDs, and error category.

Raw prompts/responses may contain sensitive provider output or future user-authored scenario text. Make raw capture opt-in, redact secrets, impose file-size/retention limits, and keep the normalized trace sufficient for ordinary research.

## 14. CLI and configuration plan

Generalize `aresim-rl rollout`, which currently dispatches only Jev, rather than creating one CLI per provider. Proposed commands:

```bash
# Validate specs without making an API call
aresim-rl llm validate configs/llm/experiments/reactive-v001.yaml

# One bounded experiment batch
aresim-rl llm run configs/llm/experiments/reactive-v001.yaml

# Resume only missing/failed episodes from immutable artifacts
aresim-rl llm run configs/llm/experiments/reactive-v001.yaml --resume

# Compare completed trials without live API calls
aresim-rl llm compare results/llm/<experiment-id>

# Propose a new declarative candidate; never launches it automatically by default
aresim-rl llm propose results/llm/<experiment-id>
```

The runner should support bounded concurrency across independent episodes with per-provider rate limiting and backoff. Never make concurrent decisions for the same live episode.

## 15. UI support

The current UI has a flat algorithm selector, a checkpoint field, and Jev-specific availability. The LLM integration should make the catalog capability driven.

### Minimum UI controls

When `LLM` is selected in Algorithm mode, show:

- provider: OpenAI, Anthropic, Gemini, OpenRouter;
- model ID: server-supplied configured choices plus an optional validated custom value;
- policy/strategy preset;
- live capability/credential status without exposing the key;
- per-step versus planner cadence when supported;
- Apply/attach action and a clear error if unavailable.

Do not send API keys from the browser or store them in frontend state. The backend reads credentials from its environment.

### Live decision inspector

Expose generic `policyMeta.llm`, not a field per provider:

```json
{
  "provider": "openai",
  "model": "<returned-model-id>",
  "policyId": "reactive-compact-history-v001",
  "strategy": "reactive",
  "action": "move_east",
  "intent": "seek visible ice",
  "confidence": 0.78,
  "fallback": null,
  "toolCalls": 0,
  "inputTokens": 1200,
  "outputTokens": 90,
  "latencyMs": 840,
  "estimatedCostUsd": 0.0012
}
```

The history/inspector can show action, concise intent/plan, provider/model, latency, usage, fallback, and tool names. It should not expose API keys or hidden chain-of-thought. Autoplay must wait for the in-flight decision and prevent overlapping calls; pause should stop scheduling the next decision but need not cancel an already accepted provider request.

### Backend API direction

- `/api/policies` returns one `llm` catalog entry and provider capability records rather than four hard-coded algorithm IDs.
- Attach-policy accepts a validated LLM profile/config ID (or a strictly bounded config object), not credentials.
- Health reports provider availability individually.
- `policyMeta` becomes a tagged generic policy metadata structure while retaining backward compatibility for Jev trajectories.
- The Save manifest includes provider/model/policy hashes and usage totals.
- LLM agent-step uses the unlocked `prepare -> decide -> commit` flow so remote inference does not hold the service lock.

## 16. Testing plan

### Unit and contract tests

- Strict config/spec decoding rejects unknown fields and unsafe paths/imports.
- Every state view is deterministic, bounded, and derived only from `aresim.obs.local.v1`.
- Tool inputs/outputs validate and cannot access hidden state.
- Memory resets, TTLs, size limits, provenance, and scope rules hold.
- Each provider adapter passes the same fake transport fixtures.
- Structured output normalization covers tool calls, refusals, truncation, malformed JSON, missing usage, and model aliases.
- Action validation covers illegal, stale, multiple, missing, and oversized decisions.
- Retries and all fallback paths are bounded and deterministic.
- Traces redact known credential patterns and satisfy their schema.

### Integration tests

- Fake-provider rollouts are deterministic for each strategy.
- Recorded provider responses can replay without network access.
- Timeouts/rate limits do not deadlock UI autoplay or batch evaluation.
- Concurrent episodes do not share memory or corrupt traces.
- The LLM agent never receives canonical state.
- The engine still validates every applied command.
- UI capability states, attach flow, one-step flow, pause, save, and replay work for available and unavailable providers.

### Research integrity tests

- Train/validation/test manifests are disjoint.
- Test artifacts cannot be inputs to `propose`.
- PolicySpec hash changes when any behavior-affecting setting or prompt changes.
- Resume never silently reruns completed episode IDs or mixes policy hashes.
- Code-experiment diffs fail the allowlist if they touch outside `engine/aresim/algorithms/llm/`.

## 17. Phased implementation

### Phase 0 — contracts and offline skeleton

- Create the `llm` package, strict schemas, fake provider, traces, and config validation.
- Define `aresim.llm.decision.v1`, state-view IDs, errors, budgets, and generic `policyMeta.llm`.
- Add unit tests and deterministic fake-provider rollouts.
- No live SDK is required for completion.

Exit criterion: a fake provider can complete and replay fixed-seed episodes with validated artifacts and no hidden-state access.

### Phase 1 — reactive providers and UI

- Implement OpenAI, Anthropic, Gemini, and OpenRouter adapters behind the common interface.
- Add one `llm` registry entry and capability-driven provider/model selection.
- Implement `local_full_v1` and `local_compact_v1`.
- Split live inference into `prepare -> decide -> commit` with stale-revision rejection.
- Add UI selection and the live decision inspector.
- Run a small matched-seed smoke matrix.

Exit criterion: all four providers can execute the same reactive PolicySpec where credentials are configured; missing providers disable cleanly.

### Phase 2 — research runner and comparisons

- Add seed-manifest batches, concurrency/rate limits, resume, artifact validation, metrics, and comparison reports.
- Add paired-seed promotion gates and immutable policy/experiment hashes.
- Add record/replay provider fixtures.

Exit criterion: one command produces a resumable, auditable provider comparison with cost and behavioral metrics.

### Phase 3 — tool-use, memory, and planning

- Add bounded read-only tools, episode memory, delta views, and planner/executor.
- Ablate full context versus model-selected queries, recent history versus retrieval, and per-step versus lower-frequency planning.
- Establish robust default budgets from validation results.

Exit criterion: each mechanism demonstrates value beyond the reactive baseline on paired seeds or is documented as rejected.

### Phase 4 — declarative autoresearch

- Add proposal, static validation, smoke gate, paired evaluation, promotion, and experiment lineage.
- Persist append-only discovery trees and add a Dream-RSI-inspired replay interface for research-scheduler experiments.
- Benchmark replay-improved scheduling against fixed and standard adaptive schedulers before enabling it by default.
- Keep launch/merge approval explicit until failure modes and costs are well understood.
- Add optional worktree automation only for allowlisted LLM-package mechanism experiments.

Exit criterion: the system can propose and evaluate bounded PolicySpec candidates without code changes or test leakage.

### Phase 5 — swarm experiments

- Add fixed-role structured analysts and coordinator.
- Measure equal-episode and equal-cost performance.
- Add role-ablation and disagreement reports.

Exit criterion: a swarm is retained only if it beats a budget-matched single agent or yields a clearly valuable safety/reliability improvement.

## 18. Recommended first experiments

Keep the first matrix small enough to explain:

1. **Jev versus generic reactive LLM:** use the closest comparable compact state and per-step cadence.
2. **Full versus compact state:** determine whether semantic compression helps or hides important information.
3. **Recent history windows:** compare 0, 4, 12, and summary-plus-salient history.
4. **Reactive versus planner/executor:** compare outcome, calls, latency, and cost.
5. **One provider-neutral policy across providers:** measure reward and action disagreement on identical seeds.
6. **Repeated frozen contexts:** sample the same decisions multiple times to estimate inference variance.
7. **Model-selected queries:** compare bounded tool use with always-supplied full context at similar token budgets.

Do not begin with prompt search, memory search, tool-use, four providers, multiple models, and a swarm all varied at once. Establish one stable reactive reference, then change one mechanism at a time.

## 19. Answers to the two central concerns

### How can an LLM change code based on its findings?

It normally should not. The efficient research surface is a validated, declarative PolicySpec, so the model can change prompts, state views, history/memory selection, strategy parameters, and budgets without executable access. That captures most useful experimentation and makes trials comparable.

When a new mechanism really requires code, use a separate worktree/branch, restrict automated writes to `engine/aresim/algorithms/llm/`, run offline tests and diff allowlists, and require review before live evaluation or merge. Never let the gameplay agent edit itself mid-episode. The evaluated commit and policy hash stay frozen for the entire run.

### What should an LLM be allowed to do?

It may inspect only permitted, partial-observation-derived state; query bounded history from its current episode; read/write bounded policy memory; form plans; request registered read-only tools; and recommend one mask-legal action.

It may not train weights, edit code, run commands, access arbitrary files/network/services, see hidden simulator state, alter rewards/rules/seeds, bypass the action mask, or persist seed-specific knowledge into held-out evaluation. Robustness comes from fixed-seed splits, paired comparisons, bounded memory, failure fallbacks, and selecting policies on validation rather than anecdotes.

## 20. Open decisions before implementation

These choices can be made during Phase 0 without changing the architecture:

- Whether the first benchmark task is open exploration, the resource mission, or both. Recommendation: use the goal-bearing resource mission for primary success metrics and open exploration for behavioral coverage.
- Whether the initial live UI accepts arbitrary model IDs. Recommendation: server-configured allowlist plus an advanced custom field, with the resolved ID always recorded.
- Whether raw provider payload retention is allowed. Recommendation: off by default; normalized redacted traces on by default.
- Whether validation may use more than one inference replicate per seed. Recommendation: one for routine search, multiple only for finalists and frozen-context variance tests.
- Maximum spend and wall-clock budget. These must be explicit before live autoresearch can launch candidates automatically.

The implementation should not block on a universal answer to these questions; they are experiment configuration and governance choices, not reasons to couple providers or weaken the agent boundary.
