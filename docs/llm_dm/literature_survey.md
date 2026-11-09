# Literature Survey: LLM Agents for AresSim

Implementation status (2026-11-09): `llm_dm` supports a Decision Model or LLM Actor. It uses persistent goals, focused Memory, and a required LLM Learner. The historical research comparisons below motivate this shared experience-driven workflow; see the [implementation guide](implementation.md) for current code, configuration and contracts.

Reviewed: **2026-11-07**. Scope: a frozen LLM acting in AresSim's partially observed, turn-based rover resource mission, with controlled history and a mandatory first-release Learner for between-episode experience learning. This replaces the earlier general game-agent survey. Use [algorithm details](../../engine/aresim/algorithms/llm_dm/DETAILS.md) for current contracts. The research review date is unchanged.

This is a focused primary-source review, not an exhaustive survey or a reproduced benchmark. Sources were checked through original papers, author project material, conference records, and official provider documentation. Method/result inspection was prioritized for ReAct, Reflexion, ExpeL, HiAgent, ReasoningBank, ACE, BALROG, MemoryArena, and REAPER; some adjacent works below were reviewed at abstract/method-overview level. Source versions are stated where relevant. Results belong to their original settings. Every proposed AresSim comparison is our inference, not a published result.

## 1. What this project actually needs evidence for

AresSim already supplies symbolic local observations and an authoritative ten-action mask. It does not need an LLM to infer an action space from screenshots, implement the simulator, or write executable skills to move the rover. Its hard dependencies are spatial recall, returning to a previously observed pad, balancing survival against delivery, remembering consumed resources, and adapting decisions to changing weather/power.

The actor can know its coordinates without knowing the rest of the map. A local crop therefore creates partial observability even in a deterministic simulator. Within-episode observed-cell memory and cross-episode strategic lessons solve different problems. The existing `local_memory.v2`, scripted policy, and Decision Models memory are serious comparators; memory is not a novel capability merely because the new implementation uses text.

The evidence should answer four questions:

| Question | Evidence required in AresSim | Common invalid shortcut |
|---|---|---|
| Can an LLM act competently? | Actual legal actions and completed mission rollouts | A plausible plan or schema-valid output |
| Does history help? | Better return/delivery when relevant facts leave the crop | Recall accuracy alone |
| Does experience teach transferable advice? | Same frozen actor improves on unused seeds after development | Replaying old rewards under new proposed actions |
| Is the system worthwhile? | Quality, failure, latency, and all-in cost against existing policies | Comparing token counts or published scores across unrelated tasks |

## 2. Action selection and grounded feedback

### ReAct — start from an interaction loop

**Source:** Yao et al., [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/html/2210.03629v3), ICLR 2023; arXiv v3 inspected. The paper interleaves reasoning, actions, and observations, with interactive evaluations including ALFWorld and WebShop. Its decision-making formulation allows sparse reasoning rather than requiring a long explanation at every action.

**Limit:** Text/task-specific actions and demonstrations differ from the rover's primitive controls. Better benchmark performance does not establish spatial or long-horizon delivery competence. Reasoning traces can still be wrong.

**AresSim implication:** The transferable mechanism is repeated decisions grounded in fresh environment feedback. Start with one action per tick and compare action-only outputs with any later bounded planning variant. Do not make a rationale mandatory or treat it as observed truth. The plan's feedback hook is necessary because the existing `Agent.act` interface alone does not deliver completed outcomes.

### BALROG — evaluate real gameplay rather than verbal plans

**Source:** Paglieri et al., [BALROG: Benchmarking Agentic LLM and VLM Reasoning On Games](https://arxiv.org/html/2411.13543v1), 2024 preprint / ICLR 2025; original evaluation version inspected, with [author implementation](https://github.com/balrog-ai/BALROG). It evaluates agents across varied interactive games and reports major difficulty in longer/harder tasks, including shortcomings with visual observations.

**Limit:** Its original model results are historical, not a ranking of providers available in November 2026. Task wrappers, prompts, progress measures, and horizons differ. Broad game coverage does not validate an AresSim policy.

**AresSim implication:** Use the already available symbolic input first, record progress and failure modes, and evaluate complete rollouts. Primitive-action loops and poor delivery planning should be measured even when outputs are legal. Do not add image parsing just to resemble a general benchmark.

## 3. Within-episode memory

### HiAgent — working memory is a separate design choice

**Source:** Hu et al., [HiAgent: Hierarchical Working Memory Management for Solving Long-Horizon Agent Tasks with Large Language Model](https://arxiv.org/html/2408.09559v1), 2024 preprint; [ACL 2025 publication](https://aclanthology.org/2025.acl-long.1575/). It chunks action-observation histories by subgoal, summarizes completed chunks, and retrieves details when needed; evaluations include planning domains and Jericho tasks.

**Limit:** Compression and subgoal control add model calls and can lose important facts. These tasks do not show that language summaries beat exact spatial tracking in a coordinate-observable rover world.

**AresSim implication:** Keep within-episode history distinct from strategic lessons. Compare recent transitions and deterministic facts before an LLM summary. Focused mode now uses Actor-selected goals for exploration, collection, delivery, service, and recovery. Actual observations determine progress. This adapts the research idea; it does not establish the same performance gains. Count every summarization/retrieval call.

### MemoryArena — useful recall must affect subsequent actions

**Source:** He et al., [MemoryArena: Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks](https://arxiv.org/html/2602.16313v2), February 2026, revised September 2026. It couples memory construction and use through explicitly interdependent subtasks. The paper finds that strong recall-benchmark performance does not ensure strong action performance, and external memory is not universally beneficial.

**Limit:** Its web/search/planning/reasoning sessions deliberately share information. Independently reset AresSim maps do not share coordinates or resource locations. Published memory systems and scores cannot be transplanted directly.

**AresSim implication:** Borrow the evaluation principle: test recall correctness, then useful decisions, then fresh complete episodes. A remembered pad should improve returning/unloading, not just answer a location question. Reset spatial memory between maps and forbid earlier test outcomes from influencing later frozen-test episodes.

### Engineering baseline that needs no new paper

The [existing observation implementation](../../engine/aresim/components/observations.py) already remembers visited locations and bearings to observed home, ice, and unscanned rock. The [scripted policy](../../engine/aresim/algorithms/baselines/scripted.py) and [unified Actor/Memory pipeline](../../engine/aresim/algorithms/llm_dm/agent.py) also retain private history. These are repository facts, not literature findings.

A small exact fact ledger may solve much of the rover's memory problem. Retain last-observed time, distinguish current visibility from remembered availability, and retire resource claims when revisiting changes them. A summary must outperform that comparator on decisions/cost to justify replacing it.

## 4. Learning from completed experience without weight updates

### Reflexion — outcome-based reflection, with an evaluator

**Source:** Shinn et al., [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/html/2303.11366v4), NeurIPS 2023; v4 inspected. Actor, evaluator, and self-reflection produce verbal feedback retained across attempts. Experiments span sequential decisions, reasoning, and programming.

**Limit:** Reflection quality depends on feedback/evaluation quality, and repeated attempts spend additional interaction and inference. A failed episode does not identify the wrong rover action or prove the suggested correction.

**AresSim implication:** Reflect between development episodes, using authoritative task outcome rather than an LLM's success judgment. Keep the failed trajectory and uncertainty. Evaluate the resulting guidance on unused seeds with the actor held fixed; count failed attempts and reflection costs. A reflected lesson is a candidate, not immediately active policy.

### ExpeL — aggregate lessons across tasks

**Source:** Zhao et al., [ExpeL: LLM Agents Are Experiential Learners](https://arxiv.org/html/2308.10144v3), 2023 preprint / AAAI 2024. It gathers experience, extracts and revises insights from successful and failed attempts, and retrieves related examples/insights at inference without model fine-tuning.

**Limit:** Evidence comes from particular language-agent tasks. Task-order effects, demonstration selection, reflection cost, and prior knowledge complicate an experience-learning claim. Transfer is not arbitrary-game generalization.

**AresSim implication:** Compare raw outcome-labeled cases with distilled lessons from the same permitted evidence. Lesson conditions should capture observable operating context, such as cargo or power state, rather than seed-specific coordinates. Keep the learning phase and frozen evaluation separate.

### ReasoningBank — include failures, but do not import its trust assumptions

**Source:** Ouyang et al., [ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory](https://arxiv.org/html/2509.25140v2), September 2025 preprint / ICLR 2026; March 2026 v2 checked alongside v1. It distills reusable strategies from successes and failures, retrieves memories, and studies extra exploration through memory-aware test-time scaling. The implementation uses model judgment for outcome signals and simple addition for consolidation; web observation histories are approximated through agent reasoning in its setup. V2 also examines judge-noise robustness in a web-task subset.

**Limit:** Model-judged success and reasoning approximations do not establish factual memory reliability. Additional attempts are additional compute. Addition alone does not resolve contradictions or stale advice.

**AresSim implication:** Use simulator outcome labels and exact permitted observations instead. Test success-only versus success-and-failure lessons with a fixed actor/context budget. Preserve counterexamples and scope advice explicitly. More attempts must compete against spending the same budget on more independent episodes.

### Dynamic Cheatsheet — a strong simple comparator

**Source:** Suzgun et al., [Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory](https://arxiv.org/abs/2504.07952), April 2025 preprint; abstract/method overview reviewed. It maintains evolving reusable context without updating model weights. Its demonstrations include reasoning tasks and reuse of discovered code.

**Limit:** Reusable arithmetic solutions are different from partially observed simulation control. A self-curated snippet is not independently verified strategy, and task ordering can influence the final memory.

**AresSim implication:** Include a bounded summary/cheatsheet as a low-complexity baseline against structured lessons. Match evidence and context budgets, repeat development ordering, and record maintenance expense. Do not infer that its code-reuse results justify generated rover code or a tool runtime.

### ACE — local updates help preserve knowledge, not guarantee truth

**Source:** Zhang et al., [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/html/2510.04618v3), October 2025 preprint / ICLR 2026; March 2026 v3 checked alongside v1. Generator, reflector, and curator maintain identified context items through incremental deltas. The paper studies context collapse from whole-context rewrites and evaluates task performance, adaptation cost, and ablations, including reflection-quality robustness.

**Limit:** Preserving an item can preserve an error. Model-reported helpful/harmful counters are not causal estimates of its gameplay value. Large playbooks may exceed the rover actor's useful context.

**AresSim implication:** Version lesson edits rather than rewrite the evidence archive. Later compare whole-summary updates with small item-level changes from the same development episodes. Keep bounded retrieval and fresh policy-value checks. Atomic publication in the plan is an engineering choice motivated by this failure mode, not a guarantee from ACE.

### No Need for Explanations — reflection is not automatically the best content

**Source:** Alazraki et al., [No Need for Explanations: LLMs can implicitly learn from mistakes in-context](https://aclanthology.org/2025.emnlp-main.1686/), EMNLP 2025. In controlled math-reasoning comparisons, omitting explicit corrective rationales can outperform including them; the study examines overconstraint, context length, and answer diversity.

**Limit:** Prepared math examples include answer information that an unplayed rover branch does not supply. The work does not show that all reflections are harmful or that failure alone identifies an optimal action.

**AresSim implication:** Make concise outcome-labeled examples a real comparator. Never invent a correct alternative action merely because an episode failed. If explanations cost more without improving new episodes, retain the simpler evidence representation.

### REAPER — 2026 evidence directly about sequential game decisions

**Source:** Rada and Lisý, [Towards Improving Sequential Decision-Making in LLM Agents via Experience Memory](https://arxiv.org/html/2608.03420v1), August 2026, IJCAI neuro-symbolic workshop; v1 inspected. The paper evaluates fully observable games, finds suboptimal play, and studies REAPER's post-game reflection, experience retrieval, and rule extraction. Its reported memory gains focus on tic-tac-toe.

**Limit:** Small adversarial games with move-optimality analysis differ from AresSim's partial observations, resource logistics, and survival. Results are not evidence that a rover's terminal failure can label every intermediate action. This is a workshop-scale study, not a demonstrated general solution.

**AresSim implication:** Ground judgment in simulator outcomes and keep credit assignment uncertain. The useful comparison is the same frozen model before/after experience memory on unseen maps. A valid mask-compliant move still needs policy-quality evaluation.

## 5. Skills, models, and trained memory: adjacent evidence

### Voyager — relevant ambition, different action abstraction

**Source:** Wang et al., [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291), 2023; author abstract/project overview reviewed. It uses an automatic curriculum, executable skill library, and iterative feedback to acquire Minecraft behaviors without fine-tuning the LLM.

**Limit:** Executable temporally extended skills and Minecraft APIs provide a very different control interface from ten rover primitives. More infrastructure does not establish improved AresSim learning.

**AresSim implication:** Retain the idea of reusable procedures only as later guidance with entry/completion/abort conditions. Do not add Python skill execution, hidden pathfinding, macro steps, or a general curriculum to the initial actor.

### WorldCoder and WALL-E — world-model errors require separate validation

**Sources:** Tang et al., [WorldCoder, a Model-Based LLM Agent](https://arxiv.org/abs/2402.12275), 2024; Zhou et al., [WALL-E: World Alignment by Rule Learning Improves World Model-based LLM Agents](https://arxiv.org/abs/2410.07484), 2024. Abstract/method overviews reviewed. WorldCoder learns executable models from interactions; WALL-E learns rules aligning LLM predictions with environment trajectories for model-predictive control.

**Limit:** Program search, transition prediction, and planning add a second learned system with its own error and compute budget. The authoritative engine is not permission for the actor to inspect or simulate hidden state.

**AresSim implication:** First measure action-selection failures without a learned model. Any later model must predict from permitted histories, score real held-out transitions, and face a direct-actor comparator at equal cost. Imagined outcomes are predictions, never evidence of achieved reward. The current plan deliberately defers this branch.

### MemoPilot — frozen player does not mean no model is trained

**Source:** Cai et al., [From Player to Master: Enhancing Test-Time Learning of LLM Agents via Reinforcement Learning over Memory](https://arxiv.org/abs/2606.08656), June 2026 preprint; abstract/method overview reviewed. It trains a memory copilot with multi-turn GRPO while leaving the player frozen, evaluating repeated RPS and Limit Texas Hold'em.

**Limit:** The memory updater receives weight training and its training cost. Opponent adaptation in repeated games differs from independent procedural maps. It is not a drop-in no-training method.

**AresSim implication:** A learned memory policy is a possible future algorithm, with explicit training data, compute, and frozen checkpoints. Keep it outside the first prompt/context-learning implementation and avoid claiming that a frozen actor makes the entire system training-free.

## 6. Provider engineering evidence

Official documentation establishes wire capabilities, not action competence or reproducibility. Documentation was checked on 2026-11-07; model/account availability still needs an implementation-time capability probe.

| Primary source | Supported conclusion | AresSim consequence |
|---|---|---|
| [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs) | Responses can request JSON-schema-constrained output; refusals/incomplete outputs need handling | Native adapter plus local identity/mask validation |
| [Anthropic structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) | Messages can use `output_config.format`; schema subsets and response termination matter | Translate the common contract through the pinned SDK, inspect stop status |
| [Google structured-output announcement](https://blog.google/innovation-and-ai/technology/developers-tools/gemini-api-structured-outputs/) and [Gemini guide](https://ai.google.dev/gemini-api/docs/structured-output) | Official schema support exists; the guide's content-type fetch failed during this review | Recheck exact SDK settings/model support during implementation; do not claim a verified live integration |
| [OpenRouter structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs) and [provider routing](https://openrouter.ai/docs/guides/routing/provider-selection) | Structured output support and routing controls depend on model/endpoint | Require explicit capabilities and fixed route; log actual upstream and avoid implicit fallback |

Schema compliance does not make an action legal under the current mask, fresh for the current step, strategically useful, or within budget. Likewise, temperature zero and local RNG seeds do not establish deterministic hosted inference. The plan separates simulation replay, recorded-decision replay, and fresh live reruns for this reason.

No provider is selected as the winner from documentation, and no current model ranking or price is asserted here. Compare exact deployed configurations on the same missions, state/history contracts, seed splits, and bounded budgets.

## 7. Synthesis for AresSim

The strongest common evidence supports an environment-feedback loop, explicit bounded memory, and evaluation of experience-derived context. The user requires the Learner in the first release: this is a product-scope decision, while its gameplay benefit remains an empirical question. It does not identify a universal winning memory design or prove that any LLM beats AresSim's existing scripted/Decision Models/RL policies.

| Design question | Review judgment | Decisive AresSim comparison |
|---|---|---|
| First-release architecture? | Direct actor, Memory, and a bounded Learner loop; actor-only runs remain controls | One primitive action per step plus independently evaluated between-episode proposals |
| Full JSON or compact state? | Symbolic input avoids unnecessary perception; optimal compression remains empirical | Same observation, actor, history; vary only deterministic view |
| Recent history or spatial memory? | Rover return logistics make exact observed facts a strong baseline | Pad-out-of-view and exhausted-resource episodes, then held-out missions |
| Summary or structured lessons? | Both credible; explanations can also mislead | Equal-evidence, equal-context raw cases / summary / checked lessons |
| Reflection immediately active? | Unsupported self-grading is weak evidence | Independent new-episode comparison before bundle publication |
| World models, code skills, trained memory? | Relevant later branches with extra costs/assumptions | Defer until simpler failure analysis establishes need |
| Other-game generalization? | Existing papers provide motivation, not this project's evidence | Outside the current AresSim implementation and claims |

The most informative failures are specific: successful resource collection without delivery; returning to stale ice; forgetting the pad; waiting away from the pad; legal but expensive loops; unsafe advice learned under different power conditions; and provider failures hidden by Wait fallback. Instrument these before adding another research subsystem.

Evidence that would change the plan: full-state direct actors consistently failing because of a demonstrated representation limit; exact fact memory losing to a cheaper reliable summary; verified lessons improving held-out completion beyond prior-only proposals; or a compact actor matching expensive reasoning at lower all-in cost. These are measurable triggers for further work, not assumed benefits.

## 8. Reading route

1. ReAct and BALROG for the acting baseline and evaluation failure modes.
2. HiAgent and MemoryArena for the difference between retained history and useful decision memory.
3. Reflexion, ExpeL, ReasoningBank, and REAPER for between-episode experience learning.
4. Dynamic Cheatsheet, ACE, and No Need for Explanations for competing content/maintenance choices.
5. Official provider documents when implementing adapters.
6. Voyager, WorldCoder, WALL-E, and MemoPilot only when considering their deferred mechanisms.

Implementation details, defaults, ownership, interfaces, and manual acceptance live exclusively in the [implementation plan](implementation.md).
