# Research Plan for the Agent Components

Status: proposed research agenda, not a final design or an implementation. Read this with the [main agent plan](experience_driven_llm_agent_learning_plan.md) and the [literature review](literature_survey.md).

## Purpose

The pipeline has several places where a clever design *might* help. We should not pick a memory system, retrieval algorithm, or research scheduler just because a paper reports a gain elsewhere. For each component, we need to answer: **What failure in a real game would it fix? What is the simplest competing solution? Can we measure an improvement on new episodes after counting its cost?**

We will survey candidate methods, implement small controlled comparisons, and keep only components that help. Memory and history receive the first deep investigation because [AresSim's policy observation](../../engine/aresim/components/observations.py) is local: useful facts can leave view while still mattering later.

Before promoting any idea, apply a realism check: identify a decision it could change, confirm the needed evidence is actually visible to the agent, compare it with a strong simple baseline, fit it within the actor's time/context budget, and verify the gain in fresh episodes. If any step fails, the idea remains a research note rather than a pipeline requirement.

## 1. What needs research?

The rows are ordered by dependency, not by how fashionable the method is. `Now` means research should start before choosing an implementation; `Later` means a simpler baseline must exist first.

| Component | Question to answer | Simple baseline | Realistic way it might help | Main risk | Priority |
|---|---|---|---|---|---|
| Observation and feedback contract | What can the actor actually see before and after an action? | Current AresSim observation, legal mask, exact transition log | Prevents learning from unavailable information | Hidden-state or timing leak invalidates all later results | Now: prerequisite |
| State preprocessing | Which visible facts should the actor receive, and in what form? | Full readable observation and a hand-built compact view | Reduces noise while preserving battery, goal, terrain and resource signals | Drops a fact needed for a safe choice | Now |
| Within-episode history | What must be remembered after it leaves the local view? | Last `k` actions plus a small deterministic map | Return to a previously seen pad; avoid repeated moves | Incorrect locations or stale facts are worse than no memory | Now: first deep study |
| Cross-episode lessons | Which patterns from earlier games should carry over? | No lessons; then a fixed list of recent episodes | Avoid a repeated failure under similar conditions | Spurious lesson or negative transfer | Now: after history baseline |
| Retrieval and decision packet | Which history or lesson is worth showing *right now*? | Recent items in a fixed-size packet | Gives a fast actor one relevant fact instead of a long archive | Retrieval cost, irrelevant advice, context overload | Now: alongside lessons |
| Fast actor and escalation | Can Jev/Laya use the packet well, and when is an LLM call worthwhile? | Existing Jev and scripted actor | Most decisions stay cheap; hard cases get more reasoning | Fast model ignores guidance or escalation costs too much | Next |
| Improvement controller | Which proposed change should be tested next? | Fixed, budgeted experiment order | Spends calls on the most promising unresolved failure | Optimizes old runs but fails online | Later |
| Cross-game transfer | Which parts survive a change of game? | Start the same learner from scratch in each game | Reuses view operators, lesson-checking and experiment method | A human-built adapter hides the actual transfer work | Later, after AresSim evidence |

Provider adapters, storage, UI controls and action validation also need careful engineering and contract tests. They do not need a new research algorithm unless their behavior changes agent decisions or the measured cost.

## 2. Memory: separate the questions before choosing a design

“Memory” currently refers to several different jobs. They should have separate data, rules and tests:

1. **Raw history:** the exact observations, legal actions, choices and outcomes the actor was allowed to see. Keep it append-only for audit and for later analysis. This is evidence, not a lesson.
2. **Episode working memory:** facts used while the current game continues, such as where the pad was last seen or which nearby cells were already explored. It is reset at the next episode.
3. **Cross-episode knowledge:** a candidate lesson, exception or short tactic inferred from several episodes. It must name supporting and conflicting records, the conditions under which it applies, and how it was checked. Only tested items enter the actor's saved policy version.
4. **Research history:** which component changes were proposed, tested, rejected or promoted, with their cost. This helps schedule *research*, not the rover's next move.

We should design memory as a **write → maintain → retrieve → use → check** loop, not as one database or one summarization prompt. The interfaces can stay stable while the methods behind them change. For example, a retrieval request should specify the current goal, visible situation, episode/policy version, allowed scope and token budget; the response should contain source IDs and a reason the item applies. A method may return *nothing* when no item is relevant. The actor must never silently receive an entire unfiltered vault.

### What to compare at each stage

| Memory decision | Candidate approaches to survey and test | What would make the extra complexity worthwhile? |
|---|---|---|
| Write within an episode | Recent `k` transitions; deterministic tracked facts; LLM-updated summary; hybrid | Better decisions when needed facts leave view, without invented or stale facts |
| Select events to consolidate | Every episode; failures only; surprise; goal-relevant changes; diverse ordinary controls | More useful lessons per LLM call, not just more dramatic stories |
| Turn episodes into lessons | Raw-case retrieval; short reflection; structured conditional rule with counterexamples | A measured improvement on independent seeds at the same packet size |
| Maintain lessons | Keep all; merge duplicates; narrow on contradiction; expire/recheck under changed game version | Less negative transfer without forgetting useful knowledge |
| Retrieve at action time | Recent-first; feature/goal match; action-relevance ranking; actor-requested query | Better decisions per token and lower latency than simpler retrieval |
| Present to the actor | Raw snippets; concise evidence-linked lesson; subgoal with abort condition | The *fast actor* can use it reliably without parsing a large narrative |

Do not let the LLM decide which **raw** events exist; record them all. The LLM may decide which episodes to summarize and which candidate lessons to propose. The evaluator—not the LLM alone—decides what is reliable enough to put in the policy packet.

### AresSim cases that can reveal whether memory matters

- **Pad leaves view:** a rover that has seen the build pad later needs to return with cargo. Test whether a remembered location improves unloading and survival. [Jev already stores a remembered pad](../../engine/aresim/algorithms/jev/agent.py), so compare against that strong baseline, not just against an amnesiac actor.
- **Repeated movement:** a local view may look attractive even though the rover has just traversed the same cells. Test whether short action/location history reduces loops. Again, Jev already tracks recent actions.
- **Explored or exhausted cells:** test whether retaining what was scanned or extracted helps exploration after those cells leave the crop. A deterministic visited-cell ledger may be more reliable than an LLM summary.
- **Changing operating conditions:** a previous wait or resource action may have succeeded under different power/weather conditions. Test whether a lesson correctly checks its conditions instead of applying blindly.
- **No memory needed:** current battery, cargo and nearby resources are already observable. If history does not change the best decision, retrieval should stay empty. Otherwise memory adds cost and distraction.

These are test *scenarios*, not claims that a particular memory system helps. In a deterministic simulation with accessible coordinates, a small structured map may solve some problems better and cheaper than language memory. If it does, use the map. Cross-episode lessons are especially easy to overstate when the game rules are fixed or the state view already gives the actor a feature such as `wait_recharges`; the research must show an added decision benefit beyond those hints.

### How to evaluate memory without fooling ourselves

Use three linked tests. Each answers a different question:

1. **Can it recall correctly?** Build answerable questions from *actual recorded trajectories*: where was the pad last seen, what action caused a cargo change, when did an earlier belief become stale? Include unanswerable questions and require abstention. This checks factual memory, not gameplay quality. [EMemBench](https://arxiv.org/abs/2601.16690) and [LongMemEval-V2](https://arxiv.org/abs/2605.12493) motivate these types of grounded probes, but their published tasks are not this simulator.
2. **Does recall change a useful decision?** Find real decision points where a fact is absent from the current local crop but present in allowed history. Replay the same visible decision context with controlled memory methods, keeping the state view and actor fixed. Record whether the action changes, whether the retrieved fact was true and whether it was relevant. This is a diagnostic screen; a recorded path cannot tell us the return of an action that was never taken.
3. **Does it improve complete new episodes?** Run the surviving methods on paired, unused development seeds. Compare objective progress, unloads, loops, failures, lower-tail outcomes, latency and total cost. Freeze a final memory version before sealed test seeds. Repeat under changed tasks and later games to detect negative transfer.

Start with `no memory`, `recent k`, `deterministic map + recent k`, `LLM summary`, `raw-case retrieval`, and `checked lessons`, using equal actor, state view and context limits. Only then test smarter selection or on-demand queries. Measure false recalls, stale facts, relevant-item recall, packet size and decision value. A memory method can be excellent at question answering yet useless for action selection; require both kinds of evidence before claiming it improves the agent.

Add stress tests with contradictory lessons, outdated observations and misleading text inside retrieved records. The actor should recognize uncertainty or ignore an unsupported item; retrieved text must not gain authority to change tools, rules or safety checks. Repeat finalist comparisons with more than one LLM learner/provider to see whether the memory method is robust or merely tuned to one model's writing style.

## 3. Method for surveying every component

For each paper or system, record a short evidence card rather than copying its architecture. The card should answer:

1. What exact problem was studied? Was it a game, a web task, chat memory, static question answering, or a trained RL policy?
2. What changed, and what remained fixed? Was the model trained, were there expert trajectories, or was an oracle/hidden state available?
3. What baseline did the authors beat, on what split and with what metric? Were latency, tokens and calls included?
4. Which failure modes and negative results were reported? What remains untested?
5. What *specific* AresSim decision could this method improve? What simpler method could do the same?
6. What would falsify its usefulness here, and what is the smallest fair test?

Group sources by **direct evidence** (interactive agent under similar constraints), **adjacent evidence** (memory retrieval or control in a different setting), and **analogy** (brain or general optimization theory). Prefer original papers and official documentation; verify reported numbers and compare them only within their own benchmark. Include negative and null findings. A new method should enter the implementation queue only after its evidence card and test specification are written.

### Starting reading map, not an exhaustive survey

The existing [literature review](literature_survey.md) already covers WorldCoder, WALL-E, Reflexion, ExpeL, Voyager, GEPA, Dream-RSI and transfer benchmarks. These additional original works sharpen the next survey:

| Research question | Starting sources | Why to read them, with caution |
|---|---|---|
| Why keep history at all? | [Deep Recurrent Q-Learning](https://arxiv.org/abs/1507.06527); [ReAct](https://arxiv.org/abs/2210.03629) | Prior observations can matter under partial observability and interleaved action; trained recurrent RL and per-step LLM reasoning are different from our first actor |
| How to manage a bounded context? | [MemGPT](https://arxiv.org/abs/2310.08560); [RAPTOR](https://arxiv.org/abs/2401.18059) | Paging and hierarchical retrieval are options; their main demonstrations are chat/document QA, not rover control |
| How to test memory itself? | [MemoryAgentBench](https://arxiv.org/abs/2507.05257); [LongMemEval](https://arxiv.org/abs/2410.10813); [LongMemEval-V2](https://arxiv.org/abs/2605.12493); [EMemBench](https://arxiv.org/abs/2601.16690) | Cover updates, abstention, environment gotchas and game-grounded recall; none alone proves an episode-return gain |
| When does more memory fail? | [BenchTrace](https://arxiv.org/abs/2605.29225); [When Does Memory Help Multi-Trajectory Inference?](https://arxiv.org/abs/2605.28224) | Misdiagnosis, forgetting, negative transfer and interaction with the inference strategy are real risks; settings differ from AresSim |
| How should lessons be tested? | [ExpeL](https://arxiv.org/abs/2308.10144); [WALL-E](https://arxiv.org/abs/2410.07484); [WorldCoder](https://arxiv.org/abs/2402.12275) | Experience-derived insights and rules are promising, but a plausible rule needs independent predictive and policy-value tests |
| Where might Dream-RSI fit? | [Dream-RSI](https://arxiv.org/abs/2609.14858); [Go-Explore](https://www.nature.com/articles/s41586-020-03157-9) | Search scheduling and real branch exploration are different from remembering a rover state; replay cannot score an unobserved action |

The fuller survey should also revisit the view-learning, fast-actor and cross-game papers already listed in the literature review. It should result in a *comparison of mechanisms and assumptions*, not a vote for the most recent paper.

## 4. Research sequence and decisions

| Step | Work | Decision to make before moving on |
|---|---|---|
| R0. Establish evidence | Define a goal-bearing AresSim task, legal feedback contract, seed splits, a cost cap and a no-memory baseline. Collect enough traces to identify actual memory-dependent situations. | Are there consequential decisions whose relevant fact is outside the current view? |
| R1. Episode history | Compare recent history, deterministic state tracking and LLM summaries on the grounded probes and complete episodes. | Does LLM-managed episode history beat simpler tracking after accuracy and cost are counted? |
| R2. Cross-episode learning | Compare raw cases, reflections and checked conditional lessons. Test contradiction handling and retrieval at a fixed packet budget. | Do lessons improve *new* episodes and not just recall tests? |
| R3. View–memory interaction | With the actor fixed, check whether a better state view removes the need for some memory, or whether memory restores facts a compact view lost. | Which pair is best, and which component causes the gain? |
| R4. Actor and escalation | Feed the same packet to current Jev, Jev-on-packet, Laya, scripted and direct-LLM actors; test when an extra LLM call pays off. | Is there a hybrid that improves quality per total cost without safety loss? |
| R5. Improvement scheduling | Compare fixed experiment order with active test selection. Record complete research trees. Test Dream-RSI only if there are enough independent trees to check replay-to-online transfer. | Does a smarter scheduler find a better policy with fewer *real* calls? |
| R6. Generality | Freeze the chosen method, then test changed AresSim tasks and held-out games against zero-shot and from-scratch controls. | Is the transferable object a lesson, a view, or only the learning procedure? |

At each step, use a pilot to estimate run-to-run variation, then predeclare a meaningful improvement and enough paired seeds to detect it. Keep separate data for creating lessons, checking them, choosing the policy and final testing. Count **all** LLM calls used to create and maintain memory, not just action-time calls. Preserve null results. A component that cannot beat a simpler baseline should be removed or kept only as an explicitly labeled experimental option.

Research artifacts should include the evidence cards, versioned experiment spec, exact policy/memory snapshot, decision traces, aggregate report, and a short conclusion: *keep, revise, or reject*. All future LLM-specific implementation remains under `engine/aresim/algorithms/llm/`, with only minimal integration outside it as described in the main plan.

## 5. Where Dream-RSI belongs—and where it does not

Dream-RSI proposes using **recorded research trees** to improve how an agent allocates future experiments. Here it could help choose which memory variant, state view or failure hypothesis to test next. It is **not** a mechanism for selecting a relevant pad memory at a rover decision, and it cannot reveal what an unplayed action would have done. Start with a transparent fixed scheduler. Revisit Dream-RSI only after we have multiple independent research trees and can compare its choices with the fixed scheduler in fresh online experiments at the same actual budget.

The principle for the whole project is simple: a memory system is valuable only if it supplies a *correct fact at the moment that fact changes a useful action*. Everything else—storage size, elegant summaries, retrieval scores, or realistic-sounding explanations—is secondary evidence.
