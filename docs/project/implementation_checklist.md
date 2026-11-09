# AresSim Implementation Checklist

**Last updated:** October 16, 2026
**Status:** Living project checklist

This document gives a high-level view of what is implemented and what remains. Detailed behavior and architecture stay in the linked reference documents.

## Status rules

- `[x]` means the feature exists in code and has automated coverage appropriate to its current stage.
- `[ ]` means the feature is planned or proposed but is not yet implemented.
- Update this checklist whenever a milestone changes status. A documentation design alone does not count as implementation.

## 1. Playable simulator — completed

- [x] Deterministic seeded 32×32 Mars environment generation.
- [x] Authoritative Python reset/step engine and grouped default configuration.
- [x] Manual Move, Scan, Extract, Build, Service, Unload, and Wait actions.
- [x] Terrain, payload, battery, power, health, livability, service, weather, reward, warning, and terminal rules.
- [x] Twelve-kilogram rover cargo limit with separate ice and sample masses.
- [x] Stable state checksums and deterministic transition tests.
- [x] FastAPI sessions used by the React UI.
- [x] Portable trajectory export, legacy gameplay import, replay step, reset, and timeline jump.
- [x] Backend error contract and unavailable-backend recovery in the UI.

## 2. Game interface — completed

- [x] Orthographic 3D Survey view as the default environment view.
- [x] North-up Top view and direction-following Rover POV.
- [x] Persistent zoom, mouse-wheel zoom, arbitrary rotation, fit mode, rover follow, and mini-map navigation.
- [x] Seamless terrain with optional grid boundaries, visibility flashlight, layers, and sound controls.
- [x] Compact grouped status header, mission/warning HUD, context inspector, guide, and analytics surfaces.
- [x] Compact three-section Action Bar for Manual, Algorithm, and Replay modes.
- [x] Build-pad status visualization, rover path direction markers, action feedback, and sound cues.
- [x] Ambient Martian day/night backdrop with Sun, moons, planets, and reduced-motion handling.
- [x] Manual play, server-driven Algorithm autoplay (baselines + all six learned checkpoint algorithms), and backend-owned Replay flow.

Algorithm mode attaches registered baselines, Decision Models, or a masked PPO/DQN checkpoint sidecar through `POST /api/sessions/{id}/attach-policy` and steps with `POST /api/sessions/{id}/agent-step`. Inference stays on the API host; the UI sends algorithm selection and an optional dev-time checkpoint path. Decision Models requires `aresim[llm_dm]` and an API key.

## 3. RL-ready environment — completed

- [x] Add the framework-neutral `AresEnvironment` composition boundary.
- [x] Add the configurable local 8×8 rover observation and edge padding.
- [x] Add the flat 10-action RL adapter and legal-action mask.
- [x] Add task/termination and reward components without duplicating core rules.
- [x] Add configurable, quota-capped sample acquisition credit; verify Scan/Unload, partial quotas, disabled targets and sparse/default compatibility. DQN and MuZero references enable +0.30 credit; DQN seed_3 shows mission improvement, while MuZero learning remains unverified.
- [x] Add the single-rover `AresGymEnv` adapter and pass Gymnasium checks.
- [x] Add the one-rover PettingZoo Parallel environment and contract tests.
- [x] Verify direct engine, Gymnasium, and PettingZoo transition parity.

## 4. Rollouts and baseline agents — partially completed

- [x] Add external episode truncation and explicit environment/agent seeds for rollouts.
- [x] Add the versioned fixed train, validation, and test seed manifest for the current scenario.
- [x] Add random, random-valid, Wait, scripted, and Decision Models baseline/inference agents.
- [x] Add a simple rollout runner for generating trajectory samples.
- [x] Add optional `aresim.trajectory.v1` recording and validation with unified, UI-loadable `aresim.trajectory.episode.v1` artifacts.
- [x] Add reproducible frozen-policy evaluation over validation/test seeds through the shared rollout path.

## 5. RLlib training and analysis — implementation present, reference acceptance run pending

- [x] Consolidate existing PPO, DQN and MuZero reference runs into one results directory per algorithm; update reference configs, plotting defaults, learnings and evaluation paths while preserving frozen experiment identities and native checkpoint contents. See [results layout](../../results/README.md).

- [ ] Verify the implemented RLlib/Ray Tune experiment registry and strict configuration with the full dev acceptance run.
- [ ] Verify the implemented action-masked PPO RLModule, learner update, and frozen evaluation with the full dev acceptance run.
- [ ] Verify the implemented canonical W&B logging, local report plots, and reproducibility artifacts with the full dev acceptance run.
- [x] Verify native RLlib checkpoint loading through the shared `Agent` interface with the full dev acceptance run.
- [x] Wire UI Algorithm mode to server-side policy attach and agent-step (baselines, Decision Models, masked PPO/DQN checkpoint path).
- [x] Add mask-aware DQN through RLlib (selection, epsilon-greedy, and TD-target masking).
- [x] Verify corrected-reward MuZero reference wiring on CPU with the reference architecture and 50 simulations/action: collection, learner updates, checkpoint reload and legal sampled inference; long-budget CUDA mission learning remains pending.
- [x] Collate MuZero replay histories on CPU before batched device transfer; local pipeline checks and CUDA tensor/update equivalence pass. An idle A30 batch-256 learner improved from 140 to 68 ms; bounded CUDA training completed 4,208 steps/420 updates and legal sampled checkpoint reload. See [world-model scaling analysis](../rl/world_model_scaling_2026-10-11.md); parallel search is implemented below, while long-budget learning acceptance remains pending.
- [x] Reassess Ray/RLlib and native model-based backends for MuZero, DreamerV3, Plan2Explore and JEPA; inspect installed RLlib 2.58 and verify Dreamer structured-observation/curiosity integration limits.
- [x] Make Ray Core the sole MuZero collection path (no backend selector or serial/local fallback), with batched MCTS/reanalysis, bounded policy age and preserved update credit, with partial-episode CPU/Ray/CUDA resume verification. A bounded same-A30 comparison measured 13.52 → 111.26 steps/s (8.2×), exceeding the 5× diagnostic promotion gate; long-budget throughput and held-out learning remain unverified. See [implementation and evidence](../rl/muzero_parallel_2026-10-11.md).
- [x] Correct MuZero checkpoint rotation/resume by numeric step and use self-normalized importance-weighted loss with unweighted loss, weight statistics and effective-sample-size reporting. Existing checks and real-replay gradient/ordering diagnostics pass; task-learning improvement remains unverified. See [October 13 corrections](../rl/muzero_learning_fixes_2026-10-13.md).
- [x] Implement actual-minibatch full-unroll MuZero reanalysis, positive rare-action legality weighting and stopped-target temporal consistency with target age/coverage/legality diagnostics. Existing checks and tiny Ray partial resume pass; no GPU benchmark/training launched. See [learning-path settings and verification](../rl/muzero_learning_v3_2026-10-13.md).
- [ ] Complete MuZero reference acceptance: the dedicated PyTorch model/search/replay path, smoke run, resume, frozen evaluation, and UI attach are implemented; long-budget held-out comparison remains.
- [x] Extract shared RLlib environment/metrics/Tune plumbing, common encoder dimensions/inventories, and registered config/runner dispatch; preserve existing public aliases and sidecar formats.
- [x] Add masked Recurrent DQN through RLlib with LSTM sequence replay/burn-in, seeded replay checkpoint state, native budget-extension resume and frozen/UI inference; CPU smoke verified.
- [x] Add symbolic DreamerV3 through PyTorch/Ray with sequence replay, categorical RSSM, imagined actor-critic, learned imagined legality and exact same-source CPU partial-episode resume diagnostics.
- [x] Add Plan2Explore reward-free disagreement pretraining, separate task adaptation, frozen parent lineage, phase-local resume and matched-budget reference presets; both CPU smoke phases verified.
- [x] Verify new CPU smoke/inference, manual evaluation/recorded trajectories, optional imports and frontend build; publish permanent workflows and usage.
- [ ] Complete multi-seed held-out mission-learning and same-hardware CUDA throughput comparisons for Recurrent DQN, DreamerV3 and Plan2Explore.

The [recurrent and world-model guide](../../engine/aresim/algorithms/README.md) records implementation details, shared ownership, phase commands, local verification and pending research acceptance. The temporary planning document has been retired.

## 6. Advanced research — later

- [x] Add crop-derived resource/home memory and visited-cell observations.
- [ ] Complete controlled comparisons of memory, stateless observations and recurrence.
- [ ] Add JEPA representation learning and reusable encoder artifacts.
- [ ] Add learned world-model experiments without replacing authoritative simulator transitions.
- [ ] Add offline learning and behavior-cloning experiments where useful.
- [x] Consolidate llm_dm: Decision Model or LLM Actor, shared observed Memory, required LLM Learner, persistent advice, bounded providers, compact logs and native replay export.
- [ ] Add LLM + scripted, LLM + RL, JEPA + RL, and world-model + RL hybrids.

## 7. Multi-rover environment — later

- [ ] Add multiple rovers with stable agent IDs.
- [ ] Add simultaneous joint actions and deterministic movement/resource conflict rules.
- [ ] Extend the existing PettingZoo Parallel contract from one rover to many.
- [ ] Add per-agent, team, coordination, utilization, and fairness metrics.
- [ ] Add RLlib multi-agent baselines such as shared-policy IPPO and MAPPO-style training.
- [ ] Add selected RLlib multi-agent research experiments where lower-level control is useful.

## 8. Packaging and platform work — before a public release

- [ ] Stabilize the public Python factory and configuration APIs.
- [ ] Add concise extension guides for observations, actions, rewards, tasks, and algorithms.
- [ ] Add validated experiment configuration files and complete reproducibility manifests.
- [ ] Define package compatibility, deprecation, and release-version policies.
- [ ] Add CI coverage for supported Python/Node versions and focused browser flows.
- [ ] Prepare installable package metadata, examples, tutorials, and a public release guide.

## 9. Optional future work

- [ ] Persistent database and server-side save library, if multi-user hosting requires them.
- [ ] Authentication and deployment hardening, if AresSim becomes a hosted service.
- [ ] WebSocket transport, only if measured remote live-streaming needs justify it.
- [ ] Add Ray Tune search spaces beyond the implemented single-trial lifecycle when a concrete sweep is designed.
- [ ] JAX, RLlib alternatives, or remote plugin discovery only for demonstrated use cases.

## Related documents

- [Environment Rules Reference](../product/environment_rules.md)
- [UI Design Reference](../product/ui_design.md)
- [Trajectory Episode and Replay Projection](../product/gameplay_save_format.md)
- [Engine Code Reference](../product/engine_code_reference.md)
- [RL Algorithms, Training, and Evaluation](../rl/rl_quickstart.md)
- [Agent Data, RL, and LLM Architecture Proposal](../rl/agent_data_rl_llm_proposal.md)
- [Algorithm Literature Survey](../rl/algorithm_literature_survey.md)


## October 6 pipeline corrections

- [x] Remove repeatable pad-progress rewards and enforce discounted potential/cutoff invariants.
- [x] Freeze numerical environment and seed/source provenance per run.
- [x] Provide held-out validation with labeled action modes and best-checkpoint preservation for separate manual analysis (automatic training evaluations removed October 7).
- [x] Report success, deliveries, action counts and Extract opportunities by episode.
- [x] Correct zero-Q DQN exploration and add MuZero learned constraints/reanalysis.
- [x] Cover small delivery diagnostics and validation targets in regression fixtures; temporary runnable presets removed.
- [x] Keep training fast: checkpoint saving only, no periodic/final evaluation or automatic reports; retain manual evaluation commands.
- [ ] Demonstrate robust full-mission learning across multiple learner seeds under the frozen v2 contracts.

Details and local verification: [pipeline corrections](../rl/pipeline_fixes_2026-10-06.md).

## November 7, 2026 cleanup and GPU tooling

- [x] Consolidate policy catalog metadata, configurable recharge projection and task-goal observation ownership across the environment/API/UI.
- [x] Consolidate shared checkpoint transport, provenance validation, loader dispatch, experiment decoding, numeric checkpoint ordering and public resume checks.
- [x] Recover the original fixed seed manifests under `configs/seeds/` and use their durable paths in active presets and guides.
- [x] Share source payload allowlists and remote dependency setup across GPU providers; add bounded Vast.ai SSH training, scoped self-stop and verified artifact recovery scripts under `scripts/vastai/`.
- [x] Rename all eight algorithm workflow documents to `DETAILS.md`; add fourteen editable draw.io PNGs showing architecture dimensions and training/inference steps beside their algorithms.
- [x] Validate with real environment/API/replay checks, seven short CPU training phases with checkpoint reload/inference, the frontend build and 17 existing desktop browser end-to-end cases. No unit tests were created or rewritten.
- [ ] Verify the new Vast.ai launcher on a live rented GPU and measure CUDA throughput. Script implementation and local checks do not establish live provider behavior or long-budget learning quality.

Decision logs: [environment/UI](refactor_environment.md), [training](refactor_training.md), [cloud](refactor_cloud.md), and [documentation](refactor_documentation.md). Start GPU work with the [Vast.ai run guide](../rl/vastai.md); inspect model data flow in the [illustrated algorithm guide](../../engine/aresim/algorithms/README.md).

## November 7, 2026 training performance

- [x] Profile MuZero, DreamerV3, both Plan2Explore phases and recurrent DQN using real simulator histories.
- [x] Optimize replay packing, recurrent reconstruction, inference publication, CUDA synchronization, scalar losses and burn-in graphs without reducing training budgets.
- [x] Add a bounded full-pipeline benchmark script to cloud payloads.
- [ ] Measure sustained throughput on the target Vast.ai NVIDIA GPU; local CPU measurements and smoke verification are documented in the [performance report](../rl/training_performance.md).
