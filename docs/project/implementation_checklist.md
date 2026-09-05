# AresSim Implementation Checklist

**Last updated:** August 31, 2026  
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
- [x] Manual play, server-driven Algorithm autoplay (baselines + masked PPO checkpoints), and backend-owned Replay flow.

Algorithm mode attaches registered baselines or a masked PPO checkpoint sidecar through `POST /api/sessions/{id}/attach-policy` and steps with `POST /api/sessions/{id}/agent-step`. Inference stays on the API host; the UI sends algorithm selection and an optional dev-time checkpoint path.

## 3. RL-ready environment — completed

- [x] Add the framework-neutral `AresEnvironment` composition boundary.
- [x] Add the configurable local 8×8 rover observation and edge padding.
- [x] Add the flat 10-action RL adapter and legal-action mask.
- [x] Add task/termination and reward components without duplicating core rules.
- [x] Add the single-rover `AresGymEnv` adapter and pass Gymnasium checks.
- [x] Add the one-rover PettingZoo Parallel environment and contract tests.
- [x] Verify direct engine, Gymnasium, and PettingZoo transition parity.

## 4. Rollouts and baseline agents — partially completed

- [x] Add external episode truncation and explicit environment/agent seeds for rollouts.
- [x] Add the versioned fixed train, validation, and test seed manifest for the current scenario.
- [x] Add random, random-valid, Wait, and scripted baseline agents.
- [x] Add a simple rollout runner for generating trajectory samples.
- [x] Add optional `aresim.trajectory.v1` recording and validation with unified, UI-loadable `aresim.trajectory.episode.v1` artifacts.
- [x] Add reproducible frozen-policy evaluation over validation/test seeds through the shared rollout path.

## 5. RLlib training and analysis — implementation present, reference acceptance run pending

- [ ] Verify the implemented RLlib/Ray Tune experiment registry and strict configuration with the full dev acceptance run.
- [ ] Verify the implemented action-masked PPO RLModule, learner update, and frozen evaluation with the full dev acceptance run.
- [ ] Verify the implemented canonical W&B logging, local report plots, and reproducibility artifacts with the full dev acceptance run.
- [x] Verify native RLlib checkpoint loading through the shared `Agent` interface with the full dev acceptance run.
- [x] Wire UI Algorithm mode to server-side policy attach and agent-step (baselines + masked PPO checkpoint path).
- [ ] Add mask-aware DQN and recurrent-policy experiments primarily through RLlib.

## 6. Advanced research — later

- [ ] Add discovered-map memory and compare it with local observations and recurrence.
- [ ] Add JEPA representation learning and reusable encoder artifacts.
- [ ] Add learned world-model experiments without replacing authoritative simulator transitions.
- [ ] Add offline learning and behavior-cloning experiments where useful.
- [ ] Add provider-neutral LLM agents with budgets, audit records, validation, and deterministic fallbacks.
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
