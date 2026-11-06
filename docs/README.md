# AresSim documentation

Canonical and planned project docs, grouped by purpose. Historical research notes live in [`design_docs/`](../design_docs/).

## Product — implemented Phase 1 contracts

| Document | Purpose |
|---|---|
| [Environment Rules](product/environment_rules.md) | Simulator semantics: terrain, actions, rewards, terminals |
| [UI Design](product/ui_design.md) | Orthographic 3D shell, HUD, modes, renderer boundary |
| [Trajectory Episode and Replay Projection](product/gameplay_save_format.md) | Unified portable trajectory export (`aresim.trajectory.episode.v1`) and legacy replay compatibility |
| [Engine Code Reference](product/engine_code_reference.md) | Maintained `engine/` file map and ownership |

## RL — agents, training, and research

| Document | Purpose |
|---|---|
| [RL Algorithms, Training, and Evaluation](rl/rl_quickstart.md) | Implemented policies, all six learned algorithms, network architecture, online sampling, W&B, checkpoints, and evaluation |
| [RL Usage Guide](rl/usage.md) | CLI training, W&B setup, baselines, trajectories, Jev, and extension contracts |
| [Results layout](../results/README.md) | Consolidated algorithm folders, trial naming and preserved historical provenance |
| [Pipeline corrections and verification](rl/pipeline_fixes_2026-10-06.md) | Reward v2, observed memory, frozen provenance, MuZero fixes, checkpoint-only training, and diagnostic evidence |
| [PPO seed 33/35 follow-up](rl/ppo_followup_2026-10-07.md) | Sampled-versus-greedy failure, successful mission learning, survival traps, and next experiments |
| [DQN seed 3 follow-up](../results/rllib_masked_double_dueling_dqn_reference/seed_3/learnings.md#october-8-follow-up-what-actually-remains-broken) | Frozen Q/action probes, learned camping, and the implemented capped sample-credit experiment |
| [MuZero workflow](../engine/aresim/algorithms/muzero/DETAILS.md) | Learned search model, replay, training, checkpoints, and inference |
| [MuZero checkpoint and loss corrections](rl/muzero_learning_fixes_2026-10-13.md) | Numeric checkpoint ordering, stable weighted loss, unweighted/replay diagnostics and fresh-run command |
| [MuZero learning-path changes](rl/muzero_learning_v3_2026-10-13.md) | Full-unroll replay reanalysis, rare-action legality balancing, temporal consistency and local verification |
| [Parallel MuZero implementation](rl/muzero_parallel_2026-10-11.md) | Ray collectors, batched search/reanalysis, partial-episode resume, A30 measurements and run commands |
| [World-model execution and framework choice](rl/world_model_scaling_2026-10-11.md) | MuZero timing evidence, Ray/RLlib alternatives, batched search, and Dreamer/Plan2Explore/JEPA integration |
| [Train on Vast.ai](rl/vastai.md) | Bounded GPU setup/training, scoped self-stop, allowlisted artifacts and recovery |
| [Illustrated algorithm guide](../engine/aresim/algorithms/README.md) | Architecture dimensions, training steps and inference flows for every implemented policy |
| [Train on JarvisLabs](rl/jarvislabs.md) | GPU/CPU instances, code upload, remote setup, and `results/` download |
| [Masked PPO training notebook](../notebooks/masked_ppo_training.ipynb) | End-to-end learned-policy training in Jupyter |
| [Algorithm Pipeline Notebook](../notebooks/rl_algorithm_pipeline.ipynb) | Baseline rollout, trajectory, validation, and determinism smoke test |
| [Evaluation Report Template](../notebooks/evaluation_report_template.ipynb) | Optional manual W&B analysis notebook |
| [Agent Data, RL, and LLM Proposal](rl/agent_data_rl_llm_proposal.md) | Observations, actions, adapters, datasets, LLM interface |
| [Experience-Driven Agent Learning Plan](llm/experience_driven_llm_agent_learning_plan.md) | Objective, workflow, component experiments, controller research, implementation, and cross-game tests |
| [LLM Agent Literature Review](llm/literature_survey.md) | Paper analyses, evidence cards, limitations, and component reading map |
| [Algorithm Literature Survey](rl/algorithm_literature_survey.md) | Algorithms worth testing and research priorities |
| [Recurrent DQN, DreamerV3, and Plan2Explore](../engine/aresim/algorithms/README.md) | Illustrated algorithm details, shared ownership, phase commands and masks/memory |

## Project — status and contributor rules

| Document | Purpose |
|---|---|
| [Implementation Checklist](project/implementation_checklist.md) | What is done vs next vs future |
| [Environment/UI refactor decisions](project/refactor_environment.md) | Inventory, ownership changes, deliberate compatibility cases and manual validation |
| [Training refactor decisions](project/refactor_training.md) | Shared checkpoint/configuration/resume ownership and CPU training evidence |
| [Cloud refactor decisions](project/refactor_cloud.md) | Provider payload/setup ownership, bounded execution and recovery contracts |
| [Documentation decisions](project/refactor_documentation.md) | Illustrated algorithm details, dimension sources and PNG verification |

Coding standards live in the always-on Cursor rule [`.cursor/rules/coding-standards.mdc`](../.cursor/rules/coding-standards.mdc).

## Suggested reading order

1. New to the product: Environment Rules → UI Design → Gameplay Save Format
2. Working on the Python backend: Engine Code Reference → Implementation Checklist
3. Starting RL work: [RL Usage Guide](rl/usage.md) (CLI + W&B) → [Train on JarvisLabs](rl/jarvislabs.md) or [Vast.ai](rl/vastai.md) for cloud GPUs → [RL Algorithms, Training, and Evaluation](rl/rl_quickstart.md) → [Jev workflow](../engine/aresim/algorithms/jev/DETAILS.md) → Agent Data proposal
4. Exploring algorithms: [Recurrent DQN, DreamerV3, and Plan2Explore](../engine/aresim/algorithms/README.md) → Algorithm Literature Survey
