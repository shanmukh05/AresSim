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
| [RL Algorithms, Training, and Evaluation](rl/rl_quickstart.md) | Implemented policies, PPO, network architecture, online sampling, W&B, checkpoints, and evaluation |
| [RL Usage Guide](rl/usage.md) | CLI training, W&B setup, baselines, trajectories, and extension contracts |
| [Masked PPO training notebook](../notebooks/masked_ppo_training.ipynb) | End-to-end learned-policy training in Jupyter |
| [Algorithm Pipeline Notebook](../notebooks/rl_algorithm_pipeline.ipynb) | Baseline rollout, trajectory, validation, and determinism smoke test |
| [Evaluation Report Template](../notebooks/evaluation_report_template.ipynb) | Optional manual W&B analysis notebook |
| [Agent Data, RL, and LLM Proposal](rl/agent_data_rl_llm_proposal.md) | Observations, actions, adapters, datasets, LLM interface |
| [Algorithm Literature Survey](rl/algorithm_literature_survey.md) | Algorithms worth testing and research priorities |

## Project — status and contributor rules

| Document | Purpose |
|---|---|
| [Implementation Checklist](project/implementation_checklist.md) | What is done vs next vs future |

Coding standards live in the always-on Cursor rule [`.cursor/rules/coding-standards.mdc`](../.cursor/rules/coding-standards.mdc).

## Suggested reading order

1. New to the product: Environment Rules → UI Design → Gameplay Save Format
2. Working on the Python backend: Engine Code Reference → Implementation Checklist
3. Starting RL work: [RL Usage Guide](rl/usage.md) (CLI + W&B) → [RL Algorithms, Training, and Evaluation](rl/rl_quickstart.md) → Agent Data proposal
4. Exploring algorithms: Algorithm Literature Survey
