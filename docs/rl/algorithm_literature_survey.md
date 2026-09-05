# AresSim Algorithm Literature Survey and Research Proposals

Last updated: 2026-08-02

Status: Short research guide. It does not change the composed environment, single-rover Gymnasium adapter, future PettingZoo multi-agent contract, or RLlib training architecture in [RL Algorithms, Training, and Evaluation](rl_quickstart.md).

## 1. What makes AresSim different

AresSim is currently a single-rover, partially observable environment with:

- an `8 x 8` local observation inside a hidden `32 x 32` world;
- a small masked `Discrete(10)` action space;
- long tasks involving exploration, collection, unloading, building, and service;
- battery, health, payload, power, and livability trade-offs;
- procedurally generated seeds and deterministic replay;
- manual, scripted, and eventually large offline trajectory datasets.

This favors discrete-action methods with masks, memory, long-horizon reasoning, and strong generalization across seeds. It does **not** currently favor continuous-control algorithms or policies trained only from rendered pixels.

## 2. Algorithms worth testing

The table separates complete control algorithms from supporting methods. JEPA, curriculum learning, and exploration bonuses are useful additions to an agent; they are not standalone policies.

| Candidate | Role in AresSim | Why it is worth testing | Priority |
|---|---|---|---|
| Random-valid, scripted, and oracle planners | Non-learning baselines | Catch environment bugs and provide lower and upper reference points. | Essential |
| Action-masked PPO in RLlib | First learned controller and pipeline-parity baseline | A tested RLModule uses the same mask in exploration, inference, and training. | Essential |
| Mask-aware DQN | Off-policy comparison | Reuses experience and directly matches discrete actions. Invalid actions must be masked during both selection and target calculation. Rainbow components can be added only after plain DQN works. | Essential |
| Recurrent PPO or recurrent DQN | Memory under partial observability | An LSTM can remember discovered ice, routes, the pad location, and earlier warnings that have left the local view. A small sequence-replay agent can borrow R2D2's recurrent replay rules without copying its distributed training system. | High |
| Gated Transformer-XL policy | Longer memory alternative | May retain longer histories than an LSTM, but should be tested only after a recurrent baseline because it is heavier and more sensitive. | Medium |
| Prioritized Level Replay (PLR) | Seed curriculum | AresSim seeds act like procedural levels. PLR revisits seeds with high learning potential instead of sampling all seeds uniformly. | High |
| Constrained PPO/CMDP agent | Safety-aware control | Treat critical battery, rover damage, or prolonged colony failure as logged costs and constraints instead of hiding every concern inside one reward. | High after PPO |
| Goal-conditioned DQN/PPO with HER | Multi-mission reuse | When tasks expose explicit goals, failed episodes can be relabeled with goals they actually achieved. This is less useful for the current open-ended task and should wait for a goal API. | Medium later |
| Behavior cloning, then IQL | Learn from saved runs | Behavior cloning is the simplest use of scripted/manual replays. IQL is a reasonable offline-RL follow-up because it avoids evaluating unseen actions during its value update. | High once data exists |
| Decision Transformer | Offline sequence baseline | Tests whether long trajectory context and desired return are useful. It needs a large, diverse dataset and should not replace simpler behavior cloning/IQL baselines. | Medium later |
| Self-predictive/JEPA-style encoder | Representation pretraining | Predict future latent observations from history and actions, then reuse the encoder in PPO or DQN. This can learn map dynamics without reconstructing every raw cell. | High research value |
| DreamerV3-style world model | Latent imagination | Learns a recurrent latent dynamics model and trains behavior in imagined rollouts. Its discrete-domain results make it more relevant than continuous-only planners, but masks and partial observations need explicit treatment. | High research value |
| MuZero-style planning | Learned search model | The 10-action space makes tree search feasible. MuZero predicts reward, policy, and value for planning without reconstructing the full world, but it is substantially more complex than Dreamer or PPO. | Medium research value |
| Plan2Explore-style agent | Reward-free exploration | Uses world-model disagreement to seek informative regions before task-specific training. This fits randomized maps and sparse discovery rewards. | Medium research value |
| Options / hierarchical RL | Long-horizon skills | A manager selects skills such as Explore, Collect, Return, Unload, Build, and Service; a validated low-level policy executes them. Option-Critic is one learned approach. | High after skills exist |
| LLM manager + RL/scripted executor | Semantic planning | An LLM can choose subgoals or explain failures, while a grounded executor handles every rover action. SayCan supports this planner/affordance split; direct per-step LLM control is not recommended. | Medium research value |

## 3. AresSim-specific research proposals

These are proposed AresSim experiments, not claims of entirely new general-purpose algorithms.

### 3.1 Mars-JEPA: action-conditioned predictive representation

Train an encoder on trajectory windows without using reward as its main target:

```text
recent local observations + action sequence
                    -> predict future latent observations
                    -> reuse the encoder in PPO or DQN
```

Use short and medium prediction distances so the representation must capture both rover movement and slower colony changes. This is closer to Self-Predictive Representations than applying image-only I-JEPA unchanged. Compare a frozen encoder, a fine-tuned encoder, and an identical randomly initialized encoder.

### 3.2 Safety-aware latent world model

Extend a Dreamer-style latent model with separate predictions for normal reward and safety costs such as critical battery, rover damage, or service failure. The policy may imagine outcomes, but the authoritative AresSim engine still validates and executes the selected action. The useful question is whether prediction improves sample efficiency and risk management—not whether the learned model can replace the simulator.

### 3.3 Event-skill hierarchy

Define a small reusable skill set around existing events:

```text
Explore -> Collect -> Return -> Unload -> Build -> Service
```

Compare three managers over the same low-level executor: scripted rules, learned options, and an LLM. This isolates the value of high-level planning from low-level navigation and gives future hybrid agents a fair comparison.

### 3.4 Curriculum plus representation transfer

Combine PLR with a shared encoder: train on seeds chosen by learning progress, then test on permanently held-out seeds and harder weather/resource distributions. Compare ordinary PPO, PPO+PLR, Mars-JEPA+PPO, and Mars-JEPA+PPO+PLR. This directly tests whether better representations and better seed selection provide complementary gains.

## 4. Recommended experiment order

| Stage | Experiments | Main question |
|---:|---|---|
| 1 | Random-valid, scripted, oracle, action-masked PPO in RLlib | Is the environment learnable, are masks/rewards correct, and do the adapters preserve the contract? |
| 2 | Recurrent PPO, recurrent DQN, discovered-map policy | How much does memory help under the local view? |
| 3 | PLR and constrained PPO | Can policies generalize across seeds while respecting safety costs? |
| 4 | Behavior cloning and IQL | Can saved manual/scripted experience reduce online interaction? |
| 5 | Mars-JEPA and SPR-style pretraining | Does predictive representation learning improve data efficiency and transfer? |
| 6 | DreamerV3-style model, then Plan2Explore or MuZero | Does learned prediction or planning beat strong model-free baselines? |
| 7 | Learned skill manager and LLM manager | Does hierarchy improve long missions without bypassing validation? |
| 8 | MAPPO/QMIX or other multi-agent methods | Only after multiple rovers and simultaneous-action rules exist. |

Do not run every combination. Promote a method only when it beats the simpler stage before it on held-out seeds, with comparable environment steps and reported compute.

## 5. Approaches to postpone

- **SAC and TD-MPC2:** primarily designed for continuous control; AresSim currently has ten discrete actions.
- **Pixel-only agents:** discard useful symbolic state and add rendering noise and compute.
- **Direct I-JEPA on screenshots:** useful only for a separate visual-agent benchmark; symbolic trajectory prediction is a better first JEPA experiment.
- **World models as simulator truth:** learned predictions must never mutate canonical state or define evaluation outcomes.
- **LLM control every step:** expensive, slow, and difficult to reproduce; use the LLM as a low-frequency manager.
- **Multi-agent algorithms now:** there is only one rover and no simultaneous-action conflict contract.
- **Large sequence models before enough data exists:** Decision Transformer and similar methods need broad, good-quality trajectories.

## 6. How to compare algorithms fairly

For every experiment, report:

- success and sparse evaluation return on fixed held-out seeds;
- environment steps, wall-clock time, and approximate compute;
- invalid-action selections before masking;
- battery/health failures and other safety-cost totals;
- sample, ice, and ore delivery; build/service progress; and payload utilization;
- performance under new seeds, weather, and resource distributions;
- ablations that remove the proposed memory, model, curriculum, or representation module.

Use the same observation, action, reward, seed, and episode-limit manifests across comparisons. Shaped rewards may train an agent, but final ranking should include frozen evaluation semantics. Use environment transitions—not training iterations, sampled batches, or optimizer steps—as the x-axis, and report team/per-agent coordination metrics once multiple rovers exist.

## 7. Primary literature

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://ojs.aaai.org/index.php/AAAI/article/view/11796)
- [Recurrent Experience Replay in Distributed Reinforcement Learning (R2D2)](https://openreview.net/pdf?id=r1lyTjAqYX)
- [Stabilizing Transformers for Reinforcement Learning (GTrXL)](https://proceedings.mlr.press/v119/parisotto20a.html)
- [Prioritized Level Replay](https://proceedings.mlr.press/v139/jiang21b.html)
- [Constrained Policy Optimization](https://proceedings.mlr.press/v70/achiam17a.html)
- [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)
- [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169)
- [Decision Transformer](https://proceedings.neurips.cc/paper/2021/hash/7f489f642a0ddb10272b5c31057f0663-Abstract.html)
- [Data-Efficient Reinforcement Learning with Self-Predictive Representations](https://arxiv.org/abs/2007.05929)
- [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html)
- [Mastering Diverse Domains through World Models (DreamerV3)](https://arxiv.org/abs/2301.04104)
- [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://arxiv.org/abs/2310.16828)
- [MuZero: Planning with a Learned Model](https://www.nature.com/articles/s41586-020-03051-4)
- [Planning to Explore via Self-Supervised World Models](https://arxiv.org/abs/2005.05960)
- [The Option-Critic Architecture](https://arxiv.org/abs/1609.05140)
- [Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://proceedings.mlr.press/v205/ichter23a.html)
- [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291)
