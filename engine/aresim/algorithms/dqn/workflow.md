# Masked DQN workflow

Implementation reference for `masked_dqn` and `local_cnn_q`. Off-policy **Double Dueling DQN** (van Hasselt 2016 + Wang 2016; n-step=3 on the reference config) trains through Ray RLlib on the same Gym env, seeds, masks, and evaluation path as masked PPO.

W&B run name is `{experiment_id}-{trial_id}`. The reference trial uses `experiment_id: rllib_masked_double_dueling_dqn_reference`, `group: double_dueling_dqn`, and checkpoint `policy_id` `{experiment_id}:{trial_id}:{label}`. Registry name stays `masked_dqn`.

Simulator rules stay in `aresim.core` and `aresim.components`. Illegal actions are excluded from greedy selection, epsilon-greedy, and Double-DQN targets by setting those Q-values to `-inf` inside `compute_q_values`.

**Source ownership**

| File | Responsibility |
|---|---|
| `config.py` | `MaskedDQNConfig`, YAML decode |
| `train.py` | `LocalMaskedQNetwork`, `AresMaskedDQNRLModule`, `MaskedDQNFactory` |
| `checkpoint.py` | `rllib_masked_dqn` loader |
| `../common/encoder.py` | Shared local CNN encoder and `mask_illegal_actions` |
| `../registry.py` | Registers `masked_dqn`, `local_cnn_q`, `rllib_masked_dqn` |

**CLI:** `aresim-rl train configs/masked_dqn/reference.yaml` (first reference: seed_1). Smoke: `aresim-rl train configs/masked_dqn/smoke.yaml`.

Replay is RLlib's `EpisodeReplayBuffer`. Online fragments are not `aresim.trajectory.v1` artifacts. Frozen eval uses `RolloutRunner` like PPO.

`rollout_batch_size` is environment steps per Tune iteration (`min_sample_timesteps_per_iteration`), **not** the per-runner collection length. `rollout_fragment_length` controls that collection length; `train_batch_size` is the replay minibatch. `training_intensity` requests the replay-to-new-sample ratio. RLlib rounds its sampling/training weights, so inspect `num_training_step_calls_per_iteration`, sampled steps, and trained samples in each run instead of assuming the requested ratio is exact. The checked-in smoke, dev, and reference configurations target roughly four replayed samples per new environment step. n-step is 1 in smoke/dev and 3 in the reference config. Prioritized replay, C51, and recurrence are out of scope here.

RLlib training uses the non-audit environment path: observations, masks, tasks, and rewards are unchanged, while per-step full-world checksums and the post-step state copy are omitted. The returned transition state is borrowed until the next step. Frozen summary evaluation uses the same path; recorded trajectories and the ordinary engine/API retain full checksums and defensive copies. Compare both environment steps per second and validation reward when tuning collection length or replay intensity.
