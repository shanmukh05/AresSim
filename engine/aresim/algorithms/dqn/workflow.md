# Masked DQN workflow

Implementation reference for `masked_dqn` and `local_cnn_q`. Off-policy Double/dueling DQN trains through Ray RLlib on the same Gym env, seeds, masks, and evaluation path as masked PPO.

Simulator rules stay in `aresim.core` and `aresim.components`. Illegal actions are excluded from greedy selection, epsilon-greedy, and Double-DQN targets by setting those Q-values to `-inf` inside `compute_q_values`.

**Source ownership**

| File | Responsibility |
|---|---|
| `config.py` | `MaskedDQNConfig`, YAML decode |
| `train.py` | `LocalMaskedQNetwork`, `AresMaskedDQNRLModule`, `MaskedDQNFactory` |
| `checkpoint.py` | `rllib_masked_dqn` loader |
| `../common/encoder.py` | Shared local CNN encoder and `mask_illegal_actions` |
| `../registry.py` | Registers `masked_dqn`, `local_cnn_q`, `rllib_masked_dqn` |

**CLI:** `aresim-rl train configs/masked_dqn/smoke.yaml`

Replay is RLlib's `EpisodeReplayBuffer`. Online fragments are not `aresim.trajectory.v1` artifacts. Frozen eval uses `RolloutRunner` like PPO.

`rollout_batch_size` is environment steps per Tune iteration (`min_sample_timesteps_per_iteration`). `train_batch_size` is the replay minibatch. n-step is 1 in smoke/dev and 3 in the reference config. Prioritized replay, C51, and recurrence are out of scope here.
