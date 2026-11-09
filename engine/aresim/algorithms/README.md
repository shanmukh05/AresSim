# Algorithm guide

The environment owns observations, action legality, rewards and termination. Every policy ultimately returns one integer in `Discrete(10)`; algorithms consume policy-visible observations rather than canonical simulator state.

| Algorithm / policy | Model and execution | Illustrated details |
|---|---|---|
| Masked PPO | Shared CNN/telemetry actor-critic; on-policy RLlib updates | [PPO](ppo/DETAILS.md) |
| Masked DQN | Double/dueling Q heads; RLlib episode replay | [DQN](dqn/DETAILS.md) |
| Recurrent DQN | Shared encoder + 256-wide LSTM; sequence replay / burn-in | [Recurrent DQN](recurrent_dqn/DETAILS.md) |
| MuZero | History representation + learned dynamics + batched PUCT | [MuZero](muzero/DETAILS.md) |
| DreamerV3 | Categorical recurrent world model + imagined actor/critic | [DreamerV3](dreamerv3/DETAILS.md) |
| Plan2Explore | Dreamer model + disagreement ensemble; pretrain then adapt | [Plan2Explore](plan2explore/DETAILS.md) |
| llm_dm | Decision Model or LLM Actor + shared Memory + mandatory LLM Learner | [llm_dm](llm_dm/DETAILS.md) |
| Baselines | Wait, random, random-valid and scripted; inference only | [Baselines](baselines/DETAILS.md) |

Each learned algorithm's details include a tensor-dimension architecture PNG and a training-flow PNG. The llm_dm details embed its workflow diagrams; baselines have inference-flow PNGs. All PNGs embed editable draw.io XML and live in each algorithm’s `diagrams/` folder. Dimensions are the default/reference configuration; smoke presets intentionally use smaller models.

## Shared ownership

- `common/`: encoder/model dimensions, mask validation, configuration decoding and tracking.
- `catalog.py`: dependency-free policy IDs, display labels and live-controller kinds used by API policy integration.
- `registry.py`: algorithm factories, model factories, frozen loaders and shared training contracts.
- `../training/rllib/`: RLlib environment adapters, callbacks, checkpoint integration and Tune lifecycle for PPO/DQN/Recurrent DQN.
- `../training/world_models/`: Dreamer/Plan2Explore real collection, sequence replay, learning and lifecycle.
- `muzero/`: native learned-search model, search, collection, replay and trainer.

See [RL usage](../../../docs/rl/usage.md) for CLI commands, [GPU runs on Vast.ai](../../../docs/rl/vastai.md), and [training configurations](../../../configs/). Training writes checkpoints without automatically evaluating held-out seeds. Invoke `aresim-rl evaluate` explicitly for frozen validation/test analysis.
