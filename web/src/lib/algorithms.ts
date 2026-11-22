/** Live-controller metadata shared by selectors, capability gating, and policy helpers. */

type Capability = "rllib" | "muzero" | "world_models" | "llm_dm";
type ActionSelection = "q" | "policy" | null;
export type AlgorithmGroup = "baseline" | "rl" | "llm";

interface AlgorithmDefinition {
  label: string;
  shortLabel: string;
  description: string;
  kind: "baseline" | "llm" | "checkpoint";
  group: AlgorithmGroup;
  capability: Capability | null;
  actionSelection: ActionSelection;
}

export const ALGORITHM_CATALOG = {
  random_valid: {
    label: "Random",
    shortLabel: "Random",
    description: "Uniformly samples among legal masked rover actions.",
    kind: "baseline",
    group: "baseline",
    capability: null,
    actionSelection: null,
  },
  scripted: {
    label: "Scripted",
    shortLabel: "Scripted",
    description: "Rule-based heuristic planner for ice, samples, and servicing.",
    kind: "baseline",
    group: "baseline",
    capability: null,
    actionSelection: null,
  },
  masked_ppo: {
    label: "Masked PPO",
    shortLabel: "PPO",
    description: "On-policy actor-critic with legal action masking.",
    kind: "checkpoint",
    group: "rl",
    capability: "rllib",
    actionSelection: "policy",
  },
  masked_dqn: {
    label: "Double Dueling DQN",
    shortLabel: "DDQN",
    description: "Off-policy value network with dueling advantage heads.",
    kind: "checkpoint",
    group: "rl",
    capability: "rllib",
    actionSelection: "q",
  },
  recurrent_dqn: {
    label: "Recurrent DQN",
    shortLabel: "R-DQN",
    description: "Memory-augmented Q-learning for partial observability.",
    kind: "checkpoint",
    group: "rl",
    capability: "rllib",
    actionSelection: "q",
  },
  muzero: {
    label: "MuZero",
    shortLabel: "MuZero",
    description: "Learned latent dynamics with MCTS lookahead planning.",
    kind: "checkpoint",
    group: "rl",
    capability: "muzero",
    actionSelection: "policy",
  },
  dreamerv3: {
    label: "DreamerV3",
    shortLabel: "DreamerV3",
    description: "RSSM world model trained via imagined rollouts.",
    kind: "checkpoint",
    group: "rl",
    capability: "world_models",
    actionSelection: "policy",
  },
  plan2explore: {
    label: "Plan2Explore",
    shortLabel: "Plan2Explore",
    description: "Curiosity-pretrained world model adapted to mission goals.",
    kind: "checkpoint",
    group: "rl",
    capability: "world_models",
    actionSelection: "policy",
  },
  llm_dm: {
    label: "LLM + Decision Model",
    shortLabel: "LLM + DM",
    description: "Decision Model / LLM actor paired with a live LLM learner.",
    kind: "llm",
    group: "llm",
    capability: "llm_dm",
    actionSelection: null,
  },
} as const satisfies Record<string, AlgorithmDefinition>;

export type AlgorithmId = keyof typeof ALGORITHM_CATALOG;
export const ALGORITHMS = Object.entries(ALGORITHM_CATALOG).map(([value, definition]) => ({
  value: value as AlgorithmId,
  ...definition,
}));
export const CAPABILITY_REQUIREMENTS: Record<Capability, string> = {
  rllib: "rllib extra required",
  muzero: "muzero extra required",
  world_models: "world_models extra required",
  llm_dm: "jsonschema required",
};

export function supportsActionSampling(algorithmId: AlgorithmId): boolean {
  return ALGORITHM_CATALOG[algorithmId].actionSelection !== null;
}

export function usesCheckpoint(algorithmId: AlgorithmId): boolean {
  return ALGORITHM_CATALOG[algorithmId].kind === "checkpoint";
}

export function usesWorldModel(algorithmId: AlgorithmId): boolean {
  return ALGORITHM_CATALOG[algorithmId].capability === "world_models";
}
