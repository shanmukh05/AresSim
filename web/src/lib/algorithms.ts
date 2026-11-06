/** Live-controller metadata shared by selectors, capability gating, and policy helpers. */

type Capability = "rllib" | "muzero" | "world_models" | "jev";
type ActionSelection = "q" | "policy" | null;
interface AlgorithmDefinition {
  label: string;
  kind: "baseline" | "llm" | "checkpoint";
  capability: Capability | null;
  actionSelection: ActionSelection;
}

export const ALGORITHM_CATALOG = {
  random: { label: "Random", kind: "baseline", capability: null, actionSelection: null },
  random_valid: { label: "Random (valid)", kind: "baseline", capability: null, actionSelection: null },
  wait: { label: "Wait", kind: "baseline", capability: null, actionSelection: null },
  scripted: { label: "Scripted", kind: "baseline", capability: null, actionSelection: null },
  jev: { label: "Jev", kind: "llm", capability: "jev", actionSelection: null },
  masked_ppo: { label: "Masked PPO", kind: "checkpoint", capability: "rllib", actionSelection: "policy" },
  masked_dqn: { label: "Double Dueling DQN", kind: "checkpoint", capability: "rllib", actionSelection: "q" },
  muzero: { label: "MuZero", kind: "checkpoint", capability: "muzero", actionSelection: "policy" },
  recurrent_dqn: { label: "Recurrent DQN", kind: "checkpoint", capability: "rllib", actionSelection: "q" },
  dreamerv3: { label: "DreamerV3", kind: "checkpoint", capability: "world_models", actionSelection: "policy" },
  plan2explore: { label: "Plan2Explore", kind: "checkpoint", capability: "world_models", actionSelection: "policy" },
} as const satisfies Record<string, AlgorithmDefinition>;

export type AlgorithmId = keyof typeof ALGORITHM_CATALOG;
export const ALGORITHMS = Object.entries(ALGORITHM_CATALOG).map(([value, definition]) => ({
  value: value as AlgorithmId,
  ...definition,
}));
export const CAPABILITY_REQUIREMENTS: Record<Capability, string> = {
  rllib: "rllib required",
  muzero: "muzero required",
  world_models: "world_models required",
  jev: "API key required",
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

export function usesJev(algorithmId: AlgorithmId): boolean {
  return ALGORITHM_CATALOG[algorithmId].kind === "llm";
}
