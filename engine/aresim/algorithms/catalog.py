"""Dependency-free policy identities and the public live-controller catalog."""

from enum import StrEnum


class AlgorithmId(StrEnum):
    RANDOM = "random"
    RANDOM_VALID = "random_valid"
    WAIT = "wait"
    SCRIPTED = "scripted"
    LLM_DM = "llm_dm"
    MASKED_PPO = "masked_ppo"
    MASKED_DQN = "masked_dqn"
    MUZERO = "muzero"
    RECURRENT_DQN = "recurrent_dqn"
    DREAMERV3 = "dreamerv3"
    PLAN2EXPLORE = "plan2explore"


_POLICY_METADATA = {
    AlgorithmId.RANDOM: ("Random", "baseline"),
    AlgorithmId.RANDOM_VALID: ("Random (valid)", "baseline"),
    AlgorithmId.WAIT: ("Wait", "baseline"),
    AlgorithmId.SCRIPTED: ("Scripted", "baseline"),
    AlgorithmId.LLM_DM: ("LLM + Decision Model", "llm"),
    AlgorithmId.MASKED_PPO: ("Masked PPO", "checkpoint"),
    AlgorithmId.MASKED_DQN: ("Double Dueling DQN", "checkpoint"),
    AlgorithmId.MUZERO: ("MuZero", "checkpoint"),
    AlgorithmId.RECURRENT_DQN: ("Recurrent DQN", "checkpoint"),
    AlgorithmId.DREAMERV3: ("DreamerV3", "checkpoint"),
    AlgorithmId.PLAN2EXPLORE: ("Plan2Explore", "checkpoint"),
}
POLICY_CATALOG: list[dict[str, object]] = [
    {"id": identifier.value, "label": label, "kind": kind, "requiresPath": kind == "checkpoint"}
    for identifier, (label, kind) in _POLICY_METADATA.items()
]
BASELINE_ALGORITHMS = frozenset(item["id"] for item in POLICY_CATALOG if item["kind"] == "baseline")
CHECKPOINT_ALGORITHMS = frozenset(item["id"] for item in POLICY_CATALOG if item["kind"] == "checkpoint")
LLM_ALGORITHMS = frozenset(item["id"] for item in POLICY_CATALOG if item["kind"] == "llm")
