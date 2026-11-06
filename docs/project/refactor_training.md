# Training refactor decisions

Reviewed November 7, 2026 against the existing working tree. Preserve algorithm behavior and checkpoint formats; do not add unit tests.

- Put RLlib checkpoint transport, shared inference, and provenance checks in `training/rllib/checkpoint.py`; keep algorithm-specific restoration and exploration beside each algorithm.
  - DQN and recurrent DQN import private module/path/inventory helpers from PPO's checkpoint loader.
  - The shared RLlib runner imports its sidecar writer from the PPO algorithm.
  - PPO, DQN, and recurrent DQN duplicate identity/hash checks, and PPO has another inventory validator despite `training/artifacts.py` owning that contract.
  - Resume path resolution independently declares the RLlib sidecar schema; the neutral checkpoint module now owns that shared identifier.
  - RLlib runner computes a target iteration count passed into a function that never reads it; remove the unused calculation and parameter.
- Give framework-neutral checkpoint dispatch one owner in `training/checkpoints.py`, retaining existing public exports.
  - Both `training/__init__.py` and `algorithms/ppo/checkpoint.py` independently dispatch the sidecar loader ID.
- Decode experiment files once in `training/experiments.py`, forwarding the selected registry through every entry point.
  - `training/__init__.py` repeats YAML loading; `experiments.load_experiment` accepts a registry but drops it.
  - The stable experiment envelope imports shared model config through PPO and imports unused MuZero configuration classes.
  - TrainingRegistry stores config decoders twice: in a private table and the algorithm registration. Decode through the registration so one record owns the contract.
- Reuse numeric checkpoint discovery and atomic metadata publication from `training/artifacts.py`.
  - MuZero duplicates numbered/latest checkpoint selection and JSON publication.
  - RLlib retention uses lexicographic names, which stop matching step order across the six-to-seven-digit boundary.
  - MuZero's sidecar wraps `file_inventory` and duplicates native-file verification.
- Give resume configuration compatibility a public framework-neutral function in `training/resume.py`.
  - The world-model runner imports a private function from a module that also owns native RLlib path classification.
- Let replay resizing enforce its own capacity invariant.
  - Plan2Explore transfer assigns `replay.capacity` then calls the private eviction method from the runner.
- Import common encoder dimensions from their owner.
  - MuZero config imports `ModelConfig` through PPO despite `algorithms/common/model_config.py` owning it.

## Validation plan and limits

Load every checked-in learned-policy preset; exercise the custom-registry loader path; use real environment transitions for inference/replay/checkpoint smoke verification. Run short training and reload checkpoints where local dependencies permit. Do not modify or generate unit tests.

The inventory found no `notebooks/` directory although presets refer to seed manifests there. Recovered both original manifests from commit `a51147f` into `configs/seeds/`, preserving their exact bytes and seeds; frozen result manifests independently confirm the full split. Updated presets/defaults to the durable path. See `configs/seeds/README.md` for sources and checksums.

## Verification results

- All 21 checked-in learned-policy presets and both recovered seed manifests load successfully. Public `training.load_experiment` is the canonical loader; a custom registered algorithm also loads through `experiments.load_experiment(registry=...)`.
- Loading configs and importing the public training surface imports neither Torch nor Ray.
- Existing trained PPO, DQN and MuZero checkpoints each restored and ran eight legal actions on real seed 2000.
- Fresh bounded CPU training completed for PPO (64 steps), DQN (128), recurrent DQN (128), DreamerV3 (256), MuZero (256), Plan2Explore pretraining (256), and adaptation (256). Every phase wrote a final checkpoint, reloaded through the generic dispatcher and executed eight legal actions. Adaptation initialized from the just-produced pretraining checkpoint.
- Temporary smoke artifacts are under `/private/var/folders/j6/xcky2w3j1zzc93frrnjxzpmr0000gn/T/ares-training-smoke-wdh_fdlg`; the local smoke log is `/tmp/ares_training_smoke.log`. Ray process enumeration required the approved local sandbox escalation. Tracking was disabled; no cloud resources were rented.
- Python compilation and whitespace checks passed. No unit tests were created, added, restored or rewritten. GPU behavior and long-budget learning quality were not measured.

## Remaining deliberate cases

The RLlib fallback for older native module state, historical frozen MuZero metadata decoding, and Ray's deprecated-module warning filter remain because they preserve external framework/checkpoint compatibility. Plan2Explore's phase handling remains intrinsic to its pretrain/adapt contract. The post-change training audit found no further evidenced structural changes worth making in these ownership boundaries.

## Final inference and analysis audit

- Keep Jev's cargo arithmetic and repeated-action pattern definition beside its named JSON-state contract, and reuse them for policy overrides.
  - `jev/state.py` and `jev/agent.py` independently sum ice/ore/sample cargo and independently recognize three-action repeats/two-action cycles; prompt advice and policy fallback must change together.
  - `JevAgent._resolve_client` catches `JevUnavailableError` only to re-raise it unchanged.
- Pass checkpoint sources explicitly to action-path analysis and derive seed defaults from frozen experiment provenance.
  - `scripts/plot_reference_action_paths.py` mutates module globals for CLI-selected experiment roots and checkpoint labels; helpers silently read those values.
  - It imports private `evaluation._environment_config`, which only delegates to public `provenance.environment_config`.
  - Validation seed ranges and the policy seed offset repeat values already owned by frozen seed manifests.
  - `PPO_TRIALS` is unused.

SDK timeout/signature fallbacks, deterministic baseline reset requirements, Jev's productive-action heuristics, action-name wire strings, and comparison/cache compatibility checks remain intentional requirements.

Final audit verification: four baselines each completed 72 real simulator steps; mask-respecting baselines stayed legal. Fake Jev completed 72 legal steps across three seeds with timeout/unknown/Wait decisions, and the checked-in Jev YAML completed its 20-step smoke. Jev request states serialize as JSON and retain one `action` Choice. Real scripted rollouts generated 48 steps for action-path cache collection/reload, three PNG charts and summary output under a temporary directory; existing results were untouched. Public experiment loading accepts both historical positional registry and keyword registry arguments. A static direct-import graph of 104 Python modules and 305 internal edges has no top-level cycles (`/tmp/ares-module-imports.json`; deferred/plugin imports are outside that graph).
