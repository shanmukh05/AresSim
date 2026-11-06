# Environment and UI refactor decisions

November 7, 2026. This records the working-tree inventory before structural edits. Existing local changes remain the starting point; no unit tests are created or rewritten.

- Own policy identifiers and catalog metadata together, and derive selection predicates from metadata.
  - `integrations/policy.py` repeats policy membership in three sets and a separate catalog; `api.py` repeats its identifier literal.
  - `web/src/types/sim.ts` repeats learned-policy membership while `ActionBar.tsx` repeats labels and translates algorithm-specific cases into dependency requirements.
  - Keep the REST catalog fields compatible. Backend and frontend each retain their language's wire-contract declaration, with manual parity verification.
- Project the authoritative Wait recharge estimate with the engine configuration, and record it in replay deltas.
  - `web/src/lib/power.ts` copies recharge constants (0.85 and 18) that diverged from engine defaults (0.35 and 8).
  - `snapshot_from_state` lacks the engine configuration, hiding the relationship between custom settings and display.
  - Old replay files lack the estimate and their originating configuration. Display an unavailable value rather than inventing historical recharge semantics.
- Build objective progress from the selected task configuration in the shared local observation builder.
  - `components/tasks.py` repeats the default mission targets from `TaskConfig`.
  - `LocalObservation` divides by those constants, while `LocalMemoryObservation` separately replaces those same objective rows using its task config.
- Remove unused transition retention and truncation aliases.
  - `AresEnvironment._last_step` and `episode_end_breakdown` have no callers; the annotation also refers to an unimported task outcome.
  - `AresTimeLimit.step` assigns reward aliases solely to return them unchanged.
- Retain the built-in compact stepping guard as a deliberate compatibility boundary.
  - Fast stepping borrows mutable engine state and skips custom reward/task hooks. Exact built-in type checks protect subclasses with overridden behavior; switching to duck-typed capabilities alone would silently bypass those hooks.
  - A new extensibility protocol is unwarranted until custom components require compact execution.

## Validation

- Passed a manual FastAPI session using custom power settings: accurate recharge projection, catalog/enum parity, rejection of an unknown policy, `wait`/`random_valid`/`scripted` agent steps, trajectory save/load/jump, and importing a legacy payload without the new estimate.
- Passed direct observations with custom ice/sample goals and a zero-disabled ice goal for both local observation builders.
- Matched audited and compact environment observations, masks, rewards, and terminal/truncation flags for real legal-action rollouts on seeds 3, 33, and 35 through 20-step cutoffs.
- Matched backend and frontend catalog order, identifiers, labels, kinds, and checkpoint requirements.
- Existing desktop browser end-to-end suite passed all 17 cases after the changes.
- `npm run build` passed. Vite reports the existing large bundle advisory; this refactor does not alter application loading strategy.
- Re-audited changed dependency boundaries and searched for removed constants/callers. Built-in compact stepping remains deliberately explicit to preserve custom hooks. No additional evidenced structural issue worth changing remains in this reviewed environment/UI scope.

No unit tests were added, generated, restored, or rewritten. Manual verification does not benchmark a long training run or exercise a paid Jev API call.
