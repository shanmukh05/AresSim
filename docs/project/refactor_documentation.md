# Algorithm documentation decisions

Reviewed November 7, 2026 against the existing working tree. Existing algorithm workflow content and user edits are preserved.

- Put architecture, input/output dimensions and learning/inference illustrations beside the implementation in each algorithm folder.
  - The eight workflow documents have inconsistent depths and mostly prose/ASCII/Mermaid rather than the requested portable PNG illustrations.
  - Current presets use memory v2, while the older PPO architecture explanation intentionally documents retained local v1. New figures explicitly identify reference/default widths and the legacy differences.
- Rename each `workflow.md` to `DETAILS.md`, update code/document references, and expose one algorithm index.
  - Cross-links refer to the old filenames, and several active guides reference an absent recurrent/world-model guide.
  - Fixed evaluation seeds belong under `configs/seeds/`; active algorithm examples now use the recovered durable inputs.
- Preserve editing ability inside the delivered PNG itself.
  - All fourteen PNGs were exported by installed draw.io Desktop with diagram XML embedded. Opening a PNG in draw.io restores native shapes and connectors; no separate editable source is needed.

## Verification

- All eight algorithm folders have `DETAILS.md`; no `workflow.md` files remain in the algorithm tree.
- Six learned algorithms each have architecture and training PNGs; Jev and baselines each have an inference PNG. Dimensions were checked against encoder/model/config source, with smoke/reference differences stated.
- Visually reviewed the figures, including a complete contact sheet and full-size architecture checks. Repaired duplicate cell IDs and connectors that crossed labels before the final export.
- Loaded every PNG, checked nonblank image dimensions, parsed its embedded XML, verified unique cell IDs and rejected self-parenting cells. All diagram links and new navigation links resolve.
- No unit tests were created, added, generated, restored or rewritten. The figures describe implementation structure, not evidence of GPU throughput or mission-learning quality.
