# AresSim llm_dm reading entry point

Updated: 2026-11-09.

- [Algorithm details](../../engine/aresim/algorithms/llm_dm/DETAILS.md): the source-adjacent architecture, runtime workflow, file ownership, configuration, commands and output guide.
- [Implementation guide](implementation.md): the short run guide for the Actor, focused Memory, required Learner, configuration, and output files.
- [Literature survey](literature_survey.md): research motivation and limitations for experience-driven agent improvement. Its LLM Actor examples also apply to the optional LLM Actor path; Decision Models now use the same observation and Memory contracts through a typed Choice adapter.

The separate `llm` and `decision_models` algorithms are superseded by `llm_dm`. Historical result files remain readable; the former standalone Decision Models heuristics are not part of the new pipeline.
