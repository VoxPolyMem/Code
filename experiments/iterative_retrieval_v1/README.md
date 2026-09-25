# Iterative Retrieval v1

Opt-in sufficiency-driven retrieval on top of frozen r12.

- Enable: `MPMEM_ITERATIVE_RETRIEVAL_V1=1`.
- Maximum rounds: 5 total (`Round 0` plus at most four refinements).
- Per-round newly admitted evidence budgets: `30,8,5,3,2`; later rounds
  cannot globally rerank or replace the high-ranked Round-0 core.
- After every action, cumulative results are exact-ID deduplicated, fused, and
  repacked to the configured final TopK before the next sufficiency decision.
- Final context size is controlled by `MPMEM_FINAL_TOP_K` (bounded to 10--60;
  default 30). Retrieval-channel candidate depth remains 30, so a TopK sweep
  changes context length without confounding candidate recall depth.
- The decision sees the current packed TopK plus all previous queries,
  layers, retrievers, sub-questions, missing-evidence gaps, and new-evidence
  counts.
- Stop on sufficient evidence, repeated action, no new evidence, parse failure,
  or the five-round cap.
- Answer and judge run only after retrieval terminates.
- Model-declared temporal questions preserve the global chronological answer
  view and do not apply event-local Evidence Bundle grouping.
- Default is off; frozen r12 remains reproducible.

The first paid validation is a preregistered 10-question H2H panel containing
four audited image-provenance misses and six difficult FR/KR/TR/evolution rows
across all five dialogues. Initial route plans are frozen from r12.

Pilot results and the promotion decision are recorded in
`RESULT_20260915.md`. The Mem-Gallery TopK experiment is reported separately
in `MEMGALLERY_TOPK_SWEEP_20260915.md`.
