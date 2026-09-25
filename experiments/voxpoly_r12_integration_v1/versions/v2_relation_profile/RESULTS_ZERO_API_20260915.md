# AudioMem v2 zero-API result (2026-09-15)

> Successor status: the frozen matched-25 answer/judge evaluation is complete
> and all four preregistered category gates pass.  See
> `RESULTS_MATCHED25_20260915.md` for final LLM scores, exact per-QA alignment,
> and cost.  This file remains the pre-evaluation retrieval gate only.

## Decision

**PASS for a matched 25-QA answer/judge evaluation.**  The decision is based
only on frozen Top-30 evidence recall.  No answer model or judge was called,
so this document does not claim an LLM-score improvement.

## Six known regressions

| Case / QA | current R12 recall@30 | v2 recall@30 | delta |
|---|---:|---:|---:|
| G014 / ATTRIBUTION_005 | 0.1667 | 0.8333 | +0.6667 |
| G014 / PERSONA_004 | 1.0000 | 1.0000 | 0.0000 |
| G014 / PERSONA_005 | 0.0000 | 1.0000 | +1.0000 |
| G015 / ATTRIBUTION_011 | 0.3333 | 0.8333 | +0.5000 |
| G015 / ATTRIBUTION_010 | 0.5000 | 0.8333 | +0.3333 |
| G019 / ATTRIBUTION_011 | 1.0000 | 1.0000 | 0.0000 |

The initial profile-state implementation still missed G014 Persona_005.  Its
fact hybrid rank preferred a later, same-speaker “tool package pickup” event.
The final implementation resolves this general collision by adding
QA-blind event coherence: an identity-relevant fact is preferred when its raw
`refer_ids` are close to high-ranked raw semantic evidence in the same
session.  The correct `S3_T008` then enters the raw context, while the existing
gold evidence in all six QAs remains preserved.

## Attribution three-arm ablation

| Arm | mean recall@30 | fully recalled QA |
|---|---:|---:|
| Current saved R12 | 0.4861 | 2/12 |
| Raw-only global Top-30 | 0.5000 | 2/12 |
| Raw anchor + same-session/reply expansion Top-30 | **0.8889** | 4/12 |

The third arm does not read the attribution label.  It observes only the
question and raw memory: a sufficiently strong raw semantic anchor activates
a bounded reply/local packet.  The labels in `matched25.json` select which
already frozen rows are summarized in this ablation table; they do not enter
retrieval.

## Full 25-QA safety result

Current R12 mean evidence recall@30 is 0.7133; v2 is 0.9400.  The paired outcome
is 11 improvements, 14 ties, and zero regressions.  Every answer-visible row
is raw and keeps exactly one bottom-level `refer_id`.  Retrieval freeze SHA256:

`5f91213674fc97a29eb1f2c1142678efbe6aba98161070a39136f9e3a4118fc2`

Incremental external LLM usage: **0 calls, 0 tokens, CNY 0**.  Local work was
two ECAPA CPU encodes plus 27 dense query requests to the already running
embedding service; local infrastructure cost is unknown, not zero.
