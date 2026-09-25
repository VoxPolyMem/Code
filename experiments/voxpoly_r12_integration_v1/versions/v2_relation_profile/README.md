# VoxPoly AudioMem retrieval adapter v2

Status: **retrieval gate and frozen matched 25-QA LLM evaluation complete; all
four preregistered category gates passed**.  This directory is an isolated
experiment.  It does not modify the
4.4 core, Mem-Gallery, H2HMem, the frozen R12 memory builder, or any existing
speaker/Qwen service.

## Minimal method

The answer-visible context remains raw Top-30.  Facts, profiles, and speaker
state are navigation indexes only and are always projected through
`refer_ids` back to raw turns.

```text
query waveform
  -> frozen ECAPA encoder (CPU by default)
  -> frozen online EMA registry
  -> stable asker_ref + confidence

question
  -> raw dense + BM25 global retrieval
  -> high-confidence semantic utterance anchor
  -> reply_to / replied_by / grounded-fact sibling / same-session +/-1,+/-2
  -> optional asker-owned or assigned-to state selected by fact relevance
     and raw event-neighborhood coherence
  -> raw-only Top-30 packet
```

The packer preserves the first 24 rows of the frozen R12 raw context and uses
at most six remaining slots for semantic anchors, relation neighbors, and
high-confidence profile/state evidence.  A structural row already present at
rank 25--30 is protected rather than accidentally evicted.  Low-confidence
speaker identity may reorder the selected membership by a capped amount, but
cannot add or remove rows.  Unresolved identity has no effect.

No retrieval function accepts an answer, gold evidence, benchmark category,
or QA type.  The attribution ablation list exists only in the audit config;
the adapter is applied without reading it.  There is no keyword-based
attribution route.

## Audio query identity

`audio_query_adapter.py` reads waveform bytes and matches an ECAPA embedding
against the frozen online EMA registry.  High confidence requires cosine at
least 0.40 and margin at least 0.02; cosine 0.35--0.40 or insufficient margin
is low confidence; below 0.35 is unresolved.  Registry SHA must match every
memory profile, otherwise the adapter fails closed.

The two actual G014 Persona query clips both resolved on CPU to
`ONLINE_SPK_005` (profile `Omar Haddad`):

| QA | cosine | margin | confidence | old `asker_name` post-hoc check |
|---|---:|---:|---|---|
| PERSONA_004 | 0.7019 | 0.4388 | high | agrees |
| PERSONA_005 | 0.7034 | 0.4708 | high | agrees |

The old `predictions_with_asker.asker_name` value is used only after inference
as an equivalence check.  It is not an input.  Although each canonical
`full_case.json` says `project.audio_included=false`, the disk contains query
audio artifacts: G014, G015, and G019 each have ten `audio_persona/*.wav`
files.  Formal inference reads the waveform and never parses identity from the
filename.

## Zero-paid-API gate

`audit_retrieval.py` first freezes and hashes all contexts, then opens the
canonical cases to calculate evidence recall.  It made 27 requests to the
existing local embedding service and **zero external LLM calls**.

| Cohort / arm | QA | mean gold recall@30 |
|---|---:|---:|
| Attribution, current R12 | 12 | 0.4861 |
| Attribution, raw-only global | 12 | 0.5000 |
| Attribution, raw semantic anchor + local/reply expansion | 12 | **0.8889** |
| Attribution, complete v2 on frozen R12 base | 12 | **0.8750** |
| Full matched panel, current R12 | 25 | 0.7133 |
| Full matched panel, complete v2 | 25 | **0.9400** |
| Six known regressions, current R12 | 6 | 0.5000 |
| Six known regressions, complete v2 | 6 | **0.9167** |

Across all 25 QAs, v2 improves evidence recall for 11, ties for 14, and
regresses for 0.  This is a retrieval gate, **not an answer-accuracy claim**.
The complete contexts, reasons, speaker fields, raw IDs, `refer_ids`, matched
gold IDs, and freeze digest are in `artifacts/retrieval_audit.json`.

## Frozen matched evaluation

Zero-API unit tests cover unique raw Top-30, first-24 safety, protected old-tail
evidence, session-bounded neighborhoods, reply packets, profile-state event
coherence, low-confidence membership preservation, forbidden QA/gold fields,
registry mismatch, and a deliberately misleading audio filename.  All 10
tests pass on dx-infer-2.

`evaluate_matched25.py` remains default-off.  The approved run froze all 25 contexts before
opening answers/gold, uses the same `gpt-4.1-mini`, answer prompt, judge prompt,
Top-30, and audio-derived runtime character for both arms, and reuses the
already frozen shared R12 route (zero new router calls).  It atomically
checkpoints each QA and has a persistent exact-message cache plus finite
retries.  Maximum logical answer/judge calls before cache hits are 100.  The
formal run made 75 actual calls because identical messages were served from
the exact cache.

| Four-way category | QA | old reference | current R12 | v2 | Gate |
|---|---:|---:|---:|---:|---|
| Persona | 2 | 0.7500 | 0.6250 | **0.8750** | pass |
| Attribution | 12 | 0.7708 | 0.6667 | **0.8542** | pass |
| Retrieval reasoning | 7 | 0.8214 | **0.9286** | **0.9286** | pass |
| Memory evolution/conflict | 4 | 1.0000 | 1.0000 | **1.0000** | pass |
| **Overall** | **25** | **0.8200** | **0.7900** | **0.9000** | — |

The same-arm Acc@0.75 is 0.80 for current R12 and **0.96** for v2.  Gold
evidence recall@30 rises from 0.7133 to **0.9400**.  Against R12, the per-QA
LLM-score comparison is 4 wins, 20 ties, and 1 loss; the recall comparison is
11 wins, 14 ties, and no losses.  See `RESULTS_MATCHED25_20260915.md` and the
machine-readable `artifacts/matched25_eval/summary.json` for exact alignment.

The old column is a historical `gpt-4.1-mini` diagnostic reference aligned by
QA ID, not a formally matched third arm: its provider, answer context, and
Persona identity input differ.  The only formal causal comparison is current
R12 versus v2 within this run.

The preregistered estimate was roughly 110k tokens and CNY 0.35--0.50.  The
completed run was lower because exact-cache hits reduced the call count.  AIGC
consumption is reported as equivalent quota, while Velen is reported only as
paid cost.

The command below reproduces a paid run and therefore remains guarded; it must
not be launched merely to inspect the completed result:

```bash
python -B evaluate_matched25.py \
  --enable-paid-evaluation \
  --config configs/matched25.json \
  --out-dir artifacts/matched25_eval
```

GPU policy: reuse the existing Qwen3-VL service on dx-infer-2 GPU0 if a later
stage needs it; do not restart it.  ECAPA defaults to CPU.  This v2 gate used
neither Qwen nor GPU2/3.

The completed run used AIGC native `gpt-4.1-mini` only, with both configured
keys rotated after one recovered 429 and no Velen fallback.  Usage was 75
calls, 92,879 prompt tokens, 3,140 completion tokens, and 96,019 total tokens.
Its AIGC equivalent quota cost is CNY 0.3090; Velen paid cost is CNY 0.
