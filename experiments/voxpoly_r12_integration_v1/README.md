# VoxPoly / AudioMemBench r12 integration v1

Status: **experimental and default-off**. This directory does not modify the
frozen Mem-Gallery r12 builder, H2H builder, evaluator, archive, or cache.

## Purpose

This adapter maps a QA-free VoxPoly view into the same three retrieval layers
used by the unified memory system:

```text
raw turn -> contextual atomic fact -> session collection
    |                 |
    +---- refer_ids --+
    +---- optional validated reply_to_turn_id
    |
    +---- predicted stable speaker profile (ECAPA + online EMA)
```

The semantic pipeline is shared: 12-turn windows with stride 6, contextual
atomic facts, model-selected `refer_ids`, content-only `retrieval_text`,
collections, and the existing `core/planner.py`, `core/query_fusion.py`, and
`core/provenance.py` retrieval path.

Audio-specific work is confined to the input adapter:

- it accepts only a manifest-declared QA-free `online_predicted` case view;
- it joins speakers by exact `(session, turn_idx)` coordinates;
- it stores the stable ECAPA/EMA identity as `speaker_ref`;
- it preserves an observable `reply_to_turn_id` only when it closes to a raw
  turn in the same validated bundle; missing fields remain `null`;
- it records profile aliases, acoustic IDs, registry SHA256, and EMA policy;
- prototype vectors remain only in the external NPZ registry;
- profile retrieval is disabled by default.

## Identity semantics

The fact extraction call emits semantics, `refer_ids`, a proposed
`source_speaker_ref`, and `addressee_refs` together. Shared deterministic code
then validates identity:

- one cited speaker: bind it directly;
- several cited speakers: accept only a proposed speaker present in cited raw
  turns;
- addressee: accept only a stable profile present in the observable window,
  `group`, or `unknown`; an invalid ID becomes `unknown`;
- `evidence_speaker_refs` is kept separately from the principal
  `source_speaker_ref`.

No GT speaker, GT addressee, QA, answer, benchmark category, or gold evidence
is opened by this pipeline.

The currently available final-alias artifacts are batch-final: acoustic
assignment is online, but names discovered later in the conversation are
backfilled before memory construction. The output records
`alias_timing=batch_final`; strict turn-causal mode rejects such an input.

## Isolation and reproducibility

The CLI requires `--enable-experimental-audio`. It has no implicit output
path. Checkpoint fingerprints include the source-view and speaker-sidecar
hashes, prediction hash/protocol, prompt and adapter versions, model,
window/stride, and retrieval-metadata switches. A mismatched checkpoint fails
closed instead of mixing Oracle and online-speaker memories.

Mem-Gallery/H2H behavior is unchanged because:

1. no production adapter or frozen source file is edited;
2. all artifacts and checkpoints use an experiment-only root;
3. `retrieval_text` remains content-only;
4. speaker/profile retrieval remains disabled.

## Zero-API validation

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest -v test_contract.py
```

The tests use a synthetic extractor and trap network access. They cover
default-off behavior, QA/addressee leakage, strict coordinate joins, causal
mode, multi-speaker attribution, addressee candidate validation, referential
closure, shared provenance selection, profile-vector exclusion, checkpoint
fingerprints, and non-overwrite behavior.

Real-bundle preflight also performs no LLM/audio/network call:

```bash
python preflight.py \
  --enable-experimental-audio \
  --case-view-dir /path/to/online_final/CASE \
  --registry-state /path/to/online_registry_state.npz \
  --out artifacts/g014_preflight.json
```

An already-built legacy memory can be enriched without re-running fact
extraction. `migrate_reply_edges.py` validates the original fingerprints and
every unchanged raw field, refuses to overwrite the source, and writes a new
candidate copy with a migration record and `api_calls=0`. This is a schema
migration only; it does not enable reply-based retrieval by itself.

## Paid memory construction (not launched by this change)

```bash
python build_memory.py \
  --enable-experimental-audio \
  --case-view-dir /path/to/online_final/CASE \
  --registry-state /path/to/online_registry_state.npz \
  --out /isolated/path/CASE.json \
  --checkpoints /isolated/path/checkpoints/CASE \
  --model gpt-4.1-mini
```

This only constructs memory. A separate feature-gated retrieval ablation is
still required before speaker/addressee metadata can influence AudioMemBench
QA ranking. The current `evaluation/voxpoly.py` ignores these fields.

For G014, `run_g014_memory_guarded.sh` pins the exact paths/model, enforces a
single-process lock and a four-hour timeout, and refuses to spend tokens unless
`CONFIRM_PAID_G014_R12=YES` is explicitly set.

## Matched soft-identity panel

The isolated matched evaluator compares two arms with one shared router plan:

- `content_only`: the unchanged unified Top-30 context;
- `soft_identity`: the exact same candidate set, reordered by a capped boost.

The boost sees only the planner's `speaker_hint` and the memory's structured
`source_speaker_ref` / `addressee_refs`. It does not inspect question text,
benchmark labels, answers, gold evidence, or categories. Source-speaker and
addressee boosts are 0.020 and 0.010 respectively, capped at 0.025. Because no
candidate can enter or leave, the raw safety-recall set is identical by
construction. Profile aliases resolve only by exact match or an unambiguous
first token.

`make_matched_panel.py` selects QA IDs by a frozen SHA256 rule; labels,
questions, answers and evidence are not selection inputs. The evaluator writes
`matched_pairs.json`, `content_only.json`, and `soft_identity.json` atomically,
and keeps an exact-message call cache so a restart does not repay completed
calls. Transport and judge-format retries are finite.

The G014 launcher is again default-off:

```bash
CONFIRM_PAID_G014_PANEL=YES ./run_g014_panel_guarded.sh
```

It requires the completed isolated G014 memory, uses one process, pins
gpt-4.1-mini, times out after two hours, and runs the zero-API verifier after
the matched panel completes.
