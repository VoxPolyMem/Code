# Reproduce the reported 18-case VoxPolyBench path

This is the **current-audio, iterative-3** path that produced the published
18-case aggregate. It is not `evaluation/voxpoly.py`, `evaluate_case.py`, or
the older `run_evaluation_lane.py`. The full run has 1,527 QA items.

## Inputs and protocol

Complete the root [environment setup](../README.md#2-prepare-a-live-environment)
first. `VOXPOLYBENCH_HOME` must point to the **full** benchmark package,
including `output/`, `output/gen2/`, `output/gen3/`, the 18-case evaluation
inputs, question audio and `audio_persona/` manifests, and the original
`tools/speaker/ecapa_encoder.py`. The GitHub preview by itself cannot run this
experiment. The memory builder reads the benchmark-provided transcript text;
ASR is **off**. Audio is used for turn-speaker identification and for
identifying the person asking each Persona question. The online speaker
pipeline uses ECAPA, guarded EMA updates, and name binding. Addressee is
inferred from the dialogue during memory writing. QA, gold answers, and gold
speaker/addressee labels are excluded from memory construction and retrieval.

For a directly comparable speaker stage, use the frozen speaker sidecars from
the original run. If generating them afresh, run the online pipeline in the
full benchmark package for **each** case, then the included final-alias
resolver. Each case needs these files under
`$VOXPOLY_SPEAKER_ROOT/CASE_G_XXX/`:

```text
online_identity_predictions.json
online_registry_state.npz
run_manifest.json
```

The alias resolver's output must be at
`experiments/voxpoly_speaker_ablation_v1/artifacts/final_alias_v1/CASE_G_XXX/online_identity_predictions_alias_resolved.json`.
For one case, the commands are:

```bash
CASE=CASE_G_001
python "$VOXPOLYBENCH_HOME/tools/speaker/online_speaker_identity_pipeline.py" \
  --case-root "$VOXPOLYBENCH_HOME/cases/$CASE" \
  --output-dir "$VOXPOLY_SPEAKER_ROOT/$CASE" \
  --device cuda:0
mkdir -p "experiments/voxpoly_speaker_ablation_v1/artifacts/final_alias_v1/$CASE"
python experiments/voxpoly_speaker_ablation_v1/resolve_final_aliases.py \
  --input "$VOXPOLY_SPEAKER_ROOT/$CASE/online_identity_predictions.json" \
  --output "experiments/voxpoly_speaker_ablation_v1/artifacts/final_alias_v1/$CASE/online_identity_predictions_alias_resolved.json"
```

Run this preparation for all 18 case IDs before proceeding. The case IDs are
`001,002,003,005,006,009,010,012,013,014,015,016,019,020,022,023,024,025`.
Do not use `--score` while creating the speaker sidecars; scoring is separate.
Generating new speaker sidecars can change downstream results, so retain and
identify the sidecar set used for any comparison.

## QA-free preparation, memory build, and evaluation

These commands are run from the repository root in the shell that sourced
`.env`. Use an **empty** output directory for the initial preparation. The
first step is zero-API and validates that all required inputs are present;
do not start the paid steps if it fails.

```bash
export AUDIO_RUN_ROOT="$PWD/artifacts/audio_v1_run"
export AUDIO_RUNNER="experiments/voxpoly_r12_integration_v1/versions/v2_relation_profile/fullcase/all18_current_audio_v1"
python "$AUDIO_RUNNER/prepare_all18_inputs.py" --run-root "$AUDIO_RUN_ROOT"
```

The frozen audio case configuration points to the embedding service at
`127.0.0.1:9981`; expose the compatible Qwen3-VL-Embedding-2B service on that
local port before evaluation. This is a deployment address, not a change to
the embedding model or scoring. Check `all18_inputs.json` reports 18 cases
and 1,527 QA before spending API calls.

The memory lane uses `gpt-4.1-mini`, contextual 12-turn windows with stride 6,
and bounded retries. Run one case first if desired; the completed memory file
is reusable. For the full run:

```bash
CASES=(CASE_G_001 CASE_G_002 CASE_G_003 CASE_G_005 CASE_G_006 CASE_G_009 \
       CASE_G_010 CASE_G_012 CASE_G_013 CASE_G_014 CASE_G_015 CASE_G_016 \
       CASE_G_019 CASE_G_020 CASE_G_022 CASE_G_023 CASE_G_024 CASE_G_025)
python "$AUDIO_RUNNER/run_memory_lane.py" \
  --run-root "$AUDIO_RUN_ROOT" --provider aigc --cases "${CASES[@]}"
for CASE in "${CASES[@]}"; do
  python "$AUDIO_RUNNER/evaluate_case_iterative3.py" \
    --run-root "$AUDIO_RUN_ROOT" --case "$CASE" --provider aigc
done
```

This evaluation uses saved round-0 routes, the shared retrieval scope,
original question text, waveform-derived asker, speaker-only asker context,
the generic known-asker answer hint, Top-30, and at most three retrieval
rounds with budgets `30,8,5`. It runs the frozen answer and LLM-judge
prompts. Each completed result is
`$AUDIO_RUN_ROOT/current_unified_iterative3/CASE_G_XXX/matched_pairs.json`.
Check `status: complete` and the per-case QA count in all 18 files before
aggregating; partial files are not results. The runner stops after a finite
retry budget rather than retrying indefinitely. Its `--provider velen` option
is only for users who have explicitly configured and authorized that provider.

The archived 18-case mean score can always be checked without API calls via
`python scripts/replay_reference_metrics.py`. Fresh calls may not produce the
same predictions because the hosted model and newly generated speaker/route
artifacts are external state.
