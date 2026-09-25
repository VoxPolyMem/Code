# VoxPolyMem v1

VoxPolyMem is the frozen, GitHub-ready implementation used for the reported
Mem-Gallery, H2HMem, and VoxPolyBench experiments. The method writes one
shared hierarchical memory representation and applies the same model-driven
retrieval policy across text, image, and audio inputs.

## Frozen method

Each dialogue is represented by three linked layers:

1. `raw`: bottom-level turns with speaker, addressee, timestamp, image IDs,
   captions, and optional audio identity metadata;
2. `fact`: context-aware atomic facts extracted from overlapping 12-turn
   windows with stride 6;
3. `collection`: event-level groups whose evidence is grounded to raw turns.

Every upper-level node stores `refer_ids` to bottom-level evidence. Retrieval
uses a model-selected combination of `raw/fact/collection` and
`dense/BM25/image/caption`, keeps a raw-recall safety path, and may refine the
query for a bounded number of rounds. The answer context contains bottom-level
evidence rather than hidden upper-layer text.

The v1 benchmark settings are frozen as follows:

| Benchmark | Final Top-K | Round budgets | Reported QA |
| --- | ---: | --- | ---: |
| Mem-Gallery | 20 | 20, 6, 4, 2, 1 | 1,711 |
| H2HMem | 30 | 30, 8, 5, 3, 2 | 190 |
| VoxPolyBench | 30 | 30, 8, 5 | 1,527 |

The LLM configuration is `gpt-4.1-mini`, temperature 0. The embedding model is
Qwen3-VL-Embedding-2B with 2,048 dimensions and the instruction
`Represent the text for retrieval.`

## Reproducibility levels

Two complementary modes are included:

- **Exact replay** recomputes every reported aggregate from score-only
  per-question records in `reference/replay_outputs_v1.tgz`. This path is
  network-free and must match the published counts and means exactly. The
  public archive deliberately excludes benchmark questions, answers,
  predictions, and dialogue evidence.
- **Live rerun** rebuilds memories and calls the configured model providers
  with the same prompts, routing policy, Top-K, and stopping rules. Hosted
  model aliases are external state, so a future live call cannot be guaranteed
  to return byte-identical text even at temperature 0.

Run the offline acceptance gate first:

```bash
python scripts/verify_release.py
python scripts/replay_reference_metrics.py
```

Expected mean LLM-judge scores are 0.860023 for Mem-Gallery, 0.731579 for
H2HMem, and 0.837754 for VoxPolyBench.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Benchmark data and model weights are intentionally not duplicated in this
repository. Configure their paths in `.env`; see `data/README.md`. The
interactive demo is intentionally not included in this code release; the
existing VoxPolyBench demo remains in its own benchmark repository.

For an OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=...
export LLM_BASE_URL=https://api.openai.com/v1
export EMBEDDING_SERVER_URL=http://localhost:9981
```

The repository contains no credentials. Private-provider endpoints and keys
must be supplied through environment variables.

## Layout

- `core/`: shared planner, fusion, temporal handling, provenance packing, and
  bounded iterative retrieval;
- `adapters/`: benchmark-neutral memory writers and input projections;
- `evaluation/`: benchmark parsers and frozen answer/judge protocol;
- `audio/speaker/`: online ECAPA clustering, guarded EMA updates, and identity
  binding;
- `experiments/voxpoly_r12_integration_v1/`: audio/profile adapter and the
  full 18-case evaluation path;
- `vendor/`: the minimal legacy runtime required by the frozen evaluator;
- `reference/`: exact replay artifact and expected metrics;
- `scripts/`: release verification and portable live launchers.

## Live runs

Prepare benchmark data and frozen memories, then run:

```bash
bash scripts/run_memgallery_v1.sh
bash scripts/run_h2hmem_v1.sh
```

The public default asks the same planner to generate round-0 routes again;
the score-only replay archive does not contain question text or route plans.
For an exact paired-route rerun, obtain the original full result archive from
the authors and run `python scripts/extract_frozen_routes.py --archive
<full-results.tgz>` first. This optional artifact is not redistributed.

The AudioMem/VoxPolyBench path is documented in
`docs/VOXPOLYBENCH_REPRODUCTION.md`; it includes online speaker identity and
the iterative three-round evaluation.

## Release status

This directory is the immutable `v1` method release. New RL policies or
changes to memory construction must use a new version directory and must not
overwrite v1 artifacts.
