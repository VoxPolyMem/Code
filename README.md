# VoxPolyMem: reproducible v1 pipeline

This repository contains the **frozen, non-RL VoxPolyMem v1 pipeline** used for the reported Mem-Gallery, H2HMem, and VoxPolyBench runs. It is the method code, not the interactive demo or a copy of the benchmark datasets. The original prompt text, memory construction, route execution, answer/judge protocol, and benchmark budgets are retained; machine-specific locations and provider credentials are supplied by the user.

The shared memory has three linked layers: `raw` turns, contextual atomic `fact` records, and event-level `collection` records. Upper layers point back to raw evidence with `refer_ids`. The planner chooses memory layers and retrieval tools; the final answer sees selected raw evidence. Contextual facts use overlapping 12-turn windows with stride 6.

| Dataset | Final Top-K | Retrieval round budgets | Questions in reported run |
| --- | ---: | --- | ---: |
| Mem-Gallery | 20 | `20,6,4,2,1` | 1,711 across 20 topics |
| H2HMem | 30 | `30,8,5,3,2` | 190 across 5 dialogues, including `session0` |
| VoxPolyBench | 30 | `30,8,5` | 1,527 across 18 cases |

The LLM model identifier is `gpt-4.1-mini` with temperature 0. The embedding model is Qwen3-VL-Embedding-2B, 2,048 dimensions. Do not substitute a model, instruction, prompt, Top-K, or route policy and call the resulting scores a v1 reproduction.

## 1. Verify the published results without API calls

Use Python **3.12**. From the repository root:

```bash
python3.12 -m venv ../.venv-voxpolymem-v1
source ../.venv-voxpolymem-v1/bin/activate
python scripts/verify_release.py
python scripts/replay_reference_metrics.py
```

The first command verifies frozen file hashes, compiles the code, runs unit tests, and checks the archived scores. The second recomputes the question-weighted mean LLM-judge scores from score-only per-question records: Mem-Gallery `0.860023`, H2HMem `0.731579`, VoxPolyBench `0.837754`. **This verifies the reported arithmetic, not a new inference run.** The archive contains no question text, model predictions, or evidence.

The virtual environment is placed *outside* the repository because the release verifier intentionally scans every in-repository file for symlinks and unsafe content.

## 2. Prepare a live environment

The following steps make *new* model calls and incur provider charges.

```bash
python -m pip install -r requirements.txt
cp .env.example .env
# Edit .env with your own provider, data, model, and output paths.
set -a
source ./.env
set +a
export PYTHON_BIN="$(command -v python)"
```

Repeat the `set -a` / `source` / `set +a` step in each new shell. The launchers read exported environment variables; they do **not** automatically parse `.env`. Keep `.env` private. At minimum, set `OPENAI_API_KEY` (or `LLM_API_KEYS`), `LLM_BASE_URL`, and `EMBEDDING_SERVER_URL`. Set `NO_VELEN_FALLBACK=1` unless you deliberately configure and authorize a different provider. `EVAL_PIN_MODEL` must remain `gpt-4.1-mini` for v1.

Provide benchmark data and model artifacts at the paths in [`data/README.md`](data/README.md), or change the corresponding `.env` paths. The dataset roots, image/caption assets, normalized QA-free H2H streams, Mem-Gallery visual-set metadata, and full VoxPolyBench audio/QA package are external inputs; this repository does not redistribute them. The public VoxPolyBench preview alone is **not** sufficient for the 18-case rerun.

Start a Qwen3-VL-Embedding-2B service before evaluation. The frozen client sends JSON `POST` requests to `EMBEDDING_SERVER_URL`:

- `{"type":"batch_text","texts":[...],"instr":"Represent the text for retrieval."}`
- `{"type":"batch_image","images":[...],"instr":"Represent the image for retrieval."}`

The service must return `{"emb":"<base64 float32 bytes>","shape":[N,2048]}` in the same joint text/image embedding space. It is an external service, not an included launcher. Keep the same model version and image preprocessing as the original service for a comparable rerun. Check its availability with a single text request before any paid full run.

## 3. Build memories and evaluate

Run from the repository root, in the shell that loaded `.env`:

```bash
bash scripts/build_memgallery_memory_v1.sh
bash scripts/run_memgallery_v1.sh

bash scripts/build_h2hmem_memory_v1.sh
bash scripts/run_h2hmem_v1.sh
```

The build steps must finish **all** dialogue turns before QA evaluation. Output defaults to `artifacts/memory/` and `artifacts/evaluation/`. The launchers pin the benchmark-specific budgets and route mode; do not override those for v1. Existing complete memory artifacts may be reused for repeated evaluations. The Mem-Gallery console `OVERALL` is a topic-macro statistic; the reported `0.860023` is the **question-weighted** mean over 1,711 scores.

The public score-only archive cannot hold original round-0 route plans without revealing question-derived text. By default, live evaluation asks the **same frozen planner** to generate them again. If you have the original full result archive, `scripts/extract_frozen_routes.py` can supply those plans for a paired-route rerun. Hosted model output may still differ between calls even at temperature 0; byte-identical live answers are not guaranteed.

For the reported 18-case audio path, use the **iterative-3** runner described in [`docs/VOXPOLYBENCH_REPRODUCTION.md`](docs/VOXPOLYBENCH_REPRODUCTION.md). It uses benchmark-provided transcript text, online ECAPA/EMA speaker identity, and waveform-derived question asker identity; ASR was not enabled in this run. Do not substitute the older non-iterative audio runner.

## What is frozen and what you configure

| Frozen for v1 | User-provided configuration |
| --- | --- |
| Prompts and answer/judge code | API key and compatible provider endpoint |
| Memory schemas, `refer_ids`, routing and retrieval implementation | Dataset, captions, image/audio asset paths |
| Round budgets, Top-K, model identifier and embedding instructions | Qwen embedding service URL, model weights, GPU placement |
| Speaker/asker protocol and score aggregation | Writable output location and Python executable |

The code-hash contract is in `reference/code_manifest.json` and the reproduction scope is in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). Do not silently modify v1 for new RL policies or ablations; create another version. The separate demo is intentionally not published in this repository.
