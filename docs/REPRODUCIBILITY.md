# Reproducibility contract

VoxPolyMem v1 freezes algorithm code, prompts, model names, retrieval budgets,
and score-only per-question replay records. Reproducibility is checked at
three levels.

## Level 1: code integrity

`scripts/verify_release.py` compiles all Python sources, rejects symlinks and
credential-like strings, checks the reference artifact hash, and runs the
network-free unit tests.

## Level 2: exact result replay

`scripts/replay_reference_metrics.py` reads the compressed score-only
per-question records without extracting them. It independently recomputes the number of
topics/dialogues/cases, total QA, and mean LLM-judge score. A mismatch at
1e-12 tolerance fails the release gate.

The public archive contains only 43 completed score documents: 20
Mem-Gallery topics, 5 H2HMem dialogues, and 18 VoxPolyBench cases. It keeps
question IDs, categories, and scores needed to verify the aggregates, but
excludes question/answer text, predictions, raw evidence, logs, caches,
partial outputs, credentials, and machine-specific paths. Maintainers can
rebuild the deterministic archive from an expanded internal result tree with
`scripts/build_public_replay_archive.py`.

## Level 3: live reproduction

Live runs use the same prompt text, model identifier, temperature, memory
construction, retrieval policy, Top-K, and judge. They require the benchmark
data and Qwen embedding server. Hosted model inference is not bitwise
deterministic external state; exact equality is therefore asserted only for
the archived outputs, while live reruns are expected to reproduce the
experimental result within normal provider variance. Exact replay verifies
the reported arithmetic, not that fresh model calls reproduce the same
predictions. Public live scripts regenerate their initial plans because the
score-only archive intentionally omits question-derived route text. Authors
with the original full result archive can optionally extract and supply the
frozen round-0 plans for a paired-route rerun.

## Portability-only changes

The GitHub package replaces private absolute paths, API keys, and provider
URLs with environment variables. These changes do not alter memory schemas,
prompts, routing logic, retrieval scoring, stopping rules, or evaluation
metrics. The public package intentionally contains no credentials.

## Versioning rule

V1 is immutable. RL training, new memory fields, different retrieval budgets,
or changes to speaker identity must be released as a new version rather than
silently changing this directory.
