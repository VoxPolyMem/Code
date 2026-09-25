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

Live scripts preserve the same prompt text, model identifier, temperature,
memory construction, retrieval policy, Top-K, and judge. They require the
complete benchmark inputs and a compatible Qwen embedding server, neither of
which is redistributed here. Hosted model inference and provider aliases are
external state: **no byte-identical live result or score tolerance is
guaranteed**. The offline replay proves the reported arithmetic only; it does
not prove that fresh model calls reproduce the predictions. Public live
scripts regenerate their initial plans because the score-only archive omits
question-derived route text. Authors with the original full result archive
can optionally extract the frozen round-0 plans for a paired-route rerun.

## Portability-only changes

The GitHub package replaces private absolute paths, API keys, and provider
URLs with environment variables. These changes do not alter frozen memory
schemas, prompts, routing logic, retrieval scoring, stopping rules, or
evaluation metrics. Environment values must be exported before live launch;
the shell scripts do not parse `.env` automatically. The public package
intentionally contains no credentials.

## Versioning rule

V1 is immutable. RL training, new memory fields, different retrieval budgets,
or changes to speaker identity must be released as a new version rather than
silently changing this directory.
