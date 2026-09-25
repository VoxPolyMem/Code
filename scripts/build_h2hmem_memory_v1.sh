#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
export V3_PIPELINE_ROOT=${V3_PIPELINE_ROOT:-$ROOT/vendor/v3_update_pipeline}
export H2HMEM_ROOT=${H2HMEM_ROOT:-$ROOT/data/H2HMEM}
export H2H_SOURCE_ROOT=${H2H_SOURCE_ROOT:-$ROOT/data/h2h_source}
export H2HMEM_CAPTIONS=${H2HMEM_CAPTIONS:-$ROOT/data/dense_captions_h2hmem.json}
export EVAL_PIN_MODEL=${EVAL_PIN_MODEL:-gpt-4.1-mini}

cd "$ROOT"
"$PYTHON_BIN" adapters/build_h2h_contextual_facts.py \
  --dialogue dialogue1 --dialogue dialogue2 --dialogue dialogue3 \
  --dialogue dialogue4 --dialogue dialogue5 "$@"

