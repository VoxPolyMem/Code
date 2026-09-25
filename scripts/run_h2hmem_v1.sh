#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
ROUTES=${ROUTES:-$ROOT/artifacts/frozen_routes_v1}
MEMORY_DIR=${MEMORY_DIR:-$ROOT/artifacts/memory/h2h_contextual_v1/no_role_metadata}
OUT_DIR=${OUT_DIR:-$ROOT/artifacts/evaluation/h2h_v1}

export V3_PIPELINE_ROOT=${V3_PIPELINE_ROOT:-$ROOT/vendor/v3_update_pipeline}
export H2HMEM_ROOT=${H2HMEM_ROOT:-$ROOT/data/H2HMEM}
export EMBEDDING_SERVER_URL=${EMBEDDING_SERVER_URL:-http://localhost:9981}
export EVAL_PIN_MODEL=${EVAL_PIN_MODEL:-gpt-4.1-mini}
export MPMEM_EVIDENCE_BUNDLE_SOFT_GATE_V1=1
export MPMEM_ITERATIVE_RETRIEVAL_V1=1
export MPMEM_ITERATIVE_MAX_ROUNDS=5
export MPMEM_ITERATIVE_ROUND_BUDGETS=30,8,5,3,2
export MPMEM_FINAL_TOP_K=30
unset MPMEM_EVIDENCE_BUNDLE_V1 || true

cd "$ROOT"
route_args=()
if [[ -d "$ROUTES/h2h/unified_hybrid_route" ]]; then
  route_args=(--route-plan-dir "$ROUTES/h2h")
fi
"$PYTHON_BIN" -m evaluation.h2h \
  --split multi-party \
  --dialogue dialogue1 --dialogue dialogue2 --dialogue dialogue3 \
  --dialogue dialogue4 --dialogue dialogue5 \
  --strategy unified_hybrid_route \
  --memory-dir "$MEMORY_DIR" \
  "${route_args[@]}" \
  --out "$OUT_DIR" "$@"
