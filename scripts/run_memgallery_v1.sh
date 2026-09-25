#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
ROUTES=${ROUTES:-$ROOT/artifacts/frozen_routes_v1}
OUT_DIR=${OUT_DIR:-$ROOT/artifacts/evaluation/memgallery_v1}

export V2_PIPELINE_ROOT=${V2_PIPELINE_ROOT:-$ROOT/vendor/v2_update_pipeline}
export DATASET_ROOT=${DATASET_ROOT:-$ROOT/data/Mem-Gallery}
export FACT_ROOT=${FACT_ROOT:-$ROOT/artifacts/memory/memgallery_contextual_v1/compatible_views/no_role_metadata}
export FUSION_EVAL_RUNS_DIR=$OUT_DIR
export VOXPOLYMEM_STORAGE_DIR=${VOXPOLYMEM_STORAGE_DIR:-$ROOT/artifacts/runtime_storage}
export EMBEDDING_SERVER_URL=${EMBEDDING_SERVER_URL:-http://localhost:9981}
export EVAL_PIN_MODEL=${EVAL_PIN_MODEL:-gpt-4.1-mini}
export REUSE_CONTROL_OUTPUT=0
export ROUTE_ABLATION_STRATEGY=unified_hybrid_route
export MPMEM_EVIDENCE_BUNDLE_SOFT_GATE_V1=1
export MPMEM_ITERATIVE_RETRIEVAL_V1=1
export MPMEM_ITERATIVE_MAX_ROUNDS=5
export MPMEM_ITERATIVE_ROUND_BUDGETS=20,6,4,2,1
export MPMEM_FINAL_TOP_K=20
export EVAL_MODE_SUFFIX=voxpolymem-v1
unset MPMEM_EVIDENCE_BUNDLE_V1 QA_IDS || true

CASES=(
  AI_Robotics_Automation_Future_Tech Academic_Animal_Pet_Research_Life
  Architecture_Art_Culture_Exhibition_Technology Astronomy_Physics_Scientific_Experiments_Cosmology
  Baking_Dessert_Daily_Life_Skill Dog_Behavior_Research_Academic_Life
  Education_Career_Research_Lifestyle Entrepreneurship_Blockchain_Economics_Logistics_Nature
  Fashion_Personal_Care_Lifestyle_Shopping Global_Travel_Cultural_Sightseeing
  Global_Travel_Sustainable_Fashion_Design Home_Health_Lifestyle_Product
  Home_Repair_Maintenance_Cleaning Landscape_Travel_Architecture_Nature
  Music_Dance_Theater_Performance_Learning Nature_Economics_Programming_Student_Life
  Parenting_Commuting_Hobbies_Travel_Gear Python_Botany_AI_Student_Life
  Real_Estate_Home_Decor_DIY_Lifestyle Technology_Ethics_Future_Society
)
INDICES=(0 1 2 3 4 5 6 7 8 9 10 16 17 18 19 20 21 22 23 24)

cd "$ROOT"
for index in "${!CASES[@]}"; do
  case_name=${CASES[$index]}
  if [[ -d "$ROUTES/memgallery" ]]; then
    route_file="$ROUTES/memgallery/$case_name.json"
    if [[ ! -f "$route_file" ]]; then
      echo "Incomplete frozen routes: missing $route_file" >&2
      exit 1
    fi
    export ROUTE_PLAN_RESULT_FILE="$route_file"
  else
    unset ROUTE_PLAN_RESULT_FILE || true
  fi
  "$PYTHON_BIN" -m evaluation.memgallery --case "${INDICES[$index]}" "$@"
done
