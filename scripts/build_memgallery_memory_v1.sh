#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
export V2_PIPELINE_ROOT=${V2_PIPELINE_ROOT:-$ROOT/vendor/v2_update_pipeline}
export MEMGALLERY_ROOT=${MEMGALLERY_ROOT:-$ROOT/data/Mem-Gallery}
export LEGACY_MEMGALLERY_FACT_ROOT=${LEGACY_MEMGALLERY_FACT_ROOT:-$ROOT/data/memgallery_legacy_facts}
export EVAL_PIN_MODEL=${EVAL_PIN_MODEL:-gpt-4.1-mini}

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
ARGS=()
for case_name in "${CASES[@]}"; do ARGS+=(--case "$case_name"); done
cd "$ROOT"
"$PYTHON_BIN" adapters/build_memgallery_contextual_facts.py "${ARGS[@]}" "$@"

