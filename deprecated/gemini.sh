#!/bin/bash
set -euo pipefail

# APIキー設定
if [ -z "${GOOGLE_API_KEY:-}" ]; then
    echo "Error: GOOGLE_API_KEY not set."
    exit 1
fi

BASE_DIR="/home/kiso_user/Documents/workspace/Research"
DATA_DIR="${BASE_DIR}/data"
OUTPUT_DIR="${DATA_DIR}/gemini_results_gen_only"
mkdir -p "${OUTPUT_DIR}"

# ★ 生成のみを行う設定
# --max-rounds -1 : 評価・修正ループに入らない
# --per-hpo 10    : 1つの症状につき10個生成する（好みの数に変えてください）

echo "=== [Patient][Gemini][Generation Only] ==="

python gemini_pipeline_v2.py \
  --mode patient \
  --hpo-csv "${DATA_DIR}/HPO_depth_ge3_self_reportable.csv" \
  --output "${OUTPUT_DIR}/patient_gen_only.jsonl" \
  --gen-model "gemini-2.5-flash" \
  --concurrency 50 \
  --per-hpo 20 \
  --max-rounds -1 \
  --gen-max-new-tokens 64

echo "Done."