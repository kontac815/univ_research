#!/bin/bash
set -euo pipefail

########################################
# 設定（必要ならここだけ書き換え）
########################################

# プロジェクトパス
BASE_DIR="/home/kiso_user/Documents/workspace/Research"
HPO_DIR="${BASE_DIR}/HPO-patient"
DATA_DIR="${BASE_DIR}/data"

# 統合パイプラインのスクリプト名
UNIFIED_SCRIPT="vllm_pipeline.py"

# モデル・LoRA
GEN_MODEL="EQUES/MedLLama3-JP-v2"                         # ベースモデル
PATIENT_LORA="${DATA_DIR}/eques_patient_expression_lora"  # 患者表現 LoRA
DOCTOR_LORA="${DATA_DIR}/eques_doctor_expression_lora"    # 医師表現 LoRA

# どの GPU を使うか（必要に応じて変更）
# 例: 3090 が CUDA:1 の場合
GPU_ID=1

# 生成パラメータ
# 患者表現
PAT_PER_HPO=40
PAT_TARGET_GOOD=20
PAT_GEN_MAX_NEW=24
PAT_JUDGE_MAX_NEW=48
PAT_REFINE_MAX_NEW=24

# 医師表現
DOC_PER_HPO=10
DOC_TARGET_GOOD=5
DOC_GEN_MAX_NEW=32
DOC_JUDGE_MAX_NEW=48
DOC_REFINE_MAX_NEW=32

########################################
# 前提ファイル:
#   - ${DATA_DIR}/HPO_symptom_depth_leq3_self_reportable_with_jp.csv  (患者用)
#   - ${DATA_DIR}/HPO_symptom_depth_leq3_with_jp.csv                  (医師用など想定)
########################################

echo "BASE_DIR  = ${BASE_DIR}"
echo "HPO_DIR   = ${HPO_DIR}"
echo "DATA_DIR  = ${DATA_DIR}"
echo "GEN_MODEL = ${GEN_MODEL}"
echo "GPU_ID    = ${GPU_ID}"
echo

cd "${HPO_DIR}"

mkdir -p "${DATA_DIR}/notLoRA_vllm"
mkdir -p "${DATA_DIR}/LoRA_vllm"

# 使う GPU を固定したい場合（3090 側に合わせて変更）
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU_ID}

########################################
# Step 1: 患者表現 (mode=patient) LoRA あり
########################################

echo "=== [Patient][LoRA][vLLM] 開始 ==="

python "${UNIFIED_SCRIPT}" \
  --mode patient \
  --hpo-csv "${DATA_DIR}/HPO_depth_ge3_self_reportable.csv" \
  --output "${DATA_DIR}/LoRA_vllm/HPO_symptom_patient_expression_judge_refine.vllm_lora.jsonl" \
  --gen-model "${GEN_MODEL}" \
  --gen-lora-path "${PATIENT_LORA}" \
  --refine-lora-path "${PATIENT_LORA}" \
  --per-hpo "${PAT_PER_HPO}" \
  --target-good "${PAT_TARGET_GOOD}" \
  --gen-max-new-tokens "${PAT_GEN_MAX_NEW}" \
  --judge-max-new-tokens "${PAT_JUDGE_MAX_NEW}" \
  --refine-max-new-tokens "${PAT_REFINE_MAX_NEW}" \
  --min-overall 4 \
  --min-match 3 \
  --min-simplicity 3 \
  --max-rounds 3 \
  --diversity-sim-high 0.9 \
  --diversity-sim-low 0.7 \
  --log-each-round \
  --round-log-prefix "${DATA_DIR}/LoRA_vllm/HPO_symptom_patient_expression_judge_refine.vllm_lora.round" \
  --wandb \
  --wandb-project "HPO_patient_lora_vllm" \
  --wandb-tags "patient,vllm,lora"

echo "=== [Patient][LoRA][vLLM] 完了 ==="
echo
########################################
# Step 2: 医師表現 (mode=doctor) LoRA あり
########################################
# ※ 医師用の HPO CSV が別ならパスを変更してください

echo "=== [Doctor][LoRA][vLLM] 開始 ==="

python "${UNIFIED_SCRIPT}" \
  --mode doctor \
  --hpo-csv "${DATA_DIR}/HPO_depth_ge3.csv" \
  --output "${DATA_DIR}/LoRA_vllm/HPO_symptom_doctor_expression_judge_refine.vllm_lora.jsonl" \
  --gen-model "${GEN_MODEL}" \
  --gen-lora-path "${DOCTOR_LORA}" \
  --refine-lora-path "${DOCTOR_LORA}" \
  --per-hpo "${DOC_PER_HPO}" \
  --target-good "${DOC_TARGET_GOOD}" \
  --gen-max-new-tokens "${DOC_GEN_MAX_NEW}" \
  --judge-max-new-tokens "${DOC_JUDGE_MAX_NEW}" \
  --refine-max-new-tokens "${DOC_REFINE_MAX_NEW}" \
  --min-overall 4 \
  --min-match 3 \
  --min-simplicity 3 \
  --max-rounds 3 \
  --diversity-sim-high 0.9 \
  --diversity-sim-low 0.7 \
  --log-each-round \
  --round-log-prefix "${DATA_DIR}/LoRA_vllm/HPO_symptom_doctor_expression_judge_refine.vllm_lora.round" \
  --wandb \
  --wandb-project "HPO_doctor_lora_vllm" \
  --wandb-tags "doctor,vllm,lora"

echo "=== [Doctor][LoRA][vLLM] 完了 ==="
echo

########################################
# Step 3: 患者表現 (mode=patient) LoRA なし（ベースモデル）
########################################

echo "=== [Patient][NoLoRA][vLLM] 開始 ==="

python "${UNIFIED_SCRIPT}" \
  --mode patient \
  --hpo-csv "${DATA_DIR}/HPO_depth_ge3_self_reportable.csv" \
  --output "${DATA_DIR}/notLoRA_vllm/HPO_symptom_patient_expression_judge_refine.vllm_base.jsonl" \
  --gen-model "${GEN_MODEL}" \
  --per-hpo "${PAT_PER_HPO}" \
  --target-good "${PAT_TARGET_GOOD}" \
  --gen-max-new-tokens "${PAT_GEN_MAX_NEW}" \
  --judge-max-new-tokens "${PAT_JUDGE_MAX_NEW}" \
  --refine-max-new-tokens "${PAT_REFINE_MAX_NEW}" \
  --min-overall 4 \
  --min-match 3 \
  --min-simplicity 3 \
  --max-rounds 3 \
  --diversity-sim-high 0.9 \
  --diversity-sim-low 0.7 \
  --log-each-round \
  --round-log-prefix "${DATA_DIR}/notLoRA_vllm/HPO_symptom_patient_expression_judge_refine.vllm_base.round" \
  --wandb \
  --wandb-project "HPO_patient_base_vllm" \
  --wandb-tags "patient,vllm,base"

echo "=== [Patient][NoLoRA][vLLM] 完了 ==="
echo



########################################
# Step 4: 医師表現 (mode=doctor) LoRA なし
########################################

echo "=== [Doctor][NoLoRA][vLLM] 開始 ==="

python "${UNIFIED_SCRIPT}" \
  --mode doctor \
  --hpo-csv "${DATA_DIR}/HPO_depth_ge3.csv" \
  --output "${DATA_DIR}/notLoRA_vllm/HPO_symptom_doctor_expression_judge_refine.vllm_base.jsonl" \
  --gen-model "${GEN_MODEL}" \
  --per-hpo "${DOC_PER_HPO}" \
  --target-good "${DOC_TARGET_GOOD}" \
  --gen-max-new-tokens "${DOC_GEN_MAX_NEW}" \
  --judge-max-new-tokens "${DOC_JUDGE_MAX_NEW}" \
  --refine-max-new-tokens "${DOC_REFINE_MAX_NEW}" \
  --min-overall 4 \
  --min-match 3 \
  --min-simplicity 3 \
  --max-rounds 3 \
  --diversity-sim-high 0.9 \
  --diversity-sim-low 0.7 \
  --log-each-round \
  --round-log-prefix "${DATA_DIR}/notLoRA_vllm/HPO_symptom_doctor_expression_judge_refine.vllm_base.round" \
  --wandb \
  --wandb-project "HPO_doctor_base_vllm" \
  --wandb-tags "doctor,vllm,base"

echo "=== [Doctor][NoLoRA][vLLM] 完了 ==="
echo "=== すべての vLLM パイプラインが完了しました ==="
