#!/bin/bash
set -euo pipefail

########################################
# 設定（必要ならここだけ書き換え）
########################################

# プロジェクトパス
BASE_DIR="/home/kiso_user/Documents/workspace/Research"
HPO_DIR="${BASE_DIR}/HPO-patient"
DATA_DIR="${BASE_DIR}/data"

# 統合パイプラインのスクリプト名（実際のファイル名に合わせて変更）
UNIFIED_SCRIPT="4.gen&judge&refine_new.py"

# モデル・LoRA
GEN_MODEL="EQUES/MedLLama3-JP-v2"                       # ベースモデル
PATIENT_LORA="${DATA_DIR}/eques_patient_expression_lora" # 患者表現 LoRA
DOCTOR_LORA="${DATA_DIR}/eques_doctor_expression_lora"   # 医師表現 LoRA

# GPU ID（0: RTX 3090, 1: RTX 3080）
GPU0=0   # 3090
GPU1=1   # 3080

# シャード総数と、各GPUに割り当てる shard ID
NUM_SHARDS_TOTAL=10

RE_GPU0_SHARDS=(1 2 3 4 5)  # 4 shards
RE_GPU1_SHARDS=(7 8 9)    # 3 shards
# 6:4 割り当て (3090:3080)
GPU0_SHARDS=(0 1 2 3 4 5)  # 6 shard
GPU1_SHARDS=(6 7 8 9)      # 4 shard

########################################
# パラメータ（max_new_tokens 等）
########################################

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
#   - ${DATA_DIR}/HPO_symptom_depth_leq3_self_reportable_with_jp.csv (医師用)
#   ※どちらも depth / symptom / self_reportable のフィルタ済みを想定
########################################

echo "BASE_DIR  = ${BASE_DIR}"
echo "HPO_DIR   = ${HPO_DIR}"
echo "DATA_DIR  = ${DATA_DIR}"
echo "GEN_MODEL = ${GEN_MODEL}"
echo "NUM_SHARDS_TOTAL = ${NUM_SHARDS_TOTAL}"
echo "GPU0_SHARDS = ${GPU0_SHARDS[*]}"
echo "GPU1_SHARDS = ${GPU1_SHARDS[*]}"
echo

cd "${HPO_DIR}"
mkdir -p "${DATA_DIR}/notLoRA"
mkdir -p "${DATA_DIR}/LoRA"



########################################
# Step 2: 医師表現 (mode=doctor) LoRAなし 2GPU 分散
########################################

echo "=== [Doctor] No LoRA 2GPU 分散生成 & Judge/Refine 開始 ==="

# GPU0 (3090) 用
(
  export CUDA_VISIBLE_DEVICES=${GPU0}

  for SID in "${RE_GPU0_SHARDS[@]}"; do
    echo "[Doctor][NoLoRA][GPU${GPU0}] shard ${SID}/${NUM_SHARDS_TOTAL} 実行中..."
    python "${UNIFIED_SCRIPT}" \
      --mode doctor \
      --hpo-csv "${DATA_DIR}/HPO_symptom_depth_leq3_with_jp.csv" \
      --output "${DATA_DIR}/notLoRA/HPO_doctor_expression_judge_refine.eques.shard${SID}_of${NUM_SHARDS_TOTAL}.jsonl" \
      --gen-model "${GEN_MODEL}" \
      --judge-model "${GEN_MODEL}" \
      --refine-model "${GEN_MODEL}" \
      --per-hpo "${DOC_PER_HPO}" \
      --target-good "${DOC_TARGET_GOOD}" \
      --gen-max-new-tokens "${DOC_GEN_MAX_NEW}" \
      --judge-max-new-tokens "${DOC_JUDGE_MAX_NEW}" \
      --refine-max-new-tokens "${DOC_REFINE_MAX_NEW}" \
      --num-shards "${NUM_SHARDS_TOTAL}" \
      --shard-id "${SID}"\
      --wandb \
      --wandb-project "HPO_doctor" \
      --wandb-tags "doctor,unified,no-lora" \
      --log-each-round \
      --batch-across-hpo

  done
) &
PID_DOC_GPU0=$!

# GPU1 (3080) 用
(
  export CUDA_VISIBLE_DEVICES=${GPU1}

  for SID in "${RE_GPU1_SHARDS[@]}"; do
    echo "[Doctor][NoLoRA][GPU${GPU1}] shard ${SID}/${NUM_SHARDS_TOTAL} 実行中..."
    python "${UNIFIED_SCRIPT}" \
      --mode doctor \
      --hpo-csv "${DATA_DIR}/HPO_symptom_depth_leq3_with_jp.csv" \
      --output "${DATA_DIR}/notLoRA/HPO_doctor_expression_judge_refine.eques.shard${SID}_of${NUM_SHARDS_TOTAL}.jsonl" \
      --gen-model "${GEN_MODEL}" \
      --judge-model "${GEN_MODEL}" \
      --refine-model "${GEN_MODEL}" \
      --per-hpo "${DOC_PER_HPO}" \
      --target-good "${DOC_TARGET_GOOD}" \
      --gen-max-new-tokens "${DOC_GEN_MAX_NEW}" \
      --judge-max-new-tokens "${DOC_JUDGE_MAX_NEW}" \
      --refine-max-new-tokens "${DOC_REFINE_MAX_NEW}" \
      --num-shards "${NUM_SHARDS_TOTAL}" \
      --shard-id "${SID}"\
      --wandb \
      --wandb-project "HPO_doctor" \
      --wandb-tags "doctor,unified,no-lora" \
      --log-each-round \
      --batch-across-hpo
  done
) &
PID_DOC_GPU1=$!

wait "${PID_DOC_GPU0}"
wait "${PID_DOC_GPU1}"

echo "=== [Doctor] No LoRA 全 shard 完了。マージします… ==="

OUT_DOC_NOTLORA="${DATA_DIR}/notLoRA/HPO_doctor_expression_judge_refine.eques.jsonl"
: > "${OUT_DOC_NOTLORA}"
for SID in $(seq 0 $((NUM_SHARDS_TOTAL-1))); do
  SHARD_PATH="${DATA_DIR}/notLoRA/HPO_doctor_expression_judge_refine.eques.shard${SID}_of${NUM_SHARDS_TOTAL}.jsonl"
  if [ ! -f "${SHARD_PATH}" ]; then
    echo "[ERROR] missing shard file: ${SHARD_PATH}" >&2
    exit 1
  fi
  cat "${SHARD_PATH}" >> "${OUT_DOC_NOTLORA}"
done


echo "=== [Doctor] No LoRA マージ完了: ${OUT_DOC_NOTLORA} ==="
echo "=== No LoRA 全処理完了 ==="
echo

########################################
# auto-batch キャッシュをクリア
########################################

AUTO_BATCH_CACHE="${HPO_DIR}/auto_batch_cache_unified.json"
if [ -f "${AUTO_BATCH_CACHE}" ]; then
  rm -f "${AUTO_BATCH_CACHE}"
  echo "auto-batch キャッシュをクリアしました: ${AUTO_BATCH_CACHE}"
fi

########################################
# Step 3: 患者表現 (mode=patient) LoRAあり 2GPU 分散
########################################

echo "=== [Patient] LoRA 2GPU 分散生成 & Judge/Refine 開始 ==="

# GPU0 (3090) 用
(
  export CUDA_VISIBLE_DEVICES=${GPU0}

  for SID in "${GPU0_SHARDS[@]}"; do
    echo "[Patient][LoRA][GPU${GPU0}] shard ${SID}/${NUM_SHARDS_TOTAL} 実行中..."
    python "${UNIFIED_SCRIPT}" \
      --mode patient \
      --hpo-csv "${DATA_DIR}/HPO_symptom_depth_leq3_self_reportable_with_jp.csv" \
      --output "${DATA_DIR}/LoRA/HPO_symptom_patient_expression_judge_refine.eques_lora.shard${SID}_of${NUM_SHARDS_TOTAL}.jsonl" \
      --gen-model "${GEN_MODEL}" \
      --gen-lora-path "${PATIENT_LORA}" \
      --judge-model "${GEN_MODEL}" \
      --refine-model "${GEN_MODEL}" \
      --refine-lora-path "${PATIENT_LORA}" \
      --per-hpo "${PAT_PER_HPO}" \
      --target-good "${PAT_TARGET_GOOD}" \
      --gen-max-new-tokens "${PAT_GEN_MAX_NEW}" \
      --judge-max-new-tokens "${PAT_JUDGE_MAX_NEW}" \
      --refine-max-new-tokens "${PAT_REFINE_MAX_NEW}" \
      --num-shards "${NUM_SHARDS_TOTAL}" \
      --shard-id "${SID}"\
      --wandb \
      --wandb-project "HPO_patient_lora" \
      --wandb-tags "patient,unified,lora" \
      --log-each-round \
      --batch-across-hpo
  done
) &
PID_PAT_GPU0=$!

# GPU1 (3080) 用
(
  export CUDA_VISIBLE_DEVICES=${GPU1}

  for SID in "${GPU1_SHARDS[@]}"; do
    echo "[Patient][LoRA][GPU${GPU1}] shard ${SID}/${NUM_SHARDS_TOTAL} 実行中..."
    python "${UNIFIED_SCRIPT}" \
      --mode patient \
      --hpo-csv "${DATA_DIR}/HPO_symptom_depth_leq3_self_reportable_with_jp.csv" \
      --output "${DATA_DIR}/LoRA/HPO_symptom_patient_expression_judge_refine.eques_lora.shard${SID}_of${NUM_SHARDS_TOTAL}.jsonl" \
      --gen-model "${GEN_MODEL}" \
      --gen-lora-path "${PATIENT_LORA}" \
      --judge-model "${GEN_MODEL}" \
      --refine-model "${GEN_MODEL}" \
      --refine-lora-path "${PATIENT_LORA}" \
      --per-hpo "${PAT_PER_HPO}" \
      --target-good "${PAT_TARGET_GOOD}" \
      --gen-max-new-tokens "${PAT_GEN_MAX_NEW}" \
      --judge-max-new-tokens "${PAT_JUDGE_MAX_NEW}" \
      --refine-max-new-tokens "${PAT_REFINE_MAX_NEW}" \
      --num-shards "${NUM_SHARDS_TOTAL}" \
      --shard-id "${SID}"\
      --wandb \
      --wandb-project "HPO_patient_lora" \
      --wandb-tags "patient,unified,lora" \
      --log-each-round \
      --batch-across-hpo
  done
) &
PID_PAT_GPU1=$!

wait "${PID_PAT_GPU0}"
wait "${PID_PAT_GPU1}"

echo "=== [Patient] LoRA 全 shard 完了。マージします… ==="

OUT_PAT_LORA="${DATA_DIR}/LoRA/HPO_symptom_patient_expression_judge_refine.eques_lora.jsonl"
: > "${OUT_PAT_LORA}"
for SID in $(seq 0 $((NUM_SHARDS_TOTAL-1))); do
  SHARD_PATH="${DATA_DIR}/LoRA/HPO_symptom_patient_expression_judge_refine.eques_lora.shard${SID}_of${NUM_SHARDS_TOTAL}.jsonl"
  if [ ! -f "${SHARD_PATH}" ]; then
    echo "[ERROR] missing shard file: ${SHARD_PATH}" >&2
    exit 1
  fi
  cat "${SHARD_PATH}" >> "${OUT_PAT_LORA}"
done

echo "=== [Patient] LoRA マージ完了: ${OUT_PAT_LORA} ==="
echo


########################################
# Step 4: 医師表現 (mode=doctor) LoRAあり 2GPU 分散
########################################

echo "=== [Doctor] LoRA 2GPU 分散生成 & Judge/Refine 開始 ==="

# GPU0 (3090) 用
(
  export CUDA_VISIBLE_DEVICES=${GPU0}

  for SID in "${GPU0_SHARDS[@]}"; do
    echo "[Doctor][LoRA][GPU${GPU0}] shard ${SID}/${NUM_SHARDS_TOTAL} 実行中..."
    python "${UNIFIED_SCRIPT}" \
      --mode doctor \
      --hpo-csv "${DATA_DIR}/HPO_symptom_depth_leq3_with_jp.csv" \
      --output "${DATA_DIR}/LoRA/HPO_doctor_expression_judge_refine.eques_lora.shard${SID}_of${NUM_SHARDS_TOTAL}.jsonl" \
      --gen-model "${GEN_MODEL}" \
      --gen-lora-path "${DOCTOR_LORA}" \
      --judge-model "${GEN_MODEL}" \
      --refine-model "${GEN_MODEL}" \
      --refine-lora-path "${DOCTOR_LORA}" \
      --per-hpo "${DOC_PER_HPO}" \
      --target-good "${DOC_TARGET_GOOD}" \
      --gen-max-new-tokens "${DOC_GEN_MAX_NEW}" \
      --judge-max-new-tokens "${DOC_JUDGE_MAX_NEW}" \
      --refine-max-new-tokens "${DOC_REFINE_MAX_NEW}" \
      --num-shards "${NUM_SHARDS_TOTAL}" \
      --shard-id "${SID}"\
      --wandb \
      --wandb-project "HPO_doctor_lora" \
      --wandb-tags "doctor,unified,lora" \
      --log-each-round \
      --batch-across-hpo
  done
) &
PID_DOC_GPU0=$!

# GPU1 (3080) 用
(
  export CUDA_VISIBLE_DEVICES=${GPU1}

  for SID in "${GPU1_SHARDS[@]}"; do
    echo "[Doctor][LoRA][GPU${GPU1}] shard ${SID}/${NUM_SHARDS_TOTAL} 実行中..."
    python "${UNIFIED_SCRIPT}" \
      --mode doctor \
      --hpo-csv "${DATA_DIR}/HPO_symptom_depth_leq3_with_jp.csv" \
      --output "${DATA_DIR}/LoRA/HPO_doctor_expression_judge_refine.eques_lora.shard${SID}_of${NUM_SHARDS_TOTAL}.jsonl" \
      --gen-model "${GEN_MODEL}" \
      --gen-lora-path "${DOCTOR_LORA}" \
      --judge-model "${GEN_MODEL}" \
      --refine-model "${GEN_MODEL}" \
      --refine-lora-path "${DOCTOR_LORA}" \
      --per-hpo "${DOC_PER_HPO}" \
      --target-good "${DOC_TARGET_GOOD}" \
      --gen-max-new-tokens "${DOC_GEN_MAX_NEW}" \
      --judge-max-new-tokens "${DOC_JUDGE_MAX_NEW}" \
      --refine-max-new-tokens "${DOC_REFINE_MAX_NEW}" \
      --num-shards "${NUM_SHARDS_TOTAL}" \
      --shard-id "${SID}"\
      --wandb \
      --wandb-project "HPO_doctor_lora" \
      --wandb-tags "doctor,unified,lora" \
      --log-each-round \
      --batch-across-hpo
  done
) &
PID_DOC_GPU1=$!

wait "${PID_DOC_GPU0}"
wait "${PID_DOC_GPU1}"

echo "=== [Doctor] LoRA 全 shard 完了。マージします… ==="

OUT_DOC_LORA="${DATA_DIR}/LoRA/HPO_doctor_expression_judge_refine.eques_lora.jsonl"
: > "${OUT_DOC_LORA}"
for SID in $(seq 0 $((NUM_SHARDS_TOTAL-1))); do
  SHARD_PATH="${DATA_DIR}/LoRA/HPO_doctor_expression_judge_refine.eques_lora.shard${SID}_of${NUM_SHARDS_TOTAL}.jsonl"
  if [ ! -f "${SHARD_PATH}" ]; then
    echo "[ERROR] missing shard file: ${SHARD_PATH}" >&2
    exit 1
  fi
  cat "${SHARD_PATH}" >> "${OUT_DOC_LORA}"
done

echo "=== [Doctor] LoRA マージ完了: ${OUT_DOC_LORA} ==="
echo "=== 全処理完了 ==="
