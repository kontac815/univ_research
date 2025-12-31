#!/bin/bash
set -euo pipefail
source ~/.bashrc

########################################
# SFT (Swallow Instruct + ELYZA) -> gen-only (6 conditions × 2 modes)
# - FULL SFT ignored
# - EQUES QLoRA adapter is assumed to exist already
########################################

# ===== Paths =====
BASE_DIR="/home/kiso_user/Documents/workspace/Research"
HPO_DIR="${BASE_DIR}/HPO-patient"
DATA_DIR="${BASE_DIR}/data"

SFT_SCRIPT="2d.train_sft_qlora.py"
DS_CONFIG="deepspeed_config_qlora_zero2.json"
GEN_SCRIPT="vllm_pipeline_2.py"   # generate-only版

# ===== Dataset =====
TRAIN_FILES=( "../data/naist_patient_hpo_sft.jsonl" "../data/manbyo_doctor_hpo_sft.jsonl" )

# ===== GPU =====
# 3090がGPU1の想定
GPU_TRAIN=1
GPU_GEN=1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ===== W&B (for SFT + gen) =====
export WANDB__SERVICE_WAIT=300
export WANDB_CONSOLE=wrap
# export WANDB_API_KEY="..."  # 既に設定済みなら不要

# ===== Models =====
EQUES_BASE="EQUES/MedLLama3-JP-v2"
ELYZA_BASE="elyza/Llama-3-ELYZA-JP-8B"
SWALLOW_BASE="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"

# ===== Adapters (output dirs) =====
# 既存EQUES（すでに学習済み想定）
EQUES_QLORA_ADAPTER="${HPO_DIR}/out/qlora-hpo-sft"

# これから学習する
ELYZA_QLORA_ADAPTER="${HPO_DIR}/out/qlora-elyza-hpo-sft"
SWALLOW_QLORA_ADAPTER="${HPO_DIR}/out/qlora-swallow-instruct-hpo-sft"

# ===== SFT hyperparams (safe defaults) =====
SFT_EPOCHS=2
SFT_LR=2e-4
SFT_MAXLEN=384
SFT_BS=4
SFT_GAS=8
SFT_BF16="--bf16"
SFT_BNB="--bnb-4bit-use-double-quant"

# ===== Generation params (same style as your script) =====
# Patient
PAT_PER_HPO=200
PAT_TARGET_GOOD=200
PAT_GEN_MAX_NEW=24
PAT_JUDGE_MAX_NEW=48
PAT_REFINE_MAX_NEW=24

# Doctor
DOC_PER_HPO=50
DOC_TARGET_GOOD=50
DOC_GEN_MAX_NEW=32
DOC_JUDGE_MAX_NEW=48
DOC_REFINE_MAX_NEW=32

# ===== Input CSV =====
PAT_HPO_CSV="${DATA_DIR}/HPO_depth_ge3_self_reportable.csv"
DOC_HPO_CSV="${DATA_DIR}/HPO_depth_ge3.csv"

# ===== Outputs =====
OUT_BASE="${DATA_DIR}/gen_only_vllm"
LOG_BASE="${OUT_BASE}/logs"
mkdir -p "${OUT_BASE}" "${LOG_BASE}"

timestamp() { date +"%Y%m%d_%H%M%S"; }

need_file() { [[ -f "$1" ]] || { echo "[ERROR] missing file: $1"; exit 1; }; }
need_dir_or_warn() { [[ -d "$1" ]] || echo "[WARN] dir not found yet: $1"; }

########################################
# SFT runner
########################################
run_sft () {
  local base_model="$1"
  local out_dir="$2"
  local tag="$3"         # for wandb tags & naming

  echo "============================================================"
  echo "[SFT] model=${base_model}"
  echo "      out  =${out_dir}"
  echo "      tag  =${tag}"
  echo "============================================================"

  export CUDA_VISIBLE_DEVICES="${GPU_TRAIN}"

  # bitsandbytes sanity check
  python -c "import bitsandbytes as bnb; print('bitsandbytes ok', bnb.__version__)" >/dev/null 2>&1 || {
    echo "[ERROR] bitsandbytes is not importable in this env. Install it first."
    exit 1
  }

  # skip if already exists (adapter dir has adapter_config.json)
  if [[ -f "${out_dir}/adapter_config.json" ]]; then
    echo "[SKIP] adapter already exists: ${out_dir}/adapter_config.json"
    echo
    return
  fi

  deepspeed --num_gpus=1 "${SFT_SCRIPT}" \
    --model-name-or-path "${base_model}" \
    --train-files "${TRAIN_FILES[@]}" \
    --output-dir "${out_dir}" \
    --deepspeed "${DS_CONFIG}" \
    --num-epochs "${SFT_EPOCHS}" \
    --learning-rate "${SFT_LR}" \
    --per-device-train-batch-size "${SFT_BS}" \
    --gradient-accumulation-steps "${SFT_GAS}" \
    --max-length "${SFT_MAXLEN}" \
    ${SFT_BF16} \
    ${SFT_BNB} \
    --wandb \
    --wandb-project "HPO_SFT" \
    --wandb-tags "${tag},qlora,sft" \
    --wandb-group "sft_${tag}" \
    --wandb-run-name "sft_${tag}_$(timestamp)"

  echo "[DONE][SFT] ${tag}"
  echo
}

########################################
# Gen-only runner (vllm_pipeline_2.py)
########################################
run_one () {
  local mode="$1"          # patient / doctor
  local model="$2"         # --gen-model
  local tag="$3"           # output tag
  local lora_path="${4:-}" # optional
  local wb_project="$5"
  local wb_tags="$6"

  export CUDA_VISIBLE_DEVICES="${GPU_GEN}"

  local hpo_csv per_hpo target_good gen_max_new judge_max_new refine_max_new
  if [[ "${mode}" == "patient" ]]; then
    hpo_csv="${PAT_HPO_CSV}"
    per_hpo="${PAT_PER_HPO}"
    target_good="${PAT_TARGET_GOOD}"
    gen_max_new="${PAT_GEN_MAX_NEW}"
    judge_max_new="${PAT_JUDGE_MAX_NEW}"
    refine_max_new="${PAT_REFINE_MAX_NEW}"
  else
    hpo_csv="${DOC_HPO_CSV}"
    per_hpo="${DOC_PER_HPO}"
    target_good="${DOC_TARGET_GOOD}"
    gen_max_new="${DOC_GEN_MAX_NEW}"
    judge_max_new="${DOC_JUDGE_MAX_NEW}"
    refine_max_new="${DOC_REFINE_MAX_NEW}"
  fi

  local out_dir="${OUT_BASE}/${tag}"
  mkdir -p "${out_dir}"

  local out_jsonl="${out_dir}/HPO_${mode}_expression.gen_only.${tag}.jsonl"
  local log_file="${LOG_BASE}/run_${mode}_${tag}_$(timestamp).log"

  echo "============================================================"
  echo "[GEN] mode=${mode}"
  echo "      model=${model}"
  echo "      lora =${lora_path:-none}"
  echo "      csv  =${hpo_csv}"
  echo "      out  =${out_jsonl}"
  echo "      log  =${log_file}"
  echo "============================================================"

  export WANDB_PROJECT="${wb_project}"
  export WANDB_TAGS="${wb_tags}"
  export WANDB_RUN_GROUP="gen_only_${tag}"
  export WANDB_NAME="gen_only_${mode}_${tag}_$(timestamp)"

  if [[ -n "${lora_path}" ]]; then
    python "${GEN_SCRIPT}" \
      --mode "${mode}" \
      --hpo-csv "${hpo_csv}" \
      --output "${out_jsonl}" \
      --gen-model "${model}" \
      --gen-lora-path "${lora_path}" \
      --per-hpo "${per_hpo}" \
      --target-good "${target_good}" \
      --gen-max-new-tokens "${gen_max_new}" \
      --judge-max-new-tokens "${judge_max_new}" \
      --refine-max-new-tokens "${refine_max_new}" \
      --min-overall 4 \
      --min-match 3 \
      --min-simplicity 3 \
      --max-rounds 3 \
      --diversity-sim-high 0.9 \
      --diversity-sim-low 0.7 \
      --wandb \
      --wandb-project "${wb_project}" \
      --wandb-tags "${wb_tags}" \
      2>&1 | tee "${log_file}"
  else
    python "${GEN_SCRIPT}" \
      --mode "${mode}" \
      --hpo-csv "${hpo_csv}" \
      --output "${out_jsonl}" \
      --gen-model "${model}" \
      --per-hpo "${per_hpo}" \
      --target-good "${target_good}" \
      --gen-max-new-tokens "${gen_max_new}" \
      --judge-max-new-tokens "${judge_max_new}" \
      --refine-max-new-tokens "${refine_max_new}" \
      --min-overall 4 \
      --min-match 3 \
      --min-simplicity 3 \
      --max-rounds 3 \
      --diversity-sim-high 0.9 \
      --diversity-sim-low 0.7 \
      --wandb \
      --wandb-project "${wb_project}" \
      --wandb-tags "${wb_tags}" \
      2>&1 | tee "${log_file}"
  fi

  echo "[DONE][GEN] ${mode} ${tag}"
  echo
}

########################################
# Main
########################################
cd "${HPO_DIR}"

need_file "${SFT_SCRIPT}"
need_file "${DS_CONFIG}"
need_file "${GEN_SCRIPT}"
need_file "${PAT_HPO_CSV}"
need_file "${DOC_HPO_CSV}"

echo "BASE_DIR  = ${BASE_DIR}"
echo "HPO_DIR   = ${HPO_DIR}"
echo "DATA_DIR  = ${DATA_DIR}"
echo "GPU_TRAIN = ${GPU_TRAIN}"
echo "GPU_GEN   = ${GPU_GEN}"
echo "OUT_BASE  = ${OUT_BASE}"
echo

# ---- 0) SFT (Swallow Instruct, ELYZA) ----
run_sft "${SWALLOW_BASE}" "${SWALLOW_QLORA_ADAPTER}" "swallow_instruct_v05"
run_sft "${ELYZA_BASE}"   "${ELYZA_QLORA_ADAPTER}"   "elyza_llama3_jp8b"

# ---- 1) Gen-only: EQUES (qlora/base) ----
need_dir_or_warn "${EQUES_QLORA_ADAPTER}"
run_one patient "${EQUES_BASE}" "eques_qlora" "${EQUES_QLORA_ADAPTER}" \
  "HPO_genonly_eques" "generate-only,patient,eques,qlora"
run_one doctor  "${EQUES_BASE}" "eques_qlora" "${EQUES_QLORA_ADAPTER}" \
  "HPO_genonly_eques" "generate-only,doctor,eques,qlora"

run_one patient "${EQUES_BASE}" "eques_base" "" \
  "HPO_genonly_eques" "generate-only,patient,eques,base"
run_one doctor  "${EQUES_BASE}" "eques_base" "" \
  "HPO_genonly_eques" "generate-only,doctor,eques,base"

# ---- 2) Gen-only: ELYZA (qlora/base) ----
need_dir_or_warn "${ELYZA_QLORA_ADAPTER}"
run_one patient "${ELYZA_BASE}" "elyza_qlora" "${ELYZA_QLORA_ADAPTER}" \
  "HPO_genonly_elyza" "generate-only,patient,elyza,qlora"
run_one doctor  "${ELYZA_BASE}" "elyza_qlora" "${ELYZA_QLORA_ADAPTER}" \
  "HPO_genonly_elyza" "generate-only,doctor,elyza,qlora"

run_one patient "${ELYZA_BASE}" "elyza_base" "" \
  "HPO_genonly_elyza" "generate-only,patient,elyza,base"
run_one doctor  "${ELYZA_BASE}" "elyza_base" "" \
  "HPO_genonly_elyza" "generate-only,doctor,elyza,base"

# ---- 3) Gen-only: Swallow Instruct (qlora/base) ----
need_dir_or_warn "${SWALLOW_QLORA_ADAPTER}"
run_one patient "${SWALLOW_BASE}" "swallow_instruct_qlora" "${SWALLOW_QLORA_ADAPTER}" \
  "HPO_genonly_swallow" "generate-only,patient,swallow,instruct,qlora"
run_one doctor  "${SWALLOW_BASE}" "swallow_instruct_qlora" "${SWALLOW_QLORA_ADAPTER}" \
  "HPO_genonly_swallow" "generate-only,doctor,swallow,instruct,qlora"

run_one patient "${SWALLOW_BASE}" "swallow_instruct_base" "" \
  "HPO_genonly_swallow" "generate-only,patient,swallow,instruct,base"
run_one doctor  "${SWALLOW_BASE}" "swallow_instruct_base" "" \
  "HPO_genonly_swallow" "generate-only,doctor,swallow,instruct,base"

echo "=== ALL DONE (SFT -> gen-only) ==="
