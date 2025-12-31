#!/bin/bash
set -euo pipefail
source ~/.bashrc
########################################
# vLLM generate-only multi-model runner
# - same data paths + generation counts as existing vllm_pipeline.sh
########################################

# ====== Paths (same as existing vllm_pipeline.sh) ======
BASE_DIR="/home/kiso_user/Documents/workspace/Research"
HPO_DIR="${BASE_DIR}/HPO-patient"
DATA_DIR="${BASE_DIR}/data"

UNIFIED_SCRIPT="vllm_pipeline_2.py"   # ← generate-only版に置き換え済みの想定

# ====== GPU ======
GPU_ID=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# ====== Generation params (same as existing vllm_pipeline.sh) ======
# Patient
PAT_PER_HPO=200
PAT_TARGET_GOOD=200
PAT_GEN_MAX_NEW=24
PAT_JUDGE_MAX_NEW=48      # 互換のため渡す（generate-onlyでは未使用でもOK）
PAT_REFINE_MAX_NEW=24     # 互換のため渡す（generate-onlyでは未使用でもOK）

# Doctor
DOC_PER_HPO=50
DOC_TARGET_GOOD=50
DOC_GEN_MAX_NEW=32
DOC_JUDGE_MAX_NEW=48      # 互換のため渡す（generate-onlyでは未使用でもOK）
DOC_REFINE_MAX_NEW=32     # 互換のため渡す（generate-onlyでは未使用でもOK）

# ====== Input CSV (same as existing vllm_pipeline.sh) ======
PAT_HPO_CSV="${DATA_DIR}/HPO_depth_ge3_self_reportable.csv"
DOC_HPO_CSV="${DATA_DIR}/HPO_depth_ge3.csv"

# ====== Outputs base dir ======
OUT_BASE="${DATA_DIR}/gen_only_vllm"
LOG_BASE="${OUT_BASE}/logs"
mkdir -p "${OUT_BASE}" "${LOG_BASE}"

# ====== W&B (set your env if needed) ======
# export WANDB_API_KEY="..."   # 既に設定済みなら不要
export WANDB__SERVICE_WAIT=300
export WANDB_CONSOLE=wrap

# ====== Models ======
# (1) QLoRA SFT: base + adapter (LoRA)
EQUES_BASE="EQUES/MedLLama3-JP-v2"
QLORA_ADAPTER_PATH="${HPO_DIR}/out/qlora-hpo-sft"   # ★必要に応じて変更（あなたの出力先）

# (2) Base (no LoRA): EQUES_BASE をそのまま使う

# (3) Full FT model (merged model dir / HF repo). ★後で指定
FULLFT_MODEL_PATH="__REPLACE_ME_FULLFT_MODEL_PATH__"

# (4) Other LLMs
SWALLOW_MODEL="tokyotech-llm/Llama-3.1-Swallow-8B-v0.5"
ELYZA_MODEL="elyza/Llama-3-ELYZA-JP-8B"

# ====== Helper ======
timestamp() { date +"%Y%m%d_%H%M%S"; }

run_one () {
  local mode="$1"          # patient / doctor
  local model="$2"         # --gen-model
  local tag="$3"           # used for output naming
  local lora_path="${4:-}" # optional
  local wb_project="$5"    # wandb project
  local wb_tags="$6"       # comma-separated tags

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
  echo "[RUN] mode=${mode}"
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

  # NOTE:
  # generate-only版でも互換のため judge/refine系引数は渡す（未使用でもOK）
  if [[ -n "${lora_path}" ]]; then
    python "${UNIFIED_SCRIPT}" \
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
    python "${UNIFIED_SCRIPT}" \
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

  echo "[DONE] ${mode} ${tag}"
  echo
}

########################################
# Main
########################################

echo "BASE_DIR  = ${BASE_DIR}"
echo "HPO_DIR   = ${HPO_DIR}"
echo "DATA_DIR  = ${DATA_DIR}"
echo "GPU_ID    = ${GPU_ID}"
echo "OUT_BASE  = ${OUT_BASE}"
echo

cd "${HPO_DIR}"

# 0) Safety checks
if [[ ! -f "${UNIFIED_SCRIPT}" ]]; then
  echo "[ERROR] ${UNIFIED_SCRIPT} not found in ${HPO_DIR}"
  exit 1
fi
if [[ ! -f "${PAT_HPO_CSV}" ]]; then
  echo "[ERROR] Patient CSV not found: ${PAT_HPO_CSV}"
  exit 1
fi
if [[ ! -f "${DOC_HPO_CSV}" ]]; then
  echo "[ERROR] Doctor CSV not found: ${DOC_HPO_CSV}"
  exit 1
fi

# ------------------------------------------------------------
# 1) QLoRA SFT model (EQUES base + adapter)
# ------------------------------------------------------------
# ※ QLoRA adapter のパスが違うなら QLORA_ADAPTER_PATH を直してください
run_one patient "${EQUES_BASE}" "eques_qlora_sft" "${QLORA_ADAPTER_PATH}" \
  "HPO_genonly_eques_qlora" "generate-only,patient,eques,qlora"
run_one doctor  "${EQUES_BASE}" "eques_qlora_sft" "${QLORA_ADAPTER_PATH}" \
  "HPO_genonly_eques_qlora" "generate-only,doctor,eques,qlora"

# ------------------------------------------------------------
# 2) Base (no SFT) EQUES/MedLLama3-JP-v2
# ------------------------------------------------------------
run_one patient "${EQUES_BASE}" "eques_base" "" \
  "HPO_genonly_eques_base" "generate-only,patient,eques,base"
run_one doctor  "${EQUES_BASE}" "eques_base" "" \
  "HPO_genonly_eques_base" "generate-only,doctor,eques,base"

# ------------------------------------------------------------
# 3) Full fine-tuned model (merged model dir / HF repo)
# ------------------------------------------------------------
# ★ここ後はで差し替え
if [[ "${FULLFT_MODEL_PATH}" != "__REPLACE_ME_FULLFT_MODEL_PATH__" ]]; then
  run_one patient "${FULLFT_MODEL_PATH}" "eques_fullft" "" \
    "HPO_genonly_eques_fullft" "generate-only,patient,eques,fullft"
  run_one doctor  "${FULLFT_MODEL_PATH}" "eques_fullft" "" \
    "HPO_genonly_eques_fullft" "generate-only,doctor,eques,fullft"
else
  echo "[SKIP] FULLFT_MODEL_PATH is not set. (tag=eques_fullft)"
  echo
fi

# ------------------------------------------------------------
# 4) Llama 3.1 Swallow 8B v0.5
# ------------------------------------------------------------
run_one patient "${SWALLOW_MODEL}" "swallow_8b_v0_5" "" \
  "HPO_genonly_swallow" "generate-only,patient,swallow"
run_one doctor  "${SWALLOW_MODEL}" "swallow_8b_v0_5" "" \
  "HPO_genonly_swallow" "generate-only,doctor,swallow"

# ------------------------------------------------------------
# 5) elyza/Llama-3-ELYZA-JP-8B
# ------------------------------------------------------------
run_one patient "${ELYZA_MODEL}" "elyza_llama3_jp_8b" "" \
  "HPO_genonly_elyza" "generate-only,patient,elyza"
run_one doctor  "${ELYZA_MODEL}" "elyza_llama3_jp_8b" "" \
  "HPO_genonly_elyza" "generate-only,doctor,elyza"

echo "=== ALL DONE (generate-only, multi-model) ==="