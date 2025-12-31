#!/bin/bash
set -euo pipefail
source ~/.bashrc || true

########################################
# End-to-end runner:
#  SFT(ELYZA+Swallow) -> gen-only(6cond x 2mode)
#  -> postprocess(5a single setting) -> embed -> annotate -> eval
#  - skip failed condition
#  - write summary TSV
#  - W&B: SFTはTrainer経由でログ、gen/evalは後段で手動wandb log
########################################

# -------------------------
# Paths
# -------------------------
BASE_DIR="/home/kiso_user/Documents/workspace/Research"
HPO_DIR="${BASE_DIR}/HPO-patient"
DATA_DIR="${BASE_DIR}/data"

# Scripts
SFT_SCRIPT="${HPO_DIR}/2d.train_sft_qlora.py"
DS_CFG="${HPO_DIR}/deepspeed_config_qlora_zero2.json"
GEN_SCRIPT="${HPO_DIR}/vllm_pipeline_2.py"                 # generate-only
POST_SCRIPT="${HPO_DIR}/5a.postprocess_expressions_sbert.py"
EMBED_SCRIPT="${HPO_DIR}/embed_with_sbert.py"
ANNOT_SCRIPT="${HPO_DIR}/annotate_with_medtxtner_hpo_batch.py"
EVAL_SCRIPT="${HPO_DIR}/eval_hpo_annotation_medtxtner.py"

# Data
TRAIN_FILES=(
  "${DATA_DIR}/naist_patient_hpo_sft.jsonl"
  "${DATA_DIR}/manbyo_doctor_hpo_sft.jsonl"
)
HPO_MASTER_CSV="${DATA_DIR}/HPO_depth_ge3.csv"

TEST_CSV="${DATA_DIR}/end2end_runs/20251214_2200/eval/test_head10.csv"
ID_COL="patient_name"
TEXT_COL="medical_history_nobr"
GOLD_COL="hpo_ids"
HPO_OBO="${DATA_DIR}/hp.obo"
# 任意: テストデータの先頭N件だけで評価したい場合に指定
TEST_HEAD_N="${TEST_HEAD_N:-}"

# -------------------------
# Models (SwallowはInstruct推奨)
# -------------------------
EQUES_BASE="EQUES/MedLLama3-JP-v2"
ELYZA_BASE="elyza/Llama-3-ELYZA-JP-8B"
SWALLOW_BASE="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"

# QLoRA adapters (SFT出力先)
# ※EQUESは既に out/qlora-hpo-sft があるならそれを使う
EQUES_LORA_DIR="${HPO_DIR}/out/qlora-hpo-sft"
ELYZA_LORA_DIR="${HPO_DIR}/out/qlora-hpo-sft_elyza"
SWALLOW_LORA_DIR="${HPO_DIR}/out/qlora-hpo-sft_swallow"

# -------------------------
# GPU (あなたの環境: GPU1=3090 を使う想定)
# -------------------------
GPU_ID="${GPU_ID:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# fragmentation対策（PyTorch新推奨名）
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# -------------------------
# Run switches
# -------------------------
DO_SFT_SWALLOW="${DO_SFT_SWALLOW:-0}"
DO_SFT_ELYZA="${DO_SFT_ELYZA:-0}"
DO_SFT_EQUES="${DO_SFT_EQUES:-0}"   # 既に学習済みなら0でOK
DO_GEN="${DO_GEN:-0}"
DO_EVAL="${DO_EVAL:-1}"
# 患者・医師を個別に評価する従来フローに加え、先にマージしてから後処理/埋め込み/評価するか
DO_MERGED="${DO_MERGED:-1}"
# patient のみ再生成したい場合は MODES="patient"（カンマ/空白区切り可）
MODES="${MODES:-patient doctor}"
read -r -a MODE_LIST <<< "${MODES//,/ }"

# -------------------------
# Generation params
# -------------------------
PAT_HPO_CSV="${DATA_DIR}/HPO_depth_ge3_self_reportable.csv"
DOC_HPO_CSV="${DATA_DIR}/HPO_depth_ge3.csv"

PAT_PER_HPO=200
PAT_TARGET_GOOD=200
PAT_GEN_MAX_NEW=24

DOC_PER_HPO=50
DOC_TARGET_GOOD=50
DOC_GEN_MAX_NEW=32

# -------------------------
# Postprocess/Embed/Annot hyperparams (single setting)
# -------------------------
SBERT_MODEL="pkshatech/GLuCoSE-base-ja-v2"
SBERT_BATCH=64

SELF_MIN_SIM=0.0
OTHER_MARGIN=0.10
SIM_HIGH=0.9
SIM_LOW=0.7

NER_MODEL="sociocom/MedTXTNER"
NER_LABEL_PREFIXES="d,cc"
USE_NER_REGION=1
MAX_N=6
TOPK=15
MIN_SCORE=0.70
# Span selection (non-overlapping DP)
# - 長いスパンを少し優遇して複合語の分裂を抑える
# - 隣接スパンを競合扱いにする（分割抑制; 必要なときだけON推奨）
SPAN_LEN_BONUS_ALPHA="${SPAN_LEN_BONUS_ALPHA:-0.15}"
SPAN_LEN_BONUS_BETA="${SPAN_LEN_BONUS_BETA:-0.05}"
TREAT_ADJACENT_AS_OVERLAP="${TREAT_ADJACENT_AS_OVERLAP:-0}"

# -------------------------
# Output base
# -------------------------
timestamp() { date +"%Y%m%d_%H%M%S"; }
RUN_TS="20251214_2200"    # 必要なら手動で指定して doctor JSONL を先にコピーできる
OUT_BASE="${DATA_DIR}/end2end_runs/${RUN_TS}"
LOG_BASE="${OUT_BASE}/logs"
GEN_BASE="${OUT_BASE}/gen"
EVAL_BASE="${OUT_BASE}/eval"
mkdir -p "${OUT_BASE}" "${LOG_BASE}" "${GEN_BASE}" "${EVAL_BASE}"

SUMMARY_TSV="${OUT_BASE}/summary.tsv"
echo -e "cond_tag\tmode\traw_gen_lines\tpost_total\tpost_gen\tpost_official\tremoved_lines\tpred_lines\texact_f1\thier_f1\trecall@1\trecall@3\trecall@5\trecall@10\tmetrics_tsv\tpred_jsonl\tpost_jsonl" \
  > "${SUMMARY_TSV}"

# -------------------------
# Optional: test CSV slicing
# -------------------------
TEST_CSV_USED="${TEST_CSV}"
if [[ -n "${TEST_HEAD_N}" ]]; then
  TEST_CSV_USED="${EVAL_BASE}/test_head${TEST_HEAD_N}.csv"
  echo "[INFO] Using first ${TEST_HEAD_N} rows of ${TEST_CSV} (with header) -> ${TEST_CSV_USED}"
  mkdir -p "${EVAL_BASE}"
  { head -n 1 "${TEST_CSV}"; tail -n +2 "${TEST_CSV}" | head -n "${TEST_HEAD_N}"; } > "${TEST_CSV_USED}"
fi

# -------------------------
# W&B (optional)
# -------------------------
ENABLE_WANDB="${ENABLE_WANDB:-1}"               # 各ステージの wandb logging
ENABLE_WANDB_SUMMARY="${ENABLE_WANDB_SUMMARY:-0}" # 手動サマリーログ（従来挙動）
export WANDB__SERVICE_WAIT=300
export WANDB_CONSOLE=wrap

WB_PROJECT_GEN="${WB_PROJECT_GEN:-HPO_gen_only}"
WB_PROJECT_POST="${WB_PROJECT_POST:-HPO_postprocess}"
WB_PROJECT_EMBED="${WB_PROJECT_EMBED:-HPO_embed}"
WB_PROJECT_ANNOT="${WB_PROJECT_ANNOT:-HPO_annotate}"
WB_PROJECT_EVAL="${WB_PROJECT_EVAL:-HPO_eval}"

wandb_log_if_possible () {
  local project="$1"
  local name="$2"
  local group="$3"
  local tags_csv="$4"
  local summary_json="$5"   # json string

  if [[ "${ENABLE_WANDB_SUMMARY}" != "1" ]]; then
    return 0
  fi

  python - <<'PY' || true
import os, json, sys
try:
    import wandb
except Exception:
    sys.exit(0)

project = os.environ.get("WB_PROJECT")
name    = os.environ.get("WB_NAME")
group   = os.environ.get("WB_GROUP")
tags    = os.environ.get("WB_TAGS","")
payload = os.environ.get("WB_JSON","{}")

tags_list = [t.strip() for t in tags.split(",") if t.strip()]
data = json.loads(payload)

wandb.init(project=project, name=name, group=group, tags=tags_list, reinit=True)
wandb.log(data)
wandb.finish()
PY
}

build_eval_summary_json () {
python - <<'PY'
import json
from pathlib import Path
import os

post = Path(os.environ["POST"])
removed = Path(os.environ["REMOVED"])
pred = Path(os.environ["PRED"])
metrics = Path(os.environ["METRICS"])

def wc(p: Path)->int:
    if not p.exists(): return 0
    return sum(1 for _ in p.open("r", encoding="utf-8", errors="ignore"))

post_total = wc(post)
removed_lines = wc(removed)
pred_lines = wc(pred)

post_gen = 0
post_official = 0
if post.exists():
    for line in post.open("r", encoding="utf-8", errors="ignore"):
        line=line.strip()
        if not line: continue
        try:
            obj=json.loads(line)
        except Exception:
            continue
        if obj.get("source","") == "official":
            post_official += 1
        else:
            post_gen += 1

m = {}
if metrics.exists():
    for line in metrics.open("r", encoding="utf-8", errors="ignore"):
        line=line.strip()
        if not line or line.startswith("#"): 
            continue
        parts=line.split("\t")
        if len(parts) >= 2:
            m[parts[0]] = parts[1]

payload = {
  "raw_gen_lines": int(os.environ["RAW_LINES"]),
  "post_total": post_total,
  "post_gen": post_gen,
  "post_official": post_official,
  "removed_lines": removed_lines,
  "pred_lines": pred_lines,
}
# numeric metrics only
for k,v in m.items():
    try:
        payload[k] = float(v)
    except Exception:
        pass

print(json.dumps(payload, ensure_ascii=False))
PY
}

# -------------------------
# helpers: run step with skip-on-error
# -------------------------
run_step () {
  local title="$1"; shift
  local log="${LOG_BASE}/${title}.${RUN_TS}.log"
  echo "============================================================"
  echo "[STEP] ${title}"
  echo "CMD : $*"
  echo "LOG : ${log}"
  echo "============================================================"
  if "$@" 2>&1 | tee -a "${log}"; then
    echo "[OK] ${title}"
    return 0
  else
    echo "[FAIL] ${title} (skipping this branch)"
    return 1
  fi
}

# -------------------------
# SFT (QLoRA + DeepSpeed, single GPU)
# -------------------------
run_sft () {
  local base_model="$1"
  local out_dir="$2"
  local tag="$3"

  mkdir -p "${out_dir}"

  # W&B name/group
  export WANDB_PROJECT="HPO_SFT_${tag}"
  export WANDB_NAME="sft_${tag}_${RUN_TS}"
  export WANDB_RUN_GROUP="sft_${tag}"

  run_step "SFT_${tag}" \
    deepspeed "${SFT_SCRIPT}" \
      --model-name-or-path "${base_model}" \
      --train-files "${TRAIN_FILES[@]}" \
      --output-dir "${out_dir}" \
      --deepspeed "${DS_CFG}" \
      --num-epochs 2 \
      --learning-rate 2e-4 \
      --per-device-train-batch-size 2 \
      --gradient-accumulation-steps 16 \
      --max-length 384 \
      --bf16 \
      --bnb-4bit-use-double-quant \
      --wandb \
      --wandb-project "${WANDB_PROJECT}" \
      --wandb-tags "sft,qlora,${tag}" \
      --wandb-run-name "${WANDB_NAME}" \
      --wandb-group "${WANDB_RUN_GROUP}"
}

# -------------------------
# Gen-only
# -------------------------
gen_one () {
  local mode="$1"          # patient/doctor
  local model="$2"         # base model
  local tag="$3"           # condition tag
  local lora_path="${4:-}" # optional

  local hpo_csv per_hpo target_good gen_max_new
  if [[ "${mode}" == "patient" ]]; then
    hpo_csv="${PAT_HPO_CSV}"
    per_hpo="${PAT_PER_HPO}"
    target_good="${PAT_TARGET_GOOD}"
    gen_max_new="${PAT_GEN_MAX_NEW}"
  else
    hpo_csv="${DOC_HPO_CSV}"
    per_hpo="${DOC_PER_HPO}"
    target_good="${DOC_TARGET_GOOD}"
    gen_max_new="${DOC_GEN_MAX_NEW}"
  fi

  local out_dir="${GEN_BASE}/${tag}"
  mkdir -p "${out_dir}"
  local out_jsonl="${out_dir}/HPO_${mode}_expression.gen_only.${tag}.jsonl"

  local -a wandb_args=()
  if [[ "${ENABLE_WANDB}" == "1" ]]; then
    wandb_args+=(--wandb --wandb-project "${WB_PROJECT_GEN}")
    wandb_args+=(--wandb-tags "end2end,gen,${tag},${mode}")
    wandb_args+=(--wandb-run-name "gen_${tag}_${mode}_${RUN_TS}" --wandb-group "gen_${tag}")
  fi

  if [[ -n "${lora_path}" ]]; then
    run_step "GEN_${mode}_${tag}" \
      python "${GEN_SCRIPT}" \
        --mode "${mode}" \
        --hpo-csv "${hpo_csv}" \
        --output "${out_jsonl}" \
        --gen-model "${model}" \
        --gen-lora-path "${lora_path}" \
        --per-hpo "${per_hpo}" \
        --target-good "${target_good}" \
        --gen-max-new-tokens "${gen_max_new}" \
        --min-overall 4 --min-match 3 --min-simplicity 3 \
        --max-rounds 3 \
        --diversity-sim-high 0.9 --diversity-sim-low 0.7 \
        "${wandb_args[@]}"
  else
    run_step "GEN_${mode}_${tag}" \
      python "${GEN_SCRIPT}" \
        --mode "${mode}" \
        --hpo-csv "${hpo_csv}" \
        --output "${out_jsonl}" \
        --gen-model "${model}" \
        --per-hpo "${per_hpo}" \
        --target-good "${target_good}" \
        --gen-max-new-tokens "${gen_max_new}" \
        --min-overall 4 --min-match 3 --min-simplicity 3 \
        --max-rounds 3 \
        --diversity-sim-high 0.9 --diversity-sim-low 0.7 \
        "${wandb_args[@]}"
  fi
}

# -------------------------
# Postprocess -> Embed -> Annotate -> Eval (single setting, per (cond,mode))
# -------------------------
eval_one () {
  local cond_tag="$1"
  local mode="$2"
  local gen_jsonl="$3"

  if [[ ! -f "${gen_jsonl}" ]]; then
    echo "[SKIP] missing gen_jsonl: ${gen_jsonl}"
    return 0
  fi

  local run_dir="${EVAL_BASE}/${cond_tag}/${mode}"
  mkdir -p "${run_dir}"

  local post_jsonl="${run_dir}/dict.post.jsonl"
  local removed_jsonl="${run_dir}/dict.removed.jsonl"
  local emb_npy="${run_dir}/dict.emb.npy"
  local meta_jsonl="${run_dir}/dict.meta.jsonl"
  local pred_jsonl="${run_dir}/pred.jsonl"
  local metrics_tsv="${run_dir}/metrics.tsv"

  # counts
  local raw_lines
  raw_lines="$(wc -l < "${gen_jsonl}" | tr -d ' ')"

  local -a wandb_post_args=()
  local -a wandb_embed_args=()
  local -a wandb_annot_args=()
  local -a wandb_eval_args=()
  if [[ "${ENABLE_WANDB}" == "1" ]]; then
    wandb_post_args+=(--wandb --wandb-project "${WB_PROJECT_POST}" --wandb-run-name "post_${cond_tag}_${mode}_${RUN_TS}")
    wandb_post_args+=(--wandb-tags "end2end,postprocess,${cond_tag},${mode}" --wandb-group "post_${cond_tag}")

    wandb_embed_args+=(--wandb --wandb-project "${WB_PROJECT_EMBED}" --wandb-run-name "embed_${cond_tag}_${mode}_${RUN_TS}")
    wandb_embed_args+=(--wandb-tags "end2end,embed,${cond_tag},${mode}" --wandb-group "embed_${cond_tag}")

    wandb_annot_args+=(--wandb --wandb-project "${WB_PROJECT_ANNOT}" --wandb-run-name "annot_${cond_tag}_${mode}_${RUN_TS}")
    wandb_annot_args+=(--wandb-tags "end2end,annot,${cond_tag},${mode}" --wandb-group "annot_${cond_tag}")

    wandb_eval_args+=(--wandb --wandb-project "${WB_PROJECT_EVAL}" --wandb-run-name "eval_${cond_tag}_${mode}_${RUN_TS}")
    wandb_eval_args+=(--wandb-tags "end2end,eval,${cond_tag},${mode}" --wandb-group "eval_${cond_tag}")
  fi

  # 1) postprocess (5a)
  if ! run_step "POST_${cond_tag}_${mode}" \
    python "${POST_SCRIPT}" \
      --input "${gen_jsonl}" \
      --hpo-master "${HPO_MASTER_CSV}" \
      --output "${post_jsonl}" \
      --output-removed "${removed_jsonl}" \
      --mode "${mode}" \
      --sbert-model "${SBERT_MODEL}" \
      --self-min-sim "${SELF_MIN_SIM}" \
      --other-margin "${OTHER_MARGIN}" \
      --ambig-gap 0.03 \
      "${wandb_post_args[@]}"
  then
    return 0
  fi

  # 2) embed dict
  if ! run_step "EMBED_${cond_tag}_${mode}" \
    python "${EMBED_SCRIPT}" \
      --input-jsonl "${post_jsonl}" \
      --output-npy "${emb_npy}" \
      --text-key "text" \
      --model-name "${SBERT_MODEL}" \
      --batch-size "${SBERT_BATCH}" \
      "${wandb_embed_args[@]}"
  then
    return 0
  fi
  cp "${post_jsonl}" "${meta_jsonl}"

  # 3) annotate
	  local -a annot_cmd=(
	    python "${ANNOT_SCRIPT}"
	      --input-csv "${TEST_CSV}"
	      --text-column "${TEXT_COL}"
	      --id-column "${ID_COL}"
	      --output-jsonl "${pred_jsonl}"
	      --emb-npy "${emb_npy}"
	      --meta-jsonl "${meta_jsonl}"
	      --text-key "text"
	      --hpo-key "hpo_id"
	      --hpo-label-ja-key "hpo_label"
	      --ner-model-name "${NER_MODEL}"
	      --embed-model-name "${SBERT_MODEL}"
	      --ner-label-prefixes "${NER_LABEL_PREFIXES}"
	      --max-n "${MAX_N}"
	      --topk "${TOPK}"
	      --min-score "${MIN_SCORE}"
	      --span-len-bonus-alpha "${SPAN_LEN_BONUS_ALPHA}"
	      --span-len-bonus-beta "${SPAN_LEN_BONUS_BETA}"
	  )
	  if [[ "${USE_NER_REGION}" == "1" ]]; then
	    annot_cmd+=(--use-ner-region)
	  fi
	  if [[ "${TREAT_ADJACENT_AS_OVERLAP}" == "1" ]]; then
	    annot_cmd+=(--treat-adjacent-as-overlap)
	  fi
	  annot_cmd+=("${wandb_annot_args[@]}")
	  if ! run_step "ANNOT_${cond_tag}_${mode}" "${annot_cmd[@]}"; then
	    return 0
	  fi

  # 4) eval
  if ! run_step "EVAL_${cond_tag}_${mode}" \
    python "${EVAL_SCRIPT}" \
      --test-csv "${TEST_CSV}" \
      --pred-jsonl "${pred_jsonl}" \
      --id-column "${ID_COL}" \
      --gold-hpo-column "${GOLD_COL}" \
      --hpo-obo "${HPO_OBO}" \
      --output-tsv "${metrics_tsv}" \
      "${wandb_eval_args[@]}"
  then
    return 0
  fi

  # gather counts + metrics (append summary)
  python - <<PY >> "${SUMMARY_TSV}"
import json
from pathlib import Path

post = Path("${post_jsonl}")
removed = Path("${removed_jsonl}")
pred = Path("${pred_jsonl}")
metrics = Path("${metrics_tsv}")

def wc(p: Path)->int:
    if not p.exists(): return 0
    return sum(1 for _ in p.open("r", encoding="utf-8", errors="ignore"))

post_total = wc(post)
removed_lines = wc(removed)
pred_lines = wc(pred)

post_gen = 0
post_official = 0
if post.exists():
    for line in post.open("r", encoding="utf-8", errors="ignore"):
        line=line.strip()
        if not line: continue
        try:
            obj=json.loads(line)
        except Exception:
            continue
        src = obj.get("source","")
        if src == "official":
            post_official += 1
        else:
            post_gen += 1

# parse metrics.tsv (expects "metric\\tvalue")
m = {}
if metrics.exists():
    for line in metrics.open("r", encoding="utf-8", errors="ignore"):
        line=line.strip()
        if not line or line.startswith("#"): 
            continue
        parts=line.split("\\t")
        if len(parts) >= 2:
            m[parts[0]] = parts[1]

def getf(key, default=""):
    return m.get(key, default)

exact_f1 = getf("exact_f1","")
hier_f1  = getf("hier_f1","")
r1 = getf("recall@1","")
r3 = getf("recall@3","")
r5 = getf("recall@5","")
r10= getf("recall@10","")

row = [
    "${cond_tag}",
    "${mode}",
    "${raw_lines}",
    str(post_total),
    str(post_gen),
    str(post_official),
    str(removed_lines),
    str(pred_lines),
    str(exact_f1),
    str(hier_f1),
    str(r1), str(r3), str(r5), str(r10),
    str(metrics),
    str(pred),
    str(post),
]
print("\\t".join(row))
PY

  # W&B: gen/evalも手動で一発 log（任意）
  if [[ "${ENABLE_WANDB_SUMMARY}" == "1" ]]; then
    export WB_PROJECT="HPO_end2end_eval"
    export WB_NAME="eval_${cond_tag}_${mode}_${RUN_TS}"
    export WB_GROUP="eval_${cond_tag}"
    export WB_TAGS="end2end,${cond_tag},${mode}"
    WB_JSON="$(
      POST="${post_jsonl}" REMOVED="${removed_jsonl}" PRED="${pred_jsonl}" METRICS="${metrics_tsv}" RAW_LINES="${raw_lines}" \
        build_eval_summary_json
    )"
    export WB_JSON
    POST="${post_jsonl}" REMOVED="${removed_jsonl}" PRED="${pred_jsonl}" METRICS="${metrics_tsv}" RAW_LINES="${raw_lines}" \
      wandb_log_if_possible "${WB_PROJECT}" "${WB_NAME}" "${WB_GROUP}" "${WB_TAGS}" "${WB_JSON}"
  fi

  return 0
}

# -------------------------
# Postprocess -> Embed -> Annotate -> Eval (patient+doctor merged)
# -------------------------
eval_merged () {
  local cond_tag="$1"
  local pat_jsonl="${GEN_BASE}/${cond_tag}/HPO_patient_expression.gen_only.${cond_tag}.jsonl"
  local doc_jsonl="${GEN_BASE}/${cond_tag}/HPO_doctor_expression.gen_only.${cond_tag}.jsonl"

  if [[ ! -f "${pat_jsonl}" || ! -f "${doc_jsonl}" ]]; then
    echo "[SKIP] merged eval for ${cond_tag}: missing ${pat_jsonl} or ${doc_jsonl}"
    return 0
  fi

  local merged_gen="${GEN_BASE}/${cond_tag}/HPO_patient_doctor_expression.gen_only.${cond_tag}.jsonl"
  cat "${pat_jsonl}" "${doc_jsonl}" > "${merged_gen}"

  local run_dir="${EVAL_BASE}/${cond_tag}/merged"
  mkdir -p "${run_dir}"

  local post_jsonl="${run_dir}/dict.post.jsonl"
  local removed_jsonl="${run_dir}/dict.removed.jsonl"
  local emb_npy="${run_dir}/dict.emb.npy"
  local meta_jsonl="${run_dir}/dict.meta.jsonl"
  local pred_jsonl="${run_dir}/pred.jsonl"
  local metrics_tsv="${run_dir}/metrics.tsv"

  local raw_lines
  raw_lines="$(( $(wc -l < "${pat_jsonl}" | tr -d ' ') + $(wc -l < "${doc_jsonl}" | tr -d ' ') ))"

  local -a wandb_post_args=()
  local -a wandb_embed_args=()
  local -a wandb_annot_args=()
  local -a wandb_eval_args=()
  if [[ "${ENABLE_WANDB}" == "1" ]]; then
    wandb_post_args+=(--wandb --wandb-project "${WB_PROJECT_POST}" --wandb-run-name "post_${cond_tag}_merged_${RUN_TS}")
    wandb_post_args+=(--wandb-tags "end2end,postprocess,${cond_tag},merged" --wandb-group "post_${cond_tag}")

    wandb_embed_args+=(--wandb --wandb-project "${WB_PROJECT_EMBED}" --wandb-run-name "embed_${cond_tag}_merged_${RUN_TS}")
    wandb_embed_args+=(--wandb-tags "end2end,embed,${cond_tag},merged" --wandb-group "embed_${cond_tag}")

    wandb_annot_args+=(--wandb --wandb-project "${WB_PROJECT_ANNOT}" --wandb-run-name "annot_${cond_tag}_merged_${RUN_TS}")
    wandb_annot_args+=(--wandb-tags "end2end,annot,${cond_tag},merged" --wandb-group "annot_${cond_tag}")

    wandb_eval_args+=(--wandb --wandb-project "${WB_PROJECT_EVAL}" --wandb-run-name "eval_${cond_tag}_merged_${RUN_TS}")
    wandb_eval_args+=(--wandb-tags "end2end,eval,${cond_tag},merged" --wandb-group "eval_${cond_tag}")
  fi

  # 1) postprocess (mixed)
  if ! run_step "POST_${cond_tag}_merged" \
    python "${POST_SCRIPT}" \
      --input "${merged_gen}" \
      --hpo-master "${HPO_MASTER_CSV}" \
      --output "${post_jsonl}" \
      --output-removed "${removed_jsonl}" \
      --mode "mixed" \
      --sbert-model "${SBERT_MODEL}" \
      --self-min-sim "${SELF_MIN_SIM}" \
      --other-margin "${OTHER_MARGIN}" \
      --ambig-gap 0.03 \
      "${wandb_post_args[@]}"
  then
    return 0
  fi

  # 2) embed
  if ! run_step "EMBED_${cond_tag}_merged" \
    python "${EMBED_SCRIPT}" \
      --input-jsonl "${post_jsonl}" \
      --output-npy "${emb_npy}" \
      --text-key "text" \
      --model-name "${SBERT_MODEL}" \
      --batch-size "${SBERT_BATCH}" \
      "${wandb_embed_args[@]}"
  then
    return 0
  fi
  cp "${post_jsonl}" "${meta_jsonl}"

  # 3) annotate
	  local -a annot_cmd=(
	    python "${ANNOT_SCRIPT}"
	      --input-csv "${TEST_CSV}"
	      --text-column "${TEXT_COL}"
	      --id-column "${ID_COL}"
	      --output-jsonl "${pred_jsonl}"
	      --emb-npy "${emb_npy}"
	      --meta-jsonl "${meta_jsonl}"
	      --text-key "text"
	      --hpo-key "hpo_id"
	      --hpo-label-ja-key "hpo_label"
	      --ner-model-name "${NER_MODEL}"
	      --embed-model-name "${SBERT_MODEL}"
	      --ner-label-prefixes "${NER_LABEL_PREFIXES}"
	      --max-n "${MAX_N}"
	      --topk "${TOPK}"
	      --min-score "${MIN_SCORE}"
	      --span-len-bonus-alpha "${SPAN_LEN_BONUS_ALPHA}"
	      --span-len-bonus-beta "${SPAN_LEN_BONUS_BETA}"
	  )
	  if [[ "${USE_NER_REGION}" == "1" ]]; then
	    annot_cmd+=(--use-ner-region)
	  fi
	  if [[ "${TREAT_ADJACENT_AS_OVERLAP}" == "1" ]]; then
	    annot_cmd+=(--treat-adjacent-as-overlap)
	  fi
	  annot_cmd+=("${wandb_annot_args[@]}")
	  if ! run_step "ANNOT_${cond_tag}_merged" "${annot_cmd[@]}"; then
	    return 0
	  fi

  # 4) eval
  if ! run_step "EVAL_${cond_tag}_merged" \
    python "${EVAL_SCRIPT}" \
      --test-csv "${TEST_CSV}" \
      --pred-jsonl "${pred_jsonl}" \
      --id-column "${ID_COL}" \
      --gold-hpo-column "${GOLD_COL}" \
      --hpo-obo "${HPO_OBO}" \
      --output-tsv "${metrics_tsv}" \
      "${wandb_eval_args[@]}"
  then
    return 0
  fi

  # gather counts + metrics (append summary)
  python - <<PY >> "${SUMMARY_TSV}"
import json
from pathlib import Path

post = Path("${post_jsonl}")
removed = Path("${removed_jsonl}")
pred = Path("${pred_jsonl}")
metrics = Path("${metrics_tsv}")

def wc(p: Path)->int:
    if not p.exists(): return 0
    return sum(1 for _ in p.open("r", encoding="utf-8", errors="ignore"))

post_total = wc(post)
removed_lines = wc(removed)
pred_lines = wc(pred)

post_gen = 0
post_official = 0
if post.exists():
    for line in post.open("r", encoding="utf-8", errors="ignore"):
        line=line.strip()
        if not line: continue
        try:
            obj=json.loads(line)
        except Exception:
            continue
        src = obj.get("source","")
        if src == "official":
            post_official += 1
        else:
            post_gen += 1

m = {}
if metrics.exists():
    for line in metrics.open("r", encoding="utf-8", errors="ignore"):
        line=line.strip()
        if not line or line.startswith("#"): 
            continue
        parts=line.split("\\t")
        if len(parts) >= 2:
            m[parts[0]] = parts[1]

def getf(key, default=""):
    return m.get(key, default)

exact_f1 = getf("exact_f1","")
hier_f1  = getf("hier_f1","")
r1 = getf("recall@1","")
r3 = getf("recall@3","")
r5 = getf("recall@5","")
r10= getf("recall@10","")

row = [
    "${cond_tag}",
    "merged",
    "${raw_lines}",
    str(post_total),
    str(post_gen),
    str(post_official),
    str(removed_lines),
    str(pred_lines),
    str(exact_f1),
    str(hier_f1),
    str(r1), str(r3), str(r5), str(r10),
    str(metrics),
    str(pred),
    str(post),
]
print("\\t".join(row))
PY

  return 0
}

########################################
# Main
########################################
echo "BASE_DIR=${BASE_DIR}"
echo "HPO_DIR=${HPO_DIR}"
echo "DATA_DIR=${DATA_DIR}"
echo "GPU_ID=${GPU_ID}"
echo "OUT_BASE=${OUT_BASE}"
echo "SUMMARY_TSV=${SUMMARY_TSV}"
echo "MODES=${MODE_LIST[*]}"
echo

cd "${HPO_DIR}"

# sanity checks
for f in "${SFT_SCRIPT}" "${DS_CFG}" "${GEN_SCRIPT}" "${POST_SCRIPT}" "${EMBED_SCRIPT}" "${ANNOT_SCRIPT}" "${EVAL_SCRIPT}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[ERROR] missing: ${f}"
    exit 1
  fi
done
for f in "${TRAIN_FILES[@]}" "${HPO_MASTER_CSV}" "${TEST_CSV}" "${PAT_HPO_CSV}" "${DOC_HPO_CSV}" "${HPO_OBO}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[ERROR] missing: ${f}"
    exit 1
  fi
done

# 0) SFT (optional)
if [[ "${DO_SFT_EQUES}" == "1" ]]; then
  run_sft "${EQUES_BASE}" "${EQUES_LORA_DIR}" "eques" || true
fi
if [[ "${DO_SFT_ELYZA}" == "1" ]]; then
  run_sft "${ELYZA_BASE}" "${ELYZA_LORA_DIR}" "elyza" || true
fi
if [[ "${DO_SFT_SWALLOW}" == "1" ]]; then
  run_sft "${SWALLOW_BASE}" "${SWALLOW_LORA_DIR}" "swallow" || true
fi

# conditions (6)
# tag format: {model}_{base|qlora}
declare -a COND_TAGS=(
  "eques_base"
  "eques_qlora"
  "elyza_base"
  "elyza_qlora"
  "swallow_base"
  "swallow_qlora"
)

MODEL_OUT=""
LORA_OUT=""

get_model_and_lora () {
  local tag="$1"
  MODEL_OUT=""
  LORA_OUT=""

  case "${tag}" in
    eques_base)    MODEL_OUT="${EQUES_BASE}";   LORA_OUT="";;
    eques_qlora)   MODEL_OUT="${EQUES_BASE}";   LORA_OUT="${EQUES_LORA_DIR}";;
    elyza_base)    MODEL_OUT="${ELYZA_BASE}";   LORA_OUT="";;
    elyza_qlora)   MODEL_OUT="${ELYZA_BASE}";   LORA_OUT="${ELYZA_LORA_DIR}";;
    swallow_base)  MODEL_OUT="${SWALLOW_BASE}"; LORA_OUT="";;
    swallow_qlora) MODEL_OUT="${SWALLOW_BASE}"; LORA_OUT="${SWALLOW_LORA_DIR}";;
    *) return 1;;
  esac
  return 0
}


# 1) gen-only
if [[ "${DO_GEN}" == "1" ]]; then
    for tag in "${COND_TAGS[@]}"; do
    if ! get_model_and_lora "${tag}"; then
        echo "[SKIP] unknown tag: ${tag}"
        continue
    fi

    model="${MODEL_OUT}"
    lora="${LORA_OUT}"

    # ついでに：QLoRA tag なのに adapter が無ければスキップ（SFT失敗時に便利）
    if [[ "${tag}" == *_qlora && ! -d "${lora}" ]]; then
        echo "[SKIP] missing LoRA dir for ${tag}: ${lora}"
        continue
    fi

    for mode in "${MODE_LIST[@]}"; do
      gen_one "${mode}" "${model}" "${tag}" "${lora}" || true
    done
    done
fi

# 2) eval (postprocess->embed->annot->eval)
if [[ "${DO_EVAL}" == "1" ]]; then
  for tag in "${COND_TAGS[@]}"; do
    for mode in "${MODE_LIST[@]}"; do
      gen_jsonl="${GEN_BASE}/${tag}/HPO_${mode}_expression.gen_only.${tag}.jsonl"
      eval_one "${tag}" "${mode}" "${gen_jsonl}" || true
    done
    if [[ "${DO_MERGED}" == "1" ]]; then
      eval_merged "${tag}" || true
    fi
  done
fi

echo
echo "=== DONE ==="
echo "Summary: ${SUMMARY_TSV}"
echo "Tip: column -t -s \$'\\t' ${SUMMARY_TSV} | less -S"
