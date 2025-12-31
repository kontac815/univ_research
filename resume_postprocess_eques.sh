#!/bin/bash
set -euo pipefail
source ~/.bashrc || true

# Resume only the annotate -> eval stages
# for existing postprocess/embedding results (all model tags supported).

########################################
# Paths and run config
########################################
BASE_DIR="/home/kiso_user/Documents/workspace/Research"
HPO_DIR="${BASE_DIR}/HPO-patient"
DATA_DIR="${BASE_DIR}/data"

RUN_TS="${RUN_TS:-20251214_2200}"              # reuse existing gen outputs
TAGS_CSV="${TAGS:-eques_base,eques_qlora,elyza_base,elyza_qlora,swallow_base,swallow_qlora}"  # comma/space separated
MODES_CSV="${MODES:-patient,doctor}"           # comma/space separated
GPU_ID="${GPU_ID:-1}"
ENABLE_WANDB="${ENABLE_WANDB:-1}"               # set 0 to disable

read -r -a TAGS <<< "${TAGS_CSV//,/ }"
read -r -a MODES <<< "${MODES_CSV//,/ }"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export WANDB__SERVICE_WAIT=300
export WANDB_CONSOLE=wrap

########################################
# Scripts / data
########################################
ANNOT_SCRIPT="${HPO_DIR}/annotate_with_medtxtner_hpo_batch.py"
EVAL_SCRIPT="${HPO_DIR}/eval_hpo_annotation_medtxtner.py"

TEST_CSV="${DATA_DIR}/hpo_annotation_test_dataset.csv"
ID_COL="patient_name"
TEXT_COL="medical_history_nobr"
GOLD_COL="hpo_ids"
HPO_OBO="${DATA_DIR}/hp.obo"

########################################
# Hyperparams (annot/eval)
########################################
SBERT_MODEL="pkshatech/GLuCoSE-base-ja-v2"

NER_MODEL="sociocom/MedTXTNER"
NER_LABEL_PREFIXES="d,cc"
USE_NER_REGION=1
MAX_N=6
TOPK=15
MIN_SCORE=0.70

########################################
# Output dirs
########################################
OUT_BASE="${DATA_DIR}/end2end_runs/${RUN_TS}"
GEN_BASE="${OUT_BASE}/gen"
EVAL_BASE="${OUT_BASE}/eval"
LOG_BASE="${OUT_BASE}/logs_resume"
SUMMARY_TSV="${OUT_BASE}/summary_resume.tsv"

mkdir -p "${LOG_BASE}"
echo -e "cond_tag\tmode\traw_gen_lines\tpost_total\tpost_gen\tpost_official\tremoved_lines\tpred_lines\texact_f1\thier_f1\trecall@1\trecall@3\trecall@5\trecall@10\tmetrics_tsv\tpred_jsonl\tpost_jsonl" \
  > "${SUMMARY_TSV}"

########################################
# Helpers
########################################
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
    echo "[FAIL] ${title}"
    return 1
  fi
}

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

  local raw_lines
  raw_lines="$(wc -l < "${gen_jsonl}" | tr -d ' ')"

  local -a wandb_annot_args=()
  local -a wandb_eval_args=()

  if [[ "${ENABLE_WANDB}" == "1" ]]; then
    wandb_annot_args+=(--wandb --wandb-project "HPO_annotate" --wandb-run-name "annot_${cond_tag}_${mode}_${RUN_TS}")
    wandb_annot_args+=(--wandb-tags "resume,annot,${cond_tag},${mode}" --wandb-group "resume_annot_${cond_tag}")

    wandb_eval_args+=(--wandb --wandb-project "HPO_eval" --wandb-run-name "eval_${cond_tag}_${mode}_${RUN_TS}")
    wandb_eval_args+=(--wandb-tags "resume,eval,${cond_tag},${mode}" --wandb-group "resume_eval_${cond_tag}")
  fi

  # sanity checks for existing artifacts
  if [[ ! -f "${emb_npy}" || ! -f "${meta_jsonl}" ]]; then
    echo "[SKIP] missing embedding/meta for annotate: ${emb_npy} or ${meta_jsonl}"
    return 0
  fi
  if [[ ! -f "${post_jsonl}" ]]; then
    echo "[WARN] post_jsonl not found (summary counts will be zero): ${post_jsonl}"
  fi
  if [[ ! -f "${removed_jsonl}" ]]; then
    echo "[WARN] removed_jsonl not found (summary counts will be zero): ${removed_jsonl}"
  fi

  # 1) annotate
  local -a annot_cmd=(
    python3 "${ANNOT_SCRIPT}"
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
  )
  if [[ "${USE_NER_REGION}" == "1" ]]; then
    annot_cmd+=(--use-ner-region)
  fi
  annot_cmd+=("${wandb_annot_args[@]}")
  if ! run_step "ANNOT_${cond_tag}_${mode}" "${annot_cmd[@]}"; then
    return 0
  fi

  # 2) eval
  if ! run_step "EVAL_${cond_tag}_${mode}" \
    python3 "${EVAL_SCRIPT}" \
      --test-csv "${TEST_CSV}" \
      --pred-jsonl "${pred_jsonl}" \
      --id-column "${ID_COL}" \
      --gold-hpo-column "${GOLD_COL}" \
      --hpo-obo "${HPO_OBO}" \
      --output-tsv "${metrics_tsv}" \
      "${wandb_eval_args[@]}"; then
    return 0
  fi

  # summary row
  python3 - <<PY >> "${SUMMARY_TSV}"
import json
from pathlib import Path

post = Path("${post_jsonl}")
removed = Path("${removed_jsonl}")
pred = Path("${pred_jsonl}")
metrics = Path("${metrics_tsv}")

def wc(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(1 for _ in p.open("r", encoding="utf-8", errors="ignore"))

post_total = wc(post)
removed_lines = wc(removed)
pred_lines = wc(pred)

post_gen = 0
post_official = 0
if post.exists():
    for line in post.open("r", encoding="utf-8", errors="ignore"):
        line=line.strip()
        if not line:
            continue
        try:
            obj=json.loads(line)
        except Exception:
            continue
        if obj.get("source","") == "official":
            post_official += 1
        else:
            post_gen += 1

# parse metrics.tsv
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

row = [
    "${cond_tag}",
    "${mode}",
    "${raw_lines}",
    str(post_total),
    str(post_gen),
    str(post_official),
    str(removed_lines),
    str(pred_lines),
    getf("exact_f1",""),
    getf("hier_f1",""),
    getf("recall@1",""), getf("recall@3",""), getf("recall@5",""), getf("recall@10",""),
    str(metrics),
    str(pred),
    str(post),
]
print("\\t".join(row))
PY
}

########################################
# Main
########################################
echo "RUN_TS=${RUN_TS}"
echo "TAGS=${TAGS[*]}"
echo "MODES=${MODES[*]}"
echo "GEN_BASE=${GEN_BASE}"
echo "EVAL_BASE=${EVAL_BASE}"
echo "LOG_BASE=${LOG_BASE}"
echo

cd "${HPO_DIR}"

for tag in "${TAGS[@]}"; do
  for mode in "${MODES[@]}"; do
    gen_jsonl="${GEN_BASE}/${tag}/HPO_${mode}_expression.gen_only.${tag}.jsonl"
    eval_one "${tag}" "${mode}" "${gen_jsonl}" || true
  done
done

echo
echo "=== RESUME DONE ==="
echo "Summary: ${SUMMARY_TSV}"
