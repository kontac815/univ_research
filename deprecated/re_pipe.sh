#!/bin/bash
set -euo pipefail
source ~/.bashrc || true

########################################
# Re-run postprocess -> embed -> annotate -> eval
# using existing generation outputs (20251214_2200)
# - runs patient & doctor separately
# - then merges patient+doctor and re-runs embed/annot/eval
# - targets all 6 conditions (eques/elyza/swallow base+qlora)
# NOTE: Fix your env first (faiss x NumPy):
#   pip install 'numpy<2.0.0' --force-reinstall
########################################

BASE_DIR="/home/kiso_user/Documents/workspace/Research"
HPO_DIR="${BASE_DIR}/HPO-patient"
DATA_DIR="${BASE_DIR}/data"

POST_SCRIPT="${HPO_DIR}/5a.postprocess_expressions_sbert.py"
EMBED_SCRIPT="${HPO_DIR}/embed_with_sbert.py"
ANNOT_SCRIPT="${HPO_DIR}/annotate_with_medtxtner_hpo_batch.py"
EVAL_SCRIPT="${HPO_DIR}/eval_hpo_annotation_medtxtner.py"

HPO_MASTER_CSV="${DATA_DIR}/HPO_depth_ge3.csv"
TEST_CSV="${DATA_DIR}/hpo_annotation_test_dataset.csv"
HPO_OBO="${DATA_DIR}/hp.obo"
ID_COL="patient_name"
TEXT_COL="medical_history_nobr"
GOLD_COL="hpo_ids"

RUN_TS="20251214_2200"
OUT_BASE="${DATA_DIR}/end2end_runs/${RUN_TS}"
LOG_BASE="${OUT_BASE}/logs"
GEN_BASE="${OUT_BASE}/gen"
EVAL_BASE="${OUT_BASE}/eval"
SUMMARY_TSV="${OUT_BASE}/summary.tsv"

COND_TAGS=("eques_base" "eques_qlora" "elyza_base" "elyza_qlora" "swallow_base" "swallow_qlora")
MODES=("patient" "doctor")

# postprocess / embed / annotate params
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

ENABLE_WANDB="${ENABLE_WANDB:-1}"
GPU_ID="${GPU_ID:-1}"
RESET_SUMMARY="${RESET_SUMMARY:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

mkdir -p "${OUT_BASE}" "${LOG_BASE}" "${GEN_BASE}" "${EVAL_BASE}"
if [[ "${RESET_SUMMARY}" == "1" ]]; then
  echo -e "cond_tag\tmode\traw_gen_lines\tpost_total\tpost_gen\tpost_official\tremoved_lines\tpred_lines\texact_f1\thier_f1\trecall@1\trecall@3\trecall@5\trecall@10\tmetrics_tsv\tpred_jsonl\tpost_jsonl" \
    > "${SUMMARY_TSV}"
elif [[ ! -f "${SUMMARY_TSV}" ]]; then
  echo -e "cond_tag\tmode\traw_gen_lines\tpost_total\tpost_gen\tpost_official\tremoved_lines\tpred_lines\texact_f1\thier_f1\trecall@1\trecall@3\trecall@5\trecall@10\tmetrics_tsv\tpred_jsonl\tpost_jsonl" \
    > "${SUMMARY_TSV}"
fi

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

  local -a wandb_post_args=()
  local -a wandb_embed_args=()
  local -a wandb_annot_args=()
  local -a wandb_eval_args=()
  if [[ "${ENABLE_WANDB}" == "1" ]]; then
    wandb_post_args+=(--wandb --wandb-project "HPO_postprocess" --wandb-run-name "post_${cond_tag}_${mode}_${RUN_TS}")
    wandb_post_args+=(--wandb-tags "resume,postprocess,${cond_tag},${mode}" --wandb-group "post_${cond_tag}")

    wandb_embed_args+=(--wandb --wandb-project "HPO_embed" --wandb-run-name "embed_${cond_tag}_${mode}_${RUN_TS}")
    wandb_embed_args+=(--wandb-tags "resume,embed,${cond_tag},${mode}" --wandb-group "embed_${cond_tag}")

    wandb_annot_args+=(--wandb --wandb-project "HPO_annotate" --wandb-run-name "annot_${cond_tag}_${mode}_${RUN_TS}")
    wandb_annot_args+=(--wandb-tags "resume,annot,${cond_tag},${mode}" --wandb-group "annot_${cond_tag}")

    wandb_eval_args+=(--wandb --wandb-project "HPO_eval" --wandb-run-name "eval_${cond_tag}_${mode}_${RUN_TS}")
    wandb_eval_args+=(--wandb-tags "resume,eval,${cond_tag},${mode}" --wandb-group "eval_${cond_tag}")
  fi

  # 1) postprocess
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
  )
  if [[ "${USE_NER_REGION}" == "1" ]]; then
    annot_cmd+=(--use-ner-region)
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

  return 0
}

merge_and_eval () {
  local cond_tag="$1"
  local pat_dir="${EVAL_BASE}/${cond_tag}/patient"
  local doc_dir="${EVAL_BASE}/${cond_tag}/doctor"
  local merged_dir="${EVAL_BASE}/${cond_tag}/merged"

  local post_pat="${pat_dir}/dict.post.jsonl"
  local post_doc="${doc_dir}/dict.post.jsonl"
  local merged_post="${merged_dir}/dict.post.jsonl"
  local merged_meta="${merged_dir}/dict.meta.jsonl"
  local merged_emb="${merged_dir}/dict.emb.npy"
  local merged_pred="${merged_dir}/pred.jsonl"
  local merged_metrics="${merged_dir}/metrics.tsv"

  if [[ ! -f "${post_pat}" || ! -f "${post_doc}" ]]; then
    echo "[SKIP] merge for ${cond_tag}: missing ${post_pat} or ${post_doc}"
    return 0
  fi

  mkdir -p "${merged_dir}"
  cat "${post_pat}" "${post_doc}" > "${merged_post}"
  cp "${merged_post}" "${merged_meta}"

  local raw_lines
  raw_lines="$(( $(wc -l < \"${GEN_BASE}/${cond_tag}/HPO_patient_expression.gen_only.${cond_tag}.jsonl\" | tr -d ' ') + $(wc -l < \"${GEN_BASE}/${cond_tag}/HPO_doctor_expression.gen_only.${cond_tag}.jsonl\" | tr -d ' ') ))"

  local -a wandb_embed_args=()
  local -a wandb_annot_args=()
  local -a wandb_eval_args=()
  if [[ "${ENABLE_WANDB}" == "1" ]]; then
    wandb_embed_args+=(--wandb --wandb-project "HPO_embed" --wandb-run-name "embed_${cond_tag}_merged_${RUN_TS}")
    wandb_embed_args+=(--wandb-tags "resume,embed,${cond_tag},merged" --wandb-group "embed_${cond_tag}")

    wandb_annot_args+=(--wandb --wandb-project "HPO_annotate" --wandb-run-name "annot_${cond_tag}_merged_${RUN_TS}")
    wandb_annot_args+=(--wandb-tags "resume,annot,${cond_tag},merged" --wandb-group "annot_${cond_tag}")

    wandb_eval_args+=(--wandb --wandb-project "HPO_eval" --wandb-run-name "eval_${cond_tag}_merged_${RUN_TS}")
    wandb_eval_args+=(--wandb-tags "resume,eval,${cond_tag},merged" --wandb-group "eval_${cond_tag}")
  fi

  if ! run_step "EMBED_${cond_tag}_merged" \
    python "${EMBED_SCRIPT}" \
      --input-jsonl "${merged_post}" \
      --output-npy "${merged_emb}" \
      --text-key "text" \
      --model-name "${SBERT_MODEL}" \
      --batch-size "${SBERT_BATCH}" \
      "${wandb_embed_args[@]}"
  then
    return 0
  fi

  local -a annot_cmd=(
    python "${ANNOT_SCRIPT}"
      --input-csv "${TEST_CSV}"
      --text-column "${TEXT_COL}"
      --id-column "${ID_COL}"
      --output-jsonl "${merged_pred}"
      --emb-npy "${merged_emb}"
      --meta-jsonl "${merged_meta}"
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
  if ! run_step "ANNOT_${cond_tag}_merged" "${annot_cmd[@]}"; then
    return 0
  fi

  if ! run_step "EVAL_${cond_tag}_merged" \
    python "${EVAL_SCRIPT}" \
      --test-csv "${TEST_CSV}" \
      --pred-jsonl "${merged_pred}" \
      --id-column "${ID_COL}" \
      --gold-hpo-column "${GOLD_COL}" \
      --hpo-obo "${HPO_OBO}" \
      --output-tsv "${merged_metrics}" \
      "${wandb_eval_args[@]}"
  then
    return 0
  fi

  python - <<PY >> "${SUMMARY_TSV}"
import json
from pathlib import Path

post = Path("${merged_post}")
removed = Path("${merged_dir}/dict.removed.jsonl")  # mergedでは未生成
pred = Path("${merged_pred}")
metrics = Path("${merged_metrics}")

def wc(p: Path)->int:
    if not p.exists(): return 0
    return sum(1 for _ in p.open("r", encoding="utf-8", errors="ignore"))

post_total = wc(post)
removed_lines = wc(removed)
pred_lines = wc(pred)

post_gen = post_total
post_official = 0

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

echo "OUT_BASE=${OUT_BASE}"
echo "SUMMARY_TSV=${SUMMARY_TSV}"
echo "MODES=${MODES[*]}"
echo "COND_TAGS=${COND_TAGS[*]}"
echo

cd "${HPO_DIR}"

for cond_tag in "${COND_TAGS[@]}"; do
  for mode in "${MODES[@]}"; do
    gen_jsonl="${GEN_BASE}/${cond_tag}/HPO_${mode}_expression.gen_only.${cond_tag}.jsonl"
    eval_one "${cond_tag}" "${mode}" "${gen_jsonl}" || true
  done
done

for cond_tag in "${COND_TAGS[@]}"; do
  merge_and_eval "${cond_tag}" || true
done

echo
echo "=== DONE ==="
echo "Summary: ${SUMMARY_TSV}"
echo "Tip: column -t -s $'\\t' ${SUMMARY_TSV} | less -S"
