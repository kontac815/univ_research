#!/bin/bash
set -euo pipefail

########################################
# パス設定
########################################

BASE_DIR="/home/kiso_user/Documents/workspace/Research"
HPO_DIR="${BASE_DIR}/HPO-patient"
DATA_DIR="${BASE_DIR}/data"

# 4.gen&judge&refine.* の最終出力（患者・医師）
# ※実際のファイル名に合わせて書き換えてください
PATIENT_JSONL="${DATA_DIR}/notLoRA_vllm/HPO_symptom_patient_expression_judge_refine.vllm_base.jsonl"
DOCTOR_JSONL="${DATA_DIR}/notLoRA_vllm/HPO_symptom_doctor_expression_judge_refine.vllm_base.jsonl"

# HPO マスタ（jp_final と depth を含む CSV）
HPO_MASTER_CSV="${DATA_DIR}/HPO_depth_ge3.csv"

# 評価用データ & OBO
TEST_CSV="${DATA_DIR}/hpo_annotation_test_dataset.csv"
HPO_OBO="${DATA_DIR}/hp.obo"

# すべての結果を置くディレクトリ
OUT_DIR="${DATA_DIR}/notsweep_5a_medtxtner"
mkdir -p "${OUT_DIR}"

cd "${HPO_DIR}"

########################################
# 0. 患者 + 医師の RAW JSONL をマージ
#    （5a は 1本の JSONL を見る想定）
########################################

MERGED_RAW="${DATA_DIR}/notLoRA_vllm/merged.jsonl"

if [ ! -f "${MERGED_RAW}" ]; then
  echo "== Merge patient & doctor raw JSONL to: ${MERGED_RAW}"
  cat "${PATIENT_JSONL}" "${DOCTOR_JSONL}" > "${MERGED_RAW}"
fi

########################################
# 1. 5a のハイパーパラメータのスイープ設定
########################################

# S-BERT モデル（とりあえず 1 種類。増やしたければ要素を増やしてください）
SBERT_MODEL_LIST=("pkshatech/GLuCoSE-base-ja-v2")

# self-min-sim（自分の HPO ラベルとの類似度の下限）
SELF_MIN_SIM_LIST=("0.5")

# other-margin（他の HPO ラベルとの margin）
OTHER_MARGIN_LIST=("0.05")

# HPO ラベルと同一の表現を落とすかどうか
# 0: drop（デフォルト） / 1: keep (= --no-drop-same-as-label 指定)
KEEP_LABEL_LIST=("0")

########################################
# 2. MedTXTNER + n-gram 側のスイープ設定
########################################

USE_NER_REGION_LIST=("1")
MAX_N_LIST=("6")
TOPK_LIST=("5")
MIN_SCORE_LIST=("0.80")

# MedTXTNER 側の固定パラメータ
MEDTXTNER_MODEL="sociocom/MedTXTNER"
SPAN_MAX_LENGTH=256

########################################
# 3. ループで 5a → embed → annotate → eval
########################################

for SBERT_MODEL in "${SBERT_MODEL_LIST[@]}"; do
  for SELF_MIN_SIM in "${SELF_MIN_SIM_LIST[@]}"; do
    for OTHER_MARGIN in "${OTHER_MARGIN_LIST[@]}"; do
      for KEEP_LABEL in "${KEEP_LABEL_LIST[@]}"; do

        ########################################
        # 3-1. 5a: 前処理パラメータのタグを作成
        ########################################
        TAG_5A="sbert$(echo "${SBERT_MODEL}" | tr '/-' '__')_self${SELF_MIN_SIM}_margin${OTHER_MARGIN}_keep${KEEP_LABEL}"

        CLEAN_JSONL="${OUT_DIR}/dict_${TAG_5A}.jsonl"
        REMOVED_JSONL="${OUT_DIR}/dict_${TAG_5A}.removed.jsonl"
        EMB_NPY="${OUT_DIR}/emb_${TAG_5A}.npy"
        EMB_TSV="${OUT_DIR}/emb_${TAG_5A}.tsv"

        ########################################
        # 3-2. 5a: postprocess_expressions_sbert.py
        ########################################

        if [ ! -f "${CLEAN_JSONL}" ]; then
          echo "-----------------------------------------"
          echo "[5a] RUN TAG_5A = ${TAG_5A}"
          echo "-----------------------------------------"

          EXTRA_5A_FLAGS=()
          if [ "${KEEP_LABEL}" = "1" ]; then
            EXTRA_5A_FLAGS+=( "--no-drop-same-as-label" )
          fi

          python 5a.postprocess_expressions_sbert.py \
            --input "${MERGED_RAW}" \
            --hpo-master "${HPO_MASTER_CSV}" \
            --output "${CLEAN_JSONL}" \
            --output-removed "${REMOVED_JSONL}" \
            --mode "patient" \
            --sbert-model "${SBERT_MODEL}" \
            --self-min-sim "${SELF_MIN_SIM}" \
            --other-margin "${OTHER_MARGIN}" \
            "${EXTRA_5A_FLAGS[@]}"
        else
          echo "[5a] skip (already exists): ${CLEAN_JSONL}"
        fi
        ########################################
        # 3-3. embed_with_sbert.py（辞書側埋め込み：GLuCoSE など）
        ########################################

        if [ ! -f "${EMB_NPY}" ]; then
          echo "[embed] encode with SBERT for TAG_5A = ${TAG_5A}"
          python embed_with_sbert.py \
            --input-jsonl "${CLEAN_JSONL}" \
            --text-key "text" \
            --output-npy "${EMB_NPY}" \
            --model-name "${SBERT_MODEL}" \
            --batch-size 64
        else
          echo "[embed] skip (already exists): ${EMB_NPY}"
        fi


        ########################################
        # 3-4. MedTXTNER 側のパラメータスイープ
        ########################################

        for USE_NER in "${USE_NER_REGION_LIST[@]}"; do
          for MAX_N in "${MAX_N_LIST[@]}"; do
            for TOPK in "${TOPK_LIST[@]}"; do
              for MIN_SCORE in "${MIN_SCORE_LIST[@]}"; do

                RUN_ID_MED="ner${USE_NER}_n${MAX_N}_top${TOPK}_min${MIN_SCORE}"
                RUN_ID="${TAG_5A}__${RUN_ID_MED}"

                PRED_JSONL="${OUT_DIR}/pred_${RUN_ID}.jsonl"
                METRIC_TSV="${OUT_DIR}/metrics_${RUN_ID}.tsv"

                echo "======================================="
                echo " 5a+MedTXTNER RUN: ${RUN_ID}"
                echo "======================================="

                # --- annotate ---
                if [ ! -f "${PRED_JSONL}" ]; then
                  echo "[annotate] annotate_with_medtxtner_hpo_batch.py"

                  EXTRA_NER_FLAGS=()
                  if [ "${USE_NER}" = "1" ]; then
                    EXTRA_NER_FLAGS+=( "--use-ner-region" )
                  fi

                  python annotate_with_medtxtner_hpo_batch.py \
                    --input-csv "${TEST_CSV}" \
                    --text-column "medical_history_nobr" \
                    --id-column "patient_name" \
                    --output-jsonl "${PRED_JSONL}" \
                    --emb-npy "${EMB_NPY}" \
                    --meta-jsonl "${CLEAN_JSONL}" \
                    --text-key "text" \
                    --hpo-key "hpo_id" \
                    --hpo-label-ja-key "hpo_label" \
                    --ner-model-name "${MEDTXTNER_MODEL}" \
                    --embed-model-name "${SBERT_MODEL}" \
                    --max-length "${SPAN_MAX_LENGTH}" \
                    "${EXTRA_NER_FLAGS[@]}" \
                    --ner-max-length "${SPAN_MAX_LENGTH}" \
                    --ner-label-prefixes "d,cc" \
                    --max-n "${MAX_N}" \
                    --topk "${TOPK}" \
                    --min-score "${MIN_SCORE}"

                  echo "[annotate] skip (already exists): ${PRED_JSONL}"
                fi

                # --- eval ---
                echo "[eval] eval_hpo_annotation_medtxtner.py"

                python eval_hpo_annotation_medtxtner.py \
                  --test-csv "${TEST_CSV}" \
                  --id-column "patient_name" \
                  --gold-hpo-column "hpo_ids" \
                  --pred-jsonl "${PRED_JSONL}" \
                  --hpo-obo "${HPO_OBO}" \
                  --k-list "1,3,5,10" \
                  --output-tsv "${METRIC_TSV}"

              done
            done
          done
        done

      done
    done
  done
done

echo "=== ALL SWEEPS FINISHED ==="
echo "metrics TSVs are under: ${OUT_DIR}/metrics_*.tsv"
