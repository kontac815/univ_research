#!/bin/bash
set -euo pipefail

########################################
# 設定エリア (環境に合わせて書き換えてください)
########################################

# 1. パイプラインの入力ディレクトリ (round_*.jsonl がある場所)
INPUT_DIR="../data/notLoRA_vllm/doctor"

# 2. 実行するPythonスクリプト (5a) のパス
SCRIPT_PATH="./5a.postprocess_expressions_sbert.py"

# 3. HPOマスタCSVのパス (depth情報が含まれているもの)
# 例: HPO_symptom_depth_leq3_self_reportable_with_jp.csv や hpo_master_all_jp.csv など
HPO_MASTER="../data/hpo_master_all_jp.weblio.with_llm_jp.v2.csv"

# 4. S-BERT モデル設定
SBERT_MODEL="pkshatech/GLuCoSE-base-ja-v2"
DEVICE="cuda"  # GPUがない場合は "cpu"

# 5. フィルタリング閾値
SELF_MIN_SIM=0.1
OTHER_MARGIN=0.15
MODE="doctor"  # "patient" or "doctor"

########################################
# 実行処理
########################################

# 出力ディレクトリ作成 (入力ディレクトリの中に cleaned フォルダを作る)
OUTPUT_DIR="${INPUT_DIR}/cleaned"
LOG_DIR="${INPUT_DIR}/removed_logs"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "=== Batch Processing Start ==="
echo "Input Dir:  $INPUT_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "HPO Master: $HPO_MASTER"
echo "------------------------------"

# round_*.jsonl にマッチするファイルをループ処理
# 見つからない場合はそのまま文字列として渡ってエラーになるのを防ぐため nullglob を設定
shopt -s nullglob
FILES=("${INPUT_DIR}"/round_*.jsonl)
shopt -u nullglob

if [ ${#FILES[@]} -eq 0 ]; then
    echo "[ERROR] No round_*.jsonl files found in $INPUT_DIR"
    exit 1
fi

for jsonl_file in "${FILES[@]}"; do
    # ファイル名を取得 (例: round_0.jsonl)
    filename=$(basename "$jsonl_file")
    
    # 出力ファイル名 (例: round_0_cleaned.jsonl)
    output_name="${filename%.*}_cleaned.jsonl"
    output_path="${OUTPUT_DIR}/${output_name}"
    
    # 削除ログファイル名 (例: round_0_removed.jsonl)
    log_name="${filename%.*}_removed.jsonl"
    log_path="${LOG_DIR}/${log_name}"

    echo "Processing: $filename ..."

    python "$SCRIPT_PATH" \
        --input "$jsonl_file" \
        --hpo-master "$HPO_MASTER" \
        --output "$output_path" \
        --output-removed "$log_path" \
        --mode "$MODE" \
        --sbert-model "$SBERT_MODEL" \
        --device "$DEVICE" \
        --self-min-sim "$SELF_MIN_SIM" \
        --other-margin "$OTHER_MARGIN"

    echo "-> Saved to: $output_path"
    echo ""
done

echo "=== All Done ==="
echo "Cleaned files are in: $OUTPUT_DIR"
echo "Removed logs are in:  $LOG_DIR"