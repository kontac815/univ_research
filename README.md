# 日本語HPO表現生成・タグ付けパイプライン

日本語の医療テキストから HPO (Human Phenotype Ontology) 表現を生成・精製し、抽出・評価するための研究コードです。日本語固有の患者/医師表現を HPO と橋渡しし、PubCaseFinder などの HPO ベース診断支援や日英LLMにつなげることを狙います。

## 研究背景（要約）
- LLM と HPO を組み合わせた診断支援が国際的に進展する一方、自由記述から安定して HPO タグを付与する前段が課題。
- 日本語の患者/医師表現は多様で、同じ HPO に対して多数の言い回しが存在。英語LLMへの機械翻訳経由ではニュアンス損失の懸念。
- 先行リソース: NAIST 大規模患者表現辞書、万病辞書、日本語HPO訳、MedTextNER 系モデル等はあるが、「日本語フリーテキスト→HPO→診断支援」の一貫パイプラインは未整備。

## 目的
1. HPOごとに日本語の患者表現・医師表現をLLMで大規模生成し、双方向辞書を構築する。
2. 日本語テキストから症候表現を抽出し、埋め込み類似度に基づいて HPO を推定するタグ付けパイプラインを作る。
3. 得られた HPO タグを PubCaseFinder や日英LLM と接続し、実運用に近い設定で有効性を検証する。

## パイプライン概要
1. **ラベル補完/翻訳**: `hpo_label_fill_llm.py` で jp_final 欠損を埋め、`hpo_definition_translate.py` で英語定義を日本語化。
2. **SFTデータ作成**: `sft_dataset_build.py` で NAIST/万病辞書と HPO マスタから患者/医師用の chat SFT JSONL を組み立て。
3. **SFT学習**: `sft_train_qlora.py` で QLoRA による日本語医療表現モデルを学習。
4. **表現生成**: `expression_generate_vllm.py` で vLLM により患者/医師表現を大量生成（定義あり/なしプロンプト切替）。
5. **表現精製**: `expression_filter_sbert.py` でラベル重複・意味重複・質の低い表現をフィルタし、`expression_master_build.py` で HPO ごとのマスタ JSONL に統合。
6. **埋め込み生成**: `embed_expressions_medtxtner.py` (MedTXTNER) や `embed_expressions_sbert.py` (SBERT) で辞書表現をベクトル化。
7. **タグ付け/評価**: `annotate_medtxtner_batch.py` + `annotate_ngram_medtxtner.py` で n-gram + NER + negation を用いた HPO 付与を実行し、`eval_medtxtner_annotations.py` で gold データと比較評価。`search_medtxtner_faiss.py` は対話的検索ツール。

## 主なスクリプト対応表
- `hpo_label_fill_llm.py`: jp_final 欠損を LLM で補完（キャッシュ対応）。
- `hpo_definition_translate.py`: HPO 英語定義を並列で日本語翻訳。
- `sft_dataset_build.py`: NAIST/万病辞書 + HPO マスタから SFT データ生成。
- `sft_train_qlora.py`: QLoRA で SFT 学習。
- `expression_generate_vllm.py`: vLLM で患者/医師表現を生成。
- `expression_filter_sbert.py`: SBERT を用いた表現フィルタリング + 削除ログ出力。
- `expression_master_build.py`: 患者/医師表現を HPO ごとに統合したマスタを作成。
- `embed_expressions_medtxtner.py` / `embed_expressions_sbert.py`: 辞書表現の埋め込み生成。
- `annotate_ngram_medtxtner.py`: n-gram + MedTXTNER + negation で候補抽出（ライブラリ的に利用）。
- `annotate_medtxtner_batch.py`: CSV バッチを読み、HPO 注釈 JSONL を出力。
- `eval_medtxtner_annotations.py`: gold CSV と照合して Precision/Recall/F1（階層版含む）を計算。
- `search_medtxtner_faiss.py`: FAISS による類似検索デバッグ用 CLI。

## 使い方の目安
1. 環境: Python 3.10 以降を想定。`pip install -r requirements.txt` 相当の依存を整備してください（Transformers, sentence-transformers, faiss, sudachipy など）。
2. ラベル補完・翻訳（任意）:
   ```bash
   python hpo_label_fill_llm.py --master ../data/hpo_master_all_jp.weblio.csv --subset ... --obo ../data/hp.obo --out-master ...
   python hpo_definition_translate.py --master ... --obo ../data/hp.obo --out-master ...
   ```
3. SFT データ生成/学習:
   ```bash
   python sft_dataset_build.py --hpo-master ... --naist-xlsx ... --manbyo-xlsx ... --out-naist ... --out-manbyo ...
   python sft_train_qlora.py --model-name-or-path <base model> --train-files <jsonl ...> --output-dir <lora_dir>
   ```
4. 表現生成と精製:
   ```bash
   python expression_generate_vllm.py --input-hpo-master ... --lora <path> --mode patient|doctor ...
   python expression_filter_sbert.py --input-jsonl ... --hpo-master ... --output-jsonl ...
   python expression_master_build.py --hpo-master ... --patient-jsonl ... --doctor-jsonl ... --out ...
   ```
5. 埋め込み・注釈・評価:
   ```bash
   python embed_expressions_medtxtner.py --input master.jsonl --input-format jsonl --text-key patient_expression_final --output-npy ... --output-tsv ...
   python annotate_medtxtner_batch.py --input-csv ... --text-column medical_history_nobr --dict-jsonl ... --dict-emb ... --out-jsonl ...
   python eval_medtxtner_annotations.py --gold hpo_annotation_test_dataset.csv --pred <annotate出力> --obo ../data/hp.obo
   ```

※ データ（HPOマスタ、辞書、テストCSVなど）はリポジトリに含まれていません。自前で用意し、パスを適宜置き換えてください。

## 既知のポイント
- 一部スクリプトは GPU 前提（vLLM, QLoRA, FAISS with GPU 等）。CPU 実行時はパラメータを調整してください。
- Sudachi に依存する処理があります。インストール/辞書設定がない場合はフォールバックする箇所があります。
- `deprecated/` 配下は旧パイプラインです。現行は本 README に記載のスクリプト群を利用してください。

## 今後の改良例
- MedTxtNerEmbedder 重複コードの共通化（search/embedding 両方で利用）。
- CLI パラメータの統一と Hydra/TOML などによる設定管理。
- 評価セットの拡充と自動レポート化。
