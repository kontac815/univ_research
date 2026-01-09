#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
annotate_medtxtner_batch.py

MedTXTNER の埋め込み + n-gram + NER + Negation を組み合わせ、
入力 CSV の各行（例: medical_history_nobr）から HPO を抽出して
JSONL で書き出すバッチスクリプト。

前提:
    - 生成済み患者表現の MedTXTNER 埋め込み (.npy)
    - それに対応する JSONL (HPO_ID 付き; patient_expression_final など)
    - 上記 2 つは embed_expressions_medtxtner.py などで作成済み

依存:
    - annotate_ngram_medtxtner.py
    - MedTxtNerHelper
    - SudachiTokenizerWrapper
    - detect_assertion
    - generate_ngrams
    - greedy_non_overlapping
    - SpanCandidate
    - load_metadata_from_jsonl

出力 JSONL の1行は、以下のような dict になる:

{
  "input_id": "行ごとの ID",
  "text": "元のテキスト (medical_history_nobr など)",
  "rag_mode": "none-medtxtner",
  "model": "MedTXTNER_n-gram_nerneg",
  "annotations": [
    {
      "hpo_id": "HP:0001954",
      "hpo_label_ja": "下痢",
      "status": "present",          # または "absent" / "uncertain"
      "evidence_span": "水のような下痢が続いている",
      "note": "medtxtner_ngram_nerneg"
    },
    ...
  ]
}

※ status は MedTXTNER 側の assertion をそのまま使うので
   "present" / "absent" / "uncertain" の3値になる。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re
from collections import Counter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# 既存の n-gram + NER + Negation 実装を利用
from annotate_ngram_medtxtner import (
    MedTxtNerHelper,
    SudachiTokenizerWrapper,
    detect_assertion,
    generate_ngrams,
    greedy_non_overlapping,
    SpanCandidate,
    load_metadata_from_jsonl,
)


# ==============================
# GPT/Gemini 版と揃えたデータ構造
# ==============================


@dataclass
class HPOAnnotation:
    hpo_id: str
    hpo_label_ja: str
    status: str                # "present" / "absent" / "uncertain"
    evidence_span: str         # 元テキスト側のスパン
    score: float               # FAISS 類似度
    dict_expr: Optional[str] = None  # ★ 辞書側のマッチした生成表現
    note: Optional[str] = None



@dataclass
class AnnotationResult:
    input_id: str
    text: str
    rag_mode: str  # ここでは "none-medtxtner"
    model: str     # 例: "MedTXTNER_n-gram_nerneg"
    annotations: List[HPOAnnotation]

class SbertEmbedder:
    """
    Sentence-BERT (例: pkshatech/GLuCoSE-base-ja-v2) を使って
    日本語テキストを埋め込むためのラッパ。
    """

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model = SentenceTransformer(model_name)
        if device:
            self.model = self.model.to(device)

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
    ):
        # SentenceTransformer 側で normalize_embeddings=True 指定
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return emb  # torch.Tensor (N, dim)

def annotation_result_to_dict(result: AnnotationResult) -> Dict[str, Any]:
    """AnnotationResult を JSONL 用の dict に変換する。"""
    return {
        "input_id": result.input_id,
        "text": result.text,
        "rag_mode": result.rag_mode,
        "model": result.model,
        "annotations": [
            {
                "hpo_id": a.hpo_id,
                "hpo_label_ja": a.hpo_label_ja,
                "status": a.status,
                "evidence_span": a.evidence_span,
                "score": a.score,
                "dict_expr": a.dict_expr,   # ★ 追加
                "note": a.note,
            }
            for a in result.annotations
        ],
    }


# ==============================
# メタ JSONL 読み込み（HPO 日本語ラベルも拾う）
# ==============================


def load_metadata_with_label(
    path: Path,
    text_key: str = "patient_expression_final",
    hpo_key: str = "HPO_ID",
    hpo_label_ja_key: Optional[str] = None,
) -> Tuple[List[str], List[str], Dict[str, str]]:
    """
    生成患者表現のメタ JSONL を読み込み、
      - texts: 患者表現テキスト（埋め込みと 1:1）
      - hpo_ids: HPO ID
      - hpo_label_ja_map: HPO ID -> 日本語ラベル（あれば）
    を返す。

    hpo_label_ja_key が None または空文字なら、
    hpo_label_ja_map は空 dict になる。
    """
    texts: List[str] = []
    hpos: List[str] = []
    hpo_label_ja_map: Dict[str, str] = {}

    use_label = bool(hpo_label_ja_key)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            t = (rec.get(text_key) or "").strip()
            h = (rec.get(hpo_key) or "").strip()
            if not t:
                continue

            texts.append(t)
            hpos.append(h)

            if use_label:
                raw_label = rec.get(hpo_label_ja_key, None)
                if raw_label is not None:
                    label_ja = str(raw_label).strip()
                    if label_ja and h not in hpo_label_ja_map:
                        hpo_label_ja_map[h] = label_ja

    return texts, hpos, hpo_label_ja_map

def char_overlap(a: str, b: str) -> float:
    """
    2つの文字列の「文字レベルの近さ」を計算する（簡易）。

    以前の set-Jaccard は「生後」vs「産後の生血」のように、
    共有文字が少しでもあると通ってしまうため誤爆しやすい。

    ここではより厳しめの「文字 bigram Jaccard」を使う。
    """

    def norm(s: str) -> str:
        return (s or "").replace(" ", "").strip(" \t\r\n、。.,()（）[]【】「」『』:：;；")

    def bigrams(s: str) -> set[str]:
        if len(s) < 2:
            return set()
        return {s[i : i + 2] for i in range(len(s) - 1)}

    a = norm(a)
    b = norm(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    ba = bigrams(a)
    bb = bigrams(b)
    if not ba or not bb:
        return 0.0
    inter = len(ba & bb)
    union = len(ba | bb)
    return inter / union if union else 0.0


def lexical_similarity(span_text: str, dict_expr: str) -> float:
    """
    span_text と dict_expr の「文字レベルでの近さ」を [0,1] で返す。
    - bigram Jaccard
    - substring の場合は長さ比で加点（短い語の過剰な特定化を避けるため、比率ベース）
    """

    def norm(s: str) -> str:
        return (s or "").replace(" ", "").strip(" \t\r\n、。.,()（）[]【】「」『』:：;；")

    s = norm(span_text)
    e = norm(dict_expr)
    if not s or not e:
        return 0.0
    if s == e:
        return 1.0

    # base: bigram overlap
    base = char_overlap(s, e)

    # substring bonus: 長さ比（短い語→長い語の過剰な特定化を抑える）
    if s in e or e in s:
        ratio = min(len(s), len(e)) / max(len(s), len(e))
        base = max(base, ratio)

    return base

# ==============================
# 1テキストを MedTXTNER でアノテーション
# ==============================


def annotate_text_with_medtxtner(
    text: str,
    helper: MedTxtNerHelper,
    tokenizer: SudachiTokenizerWrapper,
    embedder: SbertEmbedder,
    index: faiss.Index,
    dict_texts: List[str],
    dict_hpo_ids: List[str],
    hpo_label_ja_map: Dict[str, str],
    *,
    max_length: int = 256,
    max_n: int = 6,
    topk: int = 3,
    min_score: float = 0.6,
    use_ner_region: bool = False,
    ner_max_length: int = 256,
    ner_label_prefixes: Optional[List[str]] = None,
    span_len_bonus_alpha: float = 0.15,
    span_len_bonus_beta: float = 0.05,
    treat_adjacent_as_overlap: bool = False,
) -> List[HPOAnnotation]:

    """
    1 文 text に対して MedTXTNER + n-gram + FAISS + Negation を適用し、
    HPOAnnotation のリストを返す。
    """
    text = (text or "").strip()
    if not text:
        return []

    # Sudachi でトークン + オフセット
    tokens, token_spans = tokenizer.tokenize_with_offsets(text)
    if not tokens:
        return []

    # NER 領域のマスク（オプション）
    token_allowed: List[bool]
    if use_ner_region:
        char_mask = helper.get_ner_char_mask(
            text,
            max_length=ner_max_length,
            allowed_label_prefixes=ner_label_prefixes,
        )
        if not char_mask:
            # NER がうまく動かなかった場合は全トークン許可
            token_allowed = [True] * len(tokens)
        else:
            token_allowed = []
            for (s, e) in token_spans:
                allowed = False
                for pos in range(s, e):
                    if 0 <= pos < len(char_mask) and char_mask[pos]:
                        allowed = True
                        break
                token_allowed.append(allowed)
    else:
        token_allowed = [True] * len(tokens)

    # n-gram 候補生成
    spans = generate_ngrams(tokens, token_allowed=token_allowed, max_n=max_n)
    if not spans:
        return []

    span_texts = [t for (_, _, t) in spans]

    # まとめて SBERT で埋め込み
    q_emb = embedder.encode(
        span_texts,
        batch_size=32,
        normalize=True,
    )
    if q_emb.size(0) == 0:
        return []

    q_np = q_emb.cpu().numpy().astype("float32")


    # FAISS 検索
    D, I = index.search(q_np, topk)  # (num_spans, topk)

    # SpanCandidate のリストを作成
    candidates: List[SpanCandidate] = []
    n_dict = len(dict_texts)

    # 文字レベルのオーバーラップがこれ未満の候補は捨てる
    # 以前は set-Jaccard を使っていたが誤爆が多かったため、lexical_similarity に置換。
    LEX_SIM_MIN = 0.20
    LEX_WEIGHT = 0.35
    SHORT_EXPR_PENALTY = 0.40

    for idx_span, (score_vec, idx_vec) in enumerate(zip(D, I)):
        start, end, span_text = spans[idx_span]

        best_score = -1.0
        best_combined = -1.0
        best_hpo = ""
        best_expr = ""

        for score, expr_idx in zip(score_vec, idx_vec):
            if expr_idx < 0 or expr_idx >= n_dict:
                continue

            expr = dict_texts[expr_idx]

            lex = lexical_similarity(span_text, expr)
            if lex < LEX_SIM_MIN:
                continue

            # 長い span に対して短すぎる expr を選びにくくする（例: "高インスリン性低血糖" に "低血糖"）
            span_n = (span_text or "").replace(" ", "").strip(" \t\r\n、。.,()（）[]【】「」『』:：;；")
            expr_n = (expr or "").replace(" ", "").strip(" \t\r\n、。.,()（）[]【】「」『』:：;；")
            short_pen = 0.0
            if len(span_n) > 0 and len(expr_n) < len(span_n):
                short_pen = (len(span_n) - len(expr_n)) / len(span_n)

            combined = float(score) + LEX_WEIGHT * lex - SHORT_EXPR_PENALTY * short_pen

            if combined > best_combined:
                best_combined = combined
                best_score = float(score)
                best_hpo = dict_hpo_ids[expr_idx]
                best_expr = expr

        # そもそも条件を満たす候補がなかった場合
        if best_score < min_score or not best_hpo:
            continue

        assertion = detect_assertion(
            text,
            tokens,
            token_spans,
            span_start=start,
            span_end=end,
        )

        candidates.append(
            SpanCandidate(
                start=start,
                end=end,
                text=span_text,
                best_hpo=best_hpo,
                best_expr=best_expr,
                score=best_score,
                assertion=assertion,
            )
        )


    # 重なりを解決（DP）
    # - treat_adjacent_as_overlap=True: 隣接スパンも競合扱い（分割抑制）
    # - len_bonus_*: 長いスパンを少し優遇（複合語が短い語に吸われるのを抑える）
    selected = greedy_non_overlapping(
        candidates,
        len_bonus_alpha=span_len_bonus_alpha,
        len_bonus_beta=span_len_bonus_beta,
        treat_adjacent_as_overlap=treat_adjacent_as_overlap,
    )
    if not selected:
        return []

    # SpanCandidate -> HPOAnnotation
    annotations: List[HPOAnnotation] = []
    for cand in selected:
        label_ja = hpo_label_ja_map.get(cand.best_hpo, "")
        annotations.append(
            HPOAnnotation(
                hpo_id=cand.best_hpo,
                hpo_label_ja=label_ja,
                status=cand.assertion,
                evidence_span=cand.text,
                score=cand.score,
                dict_expr=cand.best_expr,          # ★ どの生成表現に引っかかったか
                note="medtxtner_ngram_nerneg",
            )
        )
    return annotations


# ==============================
# main
# ==============================


def main():
    ap = argparse.ArgumentParser()

    # 入力 CSV / 出力 JSONL
    ap.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="アノテーション対象テキストを含む CSV (例: hpo_annotation_test_dataset.csv)",
    )
    ap.add_argument(
        "--text-column",
        type=str,
        required=True,
        help="入力 CSV 中のテキスト列名 (例: medical_history_nobr)",
    )
    ap.add_argument(
        "--id-column",
        type=str,
        default=None,
        help="行を識別する ID 列名 (例: patient_name)。未指定なら row_{idx} を使用。",
    )
    ap.add_argument(
        "--output-jsonl",
        type=str,
        required=True,
        help="アノテーション結果を書き出す JSONL ファイルパス",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="先頭 N 行のみ処理（デバッグ用）。未指定なら全行。",
    )

    # 生成患者表現（辞書側）の埋め込み + メタ JSONL
    ap.add_argument(
        "--emb-npy",
        type=str,
        required=True,
        help="生成患者表現の埋め込み .npy (MedTXTNER ベース, L2 正規化済み)",
    )
    ap.add_argument(
        "--meta-jsonl",
        type=str,
        required=True,
        help="埋め込みに対応する JSONL (表現 + HPO_ID)",
    )
    ap.add_argument(
        "--text-key",
        type=str,
        default="patient_expression_final",
        help="meta-jsonl 中の表現キー名 (デフォ: patient_expression_final)",
    )
    ap.add_argument(
        "--hpo-key",
        type=str,
        default="HPO_ID",
        help="meta-jsonl 中の HPO ID キー名 (デフォ: HPO_ID)",
    )
    ap.add_argument(
        "--hpo-label-ja-key",
        type=str,
        default="",
        help=(
            "meta-jsonl 中で日本語 HPO ラベルとして使うキー名 "
            "(例: hpo_name_ja / jp_final)。空ならラベルは空文字。"
        ),
    )

    # MedTXTNER (NER + negation 用)
    ap.add_argument(
        "--ner-model-name",
        type=str,
        default="sociocom/MedTXTNER",
        help="MedTXTNER モデル名 (NER + negation 用; デフォ: sociocom/MedTXTNER)",
    )

    # 埋め込み用 Sentence-BERT
    ap.add_argument(
        "--embed-model-name",
        type=str,
        default="pkshatech/GLuCoSE-base-ja-v2",
        help="埋め込みに使う Sentence-BERT モデル名 (デフォ: pkshatech/GLuCoSE-base-ja-v2)",
    )

    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="'cuda' or 'cpu'（未指定なら自動判定）",
    )
    ap.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="span 埋め込み用 max_length",
    )

    # NER 領域利用
    ap.add_argument(
        "--use-ner-region",
        action="store_true",
        help="MedTXTNER NER で非Oラベル領域だけ n-gram を展開する",
    )
    ap.add_argument(
        "--ner-max-length",
        type=int,
        default=256,
        help="NER 用 max_length",
    )
    ap.add_argument(
        "--ner-label-prefixes",
        type=str,
        default="d,cc",
        help="症候として扱う NER ラベルのプレフィックスをカンマ区切りで指定 (空なら非O全部)",
    )

    # n-gram & FAISS
    ap.add_argument(
        "--max-n",
        type=int,
        default=6,
        help="n-gram の最大長 (トークン数)",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=3,
        help="各 n-gram に対する FAISS 上位何件を見るか",
    )
    ap.add_argument(
        "--min-score",
        type=float,
        default=0.6,
        help="この類似度以上の候補のみ採用",
    )
    ap.add_argument(
        "--span-len-bonus-alpha",
        type=float,
        default=0.15,
        help="非重複スパン選択の長さボーナス（文字長: alpha*log1p(char_len)）",
    )
    ap.add_argument(
        "--span-len-bonus-beta",
        type=float,
        default=0.05,
        help="非重複スパン選択の長さボーナス（トークン長: beta*token_len）",
    )
    ap.add_argument(
        "--treat-adjacent-as-overlap",
        action="store_true",
        help="非重複スパン選択で end==start も競合扱い（分割抑制）",
    )
    ap.add_argument("--wandb", action="store_true", help="W&B に進捗/件数を送る")
    ap.add_argument("--wandb-project", type=str, default="HPO_annotate")
    ap.add_argument("--wandb-run-name", type=str, default=None)
    ap.add_argument("--wandb-tags", type=str, default="")
    ap.add_argument("--wandb-group", type=str, default=None)

    args = ap.parse_args()

    # ===== 入力 CSV 読み込み =====
    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        sys.exit(f"[ERROR] input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if args.text_column not in df.columns:
        sys.exit(f"[ERROR] text-column '{args.text_column}' が CSV に存在しません。")
    has_id_col = args.id_column is not None and args.id_column in df.columns

    if args.max_rows is not None:
        df = df.head(args.max_rows)

    # ===== 辞書側の埋め込み & メタデータ =====
    emb_path = Path(args.emb_npy)
    meta_path = Path(args.meta_jsonl)

    emb = np.load(emb_path).astype("float32")
    n_items, dim = emb.shape

    if args.hpo_label_ja_key.strip():
        texts, hpo_ids, hpo_label_ja_map = load_metadata_with_label(
            meta_path,
            text_key=args.text_key,
            hpo_key=args.hpo_key,
            hpo_label_ja_key=args.hpo_label_ja_key.strip(),
        )
    else:
        # 既存の関数で読みつつ、ラベル map は空にする
        texts, hpo_ids = load_metadata_from_jsonl(
            meta_path, text_key=args.text_key, hpo_key=args.hpo_key
        )
        hpo_label_ja_map = {}

    if len(texts) != n_items:
        print(
            f"[WARN] embeddings ({n_items}) != metadata texts ({len(texts)})",
            file=sys.stderr,
        )

    # ===== FAISS Index =====
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    print(f"[INFO] FAISS index built: {index.ntotal} vectors, dim={dim}", file=sys.stderr)

    # ===== MedTXTNER helper (NER + negation) & tokenizer =====
    helper = MedTxtNerHelper(model_name=args.ner_model_name, device=args.device)
    tokenizer = SudachiTokenizerWrapper()

    # ===== SBERT embedder =====
    embedder = SbertEmbedder(model_name=args.embed_model_name, device=args.device)


    # NER ラベル prefix 設定
    ner_label_prefixes: Optional[List[str]] = None
    if args.ner_label_prefixes:
        ner_label_prefixes = [
            s.strip() for s in args.ner_label_prefixes.split(",") if s.strip()
        ]

    # ===== 出力ファイルを空にしておく =====
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f_out:
        pass

    print(
        f"[INFO] start batch annotation: rows={len(df)}, "
        f"text_column={args.text_column}, id_column={args.id_column}",
        file=sys.stderr,
    )

    wandb_run = None
    if args.wandb:
        if not HAS_WANDB:
            print("[WARN] wandb is not installed. Disable --wandb to silence.", file=sys.stderr)
        else:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                group=args.wandb_group,
                tags=[t for t in args.wandb_tags.split(",") if t.strip()],
                config=vars(args),
            )

    total_rows = 0
    total_annotations = 0
    status_counts: Counter = Counter()

    # ===== 各行に対してアノテーション =====
    with output_path.open("a", encoding="utf-8") as f_out:
        for idx_row, row in df.iterrows():
            if has_id_col:
                input_id = str(row[args.id_column])
            else:
                input_id = f"row_{idx_row}"

            text_raw = row[args.text_column]
            text = "" if isinstance(text_raw, float) and np.isnan(text_raw) else str(text_raw)

            try:
                anns = annotate_text_with_medtxtner(
                    text=text,
                    helper=helper,
                    tokenizer=tokenizer,
                    embedder=embedder,   # ★ 追加
                    index=index,
                    dict_texts=texts,
                    dict_hpo_ids=hpo_ids,
                    hpo_label_ja_map=hpo_label_ja_map,
                    max_length=args.max_length,
                    max_n=args.max_n,
                    topk=args.topk,
                    min_score=args.min_score,
                    use_ner_region=args.use_ner_region,
                    ner_max_length=args.ner_max_length,
                    ner_label_prefixes=ner_label_prefixes,
                    span_len_bonus_alpha=args.span_len_bonus_alpha,
                    span_len_bonus_beta=args.span_len_bonus_beta,
                    treat_adjacent_as_overlap=args.treat_adjacent_as_overlap,
                )

            except Exception as e:
                sys.stderr.write(
                    f"[ERROR] input_id={input_id}: MedTXTNER アノテーション中に例外が発生しました: {e}\n"
                )
                anns = []

            result = AnnotationResult(
                input_id=input_id,
                text=text,
                rag_mode="none-medtxtner",
                model=f"MedTXTNER_n-gram_nerneg:{args.ner_model_name}",
                annotations=anns,
            )
            out_obj = annotation_result_to_dict(result)
            f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            f_out.flush()
            total_rows += 1
            total_annotations += len(anns)
            status_counts.update(a.status for a in anns)

    print("[INFO] done.", file=sys.stderr)

    if wandb_run:
        wandb_run.log(
            {
                "rows": total_rows,
                "annotations": total_annotations,
                "status_counts": dict(status_counts),
                "output_jsonl": str(output_path),
            }
        )
        wandb_run.finish()


if __name__ == "__main__":
    main()
