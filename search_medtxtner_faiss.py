#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
search_medtxtner_faiss.py

事前に作成した MedTXTNER 埋め込み (.npy) と、
対応する JSONL (HPO_ID + 表現) を用いて FAISS で類似検索を行うスクリプト。

機能:
  - 事前計算済みの埋め込みベクトルを読み込み
  - FAISS IndexFlatIP でインデックスを構築（内積 = コサイン類似度, 前提: L2正規化済み）
  - 標準入力から日本語文章を受け取り:
      1) Sudachi で分かち書き（オプション）
      2) MedTXTNER で埋め込み
      3) FAISS で top-k の類似表現を検索
      4) 類似度, HPO_ID, 生成表現を表示
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# FAISS
import faiss

# Sudachi (あれば使う)
try:
    from sudachipy import dictionary as sudachi_dictionary
    from sudachipy import tokenizer as sudachi_tokenizer

    SUDACHI_AVAILABLE = True
except ImportError:
    SUDACHI_AVAILABLE = False


# ============================
# MedTXTNER 埋め込みラッパ
# ============================

class MedTxtNerEmbedder:
    """
    `sociocom/MedTXTNER` を使って文埋め込みを計算する簡易ラッパ。

    TokenClassification モデルだが、
    last_hidden_state を attention_mask で mean pooling して文ベクトルにする。
    """

    def __init__(self, model_name: str = "sociocom/MedTXTNER", device: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        last_hidden_state: (batch, seq_len, hidden)
        attention_mask   : (batch, seq_len)
        戻り値: (batch, hidden)
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 256,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        texts      : 埋め込みたい文字列リスト
        batch_size : バッチサイズ
        max_length : トークナイズ時の最大長
        normalize  : True の場合 L2 正規化（FAISS inner-product で cosine 用）

        戻り値: Tensor, shape = (len(texts), hidden_dim)
        """
        all_embeddings: List[torch.Tensor] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            outputs = self.model(**enc)
            token_embeddings = outputs.last_hidden_state  # (B, L, H)
            sent_embeddings = self._mean_pool(token_embeddings, enc["attention_mask"])

            if normalize:
                sent_embeddings = torch.nn.functional.normalize(sent_embeddings, p=2, dim=1)

            all_embeddings.append(sent_embeddings.cpu())

        if not all_embeddings:
            hidden = self.model.config.hidden_size
            return torch.empty(0, hidden)

        return torch.cat(all_embeddings, dim=0)


# ============================
# Sudachi 分かち書き
# ============================

class SudachiTokenizerWrapper:
    """
    Sudachi による分かち書きラッパ。

    - Sudachi がインストールされていない場合は何もしない（元の文章を返す）。
    """

    def __init__(self):
        if SUDACHI_AVAILABLE:
            self._tokenizer = sudachi_dictionary.Dictionary().create()
            self._mode = sudachi_tokenizer.Tokenizer.SplitMode.C
        else:
            self._tokenizer = None
            self._mode = None

    def tokenize(self, text: str) -> str:
        """
        入力テキストを分かち書きした文字列に変換。

        Sudachi が使えない場合はそのまま返す。
        """
        if not SUDACHI_AVAILABLE or self._tokenizer is None:
            return text

        ms = self._tokenizer.tokenize(text, self._mode)
        surfaces = [m.surface() for m in ms if m.surface().strip()]
        return " ".join(surfaces)


# ============================
# メタデータ読み込み (JSONL)
# ============================

def load_metadata_from_jsonl(
    path: Path,
    text_key: str = "patient_expression_final",
    hpo_key: str = "HPO_ID",
) -> Tuple[List[str], List[str]]:
    """
    JSONL から [テキストリスト, HPO_IDリスト] を読み込む。

    embed_with_medtxtner.py で同じ JSONL を元に埋め込みを作っていれば、
    インデックス順が揃うので、埋め込み行列と 1:1 対応する。
    """
    texts: List[str] = []
    hpos: List[str] = []

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

    print(f"[INFO] loaded metadata: {len(texts)} items from {path}")
    return texts, hpos


# ============================
# メイン
# ============================

def main():
    ap = argparse.ArgumentParser()

    # 事前計算済み埋め込み＆メタ
    ap.add_argument(
        "--emb-npy",
        type=str,
        required=True,
        help="事前計算済み MedTXTNER 埋め込み .npy ファイル",
    )
    ap.add_argument(
        "--meta-jsonl",
        type=str,
        required=True,
        help="元の JSONL ファイルパス（HPO_ID と表現が入っているもの）",
    )
    ap.add_argument(
        "--text-key",
        type=str,
        default="patient_expression_final",
        help="JSONL 中の表現のキー名 (default: patient_expression_final)",
    )
    ap.add_argument(
        "--hpo-key",
        type=str,
        default="HPO_ID",
        help="JSONL 中の HPO ID のキー名 (default: HPO_ID)",
    )

    # 検索パラメタ
    ap.add_argument(
        "--topk",
        type=int,
        default=10,
        help="検索して返す件数 (top-k)",
    )

    # 埋め込み用
    ap.add_argument(
        "--model-name",
        type=str,
        default="sociocom/MedTXTNER",
        help="MedTXTNER モデル名 (default: sociocom/MedTXTNER)",
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
        help="トークナイズ max_length (default: 256)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="クエリ埋め込みのバッチサイズ（ほぼ 1 だが一応）",
    )

    # 分かち書き
    ap.add_argument(
        "--no-sudachi",
        action="store_true",
        help="指定すると Sudachi による分かち書きを行わない",
    )

    args = ap.parse_args()

    emb_path = Path(args.emb_npy)
    meta_path = Path(args.meta_jsonl)

    # 1) 埋め込み読み込み
    print(f"[INFO] loading embeddings from {emb_path} ...")
    emb = np.load(emb_path)  # shape = (N, dim)
    if emb.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {emb.shape}")
    n_items, dim = emb.shape
    print(f"[INFO] embeddings shape = (N={n_items}, dim={dim})")

    # 2) メタデータ読み込み（表現＆HPO_ID）
    texts, hpo_ids = load_metadata_from_jsonl(
        meta_path, text_key=args.text_key, hpo_key=args.hpo_key
    )
    if len(texts) != n_items:
        print(
            f"[WARN] embeddings rows ({n_items}) != metadata items ({len(texts)}) "
            "- インデックスがずれている可能性があります"
        )

    # 3) FAISS インデックス構築 (内積ベース、ベクトルは L2 正規化済みを想定)
    print("[INFO] building FAISS IndexFlatIP ...")
    index = faiss.IndexFlatIP(dim)
    # numpy float32 前提
    if emb.dtype != np.float32:
        emb = emb.astype("float32")
    index.add(emb)
    print(f"[INFO] FAISS index built: {index.ntotal} vectors")

    # 4) MedTXTNER 埋め込み器
    print("[INFO] loading MedTXTNER model for query encoding ...")
    embedder = MedTxtNerEmbedder(model_name=args.model_name, device=args.device)

    # 5) Sudachi 分かち書きラッパ
    tokenizer = SudachiTokenizerWrapper()
    use_sudachi = (not args.no_sudachi) and SUDACHI_AVAILABLE
    if not SUDACHI_AVAILABLE and not args.no_sudachi:
        print("[WARN] sudachipy が見つからないため、分かち書きは行いません。")
    print(f"[INFO] use_sudachi = {use_sudachi}")

    # 6) 対話ループ
    print("\n=== MedTXTNER + FAISS 検索 ===")
    print("文章を入力してください（空行 or Ctrl-D で終了）\n")

    while True:
        try:
            query = input("> ").strip()
        except EOFError:
            print("\n[INFO] EOF, bye.")
            break

        if not query:
            print("[INFO] empty input, bye.")
            break

        # 分かち書き（オプション）
        if use_sudachi:
            query_tok = tokenizer.tokenize(query)
        else:
            query_tok = query

        # クエリ埋め込み
        q_emb = embedder.encode(
            [query_tok],
            batch_size=args.batch_size,
            max_length=args.max_length,
            normalize=True,  # cosine 用に正規化
        )  # (1, dim)

        q_np = q_emb.numpy().astype("float32")

        # FAISS 検索
        D, I = index.search(q_np, args.topk)  # D: (1, k), I: (1, k)
        scores = D[0]
        indices = I[0]

        print("\n--- Top-{} results ---".format(args.topk))
        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            if idx < 0 or idx >= len(texts):
                continue
            hpo = hpo_ids[idx] if idx < len(hpo_ids) else ""
            expr = texts[idx]
            print(f"[{rank}] score={score:.4f}  HPO_ID={hpo}\n    expr={expr}")
        print("----------------------\n")


if __name__ == "__main__":
    main()
