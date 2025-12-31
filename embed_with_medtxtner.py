#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
embed_with_medtxtner.py

生成済みの患者／医師表現を `sociocom/MedTXTNER` で埋め込みベクトルに変換するスクリプト。

- 入力:
    * JSONL: 各行の指定キーからテキストを取得
             例) {"patient_expression_final": "立ちくらみがする", ...}
    * TXT:   1行1表現

- 出力:
    * .npy: shape = (N, hidden_dim) の float32 行列
    * .tsv: index と元テキストの対応表
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


class MedTxtNerEmbedder:
    """
    `sociocom/MedTXTNER` を使って文埋め込みを計算する簡易ラッパー。

    - Token classification モデルだが、
      出力の last_hidden_state (各トークンのベクトル) を
      attention_mask で mean pooling して文ベクトルにする。
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
    def encode(self, texts: List[str], batch_size: int = 32, max_length: int = 256,
               normalize: bool = True) -> torch.Tensor:
        """
        texts      : 埋め込みたい文字列リスト
        batch_size : バッチサイズ
        max_length : トークナイズ時の最大長
        normalize  : True の場合 L2 正規化（FAISS の inner-product で cosine にする用）

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

            # TokenClassification モデルなので、hidden_states から最終層を使う
            outputs = self.model(**enc, output_hidden_states=True, return_dict=True)
            # outputs.hidden_states: tuple(layer_num) of (B, L, H)
            token_embeddings = outputs.hidden_states[-1]  # (B, L, H)

            sent_embeddings = self._mean_pool(token_embeddings, enc["attention_mask"])  # (B, H)

            if normalize:
                sent_embeddings = torch.nn.functional.normalize(sent_embeddings, p=2, dim=1)

            all_embeddings.append(sent_embeddings.cpu())

        if not all_embeddings:
            hidden = self.model.config.hidden_size
            return torch.empty(0, hidden)

        return torch.cat(all_embeddings, dim=0)



def load_texts_from_jsonl(path: Path, text_key: str) -> List[str]:
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            txt = (rec.get(text_key) or "").strip()
            if not txt:
                continue
            texts.append(txt)
    print(f"[INFO] loaded {len(texts)} texts from JSONL (key='{text_key}')")
    return texts


def load_texts_from_txt(path: Path) -> List[str]:
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            txt = line.strip()
            if not txt:
                continue
            texts.append(txt)
    print(f"[INFO] loaded {len(texts)} texts from TXT (1 line = 1 expression)")
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="入力ファイル (JSONL or TXT)")
    ap.add_argument(
        "--input-format",
        type=str,
        choices=["jsonl", "txt"],
        required=True,
        help="入力フォーマット",
    )
    ap.add_argument(
        "--text-key",
        type=str,
        default="patient_expression_final",
        help="JSONL のときにテキストを取るキー名",
    )
    ap.add_argument(
        "--model-name",
        type=str,
        default="sociocom/MedTXTNER",
        help="MedTextNER のモデル名",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="埋め込み計算時のバッチサイズ",
    )
    ap.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="トークナイズの max_length",
    )
    ap.add_argument(
        "--no-normalize",
        action="store_true",
        help="指定すると L2 正規化を行わない（デフォルトは正規化する）",
    )
    ap.add_argument(
        "--output-npy",
        type=str,
        required=True,
        help="埋め込み行列を書き出す .npy ファイルパス",
    )
    ap.add_argument(
        "--output-tsv",
        type=str,
        required=True,
        help="index と元表現を書き出す .tsv ファイルパス",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="'cuda' or 'cpu'（未指定なら自動判定）",
    )

    args = ap.parse_args()

    input_path = Path(args.input)

    if args.input_format == "jsonl":
        texts = load_texts_from_jsonl(input_path, text_key=args.text_key)
    else:
        texts = load_texts_from_txt(input_path)

    if not texts:
        print("[WARN] no texts found. exit.")
        return

    # 埋め込み器の用意
    embedder = MedTxtNerEmbedder(model_name=args.model_name, device=args.device)

    # 埋め込み計算
    print("[INFO] encoding texts with MedTXTNER ...")
    emb = embedder.encode(
        texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        normalize=not args.no_normalize,
    )  # (N, hidden)

    print(f"[INFO] embeddings shape: {tuple(emb.shape)}")

    # .npy で保存
    out_npy = Path(args.output_npy)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, emb.numpy().astype("float32"))
    print(f"[INFO] saved embeddings to: {out_npy}")

    # index とテキスト対応表
    out_tsv = Path(args.output_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8") as f:
        f.write("index\ttext\n")
        for i, t in enumerate(texts):
            t_safe = t.replace("\t", " ").replace("\n", " ")
            f.write(f"{i}\t{t_safe}\n")
    print(f"[INFO] saved index-text map to: {out_tsv}")


if __name__ == "__main__":
    main()
