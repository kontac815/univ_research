#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
embed_expressions_sbert.py

HPO 患者/医師表現の JSONL を読み込み、
pkshatech/GLuCoSE-base-ja-v2 などの Sentence-BERT で埋め込み (.npy) を作る。

前提:
  - 1行1レコードの JSONL
  - 表現テキストは text_key で指定 (例: patient_expression_final)
  - HPO_ID などのメタ情報は別スクリプトで JSONL 側から読む
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


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
            t = (rec.get(text_key) or "").strip()
            if t:
                texts.append(t)
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-jsonl",
        type=str,
        required=True,
        help="HPO 表現辞書の JSONL (例: patient/doctor_expression_gen_filtered.jsonl)",
    )
    ap.add_argument(
        "--text-key",
        type=str,
        default="patient_expression_final",
        help="JSONL 中のテキストキー名 (患者表現: patient_expression_final / 医師表現: doctor_expression_final)",
    )
    ap.add_argument(
        "--output-npy",
        type=str,
        required=True,
        help="出力する埋め込み .npy パス",
    )
    ap.add_argument(
        "--model-name",
        type=str,
        default="pkshatech/GLuCoSE-base-ja-v2",
        help="Sentence-BERT モデル名 (デフォルト: pkshatech/GLuCoSE-base-ja-v2)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="埋め込み時のバッチサイズ",
    )
    ap.add_argument("--wandb", action="store_true", help="W&B に進捗を送る")
    ap.add_argument("--wandb-project", type=str, default="HPO_embed")
    ap.add_argument("--wandb-run-name", type=str, default=None)
    ap.add_argument("--wandb-tags", type=str, default="")
    ap.add_argument("--wandb-group", type=str, default=None)

    args = ap.parse_args()

    input_jsonl = Path(args.input_jsonl)
    if not input_jsonl.exists():
        raise FileNotFoundError(f"input-jsonl not found: {input_jsonl}")

    wandb_run = None
    if args.wandb:
        if not HAS_WANDB:
            print("[WARN] wandb is not installed. Disable --wandb to silence.")
        else:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                group=args.wandb_group,
                tags=[t for t in args.wandb_tags.split(",") if t.strip()],
                config=vars(args),
            )

    print(f"== load texts from: {input_jsonl}")
    texts = load_texts_from_jsonl(input_jsonl, text_key=args.text_key)
    print(f"== texts: {len(texts)}")

    print(f"== load Sentence-BERT model: {args.model_name}")
    model = SentenceTransformer(args.model_name)

    print("== encode...")
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )  # shape: (N, dim), float32

    emb = emb.astype("float32")
    print(f"== embeddings shape: {emb.shape}")

    output_npy = Path(args.output_npy)
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, emb)
    print(f"== saved to: {output_npy}")

    if wandb_run:
        wandb_run.log(
            {
                "num_texts": len(texts),
                "embedding_dim": emb.shape[1] if emb.size > 0 else 0,
                "output_path": str(output_npy),
            }
        )
        wandb_run.finish()


if __name__ == "__main__":
    main()
