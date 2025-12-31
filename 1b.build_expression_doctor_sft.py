#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
build_manbyo_doctor_sft.py

MANBYO_20210602.xlsx から、医師表現生成用 LoRA の SFT データを作成するスクリプト。

- 入力: MANBYO_20210602.xlsx
    カラム例:
        出現形, 出現形よみ, ICDコード, 標準病名, 信頼度レベル, 頻度レベル

- 出力: JSONL (1行1サンプル)
    {
      "messages": [
        {"role": "system", "content": "...医師表現用の指示..."},
        {"role": "user", "content": "標準病名: 発熱"},
        {"role": "assistant", "content": "発熱"}
      ],
      "metadata": {
        "source": "MANBYO",
        "standard_name": "発熱",
        "surface": "発熱",
        "icd_code": "R509",
        "trust": "S",
        "freq": "95-100%"
      }
    }

この JSONL は、NAIST 患者表現 SFT と同じコードで、
system プロンプトだけ差し替えてそのまま SFT に使える想定。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd


def load_manbyo(path: Path) -> pd.DataFrame:
    print(f"=== load MANBYO: {path} ===")
    df = pd.read_excel(path)
    expected_cols = ["出現形", "出現形よみ", "ICDコード", "標準病名", "信頼度レベル", "頻度レベル"]
    for c in expected_cols:
        if c not in df.columns:
            raise RuntimeError(f"MANBYO に期待するカラムが見つかりません: {c}")
    return df


def filter_manbyo(
    df: pd.DataFrame,
    trust_levels: Optional[List[str]] = None,
    min_freq_rank: int = 0,
    icd_prefixes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    MANBYO の行をフィルタする。
    - trust_levels: ["S", "A"] など。None のときはフィルタしない。
    - min_freq_rank: 1〜20 を想定。
        頻度レベルを 5%刻みの 20 段階 (0-5%, 5-10%, ..., 95-100%) とみなして
        ランク付けし、rank >= min_freq_rank のものだけ残す。
        例:
            "0-5%"    -> rank 1
            "5-10%"   -> rank 2
            ...
            "90-95%"  -> rank 19
            "95-100%" -> rank 20
            "出現なし" -> rank 0
    - icd_prefixes: ["R", "G"] など。None のときは ICD コードでフィルタしない。
    """

    df2 = df.copy()

    # 信頼度フィルタ
    if trust_levels:
        df2 = df2[df2["信頼度レベル"].isin(trust_levels)]

    # ICD コードのプレフィックスフィルタ
    if icd_prefixes:
        icd_col = df2["ICDコード"].fillna("").astype(str)
        cond = False
        for p in icd_prefixes:
            cond = cond | icd_col.str.startswith(p)
        df2 = df2[cond]

    # 頻度レベルフィルタ
    if min_freq_rank > 0:
        import re

        def freq_rank(s: Any) -> int:
            if not isinstance(s, str):
                return 0
            s = s.strip()
            if s == "出現なし":
                return 0
            m = re.match(r"(\d+)-(\d+)%", s)
            if not m:
                return 0
            low = int(m.group(1))
            high = int(m.group(2))
            # 0-5% -> 1, 5-10% -> 2, ..., 95-100% -> 20
            return high // 5

        ranks = df2["頻度レベル"].map(freq_rank)
        df2 = df2[ranks >= min_freq_rank]

    return df2



def build_messages_for_doctor(standard_name: str, surface: str) -> List[Dict[str, str]]:
    """
    医師表現 SFT 用の chat messages を組み立てる。
    NAIST 患者表現 SFT と同じ構造で、system プロンプトのみ医師向けに変更。
    """
    system_prompt = (
        "あなたは日本語の医療用語と電子カルテ記載に詳しい日本人医師です。\n"
        "次に与える医学的な用語（症状名・所見名・病態名など）について、"
        "医師がカルテや診療録に記載しそうな表現を日本語で1つだけ生成してください。\n"
        "できるだけ簡潔に、通常のカルテ記載を意識した短い語句またはごく短い文にしてください。\n"
        "患者の話し言葉ではなく、医師が使用する標準的な専門用語・略語を用いてください。\n"
        "ただし、意味が過度に抽象的にならないようにし、診療録として具体的な情報が伝わる表現にしてください。\n"
        "出力は1つの表現のみとし、説明文やコメント、箇条書き、番号付けは一切行わないでください。"
    )
    user_content = f" {standard_name}"
    assistant_content = surface

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def build_sft_examples(
    df: pd.DataFrame,
    max_examples_per_name: int = 20,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    MANBYO の DataFrame から SFT 用サンプルを作る。
    - 標準病名ごとに groupby して、出現形を最大 max_examples_per_name 個まで使う。
    - 同じ (標準病名, 出現形) の重複は取り除く。
    """

    random.seed(seed)


    # ① 完全一致重複を削除
    df = df.drop_duplicates(subset=["標準病名", "出現形"])

    # ② 標準病名と出現形が「同じもの」は除外
    #    （空白や全角スペースだけ無視して比較）
    def _norm(s: Any) -> str:
        s = str(s) if s is not None else ""
        return s.replace(" ", "").replace("　", "")

    mask = df.apply(
        lambda r: _norm(r["標準病名"]) != _norm(r["出現形"]),
        axis=1,
    )
    df = df[mask]

    groups = df.groupby("標準病名")
    examples: List[Dict[str, Any]] = []

    for standard_name, g in groups:
        # 出現形をシャッフルして上位 max_examples_per_name 個だけ使う
        g_shuf = g.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        if max_examples_per_name > 0:
            g_shuf = g_shuf.iloc[:max_examples_per_name]

        for _, row in g_shuf.iterrows():
            surface = str(row["出現形"]).strip()
            if not surface:
                continue

            messages = build_messages_for_doctor(standard_name, surface)
            ex = {
                "messages": messages,
                "metadata": {
                    "source": "MANBYO",
                    "standard_name": str(standard_name),
                    "surface": surface,
                    "surface_reading": str(row.get("出現形よみ", "")),
                    "icd_code": str(row.get("ICDコード", "")),
                    "trust": str(row.get("信頼度レベル", "")),
                    "freq": str(row.get("頻度レベル", "")),
                },
            }
            examples.append(ex)

    return examples


def save_jsonl(examples: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"=== saved SFT JSONL: {out_path} (n={len(examples)}) ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manbyo-xlsx",
        type=str,
        required=True,
        help="MANBYO_20210602.xlsx のパス",
    )
    ap.add_argument(
        "--output",
        type=str,
        required=True,
        help="SFT JSONL の出力パス",
    )
    ap.add_argument(
        "--trust-levels",
        type=str,
        default="S,A",
        help="使用する信頼度レベル（カンマ区切り）例: 'S,A' / 'S,A,B' / '' (空文字で無視)",
    )
    ap.add_argument(
        "--min-freq-rank",
        type=int,
        default=3,
        help="頻度レベルの最小ランク (0〜5)。3 ならおおよそ '50-80%' 以上を残すイメージ。",
    )
    ap.add_argument(
        "--icd-prefixes",
        type=str,
        default="",
        help="ICDコードのプレフィックス（カンマ区切り）。例: 'R' なら症状・徴候。空ならフィルタしない。",
    )
    ap.add_argument(
        "--max-examples-per-name",
        type=int,
        default=20,
        help="標準病名ごとの最大サンプル数 (0 以下なら無制限)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = ap.parse_args()

    manbyo_path = Path(args.manbyo_xlsx)
    out_path = Path(args.output)

    df = load_manbyo(manbyo_path)

    # trust_levels の解釈
    if args.trust_levels.strip() == "":
        trust_levels = None
    else:
        trust_levels = [t.strip() for t in args.trust_levels.split(",") if t.strip()]

    # icd_prefixes の解釈
    if args.icd_prefixes.strip() == "":
        icd_prefixes = None
    else:
        icd_prefixes = [p.strip() for p in args.icd_prefixes.split(",") if p.strip()]

    df_f = filter_manbyo(
        df,
        trust_levels=trust_levels,
        min_freq_rank=args.min_freq_rank,
        icd_prefixes=icd_prefixes,
    )

    examples = build_sft_examples(
        df_f,
        max_examples_per_name=args.max_examples_per_name,
        seed=args.seed,
    )
    save_jsonl(examples, out_path)


if __name__ == "__main__":
    main()
