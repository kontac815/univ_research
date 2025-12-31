#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NAIST大規模患者表現辞書.xlsx から
「1標準病名 → 患者表現1つ」の SFT データを作るスクリプト。

出力: naist_patient_expression_sft.single.jsonl
  各行:
    {
      "instruction": "...標準病名: ○○",
      "input": "",
      "output": "患者さんが言いそうな表現1つ"
    }
"""

from __future__ import annotations
import json
import unicodedata
import re
from collections import Counter
from pathlib import Path

import pandas as pd

# ==============
# 設定
# ==============
BASE_DIR = Path("/home/kiso_user/Documents/workspace/Research/data")
NAIST_PATH = BASE_DIR / "NAIST大規模患者表現辞書.xlsx"
OUT_JSONL = BASE_DIR / "naist_patient_expression_sft_single.jsonl"

# 1つの標準病名から使う最大サンプル数
MAX_PER_DISEASE = 50

# 必須カラム
COL_PATIENT = "出現形（患者表現）"
COL_DISEASE = "標準病名"
MEDICAL_SUFFIX_RE = re.compile(r"(症|炎|障害|癌|腫瘍|腫瘤|病)$")

# ★ 追加：明らかに医療者寄りに寄りがちな語
MEDICAL_KEYWORDS = [
    "ヘルニア", "症候群", "再発", "手術", "術", "切除",
    "腫瘍", "腫瘤", "癌", "がん", "悪性", "良性",
    "既往", "合併症", "検査", "治療"
]


# ==============
# ユーティリティ
# ==============
def normalize_text(s: str) -> str:
    """全角半角を揃えて前後空白を削除"""
    if not isinstance(s, str):
        return ""
    return unicodedata.normalize("NFKC", s).strip()


MEDICAL_SUFFIX_RE = re.compile(r"(症|炎|障害|癌|腫瘍|腫瘤|病)$")


def is_medical_like_short(expr: str) -> bool:
    """
    「花粉症」「舌炎」「精神障害」みたいな
    短い医学用語っぽい表現をざっくり弾く。
    """
    expr = expr.strip()
    if len(expr) == 0:
        return True
    if len(expr) <= 6 and MEDICAL_SUFFIX_RE.search(expr):
        return True
    return False


# ==============
# NAIST 読み込み
# ==============
def load_naist_df(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)

    # 必須列チェック
    for col in (COL_PATIENT, COL_DISEASE):
        if col not in df.columns:
            raise ValueError(f"必要な列が見つかりません: {col}")

    df = df.dropna(subset=[COL_PATIENT, COL_DISEASE]).copy()

    df["patient_norm"] = df[COL_PATIENT].map(normalize_text)
    df["disease_norm"] = df[COL_DISEASE].map(normalize_text)

    df = df[
        (df["patient_norm"] != "") &
        (df["disease_norm"] != "")
    ].copy()

    return df


# ==============
# SFT レコード構築
# ==============
def build_sft_records(df: pd.DataFrame):
    """
    1行 = 1訓練サンプル（標準病名→患者表現1つ）のレコードを作成。
    """
    records = []
    seen_pairs = set()
    counts_per_disease = Counter()

    for _, row in df.iterrows():
        disease = row["disease_norm"]
        expr = row["patient_norm"]

        if not disease or not expr:
            continue

        # disease と完全一致する表現は除外
        if disease == expr:
            continue

        # ★ disease をそのまま（あるいはほぼそのまま）含む表現は除外
        #    例: disease="鼠径ヘルニア" → "左鼠径ヘルニア再発" など
        if disease in expr:
            continue

        # 短すぎる
        if len(expr) < 2:
            continue

        # ★ 長すぎる（文章っぽい）のも除外
        if len(expr) > 20:
            continue

        # 短い医学用語っぽい表現は除外（患者口語に寄せたい）
        if is_medical_like_short(expr):
            continue

        # ★ 明らかに医療者寄りの単語を含むものを除外
        if any(kw in expr for kw in MEDICAL_KEYWORDS):
            continue

        # ★ 敬体は避けたいなら（SFTからも消す）
        if expr.endswith("です") or expr.endswith("ます"):
            continue


        instruction = (
            "あなたは日本語の医療用語に詳しいAIアシスタントです。\n"
            "次の医学的な症状名について、患者さんが実際に言いそうな表現を"
            "日本語で1つだけ生成してください。\n"
            "できるだけ短く端的に、文章ではなく語句やごく短い表現にしてください。"
            "（例:「胸が苦しい」「頭がズキズキする」など）\n"
            "専門用語や診断名は避け、日常的な日本語に言い換えてください。\n"
            "出力は患者の発言のみを書き、説明や番号付けはしないでください。\n"
            "敬体（〜です／〜ます）はなるべく使わないでください。\n\n"
            "症状名や病名の語をそのまま含めないでください。\n"
            f"症状名: {disease}"
        )

        rec = {
            "instruction": instruction,
            "input": "",
            "output": expr,
        }
        records.append(rec)

    return records, counts_per_disease


def save_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    print(f"読み込み: {NAIST_PATH}")
    df = load_naist_df(NAIST_PATH)
    print(f"元データ行数: {len(df)}")

    records, cnt = build_sft_records(df)
    print(f"SFT レコード数: {len(records)}")
    print(f"標準病名の件数: {len(cnt)}")
    if cnt:
        vs = sorted(cnt.values())
        print(
            f"1標準病名あたりサンプル数: "
            f"min={vs[0]}, median={vs[len(vs)//2]}, max={vs[-1]}"
        )

    save_jsonl(records, OUT_JSONL)
    print(f"保存しました: {OUT_JSONL.resolve()}")


if __name__ == "__main__":
    main()
