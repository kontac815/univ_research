#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NAIST大規模患者表現辞書 & MANBYO_20210602.xlsx から、
HPO ラベル + (あれば) 定義 → 患者/医師表現生成用の SFT データを作成する。

ポイント:
- 学習に使うのは「標準病名 != 出現形」の行すべて。
  → 標準病名と完全一致する表現はここで除去する。
- HPO_depth_ge3.csv の jp_final と標準病名が一致するものは HPO ID を紐づけ。
  さらに definition_ja があるときだけ「定義付きプロンプト」にする。
- HPO にマッチしない or definition_ja がないものは、
  「症状名: 標準病名」だけを渡すプロンプトにする。
- MANBYO は信頼度レベル S/A/B の行だけ採用。

出力:
- naist_patient_hpo_sft.jsonl
- manbyo_doctor_hpo_sft.jsonl

各行は chat 形式:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user",   "content": "..."}
  ],
  "output": "ターゲット表現",
  "task": "patient_expression" or "doctor_expression",
  "hpo_id": "HP:0000001" or null,
  "has_definition": true/false,
  "standard_name": "...",
  "source": "NAIST" or "MANBYO"
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import re


# =========================
# 共通: 正規化 & HPO ロード
# =========================

def normalize_for_compare(text: str) -> str:
    """
    標準病名と出現形の「同一判定」用の簡易正規化。
    - 前後空白除去
    - 連続空白(全角/半角) → 1つの半角スペース
    - 全角スペース除去
    """
    if text is None:
        return ""
    s = str(text)
    s = s.strip()
    # 全角スペース → 半角
    s = s.replace("\u3000", " ")
    # 複数スペースを1つに
    s = re.sub(r"\s+", " ", s)
    return s


def load_hpo_master(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = ["HPO_ID", "jp_final", "definition_ja"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"HPO master is missing columns: {missing}")

    df = df.dropna(subset=["jp_final"])
    df["jp_final_norm"] = df["jp_final"].astype(str).str.strip()
    return df


def build_hpo_lookup(df_hpo: pd.DataFrame) -> Dict[str, Dict[str, Optional[str]]]:
    """
    jp_final_norm をキーにして HPO 情報を引ける dict にする。
    """
    lookup: Dict[str, Dict[str, Optional[str]]] = {}
    for _, row in df_hpo.iterrows():
        key = row["jp_final_norm"]
        definition = row.get("definition_ja", None)
        if isinstance(definition, float) and pd.isna(definition):
            definition = None
        else:
            definition = str(definition).strip() if definition is not None else None
        lookup[key] = {
            "HPO_ID": row["HPO_ID"],
            "name_ja": row["jp_final_norm"],
            "definition_ja": definition,
        }
    return lookup


# =========================
# vLLM と揃えたプロンプト関数
# =========================

def build_generate_messages_patient(symptom_name: str,
                                    definition_ja: Optional[str]) -> List[Dict[str, str]]:
    """
    患者表現生成用の messages を構築。
    definition_ja が None/空なら「定義:」ブロックなし。
    """
    definition_ja = (definition_ja or "").strip()

    system_msg = (
        "あなたは日本語の医療面接に精通したAIアシスタントです。"
        "以下に与えるHPOの情報（症状名と、あればその定義）にもとづき、"
        "患者が診察室で医師に訴えそうな日本語の発言を1つだけ生成してください。"
        "制約:"
        " - 出力は患者の発言そのもののみとし、説明文・コメント・番号付けは一切行わないこと。"
        " - できるだけ短く簡潔にし、1文またはそれより短い表現にすること。"
        " - 「〜です／〜ます」よりも自然な話し言葉（〜て、〜なんです、など）を優先すること。"
        " - 疾患名や専門用語はなるべく避け、日常的な日本語で症状を表現すること。"
        " - 与えられた症状名の語をそのまま繰り返さず、患者が使いそうな言い換えを用いること。"
        " - 不必要に複数の症状を詰め込まず、中心となる症状を1つに絞ること。"
    )

    if definition_ja:
        user_msg = (
            f"症状名: {symptom_name}\n\n"
            f"定義:\n{definition_ja}\n\n"
            "患者が医師に話すときの日本語の一言だけを出力してください。"
        )
    else:
        user_msg = (
            f"症状名: {symptom_name}\n\n"
            "患者が医師に話すときの日本語の一言だけを出力してください。"
        )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def build_generate_messages_doctor(symptom_name: str,
                                   definition_ja: Optional[str]) -> List[Dict[str, str]]:
    """
    医師表現生成用の messages。
    """
    definition_ja = (definition_ja or "").strip()

    system_msg = (
        "あなたは日本語の医療用語と電子カルテ記載に精通した日本人医師です。"
        "以下に与えるHPOの情報（症状名と、あればその定義）にもとづき、"
        "医師が電子カルテに記載しそうな日本語の表現を1つだけ生成してください。"
        "制約:"
        " - 出力はカルテに記載する1つの表現のみとし、説明文・コメント・箇条書き・番号付けは一切行わないこと。"
        " - できるだけ簡潔に、通常のカルテ記載を意識した短い語句またはごく短い文にすること。"
        " - 患者の話し言葉ではなく、医師が用いる標準的な専門用語・略語を使用してよい。"
        " - ただし意味が過度に抽象的にならないようにし、臨床的な情報が伝わる表現にすること。"
        " - 与えられた症状名の語をそのまま繰り返すのではなく、診療録として自然な形に整えること。"
        " - 不必要に多くの情報を詰め込まず、中心となる症状・所見・病態にフォーカスすること。"
    )

    if definition_ja:
        user_msg = (
            f"症状名: {symptom_name}\n\n"
            f"定義:\n{definition_ja}\n\n"
            "医師が電子カルテに記載する1つの表現だけを、日本語で出力してください。"
        )
    else:
        user_msg = (
            f"症状名: {symptom_name}\n\n"
            "医師が電子カルテに記載する1つの表現だけを、日本語で出力してください。"
        )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# =========================
# NAIST: 患者表現 SFT
# =========================

def build_patient_sft_records(
    naist_path: Path,
    hpo_lookup: Dict[str, Dict[str, Optional[str]]],
) -> List[Dict[str, Any]]:
    """
    NAIST大規模患者表現辞書.xlsx から SFT レコードを作成。

    - 標準病名と出現形（患者表現）が「同一」とみなせる行は除去。
    - HPO ラベル一致 & definition_ja あり → 定義付きプロンプト。
    - それ以外 → 症状名 = 標準病名（か HPOラベル）で定義なしプロンプト。
    """
    df = pd.read_excel(naist_path)

    # カラム名は 1a.* を前提に
    # 例: "標準病名", "出現形（患者表現）", "ICD-10" など
    if "標準病名" not in df.columns:
        raise ValueError("NAIST: '標準病名' カラムが見つかりません。")
    if "出現形（患者表現）" not in df.columns:
        raise ValueError("NAIST: '出現形（患者表現）' カラムが見つかりません。")

    df["標準病名_str"] = df["標準病名"].astype(str)
    df["患者表現_str"] = df["出現形（患者表現）"].astype(str)

    # 同一判定用ノーマライズ
    df["標準病名_norm"] = df["標準病名_str"].map(normalize_for_compare)
    df["患者表現_norm"] = df["患者表現_str"].map(normalize_for_compare)

    # 標準病名 == 患者表現 の行を除去
    before = len(df)
    df = df[df["標準病名_norm"] != df["患者表現_norm"]].copy()
    after = len(df)
    print(f"[NAIST] 標準病名=患者表現 の行を除去: {before - after} / {before}")

    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        standard_name = row["標準病名_norm"]
        patient_expr = row["患者表現_str"].strip()
        if not standard_name or not patient_expr:
            continue

        # HPO マッチ & 定義
        hpo_info = hpo_lookup.get(standard_name)
        if hpo_info is not None and hpo_info.get("definition_ja"):
            symptom_name = hpo_info["name_ja"]
            definition = hpo_info["definition_ja"]
            hpo_id = hpo_info["HPO_ID"]
            has_definition = True
        else:
            # HPO にマッチしない or 定義なし
            symptom_name = standard_name
            definition = None
            hpo_id = hpo_info["HPO_ID"] if hpo_info is not None else None
            has_definition = False

        messages = build_generate_messages_patient(
            symptom_name=symptom_name,
            definition_ja=definition,
        )

        rec = {
            "messages": messages,
            "output": patient_expr,
            "task": "patient_expression",
            "hpo_id": hpo_id,
            "has_definition": has_definition,
            "standard_name": standard_name,
            "source": "NAIST",
        }
        records.append(rec)

    print(f"[NAIST] SFT records: {len(records)}")
    return records


# =========================
# MANBYO: 医師表現 SFT
# =========================

def build_doctor_sft_records(
    manbyo_path: Path,
    hpo_lookup: Dict[str, Dict[str, Optional[str]]],
    use_trust_levels: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    MANBYO_20210602.xlsx から医師表現 SFT レコード。

    - 信頼度レベル ∈ use_trust_levels (例: S/A/B) の行のみ。
    - 標準病名 == 出現形 の行は除去。
    - HPO ラベル一致 & 定義あり → 定義付きプロンプト。
    """
    df = pd.read_excel(manbyo_path)

    required_cols = ["標準病名", "出現形", "信頼度レベル"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"MANBYO: 必須カラムが欠損しています: {missing}")

    df["標準病名_str"] = df["標準病名"].astype(str)
    df["出現形_str"] = df["出現形"].astype(str)
    df["信頼度レベル"] = df["信頼度レベル"].astype(str).str.strip()

    # 信頼度レベルフィルタ
    if use_trust_levels:
        df = df[df["信頼度レベル"].isin(use_trust_levels)].copy()
        print(f"[MANBYO] 信頼度 {use_trust_levels} の行のみ利用: {len(df)} 行")

    # 同一判定用ノーマライズ
    df["標準病名_norm"] = df["標準病名_str"].map(normalize_for_compare)
    df["出現形_norm"] = df["出現形_str"].map(normalize_for_compare)

    before = len(df)
    df = df[df["標準病名_norm"] != df["出現形_norm"]].copy()
    after = len(df)
    print(f"[MANBYO] 標準病名=出現形 の行を除去: {before - after} / {before}")

    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        standard_name = row["標準病名_norm"]
        surface = row["出現形_str"].strip()
        if not standard_name or not surface:
            continue

        # HPO マッチ & 定義
        hpo_info = hpo_lookup.get(standard_name)
        if hpo_info is not None and hpo_info.get("definition_ja"):
            symptom_name = hpo_info["name_ja"]
            definition = hpo_info["definition_ja"]
            hpo_id = hpo_info["HPO_ID"]
            has_definition = True
        else:
            symptom_name = standard_name
            definition = None
            hpo_id = hpo_info["HPO_ID"] if hpo_info is not None else None
            has_definition = False

        messages = build_generate_messages_doctor(
            symptom_name=symptom_name,
            definition_ja=definition,
        )

        rec = {
            "messages": messages,
            "output": surface,
            "task": "doctor_expression",
            "hpo_id": hpo_id,
            "has_definition": has_definition,
            "standard_name": standard_name,
            "source": "MANBYO",
        }
        records.append(rec)

    print(f"[MANBYO] SFT records: {len(records)}")
    return records


# =========================
# JSONL 保存 & main
# =========================

def save_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpo-csv", type=Path, default=Path("../data/HPO_depth_ge3.csv"))
    parser.add_argument("--naist-xlsx", type=Path, default=Path("../data/NAIST大規模患者表現辞書.xlsx"))
    parser.add_argument("--manbyo-xlsx", type=Path, default=Path("../data/MANBYO_20210602.xlsx"))
    parser.add_argument("--out-patient-jsonl", type=Path, default=Path("../data/naist_patient_hpo_sft.jsonl"))
    parser.add_argument("--out-doctor-jsonl", type=Path, default=Path("../data/manbyo_doctor_hpo_sft.jsonl"))
    parser.add_argument(
        "--manbyo-trust-levels",
        type=str,
        default="S,A,B",
        help="MANBYOで利用する信頼度レベル (カンマ区切り, 例: S,A,B)",
    )
    args = parser.parse_args()

    print(f"== load HPO master: {args.hpo_csv}")
    df_hpo = load_hpo_master(args.hpo_csv)
    hpo_lookup = build_hpo_lookup(df_hpo)
    print(f"  HPO lookup entries: {len(hpo_lookup)}")

    print(f"== build NAIST patient SFT: {args.naist_xlsx}")
    patient_records = build_patient_sft_records(args.naist_xlsx, hpo_lookup)
    save_jsonl(patient_records, args.out_patient_jsonl)
    print(f"  saved: {args.out_patient_jsonl.resolve()}")

    trust_levels = [s.strip() for s in args.manbyo_trust_levels.split(",") if s.strip()]
    print(f"== build MANBYO doctor SFT: {args.manbyo_xlsx}")
    doctor_records = build_doctor_sft_records(args.manbyo_xlsx, hpo_lookup, use_trust_levels=trust_levels)
    save_jsonl(doctor_records, args.out_doctor_jsonl)
    print(f"  saved: {args.out_doctor_jsonl.resolve()}")


if __name__ == "__main__":
    main()
