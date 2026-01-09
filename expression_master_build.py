#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
expression_master_build.py

患者表現 / 医師表現の生成・精査済み JSONL をマージし，
HPO ごとの表現マスタ JSONL を作るスクリプト。

想定する入力:
  - HPO マスタ:
      data/hpo_master_all_jp_with_self_report.csv
  - 患者表現 (PatternB, Judge+Refine 後):
      data/HPO_symptom_patient_expression_judge_refine.eques_lora.jsonl
  - 医師表現 (PatternB, Judge+Refine 後):
      data/HPO_doctor_expression_judge_refine.jsonl

出力:
  - data/HPO_expression_master.patient_doctor.jsonl

各行の例:
  {
    "HPO_ID": "HP:0001873",
    "HPO_name_ja": "貧血",
    "HPO_name_en": "Anemia",
    "category": "phenotype",
    "category_v2": "symptom",
    "sub_category": "symptom_candidate",
    "is_leaf": 1,
    "self_reportable": 1,
    "patient_expressions": ["めまいがする", "立ちくらみがする", ...],
    "doctor_expressions": ["貧血あり", "Hb低値", ...]
  }
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any


@dataclass
class HpoExpressionMaster:
    hpo_id: str
    name_ja: str = ""
    name_en: str = ""
    category: str = ""
    category_v2: str = ""
    sub_category: str = ""
    is_leaf: int = 0
    self_reportable: int = 0
    patient_expressions: List[str] = field(default_factory=list)
    doctor_expressions: List[str] = field(default_factory=list)


def load_hpo_master(path: Path) -> Dict[str, Dict[str, Any]]:
    """hpo_master_all_jp_with_self_report.csv を読み込んで dict にする。"""
    hpo_info: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hpo_id = (row.get("HPO_ID") or "").strip()
            if not hpo_id:
                continue
            hpo_info[hpo_id] = row
    return hpo_info


def _normalize_expr(text: str) -> str:
    """簡単な前処理 + 前後空白除去。"""
    if text is None:
        return ""
    t = str(text).strip()
    # ここで簡単な正規化（連続スペースの縮約など）をするなら足す
    return t


def build_expression_master(
    hpo_master_csv: Path,
    patient_jsonl: Path,
    doctor_jsonl: Path,
) -> Dict[str, HpoExpressionMaster]:
    """患者表現 + 医師表現をマージして HPO ごとの表現マスタを構築する。"""

    hpo_info = load_hpo_master(hpo_master_csv)
    masters: Dict[str, HpoExpressionMaster] = {}

    def get_master(
        hpo_id: str,
        fallback_ja: str = "",
        fallback_en: str = "",
        fallback_cat: str = "",
    ) -> HpoExpressionMaster:
        if hpo_id in masters:
            return masters[hpo_id]

        info = hpo_info.get(hpo_id, {})
        name_ja = (info.get("jp_final") or fallback_ja or "").strip()
        name_en = (info.get("name_en") or fallback_en or "").strip()
        category = (info.get("category") or fallback_cat or "").strip()
        category_v2 = (info.get("category_v2") or "").strip()
        sub_category = (info.get("sub_category") or "").strip()
        is_leaf = int(info.get("is_leaf") or 0) if "is_leaf" in info else 0
        self_reportable = int(info.get("self_reportable") or 0) if "self_reportable" in info else 0

        m = HpoExpressionMaster(
            hpo_id=hpo_id,
            name_ja=name_ja,
            name_en=name_en,
            category=category,
            category_v2=category_v2,
            sub_category=sub_category,
            is_leaf=is_leaf,
            self_reportable=self_reportable,
        )
        masters[hpo_id] = m
        return m

    # 患者表現を読み込み
    if patient_jsonl.is_file():
        with patient_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                hpo_id = (rec.get("HPO_ID") or "").strip()
                if not hpo_id:
                    continue

                expr = (
                    rec.get("patient_expression_final")
                    or rec.get("patient_expression_original")
                    or rec.get("patient_expression")
                )
                expr = _normalize_expr(expr)
                if not expr:
                    continue

                m = get_master(
                    hpo_id,
                    fallback_ja=(rec.get("HPO_name_ja") or ""),
                    fallback_en=(rec.get("HPO_name_en") or ""),
                    fallback_cat=(rec.get("category") or ""),
                )
                if expr not in m.patient_expressions:
                    m.patient_expressions.append(expr)

    # 医師表現を読み込み
    if doctor_jsonl.is_file():
        with doctor_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                hpo_id = (rec.get("HPO_ID") or "").strip()
                if not hpo_id:
                    continue

                expr = rec.get("expression") or rec.get("doctor_expression")
                expr = _normalize_expr(expr)
                if not expr:
                    continue

                m = get_master(
                    hpo_id,
                    fallback_ja=(rec.get("HPO_name_ja") or ""),
                    fallback_en=(rec.get("HPO_name_en") or ""),
                    fallback_cat=(rec.get("category") or ""),
                )
                if expr not in m.doctor_expressions:
                    m.doctor_expressions.append(expr)

    return masters


def save_expression_master(
    masters: Dict[str, HpoExpressionMaster],
    out_path: Path,
) -> None:
    """HPO ごとの表現マスタを JSONL で保存する。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for hpo_id in sorted(masters.keys()):
            m = masters[hpo_id]
            # どちらの表現も 0 件ならスキップ
            if not m.patient_expressions and not m.doctor_expressions:
                continue
            rec = {
                "HPO_ID": m.hpo_id,
                "HPO_name_ja": m.name_ja,
                "HPO_name_en": m.name_en,
                "category": m.category,
                "category_v2": m.category_v2,
                "sub_category": m.sub_category,
                "is_leaf": m.is_leaf,
                "self_reportable": m.self_reportable,
                "patient_expressions": m.patient_expressions,
                "doctor_expressions": m.doctor_expressions,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hpo-master",
        type=str,
        default="../data/hpo_master_all_jp_with_self_report.csv",
        help="HPO マスタ CSV のパス",
    )
    parser.add_argument(
        "--patient-jsonl",
        type=str,
        default="../data/HPO_symptom_patient_expression_judge_refine.eques_lora.jsonl",
        help="患者表現 JSONL（Judge+Refine 後）",
    )
    parser.add_argument(
        "--doctor-jsonl",
        type=str,
        default="../data/HPO_doctor_expression_judge_refine.jsonl",
        help="医師表現 JSONL（Judge+Refine 後）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../data/HPO_expression_master.patient_doctor.jsonl",
        help="出力先 JSONL パス",
    )
    args = parser.parse_args()

    hpo_master_csv = Path(args.hpo_master)
    patient_jsonl = Path(args.patient_jsonl)
    doctor_jsonl = Path(args.doctor_jsonl)
    out_path = Path(args.output)

    print(f"=== Load HPO master from {hpo_master_csv} ===")
    print(f"=== Load patient expressions from {patient_jsonl} ===")
    print(f"=== Load doctor expressions from {doctor_jsonl} ===")

    masters = build_expression_master(
        hpo_master_csv=hpo_master_csv,
        patient_jsonl=patient_jsonl,
        doctor_jsonl=doctor_jsonl,
    )

    num_hpo = len(masters)
    num_pat = sum(len(m.patient_expressions) for m in masters.values())
    num_doc = sum(len(m.doctor_expressions) for m in masters.values())

    print(
        f"=== Built expression master: HPO={num_hpo}, "
        f"patient_expr={num_pat}, doctor_expr={num_doc} ==="
    )

    print(f"=== Save expression master to {out_path} ===")
    save_expression_master(masters, out_path)
    print("=== DONE ===")


if __name__ == "__main__":
    main()
