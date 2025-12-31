#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
5a.postprocess_expressions_sbert.py (With Deletion Log)

HPO 患者/医師表現の「最終 JSONL」に対してフィルタリングを行い、
クリーンな辞書と、削除されたレコードのログを出力する。

処理フロー:
  1) HPO日本語ラベルと同一の表現を除外
  2) 異なるHPOにまたがる同一表現の重複解決 (Depth優先)
  3) Sentence-BERT による意味的フィルタリング
  4) 公式HPOラベルの注入
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from tqdm import tqdm

# S-BERT / Transformers
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


@dataclass
class ExprRecord:
    hpo_id: str
    hpo_label: str
    mode: str
    expr: str
    score: float = 0.0
    
    # 削除情報記録用
    removal_reason: str = ""
    removal_details: Dict[str, Any] = field(default_factory=dict)
    
    # 内部処理用
    label_norm: str = field(init=False)
    expr_norm: str = field(init=False)

    def __post_init__(self):
        self.label_norm = normalize_text(self.hpo_label)
        self.expr_norm = normalize_text(self.expr)


def normalize_text(text: str) -> str:
    """簡易正規化"""
    if not text:
        return ""
    import unicodedata
    # NFKC正規化 + 小文字化 + 空白削除
    t = unicodedata.normalize('NFKC', text)
    t = t.lower().strip()
    t = t.replace(" ", "").replace("　", "")
    return t


def load_hpo_metadata(csv_path: Path) -> Dict[str, Dict]:
    metadata = {}
    if not csv_path.exists():
        print(f"[WARN] HPO master not found: {csv_path}")
        return metadata

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hid = row.get("HPO_ID")
            label = row.get("jp_final") or row.get("jp_label") or ""
            depth_str = row.get("depth", "999")
            
            if hid:
                try:
                    depth = int(depth_str)
                except ValueError:
                    depth = 999
                
                metadata[hid] = {
                    "label": label,
                    "depth": depth
                }
    return metadata


def clean_expression(expr: str, mode: str) -> str:
    """
    Normalize a generated expression similarly to vllm_pipeline_2 postprocess:
      - keep first line only
      - drop anything after/ending with the token 'assistant'
      - trim simple role markers and surrounding quotes
    """
    if not expr:
        return ""

    s = str(expr).strip()
    s = s.splitlines()[0].strip()
    s = re.split(r"assistant[:：]?", s, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    s = re.sub(r"\s*assistant[:：]?\s*$", "", s, flags=re.IGNORECASE)

    if mode == "patient":
        for prefix in ["患者:", "患者「", "患者｢", "「", "『"]:
            if s.startswith(prefix):
                s = s[len(prefix):].lstrip()
        for suf in ["」", "『", "』", "｡"]:
            if s.endswith(suf):
                s = s[:-1].rstrip()
    else:
        for prefix in ["医師所見:", "医師記載:", "所見:", "症状:", "記載:", "カルテ:", "医師:"]:
            if s.startswith(prefix):
                s = s[len(prefix):].lstrip()
        for prefix in ["「", "『", "\"", "“", "”"]:
            if s.startswith(prefix):
                s = s[len(prefix):].lstrip()
        for suf in ["」", "『", "』", "｡"]:
            if s.endswith(suf):
                s = s[:-1].rstrip()

    return s


def detect_mode(data: Dict[str, Any], default_mode: str) -> str:
    """
    Decide per-record mode when --mode=mixed.
    """
    src = str(data.get("mode") or data.get("source") or "").lower()
    if "doctor" in src:
        return "doctor"
    if "patient" in src:
        return "patient"
    return default_mode


def load_records(jsonl_path: Path, hpo_metadata: Dict[str, Dict], mode: str) -> List[ExprRecord]:
    """JSONLからレコードを読み込み、ExprRecordオブジェクトのリストにする"""
    records = []
    if not jsonl_path.exists():
        print(f"[WARN] Input JSONL not found: {jsonl_path}")
        return records

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # --- キーのゆらぎ吸収処理 (修正部分) ---

            # 1. HPO ID
            # パターン1: "HPO_ID", パターン2: "hpo_id"
            hid = data.get("HPO_ID") or data.get("hpo_id")

            # 2. 表現テキスト (Expression)
            # 優先順: patient_expression_final > expression > final_text > text
            expr = (
                data.get("patient_expression_final") or 
                data.get("doctor_expression_final") or 
                data.get("expr") or 
                data.get("final_text") or 
                data.get("refined_text") or 
                data.get("text") or
                data.get("patient_expression_original") or
                data.get("doctor_expression_original")
            )
            record_mode = detect_mode(data, mode) if mode == "mixed" else mode
            expr = clean_expression(expr, record_mode)
            
            # 3. スコア (Overall Score)
            score_val = 0.0
            
            # パターン1: フラットな "judge_overall"
            if "judge_overall" in data:
                score_val = float(data["judge_overall"])
            
            # パターン2: ネストされた "judge": {"overall": ...}
            elif "judge" in data and isinstance(data["judge"], dict):
                score_val = float(data["judge"].get("overall", 0))
            
            # 既存パターン: "judge_result": {"overall_score": ...}
            elif "judge_result" in data and isinstance(data["judge_result"], dict):
                score_val = float(data["judge_result"].get("overall_score", 0))
                
            # フォールバック: 単純な "score"
            elif "score" in data:
                score_val = float(data["score"])

            # --------------------------------------

            if not hid or not expr:
                continue

            # マスタからラベル情報を補完
            label = ""
            if hid in hpo_metadata:
                label = hpo_metadata[hid]["label"]
            
            rec = ExprRecord(
                hpo_id=hid,
                hpo_label=label,
                mode=record_mode,
                expr=expr,
                score=score_val
            )
            records.append(rec)
    
    return records


def drop_same_as_label(records: List[ExprRecord]) -> Tuple[List[ExprRecord], List[ExprRecord]]:
    """Step 1: ラベルと完全一致を除外"""
    kept = []
    removed = []
    
    for r in records:
        if r.expr_norm == r.label_norm:
            r.removal_reason = "same_as_label"
            r.removal_details = {"label": r.hpo_label}
            removed.append(r)
        else:
            kept.append(r)
            
    print(f"[Filter] drop_same_as_label: Removed {len(removed)} records.")
    return kept, removed


# 置き換え推奨: drop_cross_hpo_duplicates

def drop_cross_hpo_duplicates(records: List[ExprRecord], hpo_metadata: Dict[str, Dict[str, Any]],
                              policy: str = "drop_all") -> Tuple[List[ExprRecord], List[ExprRecord]]:
    """
    policy:
      - drop_all: 複数HPOに跨るexpr_normは全部落とす（patient推奨）
      - keep_shallow: 一番浅いHPOに寄せる（旧挙動）
      - keep_deep: 一番深いHPOに寄せる
      - keep_best_score: score最大のHPOを残す
    """
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in records:
        grouped[r.expr_norm].append(r)

    kept, removed = [], []
    for expr_norm, rs in grouped.items():
        if len(rs) == 1:
            kept.append(rs[0])
            continue
        # 同一 expr_norm が複数行あっても、HPO が1つだけなら cross-HPO 重複ではない。
        # merged(patient+doctor) で同一HPOに同一表現が2回出るケースを誤って全削除しないため。
        if len({r.hpo_id for r in rs}) == 1:
            kept.extend(rs)
            continue

        if policy == "drop_all":
            for r in rs:
                r.removal_reason = "cross_hpo_duplicate"
                removed.append(r)
            continue

        def depth_of(r):
            return float(hpo_metadata.get(r.hpo_id, {}).get("depth", 1e9))

        if policy == "keep_shallow":
            rs_sorted = sorted(rs, key=depth_of)
            winner = rs_sorted[0]
        elif policy == "keep_deep":
            rs_sorted = sorted(rs, key=depth_of, reverse=True)
            winner = rs_sorted[0]
        elif policy == "keep_best_score":
            winner = max(rs, key=lambda x: float(x.score or 0.0))
        else:
            raise ValueError(f"unknown policy: {policy}")

        kept.append(winner)
        for r in rs:
            if r is winner:
                continue
            r.removal_reason = f"cross_hpo_duplicate_{policy}"
            removed.append(r)

    return kept, removed



def run_sbert_filter(
    records: List[ExprRecord], 
    hpo_metadata: Dict[str, Dict],
    args
) -> Tuple[List[ExprRecord], List[ExprRecord]]:
    """Step 3: S-BERTフィルタ"""
    self_min_sim: float = getattr(args, "self_min_sim", 0.0)
    other_margin: float = getattr(args, "other_margin", 0.0)
    ambig_gap: float = getattr(args, "ambig_gap", 0.0)

    if not HAS_SBERT:
        print("[ERROR] sentence-transformers not installed. Skip S-BERT filter.")
        return records, []

    print(f"[S-BERT] Loading model: {args.sbert_model} ...")
    model = SentenceTransformer(args.sbert_model, device=args.device)

    # 1. 全HPOラベル埋め込み
    unique_hids = list(hpo_metadata.keys())
    hid_to_idx = {hid: i for i, hid in enumerate(unique_hids)}
    
    print("[S-BERT] Encoding all HPO labels...")
    all_labels = [hpo_metadata[hid]["label"] for hid in unique_hids]
    label_embeddings = model.encode(all_labels, convert_to_tensor=True, normalize_embeddings=True)
    
    # 2. 表現埋め込み
    print(f"[S-BERT] Encoding {len(records)} expressions...")
    expr_texts = [r.expr for r in records]
    expr_embeddings = model.encode(expr_texts, batch_size=128, convert_to_tensor=True, normalize_embeddings=True)
    
    kept = []
    removed = []

    # 3. 計算と判定
    target_indices = []
    valid_mask = []
    for r in records:
        if r.hpo_id in hid_to_idx:
            target_indices.append(hid_to_idx[r.hpo_id])
            valid_mask.append(True)
        else:
            target_indices.append(0)
            valid_mask.append(False)
            
    target_indices = torch.tensor(target_indices, device=args.device)
    
    print("[S-BERT] Calculating similarity matrix...")
    chunk_size = 1000
    total = len(records)
    
    for start_idx in tqdm(range(0, total, chunk_size)):
        end_idx = min(start_idx + chunk_size, total)
        
        batch_expr_emb = expr_embeddings[start_idx:end_idx]
        batch_recs = records[start_idx:end_idx]
        batch_valid = valid_mask[start_idx:end_idx]
        batch_target_idx = target_indices[start_idx:end_idx]
        
        sim_matrix = util.cos_sim(batch_expr_emb, label_embeddings)
        
        own_sims = sim_matrix[torch.arange(len(batch_expr_emb)), batch_target_idx]
        
        # 自分をマスクして他人との最大を探す
        sim_matrix.scatter_(1, batch_target_idx.unsqueeze(1), -1.0)
        max_other_sims, max_other_indices = sim_matrix.max(dim=1)
        
        for i, rec in enumerate(batch_recs):
            if not batch_valid[i]:
                kept.append(rec)
                continue
                
            s_own = float(own_sims[i])
            s_other = float(max_other_sims[i])
            best_other_idx = int(max_other_indices[i])
            best_other_hid = unique_hids[best_other_idx]
            best_other_label = all_labels[best_other_idx]
            
            # 判定ロジック
            reason = None
            
            # run_sbert_filter のループ内（reason 判定）を置き換えイメージ

            gap = s_own - s_other  # 自HPOがどれだけ勝ってるか

            if s_own < self_min_sim:
                reason = "low_self_sim"
            elif (s_other - s_own) > other_margin:
                reason = "high_other_sim"
            elif ambig_gap > 0 and gap < ambig_gap:
                reason = "ambiguous_top1_top2"   # ← NEW
            else:
                reason = ""

            if reason:
                rec.removal_reason = reason
                rec.removal_details = {
                    "own_sim": round(s_own, 4),
                    "other_sim": round(s_other, 4),
                    "margin": round(s_other - s_own, 4),
                    "best_other_hpo": best_other_hid,
                    "best_other_label": best_other_label
                }
                removed.append(rec)
            else:
                # 採用する場合はスコア情報を保持しておく（分析用）
                # rec.score などを上書きしてもいいが、ここでは元のJudgeスコアを優先
                kept.append(rec)

    print(f"[Filter] S-BERT filter: Removed {len(removed)} records.")
    return kept, removed


def add_official_labels(
    records: List[ExprRecord],
    hpo_metadata: Dict[str, Dict],
    mode: str,
    *,
    hpo_ids: Optional[set[str]] = None,
) -> List[ExprRecord]:
    """公式ラベルを追加"""
    existing_set = {(r.hpo_id, r.expr_norm) for r in records}
    hpo_ids_in_data = hpo_ids if hpo_ids is not None else {r.hpo_id for r in records}

    added_count = 0
    for hid in hpo_ids_in_data:
        if hid not in hpo_metadata:
            continue
        label = hpo_metadata[hid]["label"]
        norm_label = normalize_text(label)
        
        if label and (hid, norm_label) not in existing_set:
            new_rec = ExprRecord(
                hpo_id=hid, hpo_label=label, mode=mode, expr=label, score=10.0,
                removal_reason="official_added" # 削除ではないがソース識別用
            )
            records.append(new_rec)
            existing_set.add((hid, norm_label))
            added_count += 1

    print(f"[Add] Added {added_count} official labels.")
    return records


def save_removed_records(removed_list: List[ExprRecord], output_path: str):
    """削除されたレコードをJSONLで保存"""
    print(f"Saving removed log to {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as f:
        for r in removed_list:
            out_obj = {
                "hpo_id": r.hpo_id,
                "hpo_label": r.hpo_label,
                "text": r.expr,
                "reason": r.removal_reason,
                "details": r.removal_details,
                "original_score": r.score
            }
            f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--hpo-master", required=True, help="HPO master CSV")
    parser.add_argument("--output", required=True, help="Output cleaned JSONL")
    parser.add_argument("--output-removed", default="removed_log.jsonl", help="Output removed records JSONL")
    parser.add_argument("--mode", default="patient", choices=["patient", "doctor", "mixed"], help="mixed: decide per-record from source/mode")
    
    parser.add_argument("--no-drop-same-as-label", action="store_true")
    
    # S-BERT options
    parser.add_argument("--sbert-model", default="pkshatech/GLuCoSE-base-ja-v2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--self-min-sim", type=float, default=0.4)
    parser.add_argument("--other-margin", type=float, default=0.15)
    parser.add_argument(
        "--ambig-gap",
        type=float,
        default=0.05,
        help="self_sim - other_sim がこの値未満なら曖昧として除外（0で無効）",
    )
    parser.add_argument("--wandb", action="store_true", help="W&B にステップ統計を送る")
    parser.add_argument("--wandb-project", type=str, default="HPO_postprocess")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--wandb-group", type=str, default=None)

    args = parser.parse_args()

    wandb_run = None
    if args.wandb:
        if not HAS_WANDB:
            print("[WARN] wandb is not installed. Disable --wandb to silence this message.")
        else:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                group=args.wandb_group,
                tags=[t for t in args.wandb_tags.split(",") if t.strip()],
                config=vars(args),
            )
    
    # 全削除ログを貯めるリスト
    all_removed_records = []

    print(f"Loading HPO master from {args.hpo_master} ...")
    hpo_metadata = load_hpo_metadata(Path(args.hpo_master))

    print(f"Loading records from {args.input} ...")
    records = load_records(Path(args.input), hpo_metadata, args.mode)
    initial_count = len(records)
    print(f"Total loaded: {initial_count}")
    hpo_ids_in_input = {r.hpo_id for r in records}

    # 1. Label完全一致削除
    if not args.no_drop_same_as_label:
        records, removed = drop_same_as_label(records)
        all_removed_records.extend(removed)

    # 2. 重複削除 (Depth優先)
    records, removed = drop_cross_hpo_duplicates(records, hpo_metadata)
    all_removed_records.extend(removed)

    # 3. S-BERTフィルタ
    if args.self_min_sim > 0 or args.other_margin > 0:
        records, removed = run_sbert_filter(records, hpo_metadata, args)
        all_removed_records.extend(removed)

    # 4. 公式ラベル追加
    records = add_official_labels(records, hpo_metadata, args.mode, hpo_ids=hpo_ids_in_input)

    # 保存
    print(f"Saving {len(records)} clean records to {args.output} ...")
    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            # 念のため最終出力前にもクリーニングをかけておく
            text_clean = clean_expression(r.expr, r.mode)
            out_obj = {
                "hpo_id": r.hpo_id,
                "hpo_label": r.hpo_label,
                "text": text_clean,
                "score": r.score,
                "mode": r.mode,
                "source": "official" if r.score == 10.0 else "gen_filtered"
            }
            f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            
    # 削除ログ保存
    if all_removed_records:
        save_removed_records(all_removed_records, args.output_removed)
        
    print("\n=== Summary ===")
    print(f"Initial: {initial_count}")
    print(f"Final:   {len(records)}")
    print(f"Removed: {len(all_removed_records)}")
    print("Done.")

    if wandb_run:
        reason_counts = Counter(r.removal_reason or "unspecified" for r in all_removed_records)
        added_official = sum(1 for r in records if r.removal_reason == "official_added")
        wandb_run.log(
            {
                "initial": initial_count,
                "final": len(records),
                "removed": len(all_removed_records),
                "removed_by_reason": dict(reason_counts),
                "added_official": added_official,
            }
        )
        wandb_run.finish()

if __name__ == "__main__":
    main()
