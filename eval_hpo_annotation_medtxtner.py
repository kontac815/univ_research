#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
eval_hpo_annotation_medtxtner.py

MedTXTNER + n-gram + FAISS による HPO 抽出結果を、
hpo_annotation_test_dataset.csv に対して評価するスクリプト。

評価指標:
  - Exact precision / recall / F1
  - 階層構造を加味した Hierarchical precision / recall / F1
  - recall@k (k=1,3,5,10 など)
  - 「最小の候補数で正解に到達する m」の平均 (exact / hierarchical)

前提:
  - gold CSV: hpo_annotation_test_dataset.csv
      * ID 列: family_name (または patient_name 等)
      * gold HPO 列: hpo_ids ("HP:0001250 HP:0025356" のような空白区切り)
  - pred JSONL: annotate_with_medtxtner_hpo_batch.py の出力
      * 各行: {
            "input_id": "...",
            "annotations": [
                {"hpo_id": "...", "status": "present", "score": 0.83, ...},
                ...
            ]
        }
    ※ 上記の通り、score フィールドが存在する想定（本スクリプトより前に
       annotate_with_medtxtner_hpo_batch.py を修正しておくこと）。
  - hp.obo: HPO の OBO ファイル (is_a から階層構造を構築する)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import pandas as pd
try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ============================
# HPO OBO から階層構造を構築
# ============================

def parse_hpo_obo(path: Path) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    hp.obo を読み込み、親子関係を構築する。

    戻り値:
      - parents_map: hpo_id -> [parent_ids]
      - children_map: hpo_id -> [child_ids]
    """
    parents_map: Dict[str, List[str]] = {}
    children_map: Dict[str, List[str]] = {}

    current_id: Optional[str] = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                current_id = None
                continue

            if line.startswith("id: "):
                current_id = line.split("id: ")[1].strip()
                if current_id not in parents_map:
                    parents_map[current_id] = []
                if current_id not in children_map:
                    children_map[current_id] = []
                continue

            if current_id and line.startswith("is_a: "):
                # 例: "is_a: HP:0012823 ! Nervous system abnormality"
                parts = line.split("is_a: ")[1].split(" ! ")
                parent_id = parts[0].strip()
                parents_map.setdefault(current_id, []).append(parent_id)
                children_map.setdefault(parent_id, []).append(current_id)

    return parents_map, children_map


def get_ancestors(
    hpo_id: str,
    parents_map: Dict[str, List[str]],
) -> Set[str]:
    """
    hpo_id のすべての祖先 (親・祖父母...) を集める。
    自分自身は含めない。
    """
    visited: Set[str] = set()
    stack = list(parents_map.get(hpo_id, []))

    while stack:
        p = stack.pop()
        if p in visited:
            continue
        visited.add(p)
        stack.extend(parents_map.get(p, []))

    return visited


def get_descendants(
    hpo_id: str,
    children_map: Dict[str, List[str]],
) -> Set[str]:
    """
    hpo_id のすべての子孫 (子・孫...) を集める。
    自分自身は含めない。
    """
    visited: Set[str] = set()
    stack = list(children_map.get(hpo_id, []))

    while stack:
        c = stack.pop()
        if c in visited:
            continue
        visited.add(c)
        stack.extend(children_map.get(c, []))

    return visited


# ============================
# gold / pred 読み込み
# ============================

def load_gold_csv(
    path: Path,
    id_column: str,
    gold_hpo_column: str,
) -> Dict[str, Set[str]]:
    """
    gold CSV を読み込み、id -> gold HPO set を返す。
    """
    df = pd.read_csv(path)
    if id_column not in df.columns:
        raise ValueError(f"id_column '{id_column}' not in CSV columns.")
    if gold_hpo_column not in df.columns:
        raise ValueError(f"gold_hpo_column '{gold_hpo_column}' not in CSV columns.")

    gold_map: Dict[str, Set[str]] = {}

    for _, row in df.iterrows():
        id_val = str(row[id_column])
        hpo_str = str(row[gold_hpo_column]) if not pd.isna(row[gold_hpo_column]) else ""
        hpos = set(h.strip() for h in hpo_str.split() if h.strip())
        gold_map[id_val] = hpos

    return gold_map


def load_pred_jsonl(
    path: Path,
    use_status_present_only: bool = True,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    pred JSONL を読み込み、id -> [(hpo_id, score), ...] を返す。

    use_status_present_only:
      True なら status=="present" のもののみ対象にする。
    """
    pred_map: Dict[str, List[Tuple[str, float]]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            input_id = str(obj.get("input_id", ""))
            if not input_id:
                continue

            anns = obj.get("annotations", [])
            preds: List[Tuple[str, float]] = []

            for a in anns:
                status = a.get("status", "present")
                if use_status_present_only and status != "present":
                    continue
                hpo_id = a.get("hpo_id")
                if not hpo_id:
                    continue
                score = float(a.get("score", 0.0))
                preds.append((hpo_id, score))

            # score 降順にソート
            preds.sort(key=lambda x: x[1], reverse=True)

            pred_map[input_id] = preds

    return pred_map


# ============================
# メトリクス計算
# ============================

def compute_exact_metrics(
    gold_map: Dict[str, Set[str]],
    pred_map: Dict[str, List[Tuple[str, float]]],
) -> Tuple[float, float, float]:
    """
    Exact match ベースの micro precision / recall / F1 を計算。
    """
    tp = 0
    fp = 0
    fn = 0

    for pid, gold_hpos in gold_map.items():
        preds = [h for h, _ in pred_map.get(pid, [])]

        gold_set = set(gold_hpos)
        pred_set = set(preds)

        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_hier_metrics(
    gold_map: Dict[str, Set[str]],
    pred_map: Dict[str, List[Tuple[str, float]]],
    parents_map: Dict[str, List[str]],
    children_map: Dict[str, List[str]],
) -> Tuple[float, float, float]:
    """
    親子も OK とみなした Hierarchical micro precision / recall / F1。

    定義:
      - gold h, pred p について、
        * p == h
        * または p が h の祖先 / 子孫
        ならヒット。
    """
    tp = 0
    fp = 0
    fn = 0

    # 事前に ancestors / descendants をキャッシュ
    ancestor_cache: Dict[str, Set[str]] = {}
    descendant_cache: Dict[str, Set[str]] = {}

    def get_anc(hid: str) -> Set[str]:
        if hid not in ancestor_cache:
            ancestor_cache[hid] = get_ancestors(hid, parents_map)
        return ancestor_cache[hid]

    def get_desc(hid: str) -> Set[str]:
        if hid not in descendant_cache:
            descendant_cache[hid] = get_descendants(hid, children_map)
        return descendant_cache[hid]

    for pid, gold_hpos in gold_map.items():
        preds = [h for h, _ in pred_map.get(pid, [])]
        pred_set = set(preds)

        # gold 1 個ずつに対して一致する pred を探す
        matched_gold: Set[str] = set()
        matched_pred: Set[str] = set()

        for g in gold_hpos:
            g_anc = get_anc(g)
            g_desc = get_desc(g)

            for p in preds:
                if p in matched_pred:
                    continue
                if p == g or p in g_anc or p in g_desc:
                    matched_gold.add(g)
                    matched_pred.add(p)
                    break

        tp += len(matched_gold)
        fp += len(pred_set - matched_pred)
        fn += len(gold_hpos - matched_gold)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_recall_at_k(
    gold_map: Dict[str, Set[str]],
    pred_map: Dict[str, List[Tuple[str, float]]],
    ks: List[int],
) -> Dict[int, float]:
    """
    recall@k (exact) を計算。
    """
    recalls: Dict[int, float] = {}
    for k in ks:
        total = 0.0
        count = 0
        for pid, gold_hpos in gold_map.items():
            if not gold_hpos:
                continue
            preds_k = [h for h, _ in pred_map.get(pid, [])][:k]
            hit = len(set(preds_k) & set(gold_hpos))
            total += hit / len(gold_hpos)
            count += 1
        recalls[k] = total / count if count > 0 else 0.0
    return recalls


def compute_min_k_stats(
    gold_map: Dict[str, Set[str]],
    pred_map: Dict[str, List[Tuple[str, float]]],
    parents_map: Dict[str, List[str]],
    children_map: Dict[str, List[str]],
) -> Tuple[float, float]:
    """
    各症例ごとに
      - exact で初めて gold にヒットする最小 k
      - hierarchical で初めて gold にヒットする最小 k
    を求め、全体の平均を返す。
    """

    ancestor_cache: Dict[str, Set[str]] = {}
    descendant_cache: Dict[str, Set[str]] = {}

    def get_anc(hid: str) -> Set[str]:
        if hid not in ancestor_cache:
            ancestor_cache[hid] = get_ancestors(hid, parents_map)
        return ancestor_cache[hid]

    def get_desc(hid: str) -> Set[str]:
        if hid not in descendant_cache:
            descendant_cache[hid] = get_descendants(hid, children_map)
        return descendant_cache[hid]

    min_k_exact: List[int] = []
    min_k_hier: List[int] = []

    for pid, gold_hpos in gold_map.items():
        if not gold_hpos:
            continue
        preds = [h for h, _ in pred_map.get(pid, [])]
        if not preds:
            continue

        # exact
        k_exact = None
        gold_set = set(gold_hpos)
        seen: Set[str] = set()
        for i, p in enumerate(preds, start=1):
            seen.add(p)
            if seen & gold_set:
                k_exact = i
                break

        if k_exact is not None:
            min_k_exact.append(k_exact)

        # hierarchical
        k_hier = None
        used_preds: Set[str] = set()
        for i, p in enumerate(preds, start=1):
            used_preds.add(p)
            # 任意の gold について、used_preds に親子があるか
            hit = False
            for g in gold_hpos:
                g_anc = get_anc(g)
                g_desc = get_desc(g)
                if p == g or p in g_anc or p in g_desc:
                    hit = True
                    break
            if hit:
                k_hier = i
                break

        if k_hier is not None:
            min_k_hier.append(k_hier)

    avg_k_exact = sum(min_k_exact) / len(min_k_exact) if min_k_exact else 0.0
    avg_k_hier = sum(min_k_hier) / len(min_k_hier) if min_k_hier else 0.0
    return avg_k_exact, avg_k_hier


# ============================
# main
# ============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-csv", type=str, required=True,
                    help="hpo_annotation_test_dataset.csv")
    ap.add_argument("--id-column", type=str, default="family_name",
                    help="症例 ID として使う列名 (デフォ: family_name)")
    ap.add_argument("--gold-hpo-column", type=str, default="hpo_ids",
                    help="gold HPO ID を含む列名 (デフォ: hpo_ids, 空白区切り)")
    ap.add_argument("--pred-jsonl", type=str, required=True,
                    help="annotate_with_medtxtner_hpo_batch.py の出力 JSONL")
    ap.add_argument("--hpo-obo", type=str, required=True,
                    help="HPO の OBO ファイル (hp.obo)")
    ap.add_argument("--k-list", type=str, default="1,3,5,10",
                    help="recall@k を計算する k のリスト (カンマ区切り)")
    ap.add_argument("--output-tsv", type=str, default="",
                    help="評価指標を TSV で保存する場合のパス")
    ap.add_argument("--wandb", action="store_true", help="W&B に評価結果を送る")
    ap.add_argument("--wandb-project", type=str, default="HPO_eval")
    ap.add_argument("--wandb-run-name", type=str, default=None)
    ap.add_argument("--wandb-tags", type=str, default="")
    ap.add_argument("--wandb-group", type=str, default=None)

    args = ap.parse_args()

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

    test_csv = Path(args.test_csv)
    pred_jsonl = Path(args.pred_jsonl)
    hpo_obo = Path(args.hpo_obo)

    ks = [int(x) for x in args.k_list.split(",") if x.strip()]

    print(f"[INFO] load gold from: {test_csv}")
    gold_map = load_gold_csv(test_csv, args.id_column, args.gold_hpo_column)
    print(f"[INFO] gold cases: {len(gold_map)}")

    print(f"[INFO] load predictions from: {pred_jsonl}")
    pred_map = load_pred_jsonl(pred_jsonl, use_status_present_only=True)
    print(f"[INFO] predicted cases: {len(pred_map)}")

    print(f"[INFO] parse HPO OBO: {hpo_obo}")
    parents_map, children_map = parse_hpo_obo(hpo_obo)

    # Exact metrics
    p_exact, r_exact, f1_exact = compute_exact_metrics(gold_map, pred_map)
    print("\n=== Exact metrics ===")
    print(f"Precision: {p_exact:.4f}")
    print(f"Recall   : {r_exact:.4f}")
    print(f"F1       : {f1_exact:.4f}")

    # Hierarchical metrics
    p_hier, r_hier, f1_hier = compute_hier_metrics(
        gold_map, pred_map, parents_map, children_map
    )
    print("\n=== Hierarchical metrics (parent/child OK) ===")
    print(f"Precision: {p_hier:.4f}")
    print(f"Recall   : {r_hier:.4f}")
    print(f"F1       : {f1_hier:.4f}")

    # recall@k
    recalls = compute_recall_at_k(gold_map, pred_map, ks)
    print("\n=== recall@k (exact) ===")
    for k in ks:
        print(f"recall@{k}: {recalls[k]:.4f}")

    # min-k stats
    avg_k_exact, avg_k_hier = compute_min_k_stats(
        gold_map, pred_map, parents_map, children_map
    )
    print("\n=== Minimum k to hit at least one gold ===")
    print(f"Average min k (exact)       : {avg_k_exact:.2f}")
    print(f"Average min k (hierarchical): {avg_k_hier:.2f}")

    if wandb_run:
        payload = {
            "precision_exact": p_exact,
            "recall_exact": r_exact,
            "f1_exact": f1_exact,
            "precision_hier": p_hier,
            "recall_hier": r_hier,
            "f1_hier": f1_hier,
            "avg_min_k_exact": avg_k_exact,
            "avg_min_k_hier": avg_k_hier,
            "pred_jsonl": str(pred_jsonl),
            "test_csv": str(test_csv),
        }
        for k in ks:
            payload[f"recall@{k}"] = recalls[k]
        wandb_run.log(payload)

    # TSV 保存 (オプション)
    if args.output_tsv:
        out_path = Path(args.output_tsv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for k in ks:
            rows.append({
                "metric": f"recall@{k}",
                "value": recalls[k],
            })
        rows.extend([
            {"metric": "precision_exact", "value": p_exact},
            {"metric": "recall_exact", "value": r_exact},
            {"metric": "f1_exact", "value": f1_exact},
            {"metric": "precision_hier", "value": p_hier},
            {"metric": "recall_hier", "value": r_hier},
            {"metric": "f1_hier", "value": f1_hier},
            {"metric": "avg_min_k_exact", "value": avg_k_exact},
            {"metric": "avg_min_k_hier", "value": avg_k_hier},
        ])
        df_out = pd.DataFrame(rows)
        df_out.to_csv(out_path, sep="\t", index=False)
        print(f"\n[INFO] metrics saved to: {out_path}")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
