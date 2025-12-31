#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
06_fill_hpo_jp_with_llm.py

HPO マスタ / サブセット CSV のうち、
jp_final が欠損している symptom 系 HPO について、
OpenAI API（Chat Completions）を用いて日本語ラベルを自動生成し、
CSV を更新するスクリプト。

想定している前提:
  - マスタ: ../data/hpo_master_all_jp.weblio.csv
  - サブセット: ../data/HPO_symptom_depth_leq3_without_jp.csv
  - OBO: ../data/hp.obo  (必要に応じてパスは変更してください)

使い方例:
  $ export OPENAI_API_KEY="sk-...."
  $ python 06_fill_hpo_jp_with_llm.py \
      --master ../data/hpo_master_all_jp.weblio.csv \
      --subset ../data/HPO_symptom_depth_leq3_without_jp.csv \
      --obo ../data/hp.obo \
      --out-master ../data/hpo_master_all_jp.weblio.with_llm_jp.csv \
      --out-subset ../data/HPO_symptom_depth_leq3_with_llm_jp.csv \
      --out-subset-jp-en ../data/HPO_symptom_depth_leq3_jp_en_pairs.csv \
      --log ../data/HPO_symptom_depth_leq3_llm_jp.log.jsonl \
      --model gpt-4o-mini

ポイント:
  - すでに jp_final が埋まっている行は上書きしません（安全に再実行可能）。
  - 途中で落ちても、--log JSONL に保存された結果は再利用されるため、
    既に生成済みの HPO_ID については再度 API を叩きません。
  - 追加で、subset 内の「jp_final が埋まっている行」だけを抜き出した
    JP-EN ペアの CSV (--out-subset-jp-en) も出力します。
"""

from __future__ import annotations

import argparse
import json
import os
os.environ["HTTP_PROXY"] = "http://proxy.l2.med.tohoku.ac.jp:8080"
os.environ["HTTPS_PROXY"] = "http://proxy.l2.med.tohoku.ac.jp:8080"
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import pandas as pd
import httpx  # ★追加
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # 実行時には openai パッケージが必要です


@dataclass
class HPOInfo:
    hpo_id: str
    name: Optional[str]
    definition: Optional[str]
    synonyms: List[str]
# Weblio 由来の「意味・読み方・使い方」「ピン留め追加できません」などのノイズを検出するためのパターン
WEBLIO_NOISE_PATTERNS = [
    "意味・読み方・使い方",
    "ピン留め追加できません",
    "発音を聞くプレーヤー再生",
    "単語を追加",
]


def clear_weblio_noise(df: pd.DataFrame, label_col: str = "jp_final") -> int:
    """
    Weblio の UI テキストが紛れ込んでいる jp_final を検出して NA にする。

    戻り値: クリアした行数
    """
    if label_col not in df.columns:
        return 0

    # object 型にそろえておく（既に main() でもやっていますが、念のため）
    df[label_col] = df[label_col].astype("object")

    series = df[label_col].astype(str)
    # ノイズパターンをまとめて正規表現に
    pattern = "|".join(re.escape(p) for p in WEBLIO_NOISE_PATTERNS)

    mask = series.str.contains(pattern, regex=True, na=False)
    n = int(mask.sum())
    if n > 0:
        print(f"== clear_weblio_noise: {label_col} に Weblio ノイズ行を {n} 件検出 → NA にします")
        df.loc[mask, label_col] = pd.NA  # ここで欠損扱いにして、あとで LLM で再生成
        if "jp_source" in df.columns:
            df.loc[mask, "jp_source"] = "llm_regen_from_weblio_noise"
    else:
        print(f"== clear_weblio_noise: {label_col} に Weblio ノイズは検出されませんでした")

    return n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill missing HPO jp_final labels using OpenAI Chat Completions."
    )
    parser.add_argument(
        "--master",
        type=Path,
        default=Path("../data/hpo_master_all_jp.weblio.csv"),
        help="HPO master CSV (will be read and optionally updated).",
    )
    parser.add_argument(
        "--subset",
        type=Path,
        default=Path("../data/HPO_symptom_depth_leq3_without_jp.csv"),
        help="Subset CSV (symptom depth<=3 without jp_final).",
    )
    parser.add_argument(
        "--obo",
        type=Path,
        default=Path("../data/hp.obo"),
        help="hp.obo path.",
    )
    parser.add_argument(
        "--out-master",
        type=Path,
        default=Path("../data/hpo_master_all_jp.weblio.with_llm_jp.csv"),
        help="Output path for updated master CSV.",
    )
    parser.add_argument(
        "--out-subset",
        type=Path,
        default=Path("../data/HPO_symptom_depth_leq3_with_llm_jp.csv"),
        help="Output path for updated subset CSV.",
    )
    parser.add_argument(
        "--out-subset-jp-en",
        type=Path,
        default=Path("../data/HPO_symptom_depth_leq3_jp_en_pairs.csv"),
        help=(
            "Output path for JP-EN pairs (subset rows where jp_final is not null). "
            "Columns: HPO_ID, jp_final, name_en (and optionally jp_source if存在)."
        ),
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("../data/HPO_symptom_depth_leq3_llm_jp.log.jsonl"),
        help="JSONL log file to store per-HPO generation results.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="OpenAI model name (e.g., gpt-4o-mini).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit on number of HPO rows to process (for testing).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep seconds between API calls (for rate limiting).",
    )
    return parser.parse_args()


def load_existing_log(log_path: Path) -> Dict[str, str]:
    """
    既存の JSONL ログから {HPO_ID: jp_label} の辞書を復元する。
    """
    results: Dict[str, str] = {}
    if not log_path.exists():
        return results

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            hpo_id = rec.get("HPO_ID") or rec.get("hpo_id")
            jp = rec.get("jp_label") or rec.get("jp_final")
            if hpo_id and jp:
                results[hpo_id] = jp
    return results


def append_log(log_path: Path, record: Dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_hp_obo(obo_path: Path) -> Dict[str, HPOInfo]:
    """
    hp.obo を簡易パースして、HPO_ID -> HPOInfo の辞書を返す。
    """
    hpo_dict: Dict[str, HPOInfo] = {}

    current_id: Optional[str] = None
    current_name: Optional[str] = None
    current_def: Optional[str] = None
    current_syns: List[str] = []

    def flush_current():
        nonlocal current_id, current_name, current_def, current_syns
        if current_id is not None:
            hpo_dict[current_id] = HPOInfo(
                hpo_id=current_id,
                name=current_name,
                definition=current_def,
                synonyms=current_syns[:],
            )
        current_id = None
        current_name = None
        current_def = None
        current_syns = []

    with obo_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if line == "[Term]":
                flush_current()
                continue

            if not line or line.startswith("[Typedef]"):
                # Term ブロックの区切り
                if current_id is not None and line == "":
                    flush_current()
                continue

            if line.startswith("id: HP:"):
                current_id = line.split("id: ", 1)[1].strip()
                continue

            if line.startswith("name: "):
                current_name = line.split("name: ", 1)[1].strip()
                continue

            if line.startswith("def: "):
                # def: "...." [PMID:xxx]
                text = line[len("def: ") :].strip()
                if text.startswith('"'):
                    try:
                        first_quote = text.index('"')
                        second_quote = text.index('"', first_quote + 1)
                        current_def = text[first_quote + 1 : second_quote]
                    except ValueError:
                        current_def = text.strip('"')
                else:
                    current_def = text
                continue

            if line.startswith("synonym: "):
                # synonym: "xxx" EXACT []
                text = line[len("synonym: ") :].strip()
                if text.startswith('"'):
                    try:
                        first_quote = text.index('"')
                        second_quote = text.index('"', first_quote + 1)
                        syn = text[first_quote + 1 : second_quote]
                        if syn:
                            current_syns.append(syn)
                    except ValueError:
                        pass
                continue

    # 最後のブロックを flush
    flush_current()
    return hpo_dict


def build_messages_for_hpo(
    row: pd.Series,
    info: Optional[HPOInfo],
) -> Tuple[str, str]:
    """
    1つの HPO に対して OpenAI Chat Completions に渡す
    system / user メッセージを構築する。
    """
    hpo_id = str(row.get("HPO_ID", ""))
    name_en = str(row.get("name_en", "") or "")

    definition = info.definition if info is not None else None
    syn_list = info.synonyms if info is not None else []
    syn_text = ", ".join(syn_list[:5]) if syn_list else "（同義語情報なし）"

    # 長すぎる definition はある程度で切る（プロンプト肥大防止）
    if definition and len(definition) > 400:
        definition_short = definition[:400] + " …"
    else:
        definition_short = definition or "（定義情報なし）"

    system_msg = (
        "あなたは日本の医師がカルテに記載する短い医療用語ラベルを作成する専門家AIです。"
        "Human Phenotype Ontology (HPO) の英語名と定義、同義語をもとに、"
        "日本の医療現場で自然に使われるであろう日本語の症状・所見名を1つだけ提案してください。"
        "疾患名や症候群名ではなく、あくまで症状・所見としての表現にしてください。"
    )

    user_msg = f"""
以下の HPO について、日本語の医療用語ラベルを1つだけ作成してください。

HPO ID: {hpo_id}
英語名 (name_en): {name_en}

定義 (definition):
{definition_short}

同義語 (synonyms):
{syn_text}

出力条件:
- 日本の医師が電子カルテや診療録に書きそうな、短い名詞句にしてください。
- 「〜の異常」「〜形態異常」「〜低下」「〜増加」「〜障害」など、一般的な医学用語の形に整えてください。
- 疾患名や「〜症候群」は避けてください（あくまで症状・所見レベルの表現とする）。
- 日本語のラベルのみを1行で出力し、説明文や英語訳、記号、番号などは一切付けないでください。
""".strip()

    return system_msg, user_msg


def postprocess_label(raw_text: str) -> str:
    """
    モデルから返ってきたテキストを軽く正規化して
    「日本語ラベル」だけを取り出す。
    """
    if not raw_text:
        return ""

    label = raw_text.strip()

    # 複数行返ってきた場合は先頭行だけ採用
    label = label.splitlines()[0].strip()

    # 箇条書きの記号を除去
    for prefix in ["- ", "・", "●", "1.", "1)", "①", "▶"]:
        if label.startswith(prefix):
            label = label[len(prefix) :].strip()

    # 先頭と末尾の引用符などを除去
    for ch in ['"', "「", "」", "『", "』", "'"]:
        if label.startswith(ch) and label.endswith(ch) and len(label) > 2:
            label = label[1:-1].strip()

    # 末尾の句点を除去（好みで）
    if label.endswith("。"):
        label = label[:-1].strip()

    return label


def ensure_openai_client() -> "OpenAI":
    """
    OpenAI クライアントを初期化する。
    - 通常は環境変数 OPENAI_API_KEY を使う
    - HTTP_PROXY / HTTPS_PROXY が設定されていれば、httpx.Client(proxy=...)
      を使ってプロキシ経由で接続する
    """
    if OpenAI is None:
        raise RuntimeError(
            "openai パッケージがインポートできません。`pip install openai` でインストールしてください。"
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "環境変数 OPENAI_API_KEY が設定されていません。OpenAI API キーを設定してください。"
        )

    # プロキシ設定（HTTPS優先で、なければHTTPを見る）
    proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")

    if proxy_url:
        # httpx の proxy 引数は http://... 形式のURLを受け取る
        # 認証が必要なら http://user:pass@host:port の形にする
        http_client = httpx.Client(proxy=proxy_url, timeout=60.0)
        client = OpenAI(api_key=api_key, http_client=http_client)
        print(f"== OpenAI client initialized with proxy: {proxy_url}")
    else:
        client = OpenAI(api_key=api_key)
        print("== OpenAI client initialized without proxy")

    return client


def generate_label_with_llm(
    client: "OpenAI",
    model: str,
    system_msg: str,
    user_msg: str,
) -> str:
    """
    OpenAI Chat Completions API を呼び出して日本語ラベルを生成する。
    GPT-5 系モデルでも動くように max_completion_tokens を使い、
    reasoning_effort を low にして可視テキストにトークンを回す。
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        # GPT-5 系では max_tokens は使えないので max_completion_tokens を使う
        # 「reasoning + 出力」の合計なのでやや多めに確保しておく
        max_completion_tokens=512,
        # 推論コストを下げて、可視出力にトークンを回す
        reasoning_effort="low",
        # GPT-5 は temperature をいじれないので渡さない
    )

    content = resp.choices[0].message.content or ""
    return postprocess_label(content)




def main() -> None:
    args = parse_args()

    # 入力ファイル確認
    if not args.master.exists():
        print(f"[ERROR] master CSV が見つかりません: {args.master}", file=sys.stderr)
        sys.exit(1)
    if not args.subset.exists():
        print(f"[ERROR] subset CSV が見つかりません: {args.subset}", file=sys.stderr)
        sys.exit(1)
    if not args.obo.exists():
        print(f"[ERROR] hp.obo が見つかりません: {args.obo}", file=sys.stderr)
        sys.exit(1)

    print(f"== load master: {args.master}")
    master_df = pd.read_csv(args.master)

    print(f"== load subset: {args.subset}")
    subset_df = pd.read_csv(args.subset)

    # ★ ここを追加：jp_final カラムを文字列型にしておく（存在する場合のみ）
    if "jp_final" in master_df.columns:
        master_df["jp_final"] = master_df["jp_final"].astype("object")
    if "jp_final" in subset_df.columns:
        subset_df["jp_final"] = subset_df["jp_final"].astype("object")

    # ★ Weblio ノイズを NA にして欠損扱いにする（master / subset 両方）
    cleared_master = clear_weblio_noise(master_df, label_col="jp_final")
    cleared_subset = clear_weblio_noise(subset_df, label_col="jp_final")
    print(f"== Weblio ノイズをクリア: master={cleared_master} 件, subset={cleared_subset} 件")

    print(f"== parse OBO: {args.obo}")
    hpo_dict = parse_hp_obo(args.obo)
    print(f"   parsed HPO terms: {len(hpo_dict)}")

    # 既存ログの読み込み（再実行対応）
    existing_log = load_existing_log(args.log)
    if existing_log:
        print(f"== existing log found: {args.log} ({len(existing_log)} entries)")

    # 対象行の抽出: subset 側で jp_final が欠損のもの
    need_mask = subset_df["jp_final"].isna()
    need_df = subset_df[need_mask].copy()
    print(f"== rows in subset with missing jp_final: {len(need_df)}")

    # ★ もともと HPO_symptom_depth_leq3_without_jp.csv にいた HPO_ID の集合
    target_ids = set(need_df["HPO_ID"].astype(str))

    if args.max_rows is not None:
        need_df = need_df.head(args.max_rows)
        print(f"== limited by --max-rows: process first {len(need_df)} rows")

    if need_df.empty:
        print("== no rows to process; exiting.")
        # それでも master/subset はそのままコピーしておく
        args.out_master.parent.mkdir(parents=True, exist_ok=True)
        args.out_subset.parent.mkdir(parents=True, exist_ok=True)
        master_df.to_csv(args.out_master, index=False)
        subset_df.to_csv(args.out_subset, index=False)

        # JP-EN ペア出力もここで作っておく
        # ★ もともと without_jp にいた HPO_ID (target_ids) かつ jp_final.notna()
        pairs_df = subset_df[
            subset_df["HPO_ID"].astype(str).isin(target_ids)
            & subset_df["jp_final"].notna()
        ].copy()
        if not pairs_df.empty:
            cols = [c for c in ["HPO_ID", "jp_final", "name_en", "jp_source"] if c in pairs_df.columns]
            pairs_df = pairs_df[cols]
            args.out_subset_jp_en.parent.mkdir(parents=True, exist_ok=True)
            print(f"== write jp-en pairs (subset, target_ids & jp_final not null): {args.out_subset_jp_en}")
            pairs_df.to_csv(args.out_subset_jp_en, index=False)
        else:
            print("== no jp_final rows in subset for target_ids; skip jp-en pairs output.")

        return


    # OpenAI クライアントを初期化
    client = ensure_openai_client()

    # HPO_ID をキーにした jp_label 辞書
    generated_labels: Dict[str, str] = dict(existing_log)

    # 実際に生成
    for idx, row in need_df.iterrows():
        hpo_id = str(row["HPO_ID"])
        if hpo_id in generated_labels:
            # 既にログにあるものは再利用
            jp_label = generated_labels[hpo_id]
            print(f"[SKIP] {hpo_id}: use cached label: {jp_label}")
        else:
            info = hpo_dict.get(hpo_id)
            system_msg, user_msg = build_messages_for_hpo(row, info)
            print(f"[CALL] {hpo_id}: {row.get('name_en')}")
            try:
                jp_label = generate_label_with_llm(
                    client=client,
                    model=args.model,
                    system_msg=system_msg,
                    user_msg=user_msg,
                )
            except Exception as e:
                print(f"[ERROR] OpenAI API error for {hpo_id}: {e}", file=sys.stderr)
                # エラー時は空欄のままにしておき、次回再実行で再度トライ
                append_log(
                    args.log,
                    {
                        "HPO_ID": hpo_id,
                        "name_en": row.get("name_en"),
                        "error": str(e),
                    },
                )
                if args.sleep > 0:
                    time.sleep(args.sleep)
                continue

            if not jp_label:
                print(f"[WARN] empty label for {hpo_id}", file=sys.stderr)
            else:
                print(f"  -> jp_label: {jp_label}")

            generated_labels[hpo_id] = jp_label
            append_log(
                args.log,
                {
                    "HPO_ID": hpo_id,
                    "name_en": row.get("name_en"),
                    "jp_label": jp_label,
                },
            )

            if args.sleep > 0:
                time.sleep(args.sleep)

        # subset_df / master_df をその場で更新
        if generated_labels[hpo_id]:
            subset_df.loc[subset_df["HPO_ID"] == hpo_id, "jp_final"] = generated_labels[hpo_id]
            master_df.loc[master_df["HPO_ID"] == hpo_id, "jp_final"] = generated_labels[hpo_id]

    # 出力（master / subset）
    args.out_master.parent.mkdir(parents=True, exist_ok=True)
    args.out_subset.parent.mkdir(parents=True, exist_ok=True)

    print(f"== write updated master: {args.out_master}")
    master_df.to_csv(args.out_master, index=False)

    print(f"== write updated subset: {args.out_subset}")
    subset_df.to_csv(args.out_subset, index=False)

    # 追加: jp_final が埋まっている subset 行のうち、
    # ★ もともと without_jp にいた HPO_ID だけ JP-EN ペアで出力
    pairs_df = subset_df[
        subset_df["HPO_ID"].astype(str).isin(target_ids)
        & subset_df["jp_final"].notna()
    ].copy()

    if not pairs_df.empty:
        cols = [c for c in ["HPO_ID", "jp_final", "name_en", "jp_source"] if c in pairs_df.columns]
        pairs_df = pairs_df[cols]
        args.out_subset_jp_en.parent.mkdir(parents=True, exist_ok=True)
        print(f"== write jp-en pairs (subset, target_ids & jp_final not null): {args.out_subset_jp_en}")
        pairs_df.to_csv(args.out_subset_jp_en, index=False)
    else:
        print("== no jp_final rows in subset for target_ids; skip jp-en pairs output.")

    print("== done.")




if __name__ == "__main__":
    main()
