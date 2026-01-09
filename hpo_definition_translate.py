#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
hpo_definition_translate.py

GPT-5系モデルで HPO 英語定義を日本語化する並列翻訳スクリプト。
ThreadPoolExecutor で複数行を同時処理し、既存の翻訳はログで再利用する。
"""

import argparse
import json
import os
import sys
import threading
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# ログ書き込み用のロック（並列処理時の競合防止）
log_lock = threading.Lock()

def parse_hp_obo_defs(obo_path):
    """oboファイルから {HPO_ID: definition_en} を抽出"""
    defs = {}
    current_id = None
    with open(obo_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("id: HP:"):
                current_id = line.replace("id: ", "")
            elif line.startswith("def:"):
                start = line.find('"')
                end = line.rfind('"')
                if start != -1 and end != -1 and end > start:
                    if current_id:
                        defs[current_id] = line[start+1:end]
    return defs

def translate_definition(client, label, def_en, model):
    """
    1件翻訳を実行する関数
    """
    prompt = (
        f"以下の医学用語（HPO）の定義文を、専門家（医師・遺伝カウンセラー）にとって"
        f"自然な日本語に翻訳してください。\n\n"
        f"用語名: {label}\n"
        f"英語定義: {def_en}\n\n"
        f"出力は翻訳結果の日本語文字列のみを返してください。"
    )
    
    try:
        # GPT-5 / o-series 対応パラメータ
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "developer", "content": "You are a helpful medical translator."},
                {"role": "user", "content": prompt}
            ],
            # temperature は指定しない (モデル側で制御)
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] Translation failed: {e}")
        return ""

def process_single_item(client, row, obo_defs, model, log_path):
    """
    1行分の処理をまとめた関数（並列実行の単位）
    戻り値: (index, definition_ja_string)
    """
    idx = row.name  # DataFrameのindex
    hid = str(row["HPO_ID"])
    category = str(row.get("category", ""))
    
    # フィルタ条件: symptom 以外はスキップ（空文字を返す）
    if category != "symptom":
        return idx, ""

    # OBOに定義がない場合
    def_en = obo_defs.get(hid)
    if not def_en:
        return idx, ""

    # 翻訳実行
    label = str(row.get("jp_final") or "")
    def_ja = translate_definition(client, label, def_en, model)
    
    # ログ保存（ロックを使って安全に書き込む）
    if def_ja:
        with log_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                log_entry = {"HPO_ID": hid, "def_ja": def_ja}
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    return idx, def_ja

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", required=True, help="入力CSVパス")
    parser.add_argument("--obo", required=True, help="hp.oboパス")
    parser.add_argument("--out-master", required=True, help="出力CSVパス")
    parser.add_argument("--log", default="translation_log.jsonl", help="ログファイルパス")
    parser.add_argument("--model", default="gpt-5-mini", help="使用するOpenAIモデル")
    parser.add_argument("--workers", type=int, default=10, help="並列数（デフォルト: 10）")
    
    args = parser.parse_args()

    print(f"[INFO] Using Model: {args.model}")
    print(f"[INFO] Parallel Workers: {args.workers}")
    
    # 1. データ読み込み
    df = pd.read_csv(args.master)
    print(f"[INFO] Original rows: {len(df)}")
    
    # DataFrameのindexを明示的に保持（並列処理後の結合のため）
    df["__original_idx"] = df.index

    obo_defs = parse_hp_obo_defs(args.obo)

    # 2. ログ(キャッシュ)読み込み
    translated_cache = {}
    if os.path.exists(args.log):
        with open(args.log, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("def_ja"):
                        translated_cache[d["HPO_ID"]] = d["def_ja"]
                except: pass
    print(f"[INFO] Cached items: {len(translated_cache)}")

    # 3. 処理対象の特定（翻訳が必要な行だけをリストアップ）
    #    DataFrame全体を渡すと並列処理管理が面倒なので、タスクリストを作る
    tasks = []
    
    # 結果格納用（初期値は空文字、既存orキャッシュがあれば埋める）
    # 最終的にこの辞書を DataFrame にマッピングする
    final_results = {} 
    has_def_col = "definition_ja" in df.columns

    print("[INFO] Preparing tasks...")
    for idx, row in df.iterrows():
        hid = str(row["HPO_ID"])
        
        # 既存の値 or キャッシュチェック
        val = row.get("definition_ja", "") if has_def_col else ""
        if pd.isna(val): val = ""
        
        if val:
            final_results[idx] = val
            continue
            
        if hid in translated_cache:
            final_results[idx] = translated_cache[hid]
            continue
            
        tasks.append(row) # 行データをタスクに追加

    print(f"[INFO] Tasks to process: {len(tasks)}")

    # 4. 並列処理実行
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    if tasks:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # タスクを投入
            future_to_idx = {
                executor.submit(process_single_item, client, row, obo_defs, args.model, args.log): row.name
                for row in tasks
            }
            
            # プログレスバー付きで完了を待機
            for future in tqdm(as_completed(future_to_idx), total=len(tasks)):
                idx, result_str = future.result()
                final_results[idx] = result_str
    else:
        print("[INFO] No new tasks to translate.")

    # 5. 結果を結合して保存
    # index順に並べてリスト化
    new_defs_list = [final_results.get(i, "") for i in df.index]
    
    df["definition_ja"] = new_defs_list
    if "__original_idx" in df.columns:
        del df["__original_idx"]

    df.to_csv(args.out_master, index=False)
    print(f"[INFO] Done. Saved to {args.out_master}")

if __name__ == "__main__":
    main()