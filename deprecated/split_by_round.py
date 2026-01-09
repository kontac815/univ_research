#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

def get_round_value(data: dict):
    """
    JSONデータからラウンド数を抽出する。
    複数のキーパターンに対応。
    """
    # 1. 今回提示された形式
    if "round" in data:
        return data["round"]
    
    # 2. 前回の形式 (judge_round / refine_round)
    if "judge_round" in data:
        return data["judge_round"]
    if "refine_round" in data:
        return data["refine_round"]
        
    return None

def main():
    parser = argparse.ArgumentParser(description="JSONLファイルをラウンド(round)ごとに分割保存します。")
    parser.add_argument("input_file", help="分割元のJSONLファイルパス")
    parser.add_argument("--output-dir", default="./split_output", help="出力先ディレクトリ")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    # 出力ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ファイルハンドル管理用 { round_num: file_handle }
    file_handles = {}
    counts = defaultdict(int)
    
    print(f"Reading from {input_path} ...")
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"[Warn] Line {line_no}: Invalid JSON. Skipped.")
                continue
            
            # ラウンド値を取得
            r_val = get_round_value(data)
            
            # ラウンドが特定できない場合は 'unknown' 扱い
            if r_val is None:
                round_key = "unknown"
            else:
                # 整数化 (0.0 -> 0, "1" -> 1)
                try:
                    round_key = int(float(r_val))
                except (ValueError, TypeError):
                    round_key = "unknown"
            
            # ファイルハンドルが未オープンの場合は開く
            if round_key not in file_handles:
                filename = f"round_{round_key}.jsonl"
                filepath = output_dir / filename
                print(f"Creating new split file: {filepath}")
                file_handles[round_key] = open(filepath, "w", encoding="utf-8")
            
            # 書き込み
            file_handles[round_key].write(json.dumps(data, ensure_ascii=False) + "\n")
            counts[round_key] += 1

    # 後始末
    for fh in file_handles.values():
        fh.close()
        
    print("\n=== Split Result ===")
    if not counts:
        print("No records found.")
    else:
        # キーをソートして表示 (0, 1, 2, ... unknown)
        sorted_keys = sorted(counts.keys(), key=lambda x: (isinstance(x, str), x))
        for k in sorted_keys:
            print(f"  Round {k}: {counts[k]} lines")
    
    print(f"\nSaved to: {output_dir.resolve()}")

if __name__ == "__main__":
    main()