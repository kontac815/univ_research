#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gemini_sft_train.py

Gemini 2.5 Flash の SFT(教師ありファインチューニング) を行う公式実装準拠スクリプト。

- google-generativeai (旧 SDK) を用いて SFT ジョブを作成
  * 公式チュートリアルと同じ create_tuned_model API を使用
- NAIST 患者表現 / 万病医師表現 の JSONL を読み込み、training_data を構築
- 学習完了までポーリングして最終的な tuned model ID を出力

作成された tuned model は、推論側では google-genai SDK の
  model="tunedModels/<MODEL_ID>"
としてそのまま利用できます。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, List, Any, Tuple

import google.generativeai as genai  # 公式 SFT 用ライブラリ

Example = Dict[str, str]

# ==============================
# データ読み込みユーティリティ
# ==============================

def load_naist_patient_jsonl(path: str, max_examples: int | None = None) -> List[Example]:
    """
    NAIST の患者表現 SFT データ (instruction / input / output) を
    Gemini tuning 用の {"text_input": ..., "output": ...} に変換。

    JSONL 1 行の例:
      {"instruction": "...", "input": "", "output": "できものができている"}
    """
    data: List[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_examples is not None and i >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            instruction = obj.get("instruction", "")
            input_text = obj.get("input", "")
            output = obj.get("output", "")

            # instruction + input を 1 本の text_input に結合
            if input_text:
                text_input = instruction.rstrip() + "\n\n" + input_text
            else:
                text_input = instruction

            if not output:
                continue

            data.append(
                {
                    "text_input": text_input,
                    "output": output,
                }
            )
    return data


def _flatten_messages_for_text_input(messages: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    doctor 用 JSONL の "messages" 形式を text_input / output に変換する。

    - messages: [{"role": "system"|"user"|"assistant", "content": "..."}...]
    - 最後の assistant を教師ラベル (output) とみなし、
      それ以前を text_input として 1 本のテキストにまとめる。
    """
    if not messages:
        return "", ""

    # 最後の assistant を探す
    last_ass_idx = None
    for idx in reversed(range(len(messages))):
        if messages[idx].get("role") in ("assistant", "model"):
            last_ass_idx = idx
            break

    if last_ass_idx is None:
        # assistant がなければ学習例としては使わない
        return "", ""

    # 入力側 (system + user + 途中の assistant など全部) を単純に role: content で連結
    input_msgs = messages[: last_ass_idx]
    target_msg = messages[last_ass_idx]

    parts: List[str] = []
    for m in input_msgs:
        role = m.get("role", "user")
        content = m.get("content", "")
        if not content:
            continue
        parts.append(f"{role}: {content}")
    text_input = "\n".join(parts).strip()

    output = target_msg.get("content", "").strip()
    return text_input, output


def load_manbyo_doctor_jsonl(path: str, max_examples: int | None = None) -> List[Example]:
    """
    万病医師表現の JSONL (messages + メタ情報) を
    {"text_input": ..., "output": ...} に変換。

    JSONL 1 行のイメージ:
      {
        "messages": [
          {"role": "system", "content": "..."},
          {"role": "user", "content": "..."},
          {"role": "assistant", "content": "..."}
        ],
        "term": "...",
        "reading": "...",
        "icd_code": "...",
        ...
      }
    """
    data: List[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_examples is not None and i >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            messages = obj.get("messages")
            if not isinstance(messages, list):
                continue

            text_input, output = _flatten_messages_for_text_input(messages)
            if not text_input or not output:
                continue

            data.append(
                {
                    "text_input": text_input,
                    "output": output,
                }
            )
    return data


# ==============================
# SFT 実行ロジック
# ==============================

def create_tuned_model(
    base_model: str,
    training_data: List[Example],
    model_id: str,
    epoch_count: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.001,
):
    """
    公式チュートリアルに沿った create_tuned_model 呼び出し。

    docs:
      - https://ai.google.dev/palm_docs/tuning_quickstart_python
      - https://developers.googleblog.com/en/tune-gemini-pro-in-google-ai-studio-or-with-the-gemini-api/
    """
    op = genai.create_tuned_model(
        source_model=base_model,
        training_data=training_data,
        id=model_id,
        epoch_count=epoch_count,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    return op


def wait_tuned_model(op, model_id: str, poll_interval_sec: int = 30):
    """
    Operation が完了するまでポーリングして結果の tuned model を返す。

    google-generativeai の Operation は wait_bar() / result() を持つので、
    公式サンプルと同様の書き方をする。
    """
    try:
        # Colab 用の進捗バーが使える場合
        for status in op.wait_bar():
            time.sleep(1)
    except AttributeError:
        # wait_bar がない場合は手動ポーリング
        while True:
            tuned = genai.get_tuned_model(f"tunedModels/{model_id}")
            state = tuned.state.name if hasattr(tuned.state, "name") else str(tuned.state)
            print("tuning state:", state)
            if state in ("ACTIVE", "FAILED", "STATE_UNSPECIFIED"):
                break
            time.sleep(poll_interval_sec)

    result = op.result()
    print("=== Tuning result ===")
    print(result)
    return result


# ==============================
# CLI
# ==============================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gemini 2.5 Flash SFT trainer (patient / doctor)."
    )
    p.add_argument(
        "--task",
        choices=["patient", "doctor"],
        required=True,
        help="patient: NAIST 患者表現, doctor: 万病医師表現",
    )
    p.add_argument(
        "--naist-jsonl",
        default="naist_patient_expression_sft_single.jsonl",
        help="NAIST 患者表現 JSONL のパス (task=patient のとき)",
    )
    p.add_argument(
        "--manbyo-jsonl",
        default="manbyo_doctor_sft_SAB_rank1.jsonl",
        help="万病 医師表現 JSONL のパス (task=doctor のとき)",
    )
    p.add_argument(
        "--base-model",
        default="models/gemini-2.5-flash",
        help=(
            "SFT 元のモデル名。Gemini 2.5 Flash 系を推奨 "
            "(例: models/gemini-2.5-flash)"
        ),
    )
    p.add_argument(
        "--model-id",
        default=None,
        help="tunedModels/<MODEL_ID> の MODEL_ID 部分。省略時は自動生成。",
    )
    p.add_argument(
        "--epoch-count", type=int, default=5, help="学習エポック数"
    )
    p.add_argument(
        "--batch-size", type=int, default=32, help="バッチサイズ (最大 64 程度を推奨)"
    )
    p.add_argument(
        "--learning-rate", type=float, default=0.001, help="学習率"
    )
    p.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="デバッグ用: 学習に使う最大サンプル数 (None で全件)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ==========================
    # API キー設定 (公式推奨)
    # ==========================
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("環境変数 GEMINI_API_KEY または GOOGLE_API_KEY を設定してください。")

    genai.configure(api_key=api_key)

    # 既存の tuned model 一覧を少しだけ表示 (デバッグ用)
    try:
        print("=== Existing tuned models (up to 5) ===")
        for i, m in zip(range(5), genai.list_tuned_models()):
            print(" -", m.name, getattr(m, "display_name", None))
    except Exception as e:
        print("list_tuned_models failed (続行します):", e)

    # ==========================
    # データ読み込み
    # ==========================
    if args.task == "patient":
        print("== Load NAIST patient SFT data from:", args.naist_jsonl)
        training_data = load_naist_patient_jsonl(args.naist_jsonl, max_examples=args.max_examples)
        default_id_prefix = "gemini25-patient-naist"
    else:
        print("== Load Manbyo doctor SFT data from:", args.manbyo_jsonl)
        training_data = load_manbyo_doctor_jsonl(args.manbyo_jsonl, max_examples=args.max_examples)
        default_id_prefix = "gemini25-doctor-manbyo"

    if not training_data:
        raise RuntimeError("training_data が空です。入力ファイルとフォーマットを確認してください。")

    print(f"Loaded training examples: {len(training_data)}")

    # ==========================
    # model_id の決定
    # ==========================
    if args.model_id:
        model_id = args.model_id
    else:
        # 公式サンプルと同様にランダムな suffix を付ける
        suffix = random.randint(0, 10000)
        model_id = f"{default_id_prefix}-{suffix}"

    print("Base model :", args.base_model)
    print("Tuned ID   :", model_id)

    # ==========================
    # SFT ジョブ作成
    # ==========================
    print("== Create tuning job ==")
    op = create_tuned_model(
        base_model=args.base_model,
        training_data=training_data,
        model_id=model_id,
        epoch_count=args.epoch_count,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    print("Operation:", op)

    # ==========================
    # 完了待ち
    # ==========================
    print("== Wait for tuning job to finish ==")
    result = wait_tuned_model(op, model_id=model_id)

    # ==========================
    # 最終モデル名の出力
    # ==========================
    tuned_model_name = f"tunedModels/{model_id}"
    print("===================================")
    print("Tuned model is ready!")
    print("  name:", tuned_model_name)
    print("この名前を gemini_pipeline_v2.py の --gen-model に渡してください。")


if __name__ == "__main__":
    main()
