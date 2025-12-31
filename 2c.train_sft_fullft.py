#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
2c.train_sft_fullft.py

DeepSpeed + full fine-tuning 用 SFT 学習スクリプト（assistant-only loss 対応）

入力 JSONL（複数可）: 1 行 1 サンプル
{
  "messages": [
    {"role":"system","content":"..."},
    {"role":"user","content":"..."}
  ],
  "output":"ターゲット表現",
  "task":"patient_expression" or "doctor_expression",
  ...
}

学習:
- messages + assistant(output) を 1つの会話として学習
- loss は assistant 部分だけにかける（system/user 部分は -100 マスク）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import wandb
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name-or-path", type=str, required=True)
    p.add_argument("--train-files", type=str, nargs="+", required=True)
    p.add_argument("--output-dir", type=str, required=True)

    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--num-epochs", type=float, default=2.0)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--save-strategy", type=str, default="epoch")
    p.add_argument("--deepspeed", type=str, default=None)

    p.add_argument("--bf16", action="store_true", help="bf16 を使う（推奨、だめなら fp16 にする）")
    p.add_argument("--fp16", action="store_true", help="fp16 を使う（bf16 が無理な場合）")
    p.add_argument("--local_rank", type=int, default=-1)
    return p.parse_args()


class JsonlChatSFTDataset(Dataset):
    def __init__(self, jsonl_paths: List[Path]):
        self.samples: List[Dict[str, Any]] = []
        for path in jsonl_paths:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self.samples.append(json.loads(line))
        print(f"[Dataset] loaded: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        ex = self.samples[i]
        # 必須キー
        messages = ex["messages"]
        output = str(ex["output"]).strip()
        return {"messages": messages, "output": output}


class ChatSFTDataCollator:
    """
    - chat_template を使って (system+user) の prompt 部分を token 化
    - 同じ template で (system+user+assistant) の full を token 化
    - full の labels を作り、prompt 部分までは -100 にする
    """

    def __init__(self, tokenizer: AutoTokenizer, max_length: int):
        self.tok = tokenizer
        self.max_length = max_length

        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        attn_list = []
        labels_list = []

        for ex in batch:
            messages = ex["messages"]
            output = ex["output"]

            # 1) prompt 部分（assistant 生成開始位置まで）
            # add_generation_prompt=True を使うと、
            # 末尾に assistant の開始トークンが含まれる状態になる（モデルが次を生成する位置）
            prompt_text = self.tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # 2) full（assistant 出力まで含める）
            full_messages = messages + [{"role": "assistant", "content": output}]
            full_text = self.tok.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            prompt_tok = self.tok(
                prompt_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None,
            )
            full_tok = self.tok(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None,
            )

            input_ids = full_tok["input_ids"]
            attention_mask = full_tok["attention_mask"]

            # prompt の token 長（full も同じ tokenizer なので先頭一致する）
            prompt_len = len(prompt_tok["input_ids"])
            # labels: full と同じだが prompt までは loss を外す
            labels = input_ids.copy()
            for j in range(min(prompt_len, len(labels))):
                labels[j] = -100

            input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
            attn_list.append(torch.tensor(attention_mask, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        # pad to max in batch
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tok.pad_token_id
        )
        attn_padded = torch.nn.utils.rnn.pad_sequence(
            attn_list, batch_first=True, padding_value=0
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attn_padded,
            "labels": labels_padded,
        }


def main():
    args = parse_args()

    model_name = args.model_name_or_path
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== tokenizer: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # dtype 選択
    use_bf16 = args.bf16
    use_fp16 = args.fp16
    if use_bf16 and use_fp16:
        raise ValueError("bf16 と fp16 は同時に指定できません。")
    if (not use_bf16) and (not use_fp16):
        # デフォルトは bf16（RTX30xxなら基本OK）
        use_bf16 = True

    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"== model (full FT): {model_name} dtype={torch_dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="sdpa",
        torch_dtype=torch_dtype,
    )

    # メモリ削減
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # checkpointing と相性問題回避

    train_paths = [Path(p) for p in args.train_files]
    dataset = JsonlChatSFTDataset(train_paths)

    collator = ChatSFTDataCollator(tok, max_length=args.max_length)

    train_args = TrainingArguments(
        output_dir=str(out_dir),
        optim="adafactor",
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        report_to=[],
        bf16=use_bf16,
        fp16=use_fp16,
        deepspeed=args.deepspeed,
        dataloader_num_workers=2,      # ← まず0推奨（デバッグしやすい）
        remove_unused_columns=False, 
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tok,
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"== saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
