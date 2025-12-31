#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NAIST由来の「標準病名 → 患者表現1つ」SFTデータで
EQUES/MedLLama3-JP-v2 に LoRA チューニングするスクリプト（1個版）。

MedLLama3-JP-v2 は Llama3 系マージモデルなので、
Swallow と同様に
  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
を LoRA 対象とする。
"""

from __future__ import annotations
import json
from pathlib import Path

import os
os.environ["HTTP_PROXY"] = "http://proxy.l2.med.tohoku.ac.jp:8080"
os.environ["HTTPS_PROXY"] = "http://proxy.l2.med.tohoku.ac.jp:8080"

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from auth import *  # 不要なら削除

# ==============
# 設定
# ==============
BASE_DIR = Path("/home/kiso_user/Documents/workspace/Research/data")

DATA_PATH = BASE_DIR / "naist_patient_expression_sft_single.jsonl"
BASE_MODEL = "EQUES/MedLLama3-JP-v2"
OUTPUT_DIR = BASE_DIR / "eques_patient_expression_lora"  # EQUES用 LoRA 保存先

MAX_SEQ_LEN = 512
BATCH_SIZE = 1
GRAD_ACCUM = 8
EPOCHS = 1

# 4bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


# ==============
# データ読み込み
# ==============
def load_jsonl(path: Path) -> Dataset:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return Dataset.from_list(records)


def build_formatting_func(tokenizer):

    def formatting_func(example):
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return text

    return formatting_func


def main():
    print("==== Loading dataset ====")
    dataset = load_jsonl(DATA_PATH)

    print("==== Loading tokenizer ====")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    formatting_func = build_formatting_func(tokenizer)

    print("==== Formatting dataset ====")
    dataset = dataset.map(lambda x: {"text": formatting_func(x)})

    print("==== Loading base model ====")
    max_memory = {
        0: "22GiB",  # RTX3090
        1: "9GiB",   # RTX3080
    }
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
    )
    model.config.use_cache = False

    print("==== Setting LoRA ====")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    print("==== SFT Config ====")
    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        logging_steps=20,
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=True,
        gradient_checkpointing=True,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
    )

    print("==== Trainer ====")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        peft_config=lora_config,
    )

    print("==== Training ====")
    trainer.train()

    print("==== Saving Model ====")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Done!")


if __name__ == "__main__":
    main()
