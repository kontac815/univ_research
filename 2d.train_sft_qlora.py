#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2d.train_sft_qlora.py (stable)

QLoRA + (optional DeepSpeed) SFT training script.
- JSONL input: each line is a sample:
{
  "messages": [{"role":"system|user|assistant","content":"..."} ...],
  "output": "target assistant text"
}

Key stability features:
- remove_unused_columns=False (prevents KeyError for messages/output)
- chat_template auto-fallback (for models without tokenizer.chat_template)
  - strongly recommends Swallow Instruct variant
- safe k-bit preparation:
  - tries prepare_model_for_kbit_training
  - if CUDA OOM occurs during preparation, falls back to minimal safe setup
- assistant-only loss masking
- DS config mismatch avoided by using DS "auto" config (recommended)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import time
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)


# -------------------------
# Args
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # IO
    p.add_argument("--model-name-or-path", type=str, required=True)
    p.add_argument("--train-files", type=str, nargs="+", required=True)
    p.add_argument("--output-dir", type=str, required=True)

    # Train
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--num-epochs", type=float, default=2.0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--save-strategy", type=str, default="epoch")
    p.add_argument("--seed", type=int, default=42)

    # Precision / DS
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--deepspeed", type=str, default=None)
    p.add_argument("--local_rank", type=int, default=-1)
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb-project", type=str, default="HPO_SFT")
    p.add_argument("--wandb-tags", type=str, default="")
    p.add_argument("--wandb-run-name", type=str, default="")
    p.add_argument("--wandb-group", type=str, default="")
    p.add_argument("--wandb-entity", type=str, default="")
    # Attention impl
    p.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "eager"],
        help="Use sdpa if available; eager is a safe fallback.",
    )

    # Dataloader
    p.add_argument("--dataloader-num-workers", type=int, default=4)
    p.add_argument("--no-pin-memory", action="store_true")
    p.add_argument("--no-persistent-workers", action="store_true")

    # QLoRA / Bitsandbytes
    p.add_argument("--load-in-4bit", action="store_true", default=True)
    p.add_argument("--bnb-4bit-quant-type", type=str, default="nf4", choices=["nf4", "fp4"])
    p.add_argument("--bnb-4bit-use-double-quant", action="store_true")
    p.add_argument("--bnb-4bit-compute-dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-target-modules",
        type=str,
        default="auto",
        help="Comma-separated. Use 'auto' for Llama-family defaults.",
    )

    # Stability knobs
    p.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable grad checkpointing (faster, uses more VRAM).",
    )
    p.add_argument(
        "--skip-prepare-kbit",
        action="store_true",
        help="Skip prepare_model_for_kbit_training (use if it OOMs on small GPU).",
    )

    return p.parse_args()


# -------------------------
# Dataset
# -------------------------
class JsonlChatSFTDataset(Dataset):
    def __init__(self, jsonl_paths: List[Path]):
        self.samples: List[Dict[str, Any]] = []
        bad = 0
        for path in jsonl_paths:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ex = json.loads(line)
                        if "messages" not in ex or "output" not in ex:
                            bad += 1
                            continue
                        # minimal validation
                        if not isinstance(ex["messages"], list):
                            bad += 1
                            continue
                        self.samples.append(ex)
                    except Exception:
                        bad += 1
                        continue
        print(f"[Dataset] loaded: {len(self.samples)} samples (skipped bad={bad})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        ex = self.samples[i]
        return {
            "messages": ex["messages"],
            "output": str(ex["output"]).strip(),
        }


# -------------------------
# Tokenizer chat template fallback
# -------------------------
def ensure_chat_template(tok: AutoTokenizer, model_name: str) -> None:
    """
    If tokenizer has no chat_template, inject a Llama3-style template.
    This keeps training code stable across base models.
    For Swallow, strongly recommends Instruct variant.
    """
    if getattr(tok, "chat_template", None):
        return

    # Strong warning for Swallow base (non-instruct)
    if "Swallow" in model_name and "Instruct" not in model_name:
        print(
            "[WARN] You are using a Swallow *base* model without an official chat_template.\n"
            "       For fewer accidents, strongly recommended:\n"
            "       tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5\n"
            "       Continuing by injecting a Llama3-style template..."
        )
    else:
        print("[WARN] tokenizer.chat_template is missing; injecting a Llama3-style chat template.")

    # Llama3-style Jinja2 template (works for system/user/assistant roles)
    # Note: This requires tokenizer vocab to contain these special tokens.
    # Most Llama3-family models do.
    tok.chat_template = (
        "{% set bos_token = '<|begin_of_text|>' %}"
        "{% set eot_token = '<|eot_id|>' %}"
        "{{ bos_token }}"
        "{% for message in messages %}"
        "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n"
        "{{ message['content'] }}{{ eot_token }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "{% endif %}"
    )


# -------------------------
# Collator: assistant-only loss
# -------------------------
class ChatSFTDataCollator:
    """
    - prompt: messages + generation prompt (assistant header)
    - full: messages + assistant(output)
    - labels are masked to compute loss only on assistant output tokens
    """
    def __init__(self, tokenizer: AutoTokenizer, max_length: int):
        self.tok = tokenizer
        self.max_length = max_length

        if self.tok.pad_token is None:
            # safe default for causal LM
            self.tok.pad_token = self.tok.eos_token

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        attn_list = []
        labels_list = []

        for ex in batch:
            messages = ex["messages"]
            output = ex["output"]

            # prompt: add_generation_prompt=True -> ends with assistant header
            prompt_text = self.tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

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

            prompt_len = len(prompt_tok["input_ids"])
            labels = input_ids.copy()
            # mask prompt tokens
            for j in range(min(prompt_len, len(labels))):
                labels[j] = -100

            input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
            attn_list.append(torch.tensor(attention_mask, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.tok.pad_token_id,
        )
        attn_padded = torch.nn.utils.rnn.pad_sequence(
            attn_list,
            batch_first=True,
            padding_value=0,
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=-100,
        )

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attn_padded,
            "labels": labels_padded,
        }


# -------------------------
# LoRA target modules
# -------------------------
def infer_lora_target_modules(model) -> List[str]:
    """
    For Llama-family, these names are standard.
    If your model differs, pass --lora-target-modules explicitly.
    """
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_target_modules(arg: str, model) -> List[str]:
    if arg.strip().lower() == "auto":
        return infer_lora_target_modules(model)
    mods = [x.strip() for x in arg.split(",") if x.strip()]
    if not mods:
        return infer_lora_target_modules(model)
    return mods


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    if args.wandb:
        if args.wandb_project:
            os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
        if args.wandb_tags:
            os.environ["WANDB_TAGS"] = args.wandb_tags
        if args.wandb_group:
            os.environ["WANDB_RUN_GROUP"] = args.wandb_group
        if args.wandb_run_name:
            os.environ["WANDB_NAME"] = args.wandb_run_name
        else:
            os.environ["WANDB_NAME"] = f"sft_{Path(args.output_dir).name}_{int(time.time())}"
        os.environ.setdefault("WANDB__SERVICE_WAIT", "300")
        os.environ.setdefault("WANDB_CONSOLE", "wrap")

    # Repro / performance knobs
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # dtype selection
    use_bf16 = args.bf16
    use_fp16 = args.fp16
    if use_bf16 and use_fp16:
        raise ValueError("bf16 と fp16 は同時に指定できません。")
    if (not use_bf16) and (not use_fp16):
        use_bf16 = True  # default

    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16
    compute_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.bnb_4bit_compute_dtype]

    model_name = args.model_name_or_path
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== tokenizer: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Ensure chat template exists
    ensure_chat_template(tok, model_name)

    # pad token safety
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Quantization config
    # NOTE: bitsandbytes must be installed in your env.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # Device map for single-process (DeepSpeed will handle distributed placement)
    device_map = None
    if args.deepspeed is None:
        # Non-DS single GPU / accelerate-less usage
        device_map = "auto"

    print("== load model (4bit) ==")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
        device_map=device_map,
    )

    # Make sure pad_token_id exists in config
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id

    # Gradient checkpointing (activation mem saver)
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # Prepare for k-bit training (LayerNorm fp32 etc.)
    if not args.skip_prepare_kbit:
        try:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=(not args.no_gradient_checkpointing),
            )
        except torch.cuda.OutOfMemoryError as e:
            print("[WARN] prepare_model_for_kbit_training() caused CUDA OOM. Falling back to minimal setup.")
            print(f"       OOM detail: {e}")
            # Minimal fallback: keep checkpointing, enable input grads for LoRA
            model.enable_input_require_grads()
            model.config.use_cache = False

    # LoRA
    target_modules = parse_target_modules(args.lora_target_modules, model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)

    # Print trainable params
    trainable = 0
    total = 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100*trainable/total:.4f}")

    # Dataset / collator
    train_paths = [Path(p) for p in args.train_files]
    dataset = JsonlChatSFTDataset(train_paths)
    collator = ChatSFTDataCollator(tok, max_length=args.max_length)

    # TrainingArguments
    train_args = TrainingArguments(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        report_to=(["wandb"] if args.wandb else []),
        run_name=(os.environ.get("WANDB_NAME") if args.wandb else None),
        bf16=use_bf16,
        fp16=use_fp16,
        deepspeed=args.deepspeed,
        remove_unused_columns=False,  # ★ critical for messages/output with custom collator
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=(not args.no_pin_memory),
        dataloader_persistent_workers=(not args.no_persistent_workers),
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tok,  # warning in transformers>=5.0 is fine for now
    )

    trainer.train()

    # Save adapter + tokenizer
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"== saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
