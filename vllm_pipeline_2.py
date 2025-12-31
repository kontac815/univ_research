# -*- coding: utf-8 -*-
"""
Unified Pattern B パイプライン (患者 / 医師 両対応) - vLLM Version
生成のみ (Judge/Refine なし)

- vllm_pipeline.py をベースに、Judge/Refine ループを削除
- 1c.build_hpo_sft_from_naist_manbyo_fullft.py と同じ「定義あり/なしでプロンプト切替」を反映
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# --- vLLM Imports ---
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError:
    print("[ERROR] vLLM not found. Please install it with `pip install vllm`.")
    raise

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# -------------------------
# Data classes
# -------------------------


@dataclass
class HpoItem:
    hpo_id: str
    name_ja: str
    name_en: str
    category: str
    source: str
    definition: str = ""


@dataclass
class ExpressionRecord:
    hpo_id: str
    hpo_name_ja: str
    hpo_name_en: str
    category: str
    source: str
    original_expression: str
    current_expression: str


@dataclass
class HPOGroup:
    hpo_id: str
    hpo_name_ja: str
    hpo_name_en: str
    category: str
    source: str
    definition: str = ""
    expressions: List[ExpressionRecord] = field(default_factory=list)


# -------------------------
# Utils
# -------------------------


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)


def split_by_shard(items: List[Any], num_shards: int, shard_id: int) -> List[Any]:
    if num_shards <= 1:
        return items
    return [x for i, x in enumerate(items) if i % num_shards == shard_id]


# -------------------------
# Postprocess (既存を継承)
# -------------------------


def postprocess_patient_expression(text: str) -> str:
    if not text:
        return ""
    s = text.strip()
    s = s.splitlines()[0].strip()
    # Drop anything after the first "assistant" token (LoRA世代で混入しがち)
    s = re.split(r"assistant[:：]?", s, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    s = re.sub(r"\s*assistant[:：]?\s*$", "", s, flags=re.IGNORECASE)
    for prefix in ["患者:", "患者「", "患者｢", "「", "『"]:
        if s.startswith(prefix):
            s = s[len(prefix) :].lstrip()
    for suf in ["」", "『", "』", "｡"]:
        if s.endswith(suf):
            s = s[:-1].rstrip()
    return s


def postprocess_doctor_expression(text: str) -> str:
    if not text:
        return ""
    s = text.strip()
    s = s.splitlines()[0].strip()
    # Drop anything after the first "assistant" token (LoRA世代で混入しがち)
    s = re.split(r"assistant[:：]?", s, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    s = re.sub(r"\s*assistant[:：]?\s*$", "", s, flags=re.IGNORECASE)
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


# -------------------------
# Prompt builders (1c と揃える)
# -------------------------


def build_generate_messages_patient(symptom_name: str, definition_ja: Optional[str]) -> List[Dict[str, str]]:
    """
    患者表現生成用 messages。
    definition_ja が None/空なら「定義:」ブロックなし。
    """
    definition_ja = (definition_ja or "").strip()

    system_msg = (
        "あなたは日本語の医療面接に精通したAIアシスタントです。"
        "以下に与えるHPOの情報（症状名と、あればその定義）にもとづき、"
        "患者が診察室で医師に訴えそうな日本語の発言を1つだけ生成してください。"
        "制約:"
        " - 出力は患者の発言そのもののみとし、説明文・コメント・番号付けは一切行わないこと。"
        " - できるだけ短く簡潔にし、1文またはそれより短い表現にすること。"
        " - 「〜です／〜ます」よりも自然な話し言葉（〜て、〜なんです、など）を優先すること。"
        " - 疾患名や専門用語はなるべく避け、日常的な日本語で症状を表現すること。"
        " - 与えられた症状名の語をそのまま繰り返さず、患者が使いそうな言い換えを用いること。"
        " - 不必要に複数の症状を詰め込まず、中心となる症状を1つに絞ること。"
    )

    if definition_ja:
        user_msg = (
            f"症状名: {symptom_name}\n\n"
            f"定義:\n{definition_ja}\n\n"
            "患者が医師に話すときの短い発言を1つだけ、日本語で出力してください。"
        )
    else:
        user_msg = (
            f"症状名: {symptom_name}\n\n"
            "患者が医師に話すときの短い発言を1つだけ、日本語で出力してください。"
        )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def build_generate_messages_doctor(symptom_name: str, definition_ja: Optional[str]) -> List[Dict[str, str]]:
    """
    医師表現生成用 messages。
    definition_ja が None/空なら「定義:」ブロックなし。
    """
    definition_ja = (definition_ja or "").strip()

    system_msg = (
        "あなたは日本語の医療用語と電子カルテ記載に精通した日本人医師です。"
        "以下に与えるHPOの情報（症状名と、あればその定義）にもとづき、"
        "医師が電子カルテに記載しそうな日本語の表現を1つだけ生成してください。"
        "制約:"
        " - 出力はカルテに記載する1つの表現のみとし、説明文・コメント・箇条書き・番号付けは一切行わないこと。"
        " - できるだけ簡潔に、通常のカルテ記載を意識した短い語句またはごく短い文にすること。"
        " - 患者の話し言葉ではなく、医師が用いる標準的な専門用語・略語を使用してよい。"
        " - ただし意味が過度に抽象的にならないようにし、臨床的な情報が伝わる表現にすること。"
        " - 与えられた症状名の語をそのまま繰り返すのではなく、診療録として自然な形に整えること。"
        " - 不必要に多くの情報を詰め込まず、中心となる症状・所見・病態にフォーカスすること。"
    )

    if definition_ja:
        user_msg = (
            f"症状名: {symptom_name}\n\n"
            f"定義:\n{definition_ja}\n\n"
            "医師が電子カルテに記載する1つの表現だけを、日本語で出力してください。"
        )
    else:
        user_msg = (
            f"症状名: {symptom_name}\n\n"
            "医師が電子カルテに記載する1つの表現だけを、日本語で出力してください。"
        )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# -------------------------
# vLLM batch runner (既存を継承)
# -------------------------


def run_vllm_batch(
    llm: LLM,
    prompts: List[str],
    *,
    max_tokens: int,
    temperature: float,
    top_p: float = 0.9,
    stop_tokens: Optional[List[str]] = None,
    lora_path: Optional[str] = None,
    lora_name: str = "adapter",
) -> List[str]:
    if not prompts:
        return []

    # Stop tokens のデフォルト (Llama 3 用)
    if stop_tokens is None:
        stop_tokens = ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>"]

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop_tokens,
    )

    # temperature=0 のような決定論
    if temperature < 1e-5:
        sampling_params.temperature = 0
        sampling_params.top_p = 1.0

    lora_req = None
    if lora_path:
        lora_req = LoRARequest(lora_name, 1, lora_path)

    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=lora_req,
        use_tqdm=False,
    )

    results: List[str] = []
    for output in outputs:
        results.append(output.outputs[0].text)
    return results


# -------------------------
# Diversity selector (Judgeなし版)
# -------------------------


def select_diverse_top_k_texts(
    texts: List[str],
    k: int,
    sim_thresh_low: float = 0.7,
) -> List[str]:
    """
    似すぎる表現を避けつつ k 個選ぶ。
    - スコアが無いので、短いものを優先して選択
    """
    import difflib

    uniq: List[str] = []
    seen = set()
    for t in texts:
        t2 = (t or "").strip()
        if not t2:
            continue
        if t2 in seen:
            continue
        seen.add(t2)
        uniq.append(t2)

    if len(uniq) <= k:
        return uniq

    uniq_sorted = sorted(uniq, key=lambda s: (len(s), s))

    selected: List[str] = []
    for cand in uniq_sorted:
        if not selected:
            selected.append(cand)
            if len(selected) >= k:
                break
            continue

        sims = [difflib.SequenceMatcher(None, cand, x).ratio() for x in selected]
        if max(sims) < sim_thresh_low:
            selected.append(cand)
        if len(selected) >= k:
            break

    # 足りない場合は埋める（類似度条件を無視してでも埋める）
    if len(selected) < k:
        for cand in uniq_sorted:
            if cand not in selected:
                selected.append(cand)
            if len(selected) >= k:
                break

    return selected[:k]


# -------------------------
# Load HPO CSV (既存を継承)
# -------------------------


def load_hpo_csv(path: Path, default_source: str = "doctor") -> List[HpoItem]:
    """
    HPO master CSV を読み込む。
    期待カラム (あるものを使う):
      - HPO_ID
      - jp_final / name_ja / label_ja
      - name_en / label_en
      - category_v2 / category
      - source
      - definition_ja / definition
    """
    import csv

    items: List[HpoItem] = []
    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hpo_id = safe_str(row.get("HPO_ID", "")).strip()
            if not hpo_id:
                continue

            name_ja = ""
            for key in ["jp_final", "jp_label", "name_ja", "HPO_name_ja", "ja_2023", "ja_2017_expert"]:
                if key in row and safe_str(row.get(key, "")).strip():
                    name_ja = safe_str(row.get(key, "")).strip()
                    break

            name_en = safe_str(row.get("name_en", "")).strip()
            cat = safe_str(row.get("category_v2", "")).strip() or safe_str(row.get("category", "Unknown")).strip()
            src = safe_str(row.get("source", "")).strip() or default_source

            # definition: definition_ja 優先、なければ definition
            definition = ""
            def_ja = safe_str(row.get("definition_ja", "")).strip()
            def_en = safe_str(row.get("definition", "")).strip()
            if def_ja:
                definition = def_ja
            elif def_en:
                definition = def_en

            items.append(HpoItem(hpo_id, name_ja, name_en, cat, src, definition))
    return items


# -------------------------
# Main processing (生成のみ)
# -------------------------


def process_all_groups_vllm(
    llm: LLM,
    tokenizer: AutoTokenizer,
    mode: str,
    groups: List[HPOGroup],
    args: argparse.Namespace,
    fout: Any,
    wandb_run: Any = None,
) -> None:
    """
    vLLM を用いて全HPOグループを一括処理（生成のみ）
    """
    doc_per_hpo = args.per_hpo if args.per_hpo > 0 else args.target_good
    if doc_per_hpo <= 0:
        raise ValueError("per-hpo / target-good が 0 以下です。")

    # 初期化
    for hpo in groups:
        hpo.expressions = []

    # プロンプト作成（HPOごとに1つ、候補数分複製）
    gen_prompts: List[str] = []
    gen_indices: List[Tuple[int, int]] = []  # (group_idx, expr_idx)

    for g_idx, hpo in enumerate(groups):
        symptom_name = hpo.hpo_name_ja or hpo.hpo_name_en or hpo.hpo_id
        definition = hpo.definition or ""
        msgs = (
            build_generate_messages_patient(symptom_name, definition)
            if mode == "patient"
            else build_generate_messages_doctor(symptom_name, definition)
        )

        # tokenizer に chat_template が無い場合のフォールバック
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in msgs]) + "\nASSISTANT:"

        for e_idx in range(doc_per_hpo):
            gen_prompts.append(prompt)
            gen_indices.append((g_idx, e_idx))

    # 一括生成
    gen_outputs = run_vllm_batch(
        llm,
        gen_prompts,
        max_tokens=args.gen_max_new_tokens,
        temperature=0.9,
        top_p=0.9,
        lora_path=args.gen_lora_path,
        lora_name="gen_adapter",
    )

    # 代入 + 後処理
    pbar = tqdm(total=len(groups), desc=f"vLLM {mode} generate-only")
    touched = [False] * len(groups)

    for (g_idx, _e_idx), out_text in zip(gen_indices, gen_outputs):
        hpo = groups[g_idx]
        orig = safe_str(out_text)

        final = postprocess_patient_expression(orig) if mode == "patient" else postprocess_doctor_expression(orig)

        hpo.expressions.append(
            ExpressionRecord(
                hpo_id=hpo.hpo_id,
                hpo_name_ja=hpo.hpo_name_ja,
                hpo_name_en=hpo.hpo_name_en,
                category=hpo.category,
                source=hpo.source,
                original_expression=orig,
                current_expression=final,
            )
        )

        if not touched[g_idx] and len(hpo.expressions) >= doc_per_hpo:
            touched[g_idx] = True
            pbar.update(1)

    pbar.close()

    # 書き出し（既存キーを維持）
    orig_key = "patient_expression_original" if mode == "patient" else "doctor_expression_original"
    final_key = "patient_expression_final" if mode == "patient" else "doctor_expression_final"

    for hpo in groups:
        finals = [er.current_expression for er in hpo.expressions]
        selected_texts = select_diverse_top_k_texts(
            finals,
            k=doc_per_hpo,
            sim_thresh_low=args.diversity_sim_low,
        )

        final_to_orig: Dict[str, str] = {}
        for er in hpo.expressions:
            if er.current_expression and er.current_expression not in final_to_orig:
                final_to_orig[er.current_expression] = er.original_expression

        for t in selected_texts:
            rec = {
                "HPO_ID": hpo.hpo_id,
                "HPO_name_ja": hpo.hpo_name_ja,
                "HPO_name_en": hpo.hpo_name_en,
                "category": hpo.category,
                "source": hpo.source,
                "definition": hpo.definition,
                orig_key: final_to_orig.get(t, ""),
                final_key: t,
                "gen_model": args.gen_model,
                "gen_lora_path": args.gen_lora_path,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if wandb_run:
        wandb_run.log(
            {
                "num_groups": len(groups),
                "per_hpo": doc_per_hpo,
                "total_generated": sum(len(h.expressions) for h in groups),
            }
        )
        wandb_run.summary["mode"] = mode
        wandb_run.summary["output_path"] = str(args.output)


# -------------------------
# Entry
# -------------------------


def main() -> None:
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument("--mode", type=str, required=True, choices=["patient", "doctor"])
    parser.add_argument("--hpo-csv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--gen-model", type=str, required=True)

    # Optional (互換性のため残す)
    parser.add_argument("--gen-lora-path", type=str, default=None)
    parser.add_argument("--judge-model", type=str, default=None, help="(互換) generate-only では未使用")
    parser.add_argument("--refine-model", type=str, default=None, help="(互換) generate-only では未使用")
    parser.add_argument("--refine-lora-path", type=str, default=None, help="(互換) generate-only では未使用")

    # Generation controls
    parser.add_argument("--per-hpo", type=int, default=5)
    parser.add_argument("--target-good", type=int, default=3, help="(互換) per-hpo 未指定時の生成数扱い")
    parser.add_argument("--gen-max-new-tokens", type=int, default=64)

    # Kept for compatibility (unused in generate-only)
    parser.add_argument("--judge-max-new-tokens", type=int, default=128, help="(互換) 未使用")
    parser.add_argument("--refine-max-new-tokens", type=int, default=64, help="(互換) 未使用")
    parser.add_argument("--min-overall", type=int, default=4, help="(互換) 未使用")
    parser.add_argument("--min-match", type=int, default=3, help="(互換) 未使用")
    parser.add_argument("--min-simplicity", type=int, default=3, help="(互換) 未使用")
    parser.add_argument("--max-rounds", type=int, default=3, help="(互換) 未使用")

    # Diversity
    parser.add_argument("--diversity-sim-high", type=float, default=0.9, help="(互換) 未使用")
    parser.add_argument("--diversity-sim-low", type=float, default=0.7)

    # Logging / tracking (互換性のため残すが未使用)
    parser.add_argument("--log-each-round", action="store_true", help="(互換) 未使用")
    parser.add_argument("--round-log-prefix", type=str, default=None, help="(互換) 未使用")
    parser.add_argument("--wandb", action="store_true", help="(互換) 未使用")
    parser.add_argument("--wandb-project", type=str, default="HPO_gen_only")
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)

    # Sharding
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)

    parser.add_argument("--batch-across-hpo", action="store_true", help="(互換) vLLM は常にまとめて生成")

    args = parser.parse_args()

    set_seed(42)

    if args.wandb and HAS_WANDB:
        wandb_run = wandb.init(
            project=args.wandb_project,
            tags=args.wandb_tags.split(",") if args.wandb_tags else [],
            config=vars(args),
            name=args.wandb_run_name,
            group=args.wandb_group,
        )
    else:
        wandb_run = None

    # vLLM quantization auto-detect
    quant_method = None
    lower = args.gen_model.lower()
    if "awq" in lower:
        quant_method = "awq"
        print(" -> Auto-detected AWQ model.")
    elif "gptq" in lower:
        quant_method = "gptq"

    # vLLM engine
    llm = LLM(
        model=args.gen_model,
        tokenizer=args.gen_model,
        quantization=quant_method,
        dtype="float16",
        enable_lora=True,
        max_lora_rank=64,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=4096,
        enforce_eager=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.gen_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load HPO items -> groups
    hpo_items = load_hpo_csv(Path(args.hpo_csv), default_source=("doctor" if args.mode == "doctor" else "patient"))
    groups_all = [
        HPOGroup(
            hpo_id=item.hpo_id,
            hpo_name_ja=item.name_ja,
            hpo_name_en=item.name_en,
            category=item.category,
            source=item.source,
            definition=item.definition,
        )
        for item in hpo_items
    ]

    groups = split_by_shard(groups_all, args.num_shards, args.shard_id)
    print(
        f"Loaded HPO groups: {len(groups)} (Total: {len(groups_all)}, "
        f"Mode: {args.mode}, Shard: {args.shard_id}/{args.num_shards})"
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out_path.open("w", encoding="utf-8") as fout:
            process_all_groups_vllm(llm, tokenizer, args.mode, groups, args, fout, wandb_run=wandb_run)

        print(f"Saved results to {args.output}")
    finally:
        if wandb_run:
            wandb.finish()


if __name__ == "__main__":
    main()
