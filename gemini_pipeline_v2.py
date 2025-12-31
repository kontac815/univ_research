# -*- coding: utf-8 -*-

"""
Unified Pattern B パイプライン (患者 / 医師 両対応) - Gemini 2.5 Flash Version
Uses the new 'google-genai' SDK recommended for Gemini 2.0/2.5 series.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm.asyncio import tqdm_asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

# --- New Google Gen AI SDK ---
from google import genai
from google.genai import types

from json import JSONDecoder

_DECODER = JSONDecoder()

# ==========================================
# Data Structures (Inherited)
# ==========================================

@dataclass
class HpoItem:
    hpo_id: str
    name_ja: str
    name_en: str
    category: str
    source: str
    definition: str = ""

@dataclass
class JudgeResult:
    match_score: int
    simplicity_score: int
    overall_score: int
    too_technical: bool
    comment: str = ""
    raw_output: str = ""
    parse_error: Optional[str] = None
    round: int = 0

@dataclass
class ExpressionRecord:
    hpo_id: str
    hpo_name_ja: str
    hpo_name_en: str
    category: str
    source: str
    original_expression: str
    current_expression: str
    judge: Optional[JudgeResult] = None
    refine_round: int = 0

@dataclass
class HPOGroup:
    hpo_id: str
    hpo_name_ja: str
    hpo_name_en: str
    category: str
    source: str
    definition: str = ""
    expressions: List[ExpressionRecord] = field(default_factory=list)


# ==========================================
# Helper Functions (Inherited)
# ==========================================

def set_seed(seed: int = 42):
    random.seed(seed)

def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def split_by_shard(items: List[Any], num_shards: int, shard_id: int) -> List[Any]:
    if num_shards <= 1:
        return items
    return [x for i, x in enumerate(items) if i % num_shards == shard_id]

def _extract_last_json_from_text(text: str) -> Dict[str, Any]:
    s = text.strip()
    try:
        return json.loads(s)
    except:
        pass
    match = re.search(r"```json(.*?)```", s, re.DOTALL)
    if match:
        try:
            return _DECODER.decode(match.group(1).strip())
        except:
            pass
    # Try finding the last brace pair
    last_open = s.find("{")
    last_close = s.rfind("}")
    if last_open == -1 or last_close == -1 or last_close < last_open:
        # 2.5 Flash with JSON mode might return just the JSON without markdown, handled by json.loads
        raise ValueError("JSON が見つかりません")
    fragment = s[last_open:last_close+1]
    return _DECODER.decode(fragment)

def parse_judge_output(raw: str, round_idx: int = 0) -> JudgeResult:
    try:
        data = _extract_last_json_from_text(raw)
        match_score = safe_int(data.get("match_score", 0))
        simplicity_score = safe_int(data.get("simplicity_score", 0))
        overall_score = safe_int(data.get("overall_score", 0))
        too_technical = bool(data.get("too_technical", False))
        comment = str(data.get("comment", "") or "")
        return JudgeResult(
            match_score=match_score,
            simplicity_score=simplicity_score,
            overall_score=overall_score,
            too_technical=too_technical,
            comment=comment,
            raw_output=raw,
            parse_error=None,
            round=round_idx,
        )
    except Exception as e:
        # Fallback regex (Inherited logic)
        try:
            def find_int(name: str, default: int = 1) -> int:
                m = re.search(rf"{name}\s*[:＝]\s*([1-5])", raw)
                return int(m.group(1)) if m else default

            match_score = find_int("match_score", 1)
            simplicity_score = find_int("simplicity_score", 1)
            overall_score = find_int("overall_score", 1)

            m_tt = re.search(r"too_technical\s*[:＝]\s*(true|false|真|偽)", raw, re.I)
            too_technical = m_tt and m_tt.group(1).lower() in ("true", "真")

            m_comment = re.search(r'comment\s*[:＝]\s*"([^"]{0,50})"', raw)
            comment = m_comment.group(1) if m_comment else ""

            return JudgeResult(
                match_score=match_score,
                simplicity_score=simplicity_score,
                overall_score=overall_score,
                too_technical=too_technical,
                comment=comment,
                raw_output=raw,
                parse_error=f"fallback_regex: {e}",
                round=round_idx,
            )
        except Exception as e2:
            return JudgeResult(
                match_score=1, simplicity_score=1, overall_score=1, too_technical=True,
                comment="Judge解析失敗", raw_output=raw, parse_error=f"{e} / {e2}", round=round_idx
            )

def is_good_by_threshold(j: JudgeResult, min_overall: int, min_match: int, min_simplicity: int) -> bool:
    return (j.overall_score >= min_overall and j.match_score >= min_match
            and j.simplicity_score >= min_simplicity and not j.too_technical)

def postprocess_patient_expression(text: str) -> str:
    if not text: return ""
    s = text.strip().splitlines()[0].strip()
    s = re.sub(r"\s*assistant[:：]?\s*$", "", s, flags=re.IGNORECASE)
    for prefix in ["患者:", "患者「", "患者｢", "「", "『"]:
        if s.startswith(prefix): s = s[len(prefix):].lstrip()
    for suf in ["」", "『", "』", "｡"]:
        if s.endswith(suf): s = s[:-1].rstrip()
    return s

def postprocess_doctor_expression(text: str) -> str:
    if not text: return ""
    s = text.strip().splitlines()[0].strip()
    s = re.sub(r"\s*assistant[:：]?\s*$", "", s, flags=re.IGNORECASE)
    for prefix in ["医師所見:", "医師記載:", "所見:", "症状:", "記載:", "カルテ:", "医師:"]:
        if s.startswith(prefix): s = s[len(prefix):].lstrip()
    for prefix in ["「", "『", "\"", "“", "”"]:
        if s.startswith(prefix): s = s[len(prefix):].lstrip()
    for suf in ["」", "『", "』", "｡"]:
        if s.endswith(suf): s = s[:-1].rstrip()
    return s

def select_diverse_top_k(expr_list: List[ExpressionRecord], k: int, sim_thresh_high: float = 0.9, sim_thresh_low: float = 0.7) -> List[ExpressionRecord]:
    import difflib
    if len(expr_list) <= k: return expr_list
    def pair_sim(a: str, b: str) -> float: return difflib.SequenceMatcher(None, a, b).ratio()

    sorted_expr = sorted(
        expr_list,
        key=lambda er: (
            -(er.judge.overall_score if er.judge else 0),
            -(er.judge.match_score if er.judge else 0),
            -(er.judge.simplicity_score if er.judge else 0),
        ),
    )
    selected: List[ExpressionRecord] = []
    for er in sorted_expr:
        if not selected: selected.append(er)
        if len(selected) >= k: break
        sims = [pair_sim(er.current_expression, x.current_expression) for x in selected]
        if (max(sims) if sims else 0.0) < sim_thresh_low: selected.append(er)
        if len(selected) >= k: break
    
    if len(selected) < k:
        for er in sorted_expr:
            if er not in selected: selected.append(er)
            if len(selected) >= k: break
    return selected[:k]

def load_hpo_csv(path: Path, default_source: str) -> List[HpoItem]:
    import csv
    items = []
    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hpo_id = row.get("HPO_ID", "").strip()
            if not hpo_id: continue
            
            name_ja = ""
            for key in ["jp_final", "jp_label", "name_ja", "HPO_name_ja", "ja_2023", "ja_2017_expert"]:
                if key in row and row[key].strip():
                    name_ja = row[key].strip(); break
            
            name_en = row.get("name_en", "").strip()
            cat = row.get("category_v2", "") or row.get("category", "Unknown").strip()
            src = row.get("source", "").strip() or default_source
            
            definition = ""
            if "definition_ja" in row and row["definition_ja"].strip():
                definition = row["definition_ja"].strip()
            elif "definition" in row and row["definition"].strip():
                definition = row["definition"].strip()
            
            items.append(HpoItem(hpo_id, name_ja, name_en, cat, src, definition))
    return items

# ==========================================
# Prompt Builders (Inherited)
# ==========================================
# (Note: Logic is identical to previous script, just condensed for brevity here)

def get_messages_patient(hpo: HPOGroup) -> str:
    # Gemini 2.5 prefers a single prompt or system instruction.
    # We will combine them in the call logic, returning the User Content here mostly.
    # But to keep structure, we return (system, user) tuple logic in the caller.
    pass 

# Retaining the exact text generation logic inside the pipeline loop for clarity
# ... (Same text content as before) ...

# ==========================================
# Gemini API Wrapper (New SDK)
# ==========================================

SEM = None
CLIENT: genai.Client = None

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
async def call_gemini(
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    temperature: float = 0.7,
    max_tokens: int = 128,
    json_mode: bool = False
) -> str:
    """
    Calls Gemini API using the new google-genai SDK.
    """
    async with SEM:
        # Config construction
        config_args = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        
        # Gemini 2.5 Flash supports strict JSON mode
        if json_mode:
            config_args["response_mime_type"] = "application/json"
        
        # In case temperature is 0, we can be explicit
        if temperature < 1e-5:
            config_args["temperature"] = 0.0

        config = types.GenerateContentConfig(**config_args)
        
        # System instructions need to be passed in config for the new SDK usually,
        # or as a separate argument to generate_content depending on version.
        # Ideally, we pass it in config.system_instruction or as a separate arg.
        # The new SDK allows `config=types.GenerateContentConfig(system_instruction=...)`
        if system_prompt:
            config.system_instruction = system_prompt

        try:
            # Async call via .aio property
            response = await CLIENT.aio.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=config
            )
            return response.text
        except Exception as e:
            # Tenacity will handle retries
            raise e

# ==========================================
# Main Pipeline Logic
# ==========================================

# Re-defining Prompt Getters to match exact strings from vLLM pipeline
def get_prompt_text_patient(hpo: HPOGroup) -> Tuple[str, str]:
    symptom_name = hpo.hpo_name_ja or hpo.hpo_name_en or hpo.hpo_id
    def_str = f"\n(定義: {hpo.definition})" if hpo.definition else ""
    sys = (
        "あなたは日本語の医療用語に詳しいAIアシスタントです。"
        "次に与える医学的な症状名について、患者さんが実際に言いそうな短い発言を1つだけ生成してください。"
        "できるだけ短く端的に、文章ではなく語句やごく短い表現にしてください。"
        "（例:「胸が苦しい」「頭がズキズキする」など）"
        "専門用語や診断名は避け、日常的な日本語に言い換えてください。"
        "出力は患者の発言のみを書き、説明や番号付けは一切しないでください。"
        "症状名や病名の語をそのまま含めないでください。"
        "敬体（〜です／〜ます）はなるべく使わないでください。"
    )
    user = f"症状名: {symptom_name}{def_str}"
    return sys, user

def get_prompt_text_doctor(hpo: HPOGroup) -> Tuple[str, str]:
    concept = hpo.hpo_name_ja or hpo.hpo_name_en or hpo.hpo_id
    def_str = f"\n(定義: {hpo.definition})" if hpo.definition else ""
    sys = (
        "あなたは日本語の医療用語と電子カルテ記載に詳しい日本人医師です。"
        "次に与える医学的な用語（症状名・所見名・病態名など）について、"
        "医師がカルテや診療録に記載しそうな表現を日本語で1つだけ生成してください。"
        "できるだけ簡潔に、通常のカルテ記載を意識した短い語句またはごく短い文にしてください。"
        "患者の話し言葉ではなく、医師が使用する標準的な専門用語・略語を用いてください。"
        "ただし、意味が過度に抽象的にならないようにし、診療録として具体的な情報が伝わる表現にしてください。"
        "出力は1つの表現のみとし、説明文やコメント、箇条書き、番号付けは一切行わないでください。"
    )
    user = f"用語： {concept}　{def_str}"
    return sys, user

def get_prompt_judge(hpo: HPOGroup, expr: str, mode: str) -> Tuple[str, str]:
    concept = hpo.hpo_name_ja or hpo.hpo_name_en or hpo.hpo_id
    def_block = f"\n[定義]\n{hpo.definition}\n" if hpo.definition else ""
    
    if mode == "patient":
        sys = (
            "あなたは日本語の医療コミュニケーションに詳しい評価AIです。"
            "一般の患者さんと、その保護者が医師に症状を説明する場面をイメージしてください。"
            "出力は JSON オブジェクト1つだけに限定してください。"
        )
        user = f"""
以下の「症状名」と「患者表現」を評価してください。
症状名: {concept} ({hpo.hpo_id}){def_block}
患者表現: {expr}

評価の前提:
- (定義がある場合) その意味と矛盾していないか確認してください。
- 「症状名」は医師やカルテで使われる専門用語です。
- 「患者表現」は、患者本人または保護者が医師に対して患者の状態を説明するときの言葉を想定してください。
- 患者表現が「体の構造の説明」だけで、痛み・つらさ・困りごとなどの訴えになっていない場合は match_score/overall_score を低くしてください。

評価軸（1〜5の整数）:
- match_score: 意味の一致度。
- simplicity_score: 表現の平易さ・患者らしさ。
- overall_score: 総合評価。
- too_technical: 専門用語（診断名、〜症、〜異形成など）が含まれる場合は true。
- comment: 30文字以内の修正コメント。

出力は以下のJSON形式のみ:
{{"match_score": 4, "simplicity_score": 4, "overall_score": 4, "too_technical": false, "comment": "..."}}
""".strip()
    else:
        sys = (
            "あなたは日本語の医療用語と電子カルテ記載に詳しい評価AIです。"
            "出力は JSON オブジェクト1つだけに限定してください。"
        )
        user = f"""
以下の「用語」と「医師表現」を評価してください。
用語: {concept}{def_block}
医師表現候補: {expr}

評価軸（1〜5の整数）:
- match_score: 意味の対応度。
- simplicity_score: カルテとしての簡潔さ・自然さ。
- overall_score: 総合評価。
- too_technical: 一般的な臨床で使われないほど専門的すぎる場合は true。
- comment: 30文字以内の修正コメント。

出力は以下のJSON形式のみ:
{{"match_score": 4, "simplicity_score": 4, "overall_score": 4, "too_technical": false, "comment": "..."}}
""".strip()
    return sys, user

def get_prompt_refine(hpo: HPOGroup, expr: str, jr: JudgeResult, mode: str) -> Tuple[str, str]:
    concept = hpo.hpo_name_ja or hpo.hpo_name_en or hpo.hpo_id
    def_constraint = f"・定義({hpo.definition})の意味から逸脱しないようにしてください。" if hpo.definition else ""
    
    if mode == "patient":
        sys = (
            "あなたは日本語の患者発言をわかりやすく短く言い換える専門家AIです。"
            "出力は患者の発言1つのみとし、説明文やコメントは一切書かないでください。"
        )
        user = f"""
症状名: {concept}
元の患者発言: {expr}
評価: match={jr.match_score}, simplicity={jr.simplicity_score}, technical={jr.too_technical}, comment={jr.comment}

指示:
- {def_constraint}
- 診断名や専門用語は使わず、日常会話の言葉に言い換えてください。
- 患者が実際に言いそうな短い話し言葉のフレーズを1つだけ出力してください。
""".strip()
    else:
        sys = (
            "あなたは日本語の医療用語と電子カルテ記載に詳しい日本人医師です。"
            "出力は修正後の表現1つのみとし、説明文やコメントは一切書かないでください。"
        )
        user = f"""
医学的な用語: {concept}
元の医師表現: {expr}
評価: match={jr.match_score}, simplicity={jr.simplicity_score}, technical={jr.too_technical}, comment={jr.comment}

指示:
- {def_constraint}
- カルテ記載として、冗長な説明文ではなく、簡潔な用語（または短い文）にしてください。
""".strip()
    return sys, user

async def process_single_hpo(
    hpo: HPOGroup,
    args: argparse.Namespace,
    round_log_file_obj=None,
    write_lock: asyncio.Lock = None
) -> Dict[str, Any]:
    
    mode = args.mode
    doc_per_hpo = args.target_good if args.per_hpo < args.target_good else args.per_hpo
    
    # Initialization
    hpo.expressions = [
        ExpressionRecord(hpo.hpo_id, hpo.hpo_name_ja, hpo.hpo_name_en, hpo.category, hpo.source, "", "")
        for _ in range(doc_per_hpo)
    ]

    # --- Round 0: Generation ---
    sys_p, user_p = get_prompt_text_patient(hpo) if mode == "patient" else get_prompt_text_doctor(hpo)
    
    gen_tasks = []
    for _ in range(doc_per_hpo):
        gen_tasks.append(call_gemini(
            sys_p, user_p, 
            args.gen_model, 
            temperature=0.7, 
            max_tokens=args.gen_max_new_tokens
        ))
    
    results = await asyncio.gather(*gen_tasks)
    for i, res_text in enumerate(results):
        proc_text = postprocess_patient_expression(res_text) if mode == "patient" else postprocess_doctor_expression(res_text)
        hpo.expressions[i].original_expression = proc_text
        hpo.expressions[i].current_expression = proc_text

    # --- Refine Loop ---
    current_round = 0
    while current_round <= args.max_rounds:
        # Judge
        judge_indices = []
        judge_tasks = []
        for i, rec in enumerate(hpo.expressions):
            if rec.refine_round == current_round:
                sys_j, user_j = get_prompt_judge(hpo, rec.current_expression, mode)
                judge_indices.append(i)
                judge_tasks.append(call_gemini(
                    sys_j, user_j,
                    args.gen_model,
                    temperature=0.0,
                    max_tokens=args.judge_max_new_tokens,
                    json_mode=True  # Use JSON mode for Judge
                ))
        
        if not judge_tasks: break
        
        judge_outputs = await asyncio.gather(*judge_tasks)
        for idx_in_list, raw_json in zip(judge_indices, judge_outputs):
            rec = hpo.expressions[idx_in_list]
            rec.judge = parse_judge_output(raw_json, round_idx=current_round)

            # Log
            if args.log_each_round and round_log_file_obj and write_lock:
                out_obj = {
                    "hpo_id": hpo.hpo_id,
                    "expression": rec.current_expression,
                    "judge": vars(rec.judge),
                    "is_good": is_good_by_threshold(rec.judge, args.min_overall, args.min_match, args.min_simplicity)
                }
                async with write_lock:
                    round_log_file_obj.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    round_log_file_obj.flush()

        # Check Good Count
        good_count = sum(1 for rec in hpo.expressions if rec.judge and is_good_by_threshold(rec.judge, args.min_overall, args.min_match, args.min_simplicity))
        if good_count >= args.target_good or current_round >= args.max_rounds:
            break
            
        # Refine
        refine_indices = []
        refine_tasks = []
        for i, rec in enumerate(hpo.expressions):
            if not (rec.judge and is_good_by_threshold(rec.judge, args.min_overall, args.min_match, args.min_simplicity)):
                sys_r, user_r = get_prompt_refine(hpo, rec.current_expression, rec.judge, mode)
                refine_indices.append(i)
                refine_tasks.append(call_gemini(
                    sys_r, user_r,
                    args.gen_model,
                    temperature=0.7,
                    max_tokens=args.refine_max_new_tokens
                ))
        
        if not refine_tasks: break
        refine_outputs = await asyncio.gather(*refine_tasks)
        
        for idx_in_list, new_text in zip(refine_indices, refine_outputs):
            rec = hpo.expressions[idx_in_list]
            proc_text = postprocess_patient_expression(new_text) if mode == "patient" else postprocess_doctor_expression(new_text)
            rec.current_expression = proc_text
            rec.refine_round = current_round + 1
            
        current_round += 1

    # Final Select
    selected = select_diverse_top_k(
        [e for e in hpo.expressions if e.judge and is_good_by_threshold(e.judge, args.min_overall, args.min_match, args.min_simplicity)] or hpo.expressions,
        k=min(args.target_good, len(hpo.expressions)),
        sim_thresh_high=args.diversity_sim_high,
        sim_thresh_low=args.diversity_sim_low
    )

    out_list = []
    for er in selected:
        jr = er.judge or JudgeResult(0,0,0,False)
        out_list.append({
            "HPO_ID": er.hpo_id,
            "original_expression": er.original_expression,
            "final_expression": er.current_expression,
            "judge_overall": jr.overall_score,
            "judge_comment": jr.comment,
            "rounds": er.refine_round
        })
    return out_list

async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["patient", "doctor"])
    parser.add_argument("--hpo-csv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    # Default updated to 2.5 Flash
    parser.add_argument("--gen-model", type=str, default="gemini-2.5-flash", help="Gemini 2.5 Flash model ID")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--concurrency", type=int, default=50) # Flash is fast, can handle higher concurrency
    
    # Logic Params
    parser.add_argument("--per-hpo", type=int, default=5)
    parser.add_argument("--target-good", type=int, default=3)
    parser.add_argument("--gen-max-new-tokens", type=int, default=64)
    parser.add_argument("--judge-max-new-tokens", type=int, default=128)
    parser.add_argument("--refine-max-new-tokens", type=int, default=64)
    parser.add_argument("--min-overall", type=int, default=4)
    parser.add_argument("--min-match", type=int, default=3)
    parser.add_argument("--min-simplicity", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--diversity-sim-high", type=float, default=0.9)
    parser.add_argument("--diversity-sim-low", type=float, default=0.7)
    parser.add_argument("--log-each-round", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)

    args = parser.parse_args()

    # --- Setup Client with New SDK ---
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY or GEMINI_API_KEY needed.")
        return
    
    global CLIENT, SEM
    CLIENT = genai.Client(api_key=api_key)
    SEM = asyncio.Semaphore(args.concurrency)

    # Load & Split
    hpo_items = load_hpo_csv(Path(args.hpo_csv), "doctor" if args.mode=="doctor" else "patient")
    groups = split_by_shard([HPOGroup(h.hpo_id, h.name_ja, h.name_en, h.category, h.source, h.definition) for h in hpo_items], args.num_shards, args.shard_id)
    
    print(f"Start: {args.gen_model} | Items: {len(groups)} | Mode: {args.mode}")

    # Output Init
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    round_log_f = None
    if args.log_each_round:
        round_log_f = Path(f"{args.output}.round_log.jsonl").open("w", encoding="utf-8")
    write_lock = asyncio.Lock()

    # Run
    async def run_safe(g):
        try: return await process_single_hpo(g, args, round_log_f, write_lock)
        except Exception as e:
            print(f"Error {g.hpo_id}: {e}")
            return []

    results = await tqdm_asyncio.gather(*[run_safe(g) for g in groups], desc="Processing")
    
    # Save
    with out_path.open("w", encoding="utf-8") as f:
        for res_list in results:
            for r in res_list:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    if round_log_f: round_log_f.close()
    print("Done.")

if __name__ == "__main__":
    asyncio.run(main_async())