# -*- coding: utf-8 -*-

"""
Unified Pattern B パイプライン (患者 / 医師 両対応) - vLLM Version (Fixed)
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
    HAS_VLLM = True
except ImportError:
    print("[ERROR] vLLM not found. Please install it with `pip install vllm`.")
    exit(1)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from json import JSONDecoder

_DECODER = JSONDecoder()

# Proxy settings (if needed)
# os.environ["HTTP_PROXY"] = "..."
# os.environ["HTTPS_PROXY"] = "..."

@dataclass
class HpoItem:
    hpo_id: str
    name_ja: str
    name_en: str
    category: str
    source: str
    # ▼ 追加: 定義文用のフィールド
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
    # ▼ 追加
    definition: str = ""
    expressions: List[ExpressionRecord] = field(default_factory=list)


def set_seed(seed: int = 42):
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
    
    # 1. シンプルなJSON検索
    try:
        return json.loads(s)
    except:
        pass
        
    # 2. コードブロック ```json ... ``` 検索
    match = re.search(r"```json(.*?)```", s, re.DOTALL)
    if match:
        try:
            return _DECODER.decode(match.group(1).strip())
        except:
            pass

    # 3. 最も外側の {} を検索
    last_open = s.find("{") # 最初に見つかる {
    last_close = s.rfind("}") # 最後に見つかる }
    
    if last_open == -1 or last_close == -1 or last_close < last_open:
        # 見つからない場合はエラー
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
        # 正規表現によるフォールバック
        try:
            def find_int(name: str, default: int = 1) -> int:
                m = re.search(rf"{name}\s*[:＝]\s*([1-5])", raw)
                return int(m.group(1)) if m else default

            match_score = find_int("match_score", 1)
            simplicity_score = find_int("simplicity_score", 1)
            overall_score = find_int("overall_score", 1)

            m_tt = re.search(r"too_technical\s*[:＝]\s*(true|false|真|偽)", raw, re.I)
            if m_tt:
                val = m_tt.group(1).lower()
                too_technical = val in ("true", "真")
            else:
                too_technical = False

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
                match_score=1,
                simplicity_score=1,
                overall_score=1,
                too_technical=True,
                comment="Judge出力の解析に失敗",
                raw_output=raw,
                parse_error=f"{e} / fallback: {e2}",
                round=round_idx,
            )


def is_good_by_threshold(
    j: JudgeResult,
    min_overall: int,
    min_match: int,
    min_simplicity: int,
) -> bool:
    return (
        j.overall_score >= min_overall
        and j.match_score >= min_match
        and j.simplicity_score >= min_simplicity
        and not j.too_technical
    )


# --- vLLM Wrapper Function ---

def run_vllm_batch(
    llm: LLM,
    prompts: List[str],
    max_tokens: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
    lora_path: Optional[str] = None,
    lora_name: str = "adapter",
    stop_tokens: Optional[List[str]] = None
) -> List[str]:
    """
    vLLMを使って一括生成を行う関数。
    """
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

    # Judgeなどで temperature=0 にしたい場合
    if temperature < 1e-5:
        sampling_params.temperature = 0
        sampling_params.top_p = 1.0

    lora_req = None
    if lora_path:
        # 名前は一意である必要があるので、パスや用途で区別
        lora_req = LoRARequest(lora_name, 1, lora_path)

    # vLLM実行
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=lora_req,
        use_tqdm=False
    )

    # 結果抽出 (生成テキストのみ)
    results = []
    for output in outputs:
        text = output.outputs[0].text
        results.append(text)

    return results


# --- Prompt Builders ---

def build_generate_messages_patient(hpo: HPOGroup) -> List[Dict[str, str]]:
    # 【修正】HPOGroupは hpo_name_ja 属性を持つ
    symptom_name = hpo.hpo_name_ja or hpo.hpo_name_en or hpo.hpo_id
    # ▼ 追加: 定義文の準備
    def_str = f"\n(定義: {hpo.definition})" if hpo.definition else ""
    system_msg = (
        "あなたは日本語の医療用語に詳しいAIアシスタントです。"
        "次に与える医学的な症状名について、患者さんが実際に言いそうな短い発言を1つだけ生成してください。"
        "できるだけ短く端的に、文章ではなく語句やごく短い表現にしてください。"
        "（例:「胸が苦しい」「頭がズキズキする」など）"
        "専門用語や診断名は避け、日常的な日本語に言い換えてください。"
        "出力は患者の発言のみを書き、説明や番号付けは一切しないでください。"
        "症状名や病名の語をそのまま含めないでください。"
        "敬体（〜です／〜ます）はなるべく使わないでください。"
    )
    user_msg = f"症状名: {symptom_name}{def_str}"
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def postprocess_patient_expression(text: str) -> str:
    if not text:
        return ""
    s = text.strip()
    s = s.splitlines()[0].strip()
    s = re.sub(r"\s*assistant[:：]?\s*$", "", s, flags=re.IGNORECASE)
    for prefix in ["患者:", "患者「", "患者｢", "「", "『"]:
        if s.startswith(prefix):
            s = s[len(prefix):].lstrip()
    for suf in ["」", "『", "』", "｡"]:
        if s.endswith(suf):
            s = s[:-1].rstrip()
    return s


def build_generate_messages_doctor(hpo: HPOGroup) -> List[Dict[str, str]]:
    # 【修正】HPOGroupは hpo_name_ja 属性を持つ
    concept = hpo.hpo_name_ja or hpo.hpo_name_en or hpo.hpo_id
    # ▼ 追加
    def_str = f"\n(定義: {hpo.definition})" if hpo.definition else ""
    system_msg = (
        "あなたは日本語の医療用語と電子カルテ記載に詳しい日本人医師です。"
        "次に与える医学的な用語（症状名・所見名・病態名など）について、"
        "医師がカルテや診療録に記載しそうな表現を日本語で1つだけ生成してください。"
        "できるだけ簡潔に、通常のカルテ記載を意識した短い語句またはごく短い文にしてください。"
        "患者の話し言葉ではなく、医師が使用する標準的な専門用語・略語を用いてください。"
        "ただし、意味が過度に抽象的にならないようにし、診療録として具体的な情報が伝わる表現にしてください。"
        "出力は1つの表現のみとし、説明文やコメント、箇条書き、番号付けは一切行わないでください。"
    )
    user_msg = f"用語： {concept}　{def_str}"
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def postprocess_doctor_expression(text: str) -> str:
    if not text:
        return ""
    s = text.strip()
    s = s.splitlines()[0].strip()
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


def build_judge_messages_patient(hpo: HPOGroup, expr: str) -> List[Dict[str, str]]:
    symptom_name = hpo.hpo_name_ja or hpo.hpo_name_en or hpo.hpo_id
    # ▼ 追加: 定義ブロック
    def_block = f"\n[定義]\n{hpo.definition}\n" if hpo.definition else ""
    system_msg = (
        "あなたは日本語の医療コミュニケーションに詳しい評価AIです。"
        "一般の患者さん（年齢は小児から高齢者まで）と、その保護者が医師に症状を説明する場面をイメージしてください。"
        "与えられた「症状名」は医師が使う医学用語です。"
        "与えられた「患者表現」が、患者さん本人または保護者の口から自然に出てきそうかどうかを重視して評価してください。"
        "出力は JSON オブジェクト1つだけに限定してください。"
        "応答は必ず '{' から始まり '}' で終わる1行のみとし、"
        "日本語の説明文や前置き・後置き、コードブロック記法(```json など)は絶対に書かないでください。"
    )
    user_content = f"""
以下の「症状名」と「患者表現」を評価してください。
症状名: {symptom_name} ({hpo.hpo_id}){def_block}
患者表現: {expr}

評価の前提:
- (定義がある場合) その意味と矛盾していないか確認してください。
- 「症状名」は医師やカルテで使われる専門用語です。
- 「患者表現」は、患者本人または保護者が医師に対して患者の状態を説明するときの言葉を想定してください。
- 一般的な患者は「多嚢胞腎」「膀胱憩室」「神経因性膀胱」のような診断名はほとんど使いません。
- 患者表現が「体の構造の説明」だけで、痛み・つらさ・困りごとなどの訴えになっていない場合は、
  症状名と関係していても match_score と overall_score を低くしてください（1〜2）。

評価軸（1〜5の整数）:
- match_score:
    - 症状名が表す内容と患者表現の意味がどれだけ一致しているか。
    - 5: ほぼ完全に同じ意味。重要な情報の欠落やズレがない。
    - 3: だいたい合っているが、一部あいまい・情報不足。
    - 1: ほとんど意味が合っていない、または症状になっていない説明だけの表現。
- simplicity_score:
    - 表現がどれだけ短く平易で、患者らしい話し言葉になっているか。
    - 5: 「背が低い」「夜にトイレに何回も起きる」のように、短く自然な日常会話の表現。
    - 3: 少し説明的だが、一般の人でも理解できる。「女性の性器の異常」など。
    - 1: 医学用語が多く、教科書や診断名に近い表現。「多嚢胞腎」「膀胱憩室」「神経因性膀胱症」など。
- overall_score:
    - 上記を総合した評価（1〜5）。意味の一致度と患者らしさのバランスを見て決めてください。
    - 症状として不自然な場合（体の構造だけの説明など）は overall_score を 1〜2 にしてください。

too_technical:
    - 患者表現が、医師や医療者が主に使う専門用語に近いかどうか。
    - 次のような場合は、短くても必ず true にしてください（迷ったら true）:
        - 病名・診断名そのものを名乗っている
          例: 「多嚢胞腎」「膀胱憩室」「神経因性膀胱」「子宮低形成」
        - 「〜症」「〜症候群」「〜腎症」「〜性膀胱」「〜機能不全」
          「〜異形成」「〜低形成」「憩室」「嚢胞」などの語をそのまま使っている。
        - カルテや教科書の見出しとして使えるような表現。
    - 逆に、一般の人の日常会話で出てきそうな言葉だけで書かれている場合だけ false にしてください。

comment:
    - 必要であれば、「どこをどう直すとよいか」を日本語で30文字以内で一言だけ書いてください。
    - 例: 「診断名なので患者はあまり言わない」「日常会話としてはやや硬い」「症状ではなく構造の説明です」など。

良い例:
- 症状名: 膀胱憩室
  患者表現: 膀胱に袋みたいなのができてる
  → match_score: 4, simplicity_score: 4, overall_score: 4, too_technical: false

悪い例（専門的すぎる）:
- 症状名: 膀胱憩室
  患者表現: 膀胱憩室があります
  → match_score: 5, simplicity_score: 2, overall_score: 3, too_technical: true

悪い例（意味が症状になっていない）:
- 症状名: 膀胱機能異常
  患者表現: 尿道が膀胱から出る
  → 解剖学の説明であり症状ではないので、
     match_score: 1, simplicity_score: 2, overall_score: 1, too_technical: false

出力フォーマット（この1行だけを返してください。以下の数値やコメントは必ず今回の評価結果に合わせて書き換えてください。例をそのままコピーしてはいけません）:
{{"match_score": 4, "simplicity_score": 4, "overall_score": 4, "too_technical": false, "comment": "コメント"}}

""".strip()
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]


def build_judge_messages_doctor(hpo: HPOGroup, expr: str) -> List[Dict[str, str]]:
    concept = hpo.hpo_name_ja or hpo.hpo_name_en or hpo.hpo_id
    # ▼ 追加
    def_block = f"\n[定義]\n{hpo.definition}\n" if hpo.definition else ""
    system_msg = (
        "あなたは日本語の医療用語と電子カルテ記載に詳しい評価AIです。"
        "日本の一般的な臨床現場で用いられる診療録・カルテの記載を知っている前提で、"
        "与えられた用語と医師表現候補が、カルテ記載として妥当かどうかを評価してください。"
        "出力は JSON オブジェクト1つだけに限定してください。"
        "必ず '{' から始まり '}' で終わる1行のみを出力し、日本語の説明文や前置き・後置き、"
        "コードブロック記法（```json など）は絶対に書かないでください。"
    )
    user_content = f"""
以下の「用語」と「医師表現」を評価してください。

用語（症状名・所見名・病態名など）:
{concept}{def_block}

医師表現候補:
{expr}

評価軸（1〜5の整数）:
- match_score:
    - 医学的な用語が表す内容と、医師表現の意味がどれだけ対応しているか。
    - 5: ほぼ完全に同じ意味。重要な情報の欠落やズレがない。
    - 3: おおむね対応しているが、一部あいまい・情報不足。
    - 1: ほとんど意味が合っていない、または別の概念を示している。
- simplicity_score:
    - 表現がどれだけ簡潔で、実際のカルテにそのまま書ける自然な記載になっているか。
    - 5: 「高血圧」「心不全増悪」「発熱（38℃台）」のように、短く簡潔で読みやすい。
    - 3: 少し冗長だが、診療録として許容範囲。
    - 1: 不必要に長い説明文になっている、または口語的すぎてカルテとして不自然。
- overall_score:
    - 上記を総合した評価（1〜5）。意味の一致度とカルテとしての自然さのバランスで決めてください。

too_technical:
    - 表現があまりにも専門的・教科書的で、一般的な臨床現場のカルテではほとんど用いられない場合は true にしてください。
    - 例: 細かすぎる病理学用語や、日常診療ではほぼ使わない英語・ラテン語表現など。
    - 一般的な専門用語（「心不全」「気管支喘息」「左室肥大」など）は false として構いません。

comment:
    - 必要であれば、「どこをどう直すとよいか」を日本語で30文字以内で一言だけ書いてください。
    - 例: 「冗長なので短くまとめてよい」「用語の意味が少しずれている」など。
    - 特にコメントが不要な場合は空文字でも構いません。

出力フォーマット（この1行だけを返してください。以下の数値やコメントは必ず今回の評価結果に合わせて書き換えてください。例をそのままコピーしてはいけません）:
{{"match_score": 4, "simplicity_score": 4, "overall_score": 4, "too_technical": false, "comment": "コメント"}}
""".strip()
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]


def build_refine_messages_patient(hpo: HPOGroup, expr: str, jr: JudgeResult) -> List[Dict[str, str]]:
    symptom_name = hpo.hpo_name_ja or hpo.hpo_name_en or hpo.hpo_id
    # ▼ 追加: 定義による制約メッセージ
    def_constraint = f"・【重要】定義({hpo.definition})の意味から逸脱しないようにしてください。" if hpo.definition else ""
    system_msg = (
        "あなたは日本語の患者発言をわかりやすく短く言い換える専門家AIです。"
        "与えられた症状名と患者の元の発言、および評価結果にもとづいて、"
        "必要に応じて患者の発言をより自然でわかりやすい短いフレーズに書き直してください。"
        "すでに十分妥当な場合は、意味を変えずにわずかな修正にとどめてください。"
        "出力は患者の発言1つのみとし、説明文やコメントは一切書かないでください。"
    )
    user_content = f"""
症状名:
{symptom_name}

元の患者発言:
{expr}

評価:
- match_score: {jr.match_score}
- simplicity_score: {jr.simplicity_score}
- overall_score: {jr.overall_score}
- too_technical: {jr.too_technical}
- comment: {jr.comment}

指示:
- 症状の意味はできるだけ保ち、何がつらい／困っているのかが伝わるようにしてください。
{def_constraint}
- 元の表現が診断名や専門用語（例: 憩室、嚢胞、低形成、異形成、〜症、〜症候群、〜機能不全 など）を含む場合は、
  それらの専門用語は使わず、日常会話の言葉に必ず言い換えてください。
- 患者が実際に言いそうな、短い話し言葉のフレーズを1つだけ出力してください。
  （例: 「トイレが近い」「背が低い」「何度もトイレに起きる」など）
- 1つの症状だけを表す表現に絞ってください。
- 敬語（〜です、〜ます）は使わないでください。
- 「。」「、」などで2文以上に分けないでください。
最終的な患者発言を1つだけ出力してください。
""".strip()
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]


def build_refine_messages_doctor(hpo: HPOGroup, expr: str, jr: JudgeResult) -> List[Dict[str, str]]:
    concept = hpo.hpo_name_ja or hpo.hpo_name_en or hpo.hpo_id
    # ▼ 追加
    def_constraint = f"・定義({hpo.definition})の意味から逸脱しないようにしてください。" if hpo.definition else ""
    system_msg = (
        "あなたは日本語の医療用語と電子カルテ記載に詳しい日本人医師です。"
        "与えられた医学的な用語と元の医師表現、および評価結果にもとづいて、"
        "カルテ記載としてより自然で適切な表現に書き直してください。"
        "出力は修正後の表現1つのみとし、説明文やコメントは一切書かないでください。"
    )
    user_content = f"""
医学的な用語:
{concept}

元の医師表現:
{expr}

評価:
- match_score: {jr.match_score}
- simplicity_score: {jr.simplicity_score}
- overall_score: {jr.overall_score}
- too_technical: {jr.too_technical}
- comment: {jr.comment}

指示:
{def_constraint}
- カルテ記載として、冗長な説明文ではなく、簡潔な用語（または短い文）にしてください。
- 意味がずれている場合は修正してください。
- 専門的すぎる（一般的な臨床で使われない）用語は避けてください。
""".strip()
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]


def select_diverse_top_k(
    expr_list: List[ExpressionRecord],
    k: int,
    sim_thresh_high: float = 0.9,
    sim_thresh_low: float = 0.7,
) -> List[ExpressionRecord]:
    import difflib
    if len(expr_list) <= k:
        return expr_list

    def pair_sim(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()

    # スコア順にソート (Overall > Match > Simplicity)
    sorted_expr = sorted(
        expr_list,
        key=lambda er: (
            -(er.judge.overall_score if er.judge else 0),
            -(er.judge.match_score if er.judge else 0),
            -(er.judge.simplicity_score if er.judge else 0),
        ),
    )

    selected: List[ExpressionRecord] = []
    # 1位は無条件採用
    for er in sorted_expr:
        if not selected:
            selected.append(er)
        if len(selected) >= k:
            break

        # 2つ目以降: 既存のものと似すぎていないかチェック
        # (厳密にやると重いので、簡易的に既存選抜済みとの最大類似度を見る)
        sims = [pair_sim(er.current_expression, x.current_expression) for x in selected]
        max_sim = max(sims)

        if max_sim < sim_thresh_low:
            selected.append(er)
        if len(selected) >= k:
            break

    # まだ足りない場合: 重複覚悟でスコア高い順に埋める
    if len(selected) < k:
        for er in sorted_expr:
            if er not in selected:
                selected.append(er)
            if len(selected) >= k:
                break

    return selected[:k]


def log_round_details(
    mode: str,
    hpo: HPOGroup,
    stage: str,
    round_idx: int,
    expr_recs: List[ExpressionRecord],
    indices: List[int],
    judge_results: List[JudgeResult],
    log_each_round: bool,
    round_log_fout: Any,
    wandb_run: Any,
    min_overall: int,
    min_match: int,
    min_simplicity: int,
):
    """
    ラウンドごとの詳細ログを出力する補助関数
    """
    if not (log_each_round and round_log_fout):
        return

    # JSONL書き出し
    # 各候補について1行ずつ
    for i, jr in zip(indices, judge_results):
        rec = expr_recs[i]
        out_obj = {
            "mode": mode,
            "stage": stage,
            "round": round_idx,
            "hpo_id": hpo.hpo_id,
            "expression": rec.current_expression,
            "judge": {
                "match": jr.match_score,
                "simplicity": jr.simplicity_score,
                "overall": jr.overall_score,
                "too_technical": jr.too_technical,
                "comment": jr.comment
            },
            "is_good": is_good_by_threshold(jr, min_overall, min_match, min_simplicity)
        }
        round_log_fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")


def load_hpo_csv(path: Path, default_source: str = "doctor") -> List[HpoItem]:
    import csv
    items = []
    # encodingは環境に合わせて utf-8 または utf-8-sig
    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hpo_id = row.get("HPO_ID", "").strip()
            if not hpo_id:
                continue
            
            # 【変更点1】日本語ラベルの取得ロジック
            # マスタ整備で作った 'jp_final' を最優先にする
            name_ja = ""
            # 優先度順: jp_final > jp_label > name_ja > ...
            target_cols = ["jp_final", "jp_label", "name_ja", "HPO_name_ja", "ja_2023", "ja_2017_expert"]
            for key in target_cols:
                if key in row and row[key].strip():
                    name_ja = row[key].strip()
                    break
            
            name_en = row.get("name_en", "").strip()
            
            # カテゴリも v2 があれば優先など必要に応じて調整
            cat = row.get("category_v2", "") or row.get("category", "Unknown").strip()
            src = row.get("source", "").strip() or default_source
            
            # 【変更点2】定義文 (Definition) の取得
            # 翻訳スクリプトで作った 'definition_ja' を優先、なければ英語 'definition'
            definition = ""
            if "definition_ja" in row and row["definition_ja"].strip():
                definition = row["definition_ja"].strip()
            elif "definition" in row and row["definition"].strip():
                definition = row["definition"].strip()
            
            # HpoItem の定義に合わせて definition を渡す
            items.append(HpoItem(hpo_id, name_ja, name_en, cat, src, definition))

    return items


def process_all_groups_vllm(
    llm: LLM,
    tokenizer: AutoTokenizer,
    mode: str,
    groups: List[HPOGroup],
    args: argparse.Namespace,
    fout: Any,
    round_log_fout: Any = None,
    wandb_run: Any = None
):
    """
    vLLMを用いて全HPOグループを一括処理するメインループ
    """
    # 設定値の展開
    doc_per_hpo = args.target_good  # ここでは doc_per_hpo = target_good として扱っていますが、必要なら args.per_hpo を使用
    if args.per_hpo > 0:
        doc_per_hpo = args.per_hpo

    target_good = args.target_good
    max_rounds = args.max_rounds

    min_overall = args.min_overall
    min_match = args.min_match
    min_simplicity = args.min_simplicity

    diversity_sim_high = args.diversity_sim_high
    diversity_sim_low = args.diversity_sim_low

    # 全グループに対して ExpressionRecord を初期化 (空の expression で)
    for hpo in groups:
        hpo.expressions = [
            ExpressionRecord(
                hpo_id=hpo.hpo_id,
                hpo_name_ja=hpo.hpo_name_ja,
                hpo_name_en=hpo.hpo_name_en,
                category=hpo.category,
                source=hpo.source,
                original_expression="",
                current_expression=""
            )
            for _ in range(doc_per_hpo)
        ]

    # 進捗バー
    pbar = tqdm(total=len(groups), desc=f"vLLM {mode} processing")
    finished_count = 0
    finished_flags = [False] * len(groups) # 各HPOが完了したか

    # ---------------------------------------------------------
    # Round 0: Initial Generation
    # ---------------------------------------------------------
    print("=== Round 0: Generating initial expressions ===")

    # 全HPO x 全候補 のプロンプト作成
    gen_prompts = []
    gen_indices = [] # (group_idx, expr_idx)

    for g_idx, hpo in enumerate(groups):
        msgs = build_generate_messages_patient(hpo) if mode == "patient" else build_generate_messages_doctor(hpo)
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for e_idx in range(doc_per_hpo):
            gen_prompts.append(prompt)
            gen_indices.append((g_idx, e_idx))

    # 一括生成
    gen_outputs = run_vllm_batch(
        llm,
        gen_prompts,
        max_tokens=args.gen_max_new_tokens,
        temperature=0.9,
        lora_path=args.gen_lora_path,
        lora_name="gen_adapter"
    )

    # 結果格納
    for (g_idx, e_idx), text in zip(gen_indices, gen_outputs):
        processed_text = postprocess_patient_expression(text) if mode == "patient" else postprocess_doctor_expression(text)
        rec = groups[g_idx].expressions[e_idx]
        rec.original_expression = processed_text
        rec.current_expression = processed_text
        rec.refine_round = 0

    # ---------------------------------------------------------
    # Refine Loop (Judge -> Refine -> Judge ...)
    # ---------------------------------------------------------
    current_round = 0
    while current_round <= max_rounds:
        print(f"=== Round {current_round}: Judging ===")

        # 1. Judge (全候補 または Refineされた候補)
        # 今回Judgeが必要な候補を特定
        # Round 0 は全員。Round > 0 は Refine されたものだけ
        judge_targets = [] # (group_idx, expr_idx, expression_text)

        for g_idx, hpo in enumerate(groups):
            if finished_flags[g_idx]:
                continue
            for e_idx, rec in enumerate(hpo.expressions):
                # Round 0なら全員Judge。それ以降は、直前にRefineされたものだけJudgeしたいが、
                # ここではシンプルに「まだGoodになっていないものはJudgeし直す」戦略でもよい。
                # 効率化のため「直前のラウンドでRefineされた」または「Round0」のみ対象にする。
                if rec.refine_round == current_round:
                    judge_targets.append((g_idx, e_idx, rec.current_expression))

        if not judge_targets:
            print("No candidates to judge. Loop finished.")
            break

        judge_prompts = []
        for (g_idx, e_idx, expr_text) in judge_targets:
            hpo = groups[g_idx]
            msgs = build_judge_messages_patient(hpo, expr_text) if mode == "patient" else build_judge_messages_doctor(hpo, expr_text)
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            judge_prompts.append(prompt)

        # Judge実行 (Base Model / No LoRA, temperature=0)
        judge_outputs = run_vllm_batch(
            llm,
            judge_prompts,
            max_tokens=args.judge_max_new_tokens,
            temperature=0.0,
            lora_path=None, # JudgeはLoRAなし
            lora_name=None
        )

        # Judge結果格納
        for (g_idx, e_idx, _), j_raw in zip(judge_targets, judge_outputs):
            rec = groups[g_idx].expressions[e_idx]
            jr = parse_judge_output(j_raw, round_idx=current_round)
            rec.judge = jr

            # ログ出力
            if args.log_each_round and round_log_fout:
                log_round_details(
                    mode, groups[g_idx], "judge", current_round,
                    [rec], [0], [jr], # ダミーリスト構造で渡す
                    True, round_log_fout, None,
                    min_overall, min_match, min_simplicity
                )

        # 完了判定
        # 各HPOについて、Goodな表現が target_good 個以上集まったら完了フラグを立てる
        for g_idx, hpo in enumerate(groups):
            if finished_flags[g_idx]:
                continue

            good_count = 0
            for rec in hpo.expressions:
                if rec.judge and is_good_by_threshold(rec.judge, min_overall, min_match, min_simplicity):
                    good_count += 1

            if good_count >= target_good:
                finished_flags[g_idx] = True
                finished_count += 1
                pbar.update(1)

        if all(finished_flags):
            print("All HPO groups reached target good count.")
            break

        if current_round >= max_rounds:
            print("Max rounds reached.")
            break

        # ---------------------------------------------------------
        # 2. Refine (必要なものだけ)
        # ---------------------------------------------------------
        print(f"=== Round {current_round + 1}: Refining ===")

        refine_targets = [] # (group_idx, expr_idx)

        for g_idx, hpo in enumerate(groups):
            if finished_flags[g_idx]:
                continue

            # GoodなものはRefineしない。BadなものだけRefine。
            # ただし、すでに十分なGoodがある場合はRefine不要だが、それはfinished_flagsで弾いている。
            # ここでは「現時点でGoodでないもの」をすべてRefine対象にする。
            for e_idx, rec in enumerate(hpo.expressions):
                is_good = False
                if rec.judge and is_good_by_threshold(rec.judge, min_overall, min_match, min_simplicity):
                    is_good = True

                if not is_good:
                    refine_targets.append((g_idx, e_idx))

        if not refine_targets:
            print("No candidates to refine.")
            break

        refine_prompts = []
        for (g_idx, e_idx) in refine_targets:
            hpo = groups[g_idx]
            rec = hpo.expressions[e_idx]
            jr = rec.judge
            # jrが無い(エラー等)場合はデフォルトを入れる
            if jr is None:
                jr = JudgeResult(0, 0, 0, False, "Not judged yet")

            msgs = build_refine_messages_patient(hpo, rec.current_expression, jr) if mode == "patient" else \
                   build_refine_messages_doctor(hpo, rec.current_expression, jr)
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            refine_prompts.append(prompt)

        # Refine実行 (LoRAあり)
        refine_outputs = run_vllm_batch(
            llm,
            refine_prompts,
            max_tokens=args.refine_max_new_tokens,
            temperature=0.7,
            lora_path=args.refine_lora_path,
            lora_name="refine_adapter"
        )

        # Refine結果更新
        for (g_idx, e_idx), new_text in zip(refine_targets, refine_outputs):
            processed_text = postprocess_patient_expression(new_text) if mode == "patient" else postprocess_doctor_expression(new_text)
            rec = groups[g_idx].expressions[e_idx]
            rec.current_expression = processed_text
            rec.refine_round = current_round + 1 # 次のラウンドへ

        current_round += 1

    pbar.close()

    # ---------------------------------------------------------
    # Finalize & Output
    # ---------------------------------------------------------
    print("=== Finalizing and saving results ===")

    orig_key = "patient_expression_original" if mode == "patient" else "doctor_expression_original"
    final_key = "patient_expression_final" if mode == "patient" else "doctor_expression_final"

    for hpo in groups:
        # Goodなもの、あるいはGoodがなければスコアが良いものを選抜
        good_exprs = [e for e in hpo.expressions if e.judge and is_good_by_threshold(e.judge, min_overall, min_match, min_simplicity)]

        # Goodが無ければ全候補から選ぶ
        if not good_exprs:
            candidates = hpo.expressions
        else:
            candidates = good_exprs

        selected = select_diverse_top_k(
            candidates,
            k=min(target_good, len(candidates)),
            sim_thresh_high=diversity_sim_high,
            sim_thresh_low=diversity_sim_low,
        )

        for er in selected:
            jr = er.judge or JudgeResult(0, 0, 0, False)
            rec = {
                "HPO_ID": er.hpo_id,
                "HPO_name_ja": er.hpo_name_ja,
                "HPO_name_en": er.hpo_name_en,
                "category": er.category,
                "source": er.source,
                "judge_overall": jr.overall_score,
                "judge_match": jr.match_score,
                "judge_simplicity": jr.simplicity_score,
                "judge_too_technical": jr.too_technical,
                "judge_comment": jr.comment,
                "judge_round": jr.round,
                "refine_round": er.refine_round,
            }
            rec[orig_key] = er.original_expression
            rec[final_key] = er.current_expression

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved results to {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["patient", "doctor"])
    parser.add_argument("--hpo-csv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--gen-model", type=str, required=True)
    parser.add_argument("--gen-lora-path", type=str, default=None)
    parser.add_argument("--judge-model", type=str, default=None, help="Ignored in vLLM version (uses gen-model)")
    parser.add_argument("--refine-model", type=str, default=None, help="Ignored in vLLM version (uses gen-model)")
    parser.add_argument("--refine-lora-path", type=str, default=None)
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
    parser.add_argument("--round-log-prefix", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="HPO_gen_judge_refine")
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--batch-across-hpo", action="store_true", help="Ignored in vLLM version (always batched)")

    args = parser.parse_args()

    set_seed(42)

    if args.wandb and HAS_WANDB:
        wandb_run = wandb.init(
            project=args.wandb_project,
            tags=args.wandb_tags.split(",") if args.wandb_tags else [],
            config=vars(args)
        )
    else:
        wandb_run = None

    # Load HPO items
    hpo_items = load_hpo_csv(Path(args.hpo_csv), default_source=("doctor" if args.mode == "doctor" else "patient"))
    groups_all = [
        HPOGroup(
            hpo_id=item.hpo_id,
            hpo_name_ja=item.name_ja,
            hpo_name_en=item.name_en,
            category=item.category,
            source=item.source
        )
        for item in hpo_items
    ]

    # Sharding
    groups = split_by_shard(groups_all, args.num_shards, args.shard_id)
    print(f"Loaded HPO groups: {len(groups)} (Total: {len(groups_all)}, Mode: {args.mode}, Shard: {args.shard_id}/{args.num_shards})")

    # Output setup
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("w", encoding="utf-8")

    round_log_fout = None
    if args.log_each_round:
        base = args.round_log_prefix if args.round_log_prefix else args.output + ".round"
        round_log_path = Path(f"{base}.mode-{args.mode}.shard{args.shard_id}.jsonl")
        round_log_path.parent.mkdir(parents=True, exist_ok=True)
        round_log_fout = round_log_path.open("w", encoding="utf-8")
        print(f"[round-log] write per-round logs to: {round_log_path}")

    # ==========================================
    # vLLM Model Loading
    # ==========================================
    print(f"=== [vLLM] Loading Model: {args.gen_model} ===")

    # 量子化の自動判定
    quant_method = None
    if "awq" in args.gen_model.lower():
        quant_method = "awq"
        print(" -> Auto-detected AWQ model.")
    elif "gptq" in args.gen_model.lower():
        quant_method = "gptq"

    # vLLMエンジンの初期化
    llm = LLM(
        model=args.gen_model,
        tokenizer=args.gen_model,
        quantization=quant_method,     # AWQなら "awq"
        dtype="float16",               # 基本はfp16
        enable_lora=True,              # LoRA有効化
        max_lora_rank=64,              # 使用するLoRAのRankに合わせて調整
        gpu_memory_utilization=0.9,    # メモリを90%確保
        trust_remote_code=True,
        tensor_parallel_size=1,        # シングルGPU
        max_model_len=4096,            # 長すぎる場合は調整
        enforce_eager=True             # エラー回避のためTrue推奨の場合あり
    )

    # トークナイザー (チャットテンプレート適用のため)
    tokenizer = AutoTokenizer.from_pretrained(args.gen_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    try:
        process_all_groups_vllm(
            llm=llm,
            tokenizer=tokenizer,
            mode=args.mode,
            groups=groups,
            args=args,
            fout=fout,
            round_log_fout=round_log_fout,
            wandb_run=wandb_run
        )
    finally:
        fout.close()
        if round_log_fout:
            round_log_fout.close()
        if wandb_run:
            wandb.finish()

if __name__ == "__main__":
    main()