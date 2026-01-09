
# -*- coding: utf-8 -*-

"""
Unified Pattern B パイプライン (患者 / 医師 両対応)
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

    # peft が無い環境でも isinstance(..., PeftModel) がエラーにならないようにダミー定義
    class PeftModel:  # type: ignore
        pass


try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

from json import JSONDecoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if HAS_BNB:
    BNB_CONFIG = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
else:
    BNB_CONFIG = None

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_DECODER = JSONDecoder()

AUTO_BATCH_CACHE_FILE = Path(__file__).with_name("auto_batch_cache_unified.json")

from contextlib import contextmanager


@contextmanager
def maybe_disable_lora_for_judge(model):
    """
    LoRA 付き PeftModel の場合だけ、一時的に adapter を無効化して
    ベースモデルとして推論するためのコンテキストマネージャ。
    LoRA なし / peft なしの場合は何もしない。
    """
    # peft が無ければ何もしない
    if not HAS_PEFT:
        yield
        return

    # PeftModel で、disable_adapter / enable_adapter を持っている場合だけ操作する
    if isinstance(model, PeftModel) and hasattr(model, "disable_adapter") and hasattr(model, "enable_adapter"):
        model.disable_adapter()
        try:
            yield
        finally:
            model.enable_adapter()
    else:
        # 通常モデルの場合はそのまま
        yield


@dataclass
class HpoItem:
    hpo_id: str
    name_ja: str
    name_en: str
    category: str
    source: str


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


def _get_gpu_key_for_cache() -> str:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return cvd.strip()
    return os.environ.get("JUDGE_GPU", "0")


def load_cached_batch_size(mode: str, stage: str) -> Optional[int]:
    try:
        if not AUTO_BATCH_CACHE_FILE.exists():
            return None
        with AUTO_BATCH_CACHE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    gpu_key = _get_gpu_key_for_cache()
    try:
        return int(data.get(mode, {}).get(stage, {}).get(gpu_key))
    except Exception:
        return None


def save_cached_batch_size(mode: str, stage: str, batch_size: int) -> None:
    gpu_key = _get_gpu_key_for_cache()
    data: Dict[str, Dict[str, Dict[str, int]]] = {}

    if AUTO_BATCH_CACHE_FILE.exists():
        try:
            with AUTO_BATCH_CACHE_FILE.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                data = {mk: {sk: dict(vv) for sk, vv in mv.items()}
                        for mk, mv in raw.items()}  # type: ignore[arg-type]
        except Exception:
            data = {}

    if mode not in data:
        data[mode] = {}
    if stage not in data[mode]:
        data[mode][stage] = {}
    data[mode][stage][gpu_key] = int(batch_size)

    AUTO_BATCH_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with AUTO_BATCH_CACHE_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def split_by_shard(items: List[Any], num_shards: int, shard_id: int) -> List[Any]:
    if num_shards <= 1:
        return items
    return [x for i, x in enumerate(items) if i % num_shards == shard_id]


def _extract_last_json_from_text(text: str) -> Dict[str, Any]:
    s = text.strip()
    last_open = s.rfind("{")
    last_close = s.rfind("}")
    if last_open == -1 or last_close == -1 or last_close < last_open:
        raise ValueError("JSON が見つかりません")
    fragment = s[last_open:last_close+1]
    return _DECODER.decode(fragment)


def parse_judge_output(raw: str, round_idx: int = 0) -> JudgeResult:
    try:
        # まずは素直に JSON を探す
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
        # ★ ここからフォールバック: JSONでなくても "match_score: 4" などを拾う
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
                # よくわからないときは「とりあえず False」として弾きすぎない
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
            # それでもダメな場合だけ本当に「全落ち」扱い
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
        and not j.too_technical  # ★ ここを追加
    )


def load_model_with_optional_lora(
    base_model_name: str,
    lora_path: Optional[str] = None,
    for_judge: bool = False,  # 引数は互換性のため残すが、挙動には使わない
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    print(f"=== load model on {DEVICE} ===")
    print(f"  base : {base_model_name}")
    print(f"  lora : {lora_path or '(none)'}")

    tok = AutoTokenizer.from_pretrained(base_model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # ★ ここを修正：judge も含めて、利用可能なら必ず 4bit 量子化
    if HAS_BNB and BNB_CONFIG is not None:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=BNB_CONFIG,
            device_map={"": DEVICE},
        )
    else:
        # BitsAndBytes が使えない環境では通常ロード
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map={"": DEVICE},
        )

    if lora_path:
        if not HAS_PEFT:
            raise RuntimeError("LoRA パスが指定されていますが peft がインストールされていません。")
        model = PeftModel.from_pretrained(base_model, lora_path)
        active = getattr(model, "active_adapter", None)
        if (isinstance(active, (list, tuple, set)) and len(active) == 0) or active in (None, ""):
            if hasattr(model, "peft_config") and len(model.peft_config) > 0:
                first = next(iter(model.peft_config.keys()))
                if hasattr(model, "set_adapter"):
                    model.set_adapter(first)  # type: ignore[attr-defined]
        model = model.to(DEVICE)
    else:
        model = base_model

    model.eval()
    return tok, model



def build_generate_messages_patient(hpo: HpoItem) -> List[Dict[str, str]]:
    symptom_name = hpo.name_ja or hpo.name_en or hpo.hpo_id
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
    user_msg = f"症状名: {symptom_name}"
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def postprocess_patient_expression(text: str) -> str:
    if not text:
        return ""
    s = text.strip()
    # 1行目だけ採用
    s = s.splitlines()[0].strip()

    # 以前の assistant ラベルを除去
    s = re.sub(r"\s*assistant[:：]?\s*$", "", s, flags=re.IGNORECASE)

    # 「患者:」「患者「」」などの接頭辞を除去
    for prefix in ["患者:", "患者「", "患者｢", "「", "『"]:
        if s.startswith(prefix):
            s = s[len(prefix):].lstrip()

    # 末尾の閉じカギかっこなどを除去（4a_new と同じ）
    for suf in ["」", "『", "』", "｡"]:
        if s.endswith(suf):
            s = s[:-1].rstrip()

    return s



def build_generate_messages_doctor(hpo: HpoItem) -> List[Dict[str, str]]:
    concept = hpo.name_ja or hpo.name_en or hpo.hpo_id

    system_msg = (
        "あなたは日本語の医療用語と電子カルテ記載に詳しい日本人医師です。"
        "次に与える医学的な用語（症状名・所見名・病態名など）について、"
        "医師がカルテや診療録に記載しそうな表現を日本語で1つだけ生成してください。"
        "できるだけ簡潔に、通常のカルテ記載を意識した短い語句またはごく短い文にしてください。"
        "患者の話し言葉ではなく、医師が使用する標準的な専門用語・略語を用いてください。"
        "ただし、意味が過度に抽象的にならないようにし、診療録として具体的な情報が伝わる表現にしてください。"
        "出力は1つの表現のみとし、説明文やコメント、箇条書き、番号付けは一切行わないでください。"
    )
    user_msg = f" {concept}"
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

    # 4b_new と同じ末尾処理
    for suf in ["」", "『", "』", "｡"]:
        if s.endswith(suf):
            s = s[:-1].rstrip()

    return s




def build_judge_messages_patient(hpo: HPOGroup, expr: str) -> List[Dict[str, str]]:
    """
    症状名と患者表現を評価する Judge 用プロンプト
    - 視点は「一般の患者が医師に話すときの口調」
    - too_technical は「医療者が使う診断名・所見名なら原則 true」
    """
    symptom_name = hpo.hpo_name_ja or hpo.hpo_name_en or hpo.hpo_id

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

評価の前提:
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
  患者表現: 膀胱に袋みたいなのができてるって言われた
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

症状名: {symptom_name}
患者表現: {expr}
""".strip()

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]

def build_judge_messages_doctor(hpo: HPOGroup, expr: str) -> List[Dict[str, str]]:
    """
    医師表現の Judge 用プロンプト
    - 視点は「日本の一般的なカルテ記載として妥当かどうか」
    - HPO や 標準病名 という語は使わず、「医学的な用語」として与える
    """
    concept = hpo.hpo_name_ja or hpo.hpo_name_en or hpo.hpo_id

    system_msg = (
        "あなたは日本語の医療用語と電子カルテ記載に詳しい評価AIです。"
        "日本の一般的な臨床現場で用いられる診療録・カルテの記載を知っている前提で、"
        "与えられた医学的な用語と医師表現候補が、カルテ記載として妥当かどうかを評価してください。"
        "出力は JSON オブジェクト1つだけに限定してください。"
        "必ず '{' から始まり '}' で終わる1行のみを出力し、日本語の説明文や前置き・後置き、"
        "コードブロック記法（```json など）は絶対に書かないでください。"
    )

    user_content = f"""
以下の「医学的な用語」と「医師表現」を評価してください。

医学的な用語（症状名・所見名・病態名など）:
{concept}

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



def build_refine_messages_patient(
    hpo: HPOGroup,
    expr: str,
    jr: JudgeResult,
) -> List[Dict[str, str]]:
    """
    Judge 結果を踏まえて、患者表現を「患者が実際に言いそうな短い発言」に
    書き直すプロンプト。
    doctor 用と同じ構造のメッセージ形式に統一。
    """
    symptom_name = hpo.hpo_name_ja or hpo.hpo_name_en or hpo.hpo_id

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
    """
    Judge 結果を踏まえて、医師表現を「カルテとしてより自然に」書き直すプロンプト。
    ここでも HPO や 標準病名という語は使わず、「医学的な用語」として渡す。
    """
    concept = hpo.hpo_name_ja or hpo.hpo_name_en or hpo.hpo_id

    system_msg = (
        "あなたは日本語の医療用語と電子カルテ記載に詳しい日本人医師です。"
        "与えられた医学的な用語と医師表現、および評価結果にもとづいて、"
        "必要に応じて医師表現をより自然で妥当なカルテ記載に書き直してください。"
        "すでに十分妥当な場合は、意味を変えずにわずかな修正にとどめてください。"
        "出力は1つの医師表現のみとし、説明文やコメントは一切書かないでください。"
    )

    user_content = f"""
医学的な用語（症状名・所見名・病態名など）:
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
- 医学的な用語の意味と対応していない場合は、意味が合うように表現を修正してください。
- 冗長な場合は、同じ意味を保ったまま簡潔なカルテ記載にしてください。
- 不自然なカルテ表現の場合は、日本の一般的な診療録に実際に書かれそうな自然な表現に書き直してください。
- すでに十分妥当な場合は、意味を変えずに、語順や助詞などをわずかに整える程度にとどめてください。

最終的な医師表現を1つだけ出力してください。
""".strip()

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]



@torch.inference_mode()
def generate_batch(
    mode: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    hpo_batch: List[HpoItem],
    max_new_tokens: int,
) -> List[str]:
    if mode == "patient":
        messages_list = [build_generate_messages_patient(h) for h in hpo_batch]
    else:
        messages_list = [build_generate_messages_doctor(h) for h in hpo_batch]

    prompts = [
        tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        for msgs in messages_list
    ]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    input_len = inputs["input_ids"].shape[1]

    results: List[str] = []
    for i in range(len(hpo_batch)):
        out_ids = outputs[i, input_len:]
        text = tokenizer.decode(out_ids, skip_special_tokens=True)
        results.append(text)
    return results


@torch.inference_mode()
def auto_tune_batch_size_for_generate(
    mode: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    hpo_sample: List[HpoItem],
    max_new_tokens: int,
    init_batch_size: int = 16,   # いまは使っていないがインターフェース維持のため残す
    max_batch_size: int = 1024,
) -> int:
    """
    生成用バッチサイズの自動チューニング。
    - 代表 HPO 1つからプロンプトを作成
    - それを max_batch_size 回コピーして OOM が出ない最大 bs を探索
      (指数探索 + 二分探索)
    """

    # 代表 HPO からプロンプト 1 つ作成
    if hpo_sample:
        example_hpo = hpo_sample[0]
        if mode == "patient":
            msgs = build_generate_messages_patient(example_hpo)
        else:
            msgs = build_generate_messages_doctor(example_hpo)
        prompt = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = "テスト用のプロンプトです。"

    probe_prompts = [prompt] * max_batch_size

    def try_bs(bs: int) -> bool:
        try:
            torch.cuda.empty_cache()
            inputs = tokenizer(
                probe_prompts[:bs],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(DEVICE)

            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            return True
        except torch.cuda.OutOfMemoryError:
            print(f"[auto-batch(generate)] OOM at batch-size={bs}")
            return False
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"[auto-batch(generate)] OOM at batch-size={bs}")
                return False
            print(f"[auto-batch(generate)] error at batch-size={bs}: {e}")
            return False
        except Exception as e:
            print(f"[auto-batch(generate)] error at batch-size={bs}: {e}")
            return False

    # まずキャッシュを試す
    cached = load_cached_batch_size(mode, "gen")
    if cached is not None:
        print(f"[auto-batch(generate)] use cached={cached}")
        if 1 <= cached <= max_batch_size and try_bs(cached):
            return cached
        else:
            print("[auto-batch(generate)] cached batch-size failed, re-tune from scratch.")

    # ここから新規探索: 1,2,4,8,... と指数的に増やす
    bs = 1
    last_ok = 0
    upper_fail = None

    while bs <= max_batch_size:
        print(f"[auto-batch(generate)] try batch-size={bs}")
        if try_bs(bs):
            last_ok = bs
            bs *= 2
        else:
            upper_fail = bs
            break

    if last_ok == 0:
        # 1 でもダメだった場合（ほぼ無いと思うがフォールバック）
        raw_best = 1
    elif upper_fail is None:
        # max_batch_size まで一度もこけなかった
        raw_best = last_ok
    else:
        # [last_ok, upper_fail) の範囲で二分探索
        lower_ok = last_ok
        hi = upper_fail
        while hi - lower_ok > 1:
            mid = (lower_ok + hi) // 2
            print(f"[auto-batch(generate)] refine try batch-size={mid}")
            if try_bs(mid):
                lower_ok = mid
            else:
                hi = mid
        raw_best = lower_ok

    safe_bs = max(1, int(raw_best * 0.8))
    save_cached_batch_size(mode, "gen", safe_bs)
    print(f"[auto-batch(generate)] final safe batch-size={safe_bs}")
    return safe_bs



@torch.inference_mode()
def judge_expressions(
    mode: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    hpo: HPOGroup,
    expressions: List[str],
    batch_size: int,
    max_new_tokens: int,
    round_idx: int,
) -> List[JudgeResult]:
    messages_list: List[List[Dict[str, str]]] = []
    for expr in expressions:
        if mode == "patient":
            messages_list.append(build_judge_messages_patient(hpo, expr))
        else:
            messages_list.append(build_judge_messages_doctor(hpo, expr))

    prompts = [
        tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        for msgs in messages_list
    ]

    results: List[JudgeResult] = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start+batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(DEVICE)

        # ★ LoRA 付きモデルでも、Judge のときは一時的に LoRA を無効化してベースで推論
        with maybe_disable_lora_for_judge(model):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # 貪欲デコード
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        input_len = inputs["input_ids"].shape[1]
        for i in range(len(batch_prompts)):
            out_ids = outputs[i, input_len:]
            text = tokenizer.decode(out_ids, skip_special_tokens=True)
            jr = parse_judge_output(text, round_idx=round_idx)
            results.append(jr)

    assert len(results) == len(expressions)
    return results


@torch.inference_mode()
def auto_tune_batch_size_for_judge(
    mode: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    hpo_sample: HPOGroup,
    expr_sample: List[str],
    max_new_tokens: int,
    init_batch_size: int = 16,   # インターフェース維持用
    max_batch_size: int = 1024,
) -> int:
    """
    Judge 用バッチサイズの自動チューニング。
    - 代表 HPO + 表現 1つから Judge プロンプトを作成
    - それを max_batch_size 回コピーして OOM が出ない最大 bs を探索
      (指数探索 + 二分探索)
    """

    # 代表の表現
    example_expr = expr_sample[0] if expr_sample else "サンプルの表現です。"

    # hpo_sample が None の場合のフォールバック
    if hpo_sample is None:
        hpo_sample = HPOGroup(
            hpo_id="HP:0000000",
            hpo_name_ja="サンプル",
            hpo_name_en="sample",
            category="symptom",
            source="leaf",
            expressions=[],
        )

    # プロンプト 1 つ作成
    if mode == "patient":
        msgs = build_judge_messages_patient(hpo_sample, example_expr)
    else:
        msgs = build_judge_messages_doctor(hpo_sample, example_expr)

    prompt = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    probe_prompts = [prompt] * max_batch_size

    def try_bs(bs: int) -> bool:
        try:
            torch.cuda.empty_cache()
            batch_prompts = probe_prompts[:bs]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(DEVICE)

            # Judge は LoRA 無効化してベースモデルで推論
            with maybe_disable_lora_for_judge(model):
                _ = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            return True
        except torch.cuda.OutOfMemoryError:
            print(f"[auto-batch(judge)] OOM at batch-size={bs}")
            return False
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"[auto-batch(judge)] OOM at batch-size={bs}")
                return False
            print(f"[auto-batch(judge)] error at batch-size={bs}: {e}")
            return False
        except Exception as e:
            print(f"[auto-batch(judge)] error at batch-size={bs}: {e}")
            return False

    # キャッシュを試す
    cached = load_cached_batch_size(mode, "judge")
    if cached is not None:
        print(f"[auto-batch(judge)] use cached={cached}")
        if 1 <= cached <= max_batch_size and try_bs(cached):
            return cached
        else:
            print("[auto-batch(judge)] cached batch-size failed, re-tune from scratch.")

    # 新規探索 (指数 + 二分)
    bs = 1
    last_ok = 0
    upper_fail = None

    while bs <= max_batch_size:
        print(f"[auto-batch(judge)] try batch-size={bs}")
        if try_bs(bs):
            last_ok = bs
            bs *= 2
        else:
            upper_fail = bs
            break

    if last_ok == 0:
        raw_best = 1
    elif upper_fail is None:
        raw_best = last_ok
    else:
        lower_ok = last_ok
        hi = upper_fail
        while hi - lower_ok > 1:
            mid = (lower_ok + hi) // 2
            print(f"[auto-batch(judge)] refine try batch-size={mid}")
            if try_bs(mid):
                lower_ok = mid
            else:
                hi = mid
        raw_best = lower_ok

    safe_bs = max(1, int(raw_best * 0.8))
    save_cached_batch_size(mode, "judge", safe_bs)
    print(f"[auto-batch(judge)] final safe batch-size={safe_bs}")
    return safe_bs


@torch.inference_mode()
def refine_expressions(
    mode: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    hpo: HPOGroup,
    expr_list: List[str],
    judge_results: List[JudgeResult],
    batch_size: int,
    max_new_tokens: int,
) -> List[str]:
    """
    Judge 結果を踏まえて表現を refine する。
    追加仕様:
      - 同一 HPO 内での重複をできるだけ避ける
      - 重複していた場合は、最大 max_regen_attempts 回まで単発再生成
      - それでも重複が解消しない場合は元の表現を採用（フォールバック）
    """
    # HPO 全体で現在使われている表現（このラウンドで refine 対象でないものも含む）
    used_global = {er.current_expression for er in hpo.expressions}

    # 再生成の最大試行回数
    max_regen_attempts = 3

    # 各 expr に対応する refine 用メッセージを準備
    messages_list: List[List[Dict[str, str]]] = []
    for expr, jr in zip(expr_list, judge_results):
        if mode == "patient":
            messages_list.append(build_refine_messages_patient(hpo, expr, jr))
        else:
            messages_list.append(build_refine_messages_doctor(hpo, expr, jr))

    # バッチ生成用のプロンプト（最初の 1 回分）
    prompts = [
        tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        for msgs in messages_list
    ]

    results: List[str] = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(DEVICE)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        input_len = inputs["input_ids"].shape[1]

        for i in range(len(batch_prompts)):
            global_idx = start + i  # expr_list / messages_list のインデックスに対応
            base_msgs = messages_list[global_idx]
            original_expr = expr_list[global_idx]

            # まずはバッチ生成結果を 1 回受け取る
            out_ids = outputs[i, input_len:]
            text = tokenizer.decode(out_ids, skip_special_tokens=True)
            if mode == "patient":
                text = postprocess_patient_expression(text)
            else:
                text = postprocess_doctor_expression(text)

            # ここから重複チェック & 単発再生成ループ
            # すでに使われている表現（他の HPO 内は気にせず、この HPO 内での重複を避けるイメージ）
            used_here = used_global | set(results)

            attempt = 0
            # 空文字 or 完全重複なら再生成を試みる
            while (not text or text in used_here) and attempt < max_regen_attempts:
                single_prompt = tokenizer.apply_chat_template(
                    base_msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                single_inputs = tokenizer(
                    single_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(DEVICE)
                single_outputs = model.generate(
                    **single_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                out_ids2 = single_outputs[0, single_inputs["input_ids"].shape[1]:]
                text2 = tokenizer.decode(out_ids2, skip_special_tokens=True)
                if mode == "patient":
                    text2 = postprocess_patient_expression(text2)
                else:
                    text2 = postprocess_doctor_expression(text2)

                text = text2
                attempt += 1
                used_here = used_global | set(results)

            # それでもダメ（重複 or 空）の場合は、元の表現をフォールバックとして採用
            if not text or text in used_here:
                text = original_expr

            results.append(text)
            used_global.add(text)

    assert len(results) == len(expr_list)
    return results


@torch.inference_mode()
def auto_tune_batch_size_for_refine(
    mode: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    hpo_sample: HPOGroup,
    expr_sample: List[str],
    judge_sample: List[JudgeResult],
    max_new_tokens: int,
    init_batch_size: int = 16,   # インターフェース維持用
    max_batch_size: int = 1024,
) -> int:
    """
    Refine 用バッチサイズの自動チューニング。
    - 代表 HPO + 表現 + JudgeResult 1つから Refine プロンプトを作成
    - それを max_batch_size 回コピーして OOM が出ない最大 bs を探索
      (指数探索 + 二分探索)
    """

    # 代表の表現 & JudgeResult
    example_expr = expr_sample[0] if expr_sample else "サンプルの表現です。"
    example_jr = judge_sample[0] if judge_sample else JudgeResult(
        match_score=3,
        simplicity_score=3,
        overall_score=3,
        too_technical=False,
        comment="サンプル",
        round=0,
    )

    # hpo_sample が None の場合のフォールバック
    if hpo_sample is None:
        hpo_sample = HPOGroup(
            hpo_id="HP:0000000",
            hpo_name_ja="サンプル",
            hpo_name_en="sample",
            category="symptom",
            source="leaf",
            expressions=[],
        )

    # refine 用メッセージ 1 つ作成
    if mode == "patient":
        msgs = build_refine_messages_patient(hpo_sample, example_expr, example_jr)
    else:
        msgs = build_refine_messages_doctor(hpo_sample, example_expr, example_jr)

    prompt = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    probe_prompts = [prompt] * max_batch_size

    def try_bs(bs: int) -> bool:
        try:
            torch.cuda.empty_cache()
            batch_prompts = probe_prompts[:bs]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(DEVICE)

            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            return True
        except torch.cuda.OutOfMemoryError:
            print(f"[auto-batch(refine)] OOM at batch-size={bs}")
            return False
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"[auto-batch(refine)] OOM at batch-size={bs}")
                return False
            print(f"[auto-batch(refine)] error at batch-size={bs}: {e}")
            return False
        except Exception as e:
            print(f"[auto-batch(refine)] error at batch-size={bs}: {e}")
            return False

    # キャッシュを試す
    cached = load_cached_batch_size(mode, "refine")
    if cached is not None:
        print(f"[auto-batch(refine)] use cached={cached}")
        if 1 <= cached <= max_batch_size and try_bs(cached):
            return cached
        else:
            print("[auto-batch(refine)] cached batch-size failed, re-tune from scratch.")

    # 新規探索 (指数 + 二分)
    bs = 1
    last_ok = 0
    upper_fail = None

    while bs <= max_batch_size:
        print(f"[auto-batch(refine)] try batch-size={bs}")
        if try_bs(bs):
            last_ok = bs
            bs *= 2
        else:
            upper_fail = bs
            break

    if last_ok == 0:
        raw_best = 1
    elif upper_fail is None:
        raw_best = last_ok
    else:
        lower_ok = last_ok
        hi = upper_fail
        while hi - lower_ok > 1:
            mid = (lower_ok + hi) // 2
            print(f"[auto-batch(refine)] refine try batch-size={mid}")
            if try_bs(mid):
                lower_ok = mid
            else:
                hi = mid
        raw_best = lower_ok

    safe_bs = max(1, int(raw_best * 0.8))
    save_cached_batch_size(mode, "refine", safe_bs)
    print(f"[auto-batch(refine)] final safe batch-size={safe_bs}")
    return safe_bs


def load_hpo_csv(path: Path, default_source: str = "leaf") -> List[HpoItem]:
    import csv

    items: List[HpoItem] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hpo_id = safe_str(row.get("HPO_ID", "")).strip()
            if not hpo_id.startswith("HP:"):
                continue
            name_en = safe_str(row.get("name_en", "")).strip()
            name_ja = safe_str(row.get("jp_final", "")).strip()
            category = safe_str(row.get("category", "")).strip() or "symptom"
            source =default_source

            items.append(
                HpoItem(
                    hpo_id=hpo_id,
                    name_ja=name_ja,
                    name_en=name_en,
                    category=category,
                    source=source,
                )
            )
    return items


def group_expressions_by_hpo(records: List[ExpressionRecord]) -> List[HPOGroup]:
    groups: Dict[str, HPOGroup] = {}
    for er in records:
        if er.hpo_id not in groups:
            groups[er.hpo_id] = HPOGroup(
                hpo_id=er.hpo_id,
                hpo_name_ja=er.hpo_name_ja,
                hpo_name_en=er.hpo_name_en,
                category=er.category,
                source=er.source,
                expressions=[],
            )
        groups[er.hpo_id].expressions.append(er)
    return list(groups.values())


def generate_all_expressions(
    mode: str,
    hpo_items: List[HpoItem],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_new_tokens: int,
    per_hpo: int,
    max_attempts_per_hpo: int,
    batch_size: int,
) -> List[ExpressionRecord]:
    n = len(hpo_items)
    collected: List[List[str]] = [[] for _ in range(n)]
    seen: List[set] = [set() for _ in range(n)]
    attempts: List[int] = [0] * n

    pbar = tqdm(total=n * per_hpo, desc=f"{mode} generate", unit="expr")

    while True:
        pending = [
            i for i in range(n)
            if len(collected[i]) < per_hpo and attempts[i] < max_attempts_per_hpo
        ]
        if not pending:
            break

        for start in range(0, len(pending), batch_size):
            batch_idx = pending[start:start+batch_size]
            hpo_batch = [hpo_items[i] for i in batch_idx]
            gen_texts = generate_batch(
                mode=mode,
                tokenizer=tokenizer,
                model=model,
                hpo_batch=hpo_batch,
                max_new_tokens=max_new_tokens,
            )

            for local_i, h in enumerate(hpo_batch):
                idx = batch_idx[local_i]
                raw = gen_texts[local_i]
                if mode == "patient":
                    expr = postprocess_patient_expression(raw)
                else:
                    expr = postprocess_doctor_expression(raw)
                attempts[idx] += 1
                if not expr or expr in seen[idx]:
                    continue
                seen[idx].add(expr)
                collected[idx].append(expr)
                pbar.update(1)

    pbar.close()

    records: List[ExpressionRecord] = []
    for h, exprs in zip(hpo_items, collected):
        for e in exprs:
            records.append(
                ExpressionRecord(
                    hpo_id=h.hpo_id,
                    hpo_name_ja=h.name_ja,
                    hpo_name_en=h.name_en,
                    category=h.category,
                    source=h.source,
                    original_expression=e,
                    current_expression=e,
                )
            )
    return records


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
        if not selected:
            selected.append(er)
            if len(selected) >= k:
                break
            continue
        sims = [pair_sim(er.current_expression, x.current_expression) for x in selected]
        max_sim = max(sims)
        if max_sim < sim_thresh_low:
            selected.append(er)
        if len(selected) >= k:
            break
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
    round_log_fout,
    wandb_run,
    min_overall: int,
    min_match: int,
    min_simplicity: int,
) -> None:
    """
    1回分の Judge / Refine の結果を
      - JSONL (round_log_fout)
      - W&B (wandb_run)
    に記録するヘルパー
    """
    if not indices or not judge_results:
        return

    # JSONL ログ（1 expr = 1 行）
    if log_each_round and round_log_fout is not None:
        for local_i, global_idx in enumerate(indices):
            er = expr_recs[global_idx]
            jr = judge_results[local_i]
            rec = {
                "mode": mode,
                "stage": stage,
                "round": round_idx,
                "HPO_ID": er.hpo_id,
                "HPO_name_ja": er.hpo_name_ja,
                "HPO_name_en": er.hpo_name_en,
                "category": er.category,
                "source": er.source,
                "expression_original": er.original_expression,
                "expression_current": er.current_expression,
                "judge_overall": jr.overall_score,
                "judge_match": jr.match_score,
                "judge_simplicity": jr.simplicity_score,
                "judge_too_technical": jr.too_technical,
                "judge_comment": jr.comment,
                "judge_round": jr.round,
                "refine_round": er.refine_round,
            }
            if jr.parse_error:
                rec["judge_parse_error"] = jr.parse_error
            round_log_fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # W&B ログ（統計値だけ）
    if wandb_run is not None:
        ovs = [jr.overall_score for jr in judge_results]
        ms = [jr.match_score for jr in judge_results]
        ss = [jr.simplicity_score for jr in judge_results]
        n = len(judge_results)
        num_good = sum(
            1 for jr in judge_results
            if is_good_by_threshold(jr, min_overall, min_match, min_simplicity)
        )
        metrics = {
            "mode": mode,
            "hpo_id": hpo.hpo_id,
            "stage": stage,
            "round": round_idx,
            "num_expr": n,
            "overall_mean": sum(ovs) / n,
            "match_mean": sum(ms) / n,
            "simplicity_mean": sum(ss) / n,
            "num_good": num_good,
        }
        wandb_run.log(metrics)


def process_hpo_group_multi_round(
    mode: str,
    hpo: HPOGroup,
    judge_tok: AutoTokenizer,
    judge_model: AutoModelForCausalLM,
    judge_batch_size: int,
    judge_max_new_tokens: int,
    refine_tok: AutoTokenizer,
    refine_model: AutoModelForCausalLM,
    refine_batch_size: int,
    refine_max_new_tokens: int,
    min_overall: int,
    min_match: int,
    min_simplicity: int,
    target_good: int,
    max_rounds: int,
    diversity_sim_high: float,
    diversity_sim_low: float,
    log_each_round: bool = False,
    round_log_fout=None,
    wandb_run=None,
) -> List[Dict[str, Any]]:
    expr_recs = list(hpo.expressions)
    if not expr_recs:
        return []
    

    orig_key = "patient_expression_original" if mode == "patient" else "doctor_expression_original"
    final_key = "patient_expression_final" if mode == "patient" else "doctor_expression_final"
    current_round = 0
    expressions = [er.current_expression for er in expr_recs]
    judges_round0 = judge_expressions(
        mode=mode,
        tokenizer=judge_tok,
        model=judge_model,
        hpo=hpo,
        expressions=expressions,
        batch_size=judge_batch_size,
        max_new_tokens=judge_max_new_tokens,
        round_idx=current_round,
    )
    # round 0 (初回 Judge) のログ
    log_round_details(
        mode=mode,
        hpo=hpo,
        stage="judge",
        round_idx=current_round,
        expr_recs=expr_recs,
        indices=list(range(len(expr_recs))),
        judge_results=judges_round0,
        log_each_round=log_each_round,
        round_log_fout=round_log_fout,
        wandb_run=wandb_run,
        min_overall=min_overall,
        min_match=min_match,
        min_simplicity=min_simplicity,
    )

    good_indices: List[int] = []
    bad_indices: List[int] = []
    for idx, (er, jr) in enumerate(zip(expr_recs, judges_round0)):
        er.judge = jr
        er.refine_round = 0
        if is_good_by_threshold(jr, min_overall, min_match, min_simplicity):
            good_indices.append(idx)
        else:
            bad_indices.append(idx)

    if len(good_indices) >= target_good or max_rounds == 0 or not bad_indices:
        good_exprs = [expr_recs[i] for i in good_indices]
        selected_exprs = select_diverse_top_k(
            good_exprs,
            k=min(target_good, len(good_exprs)),
            sim_thresh_high=diversity_sim_high,
            sim_thresh_low=diversity_sim_low,
        )
        outputs: List[Dict[str, Any]] = []
        for er in selected_exprs:
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

            outputs.append(rec)

        return outputs


    current_round = 1
    while current_round <= max_rounds and len(good_indices) < target_good and bad_indices:
        refine_indices = list(bad_indices)
        expr_list = [expr_recs[i].current_expression for i in refine_indices]
        judge_list = [expr_recs[i].judge or JudgeResult(0, 0, 0, False) for i in refine_indices]

        refined_exprs = refine_expressions(
            mode=mode,
            tokenizer=refine_tok,
            model=refine_model,
            hpo=hpo,
            expr_list=expr_list,
            judge_results=judge_list,
            batch_size=refine_batch_size,
            max_new_tokens=refine_max_new_tokens,
        )

        new_judges = judge_expressions(
            mode=mode,
            tokenizer=judge_tok,
            model=judge_model,
            hpo=hpo,
            expressions=refined_exprs,
            batch_size=judge_batch_size,
            max_new_tokens=judge_max_new_tokens,
            round_idx=current_round,
        )

        new_bad_indices: List[int] = []
        for local_idx, global_idx in enumerate(refine_indices):
            er = expr_recs[global_idx]
            old_j = er.judge or JudgeResult(0, 0, 0, False)
            new_j = new_judges[local_idx]

            if new_j.overall_score > old_j.overall_score:
                er.current_expression = refined_exprs[local_idx]
                er.judge = new_j
                er.refine_round = current_round

            if is_good_by_threshold(er.judge or new_j, min_overall, min_match, min_simplicity):
                if global_idx not in good_indices:
                    good_indices.append(global_idx)
            else:
                new_bad_indices.append(global_idx)
        # refine 後の Judge 結果（current_round）のログ
        final_judges: List[JudgeResult] = []
        for local_idx, global_idx in enumerate(refine_indices):
            er = expr_recs[global_idx]
            # er.judge は上のループで必要に応じて new_j に更新済み
            final_judges.append(er.judge or new_judges[local_idx])

        log_round_details(
            mode=mode,
            hpo=hpo,
            stage="refine",
            round_idx=current_round,
            expr_recs=expr_recs,
            indices=refine_indices,
            judge_results=final_judges,
            log_each_round=log_each_round,
            round_log_fout=round_log_fout,
            wandb_run=wandb_run,
            min_overall=min_overall,
            min_match=min_match,
            min_simplicity=min_simplicity,
        )

        bad_indices = new_bad_indices
        if len(good_indices) >= target_good or not bad_indices:
            break
        current_round += 1

    good_exprs = [expr_recs[i] for i in good_indices]
    if not good_exprs:
        return []

    selected_exprs = select_diverse_top_k(
        good_exprs,
        k=min(target_good, len(good_exprs)),
        sim_thresh_high=diversity_sim_high,
        sim_thresh_low=diversity_sim_low,
    )
    outputs: List[Dict[str, Any]] = []
    for er in selected_exprs:
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

        outputs.append(rec)

    return outputs



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["patient", "doctor"], required=True)
    parser.add_argument("--hpo-csv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--gen-model", type=str, default="EQUES/MedLLama3-JP-v2")
    parser.add_argument("--gen-lora-path", type=str, default="")
    parser.add_argument("--judge-model", type=str, default="EQUES/MedLLama3-JP-v2")
    parser.add_argument("--refine-model", type=str, default="EQUES/MedLLama3-JP-v2")
    parser.add_argument("--refine-lora-path", type=str, default="")

    parser.add_argument("--gen-max-new-tokens", type=int, default=32)
    parser.add_argument("--judge-max-new-tokens", type=int, default=128)
    parser.add_argument("--refine-max-new-tokens", type=int, default=64)

    parser.add_argument("--per-hpo", type=int, default=8)
    parser.add_argument("--max-attempts-per-hpo", type=int, default=16)

    parser.add_argument("--min-overall", type=int, default=4)
    parser.add_argument("--min-match", type=int, default=4)
    parser.add_argument("--min-simplicity", type=int, default=4)
    parser.add_argument("--target-good", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=3)

    parser.add_argument("--diversity-sim-high", type=float, default=0.95)
    parser.add_argument("--diversity-sim-low", type=float, default=0.8)

    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)
    # ラウンドごとのログ
    parser.add_argument("--log-each-round", action="store_true")
    parser.add_argument("--round-log-prefix", type=str, default="")

    # W&B ログ
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="HPO_unified_patternB")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default="",
        help="カンマ区切りタグ (例: patient,doctor,unified)",
    )

    args = parser.parse_args()
    # W&B 初期化
    wandb_run = None
    if args.wandb:
        if not HAS_WANDB:
            print("WARNING: --wandb が指定されましたが wandb がインストールされていません。W&B ログは無効化されます。")
        else:
            tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
            wandb_config = {
                "mode": args.mode,
                "hpo_csv": args.hpo_csv,
                "gen_model": args.gen_model,
                "gen_lora_path": args.gen_lora_path,
                "judge_model": args.judge_model,
                "refine_model": args.refine_model,
                "refine_lora_path": args.refine_lora_path,
                "gen_max_new_tokens": args.gen_max_new_tokens,
                "judge_max_new_tokens": args.judge_max_new_tokens,
                "refine_max_new_tokens": args.refine_max_new_tokens,
                "per_hpo": args.per_hpo,
                "max_attempts_per_hpo": args.max_attempts_per_hpo,
                "min_overall": args.min_overall,
                "min_match": args.min_match,
                "min_simplicity": args.min_simplicity,
                "target_good": args.target_good,
                "max_rounds": args.max_rounds,
                "diversity_sim_high": args.diversity_sim_high,
                "diversity_sim_low": args.diversity_sim_low,
                "num_shards": args.num_shards,
                "shard_id": args.shard_id,
                "seed": args.seed,
            }
            run_name = args.wandb_run_name or f"{args.mode}-unified-shard{args.shard_id}"
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=wandb_config,
                tags=tags or None,
            )

    set_seed(args.seed)

    mode = args.mode
    round_log_fout = None
    if args.log_each_round:
        if args.round_log_prefix:
            base = args.round_log_prefix
        else:
            # 未指定なら output をベースにする
            base = args.output + ".round"
        round_log_path = Path(f"{base}.mode-{mode}.shard{args.shard_id}.jsonl")
        round_log_path.parent.mkdir(parents=True, exist_ok=True)
        round_log_fout = round_log_path.open("w", encoding="utf-8")
        print(f"[round-log] write per-round logs to: {round_log_path}")
    hpo_items = load_hpo_csv(Path(args.hpo_csv), default_source=("doctor" if mode == "doctor" else "patient"))
    hpo_items = split_by_shard(hpo_items, args.num_shards, args.shard_id)
    print(f"Loaded HPO items: {len(hpo_items)} (mode={mode})")

    # ... hpo_items 読み込みのあと ...

    # ============================
    # モデルロード（LoRA 有無で分岐）
    # ============================
    use_shared_no_lora = (
        (args.gen_model == args.judge_model == args.refine_model)
        and (not args.gen_lora_path)
        and (not args.refine_lora_path)
    )

    # ★ LoRA ありでも「同じベース + 同じ LoRA」でよければ 1 モデル共有
    use_shared_with_lora = (
        (args.gen_model == args.judge_model == args.refine_model)
        and (args.gen_lora_path == args.refine_lora_path)
        and bool(args.gen_lora_path)  # LoRA パスが指定されている
    )

    if use_shared_no_lora:
        # LoRA なし & モデル名も全部同じ → 1 回だけロードして共有
        print("=== use SINGLE shared model (no LoRA) ===")
        shared_tok, shared_model = load_model_with_optional_lora(
            base_model_name=args.gen_model,
            lora_path=None,
            for_judge=False,  # BNB があれば 4bit でロード
        )
        gen_tok = judge_tok = refine_tok = shared_tok
        gen_model = judge_model = refine_model = shared_model

    elif use_shared_with_lora:
        # LoRA あり & モデル名・LoRA パスも同じ → 1 回だけロードして共有
        # gen / refine: LoRA 有効, judge: LoRA 無効（maybe_disable_lora_for_judge で制御）
        print("=== use SINGLE shared model (with LoRA for gen/refine, base-only for judge) ===")
        shared_tok, shared_model = load_model_with_optional_lora(
            base_model_name=args.gen_model,
            lora_path=args.gen_lora_path,
            for_judge=False,  # ここも 4bit ロード
        )
        gen_tok = judge_tok = refine_tok = shared_tok
        gen_model = judge_model = refine_model = shared_model

    else:
        # それ以外（ベースモデルが違う / LoRA が違う 等）は従来通り別々にロード
        gen_tok, gen_model = load_model_with_optional_lora(
            base_model_name=args.gen_model,
            lora_path=args.gen_lora_path or None,
            for_judge=False,
        )
        judge_tok, judge_model = load_model_with_optional_lora(
            base_model_name=args.judge_model,
            lora_path=None,          # Judge は常に LoRA 無し
            for_judge=True,
        )
        refine_tok, refine_model = load_model_with_optional_lora(
            base_model_name=args.refine_model,
            lora_path=args.refine_lora_path or None,
            for_judge=False,
        )


    if hpo_items:
        sample_hpo_items = hpo_items[: min(8, len(hpo_items))]
    else:
        sample_hpo_items = []
    gen_batch_size = auto_tune_batch_size_for_generate(
        mode=mode,
        tokenizer=gen_tok,
        model=gen_model,
        hpo_sample=sample_hpo_items,
        max_new_tokens=args.gen_max_new_tokens,
    )

    expr_records = generate_all_expressions(
        mode=mode,
        hpo_items=hpo_items,
        tokenizer=gen_tok,
        model=gen_model,
        max_new_tokens=args.gen_max_new_tokens,
        per_hpo=args.per_hpo,
        max_attempts_per_hpo=args.max_attempts_per_hpo,
        batch_size=gen_batch_size,
    )

    groups = group_expressions_by_hpo(expr_records)
    print(f"Total HPO groups (after shard): {len(groups)}")

    if groups:
        sample_group = groups[0]
        sample_expr = [er.current_expression for er in sample_group.expressions]
        if not sample_expr:
            sample_expr = ["サンプル"]
        sample_judge = [JudgeResult(3, 3, 3, False)]
    else:
        sample_group = HPOGroup(
            hpo_id="HP:0000000",
            hpo_name_ja="サンプル",
            hpo_name_en="sample",
            category="symptom",
            source="leaf",
            expressions=[],
        )
        sample_expr = ["サンプル"]
        sample_judge = [JudgeResult(3, 3, 3, False)]

    judge_batch_size = auto_tune_batch_size_for_judge(
        mode=mode,
        tokenizer=judge_tok,
        model=judge_model,
        hpo_sample=sample_group,
        expr_sample=sample_expr,
        max_new_tokens=args.judge_max_new_tokens,
    )
    refine_batch_size = auto_tune_batch_size_for_refine(
        mode=mode,
        tokenizer=refine_tok,
        model=refine_model,
        hpo_sample=sample_group,
        expr_sample=sample_expr,
        judge_sample=sample_judge,
        max_new_tokens=args.refine_max_new_tokens,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("w", encoding="utf-8")

    total_groups = len(groups)

    for idx, hpo_group in enumerate(
        tqdm(groups, desc=f"{mode} judge+refine", unit="hpo"),
        start=1,
    ):
        # judge+refine 本体
        outputs = process_hpo_group_multi_round(
            mode=mode,
            hpo=hpo_group,
            judge_tok=judge_tok,
            judge_model=judge_model,
            judge_batch_size=judge_batch_size,
            judge_max_new_tokens=args.judge_max_new_tokens,
            refine_tok=refine_tok,
            refine_model=refine_model,
            refine_batch_size=refine_batch_size,
            refine_max_new_tokens=args.refine_max_new_tokens,
            min_overall=args.min_overall,
            min_match=args.min_match,
            min_simplicity=args.min_simplicity,
            target_good=args.target_good,
            max_rounds=args.max_rounds,
            diversity_sim_high=args.diversity_sim_high,
            diversity_sim_low=args.diversity_sim_low,
            log_each_round=args.log_each_round,
            round_log_fout=round_log_fout,
            wandb_run=wandb_run,
        )
        for rec in outputs:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # W&B に進捗も送る
        if wandb_run is not None:
            wandb_run.log(
                {
                    "progress/hpo_processed": idx,
                    "progress/hpo_total": total_groups,
                    "progress/hpo_ratio": idx / total_groups if total_groups > 0 else 0.0,
                }
            )


    fout.close()
    if round_log_fout is not None:
        round_log_fout.close()

    if wandb_run is not None:
        wandb_run.finish()

    print(f"Done. Wrote to {out_path}")


if __name__ == "__main__":
    main()
