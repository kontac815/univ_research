#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
annotate_with_ngrams_medtxtner_nerneg.py

- 生成済み患者表現の MedTXTNER 埋め込み (.npy) + JSONL (HPO_ID 付き) を読み込み
- 入力文を Sudachi で分かち書き（トークン＋文字オフセットを取得）
- MedTXTNER を NER として使い、「医療表現っぽい領域」の char マスクを作る
- その領域にかかるトークンだけを対象に n-gram (1〜max_n) を生成
- 極端に短い / ひらがな1〜2文字だけ 等の n-gram は事前に除外
- 各 n-gram を MedTXTNER で埋め込み、FAISS で生成患者表現に類似検索
- 類似度が閾値以上の候補だけを採用し、スコア順に重なりを解決
- Negation は
    - 周辺文字列に対する正規表現
    - 周辺トークンに対する否定トリガー
  の両方で判定し、assertion = "present" / "absent" / "uncertain" を付与
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import faiss

# Sudachi
try:
    from sudachipy import dictionary as sudachi_dictionary
    from sudachipy import tokenizer as sudachi_tokenizer

    SUDACHI_AVAILABLE = True
except ImportError:
    SUDACHI_AVAILABLE = False


# ============================
# MedTXTNER Embedding + NER
# ============================

class MedTxtNerHelper:
    """
    MedTXTNER を使って
      - 文埋め込み（mean pooling）
      - NER（token label + offset）
    の両方を行うヘルパ。
    """

    def __init__(self, model_name: str = "sociocom/MedTXTNER", device: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_emb = torch.sum(last_hidden_state * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return sum_emb / sum_mask

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 256,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        MedTXTNER を使って文ベクトルを作る。

        AutoModelForTokenClassification は通常 TokenClassifierOutput を返し、
        last_hidden_state を直接持たないので、
        output_hidden_states=True で hidden_states[-1] を使う。
        """
        all_embs: List[torch.Tensor] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            # hidden_states を出力させる
            out = self.model(**enc, output_hidden_states=True, return_dict=True)
            # 最終層の隠れ状態を使う: (B, L, H)
            token_emb = out.hidden_states[-1]

            # attention_mask で mean pooling
            sent_emb = self._mean_pool(token_emb, enc["attention_mask"])

            if normalize:
                sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)

            all_embs.append(sent_emb.cpu())

        if not all_embs:
            hidden = self.model.config.hidden_size
            return torch.empty(0, hidden)

        return torch.cat(all_embs, dim=0)


    @torch.no_grad()
    def get_ner_char_mask(
        self,
        text: str,
        max_length: int = 256,
        allowed_label_prefixes: Optional[List[str]] = None,
    ) -> List[bool]:
        """
        MedTXTNER を NER として動かし、
        - label != "O" かつ（必要なら）allowed_label_prefixes に合致する token の offset を
          True とする char マスクを返す。

        fast tokenizer でない場合（offset_mapping 非対応）のときは、
        マスクなし（= None 相当）を返し、呼び出し側で「全文許可」にフォールバックする。
        """
        if not text:
            return []

        # fast tokenizer でない場合は offset_mapping が使えないので
        # 「マスクなし」を示すために空リストを返す
        if not getattr(self.tokenizer, "is_fast", False):
            # 呼び出し側では char_mask が空のとき token_allowed = [True] * len(tokens)
            # にフォールバックする実装になっている
            return []

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}

        out = self.model(**enc)
        logits = out.logits  # (1, seq_len, num_labels)
        pred_ids = logits.argmax(-1)[0].tolist()

        mask = [False] * len(text)

        for idx, (start, end) in enumerate(offsets):
            if start == end:
                # [CLS] / [SEP] 等
                continue
            label = self.id2label[pred_ids[idx]]
            if label == "O":
                continue

            # "B-disease" / "I-disease" → "disease" のように suffix を取る
            base_label = label
            if "-" in label:
                _, base_label = label.split("-", 1)

            if allowed_label_prefixes:
                # 例: ["disease", "symptom"] のような prefix でフィルタ
                if not any(base_label.startswith(p) for p in allowed_label_prefixes):
                    continue

            # 該当 token の char 範囲を True にする
            for pos in range(start, min(end, len(mask))):
                mask[pos] = True

        return mask



# ============================
# Sudachi Wrapper
# ============================

class SudachiTokenizerWrapper:
    """
    Sudachi で分かち書きしつつ、文字オフセットも返す。
    """

    def __init__(self):
        if SUDACHI_AVAILABLE:
            self._dict = sudachi_dictionary.Dictionary()
            self._tokenizer = self._dict.create()
            self._mode = sudachi_tokenizer.Tokenizer.SplitMode.C
        else:
            self._dict = None
            self._tokenizer = None
            self._mode = None

    def tokenize_with_offsets(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        if not SUDACHI_AVAILABLE or self._tokenizer is None:
            # Sudachi がない場合の簡易版（空白区切り＋適当な offset）
            tokens = text.split()
            spans: List[Tuple[int, int]] = []
            offset = 0
            for tok in tokens:
                start = text.find(tok, offset)
                if start < 0:
                    start = offset
                end = start + len(tok)
                spans.append((start, end))
                offset = end
            return tokens, spans

        ms = self._tokenizer.tokenize(text, self._mode)
        tokens = []
        spans = []
        for m in ms:
            s = m.surface()
            if not s.strip():
                continue
            tokens.append(s)
            spans.append((m.begin(), m.end()))
        return tokens, spans


# ============================
# Negation Detection (強化版)
# ============================

# token ベースで見たい否定トリガー
NEG_TOKEN_SET = {
    "ない", "ありません", "ないです", "無し", "なし",
    "見られない", "みられない",
    "認めない", "認められない", "認められず",
    "所見なし", "異常なし", "症状なし", "問題なし", "問題ない",
    "陰性",
}

# 文字列ベースで見る否定パターン
NEG_REGEXES = [
    r"(所見|異常|症状|問題)なし",
    r"(所見|異常|症状|問題)は?認められない",
    r"(?:は|が)?認められない",
    r"(?:は|が)?見られない",
    r"(?:は|が)?ない",
    r"陰性",
    r"訴え[^。]*なし",
    r"症状なく",
]

# 「否定できない」= absence ではない（不確実）とみなす
UNCERTAIN_REGEXES = [
    r"否定できない",
    r"否定はできない",
    r"可能性[^。]*否定できない",
]

CHAR_WINDOW = 20  # span 周辺の文字ウィンドウ


def detect_assertion(
    text: str,
    tokens: List[str],
    token_spans: List[Tuple[int, int]],
    span_start: int,
    span_end: int,
) -> str:
    """
    span_start, span_end: token index での [start, end)。
    戻り値: "present" / "absent" / "uncertain"
    """
    if not tokens:
        return "present"

    # スパンの文字範囲
    span_char_start = token_spans[span_start][0]
    span_char_end = token_spans[span_end - 1][1]

    left_char = max(0, span_char_start - CHAR_WINDOW)
    right_char = min(len(text), span_char_end + CHAR_WINDOW)
    ctx = text[left_char:right_char]

    # まず「否定できない」系 → uncertain
    for pat in UNCERTAIN_REGEXES:
        if re.search(pat, ctx):
            return "uncertain"

    # 明確な negation パターン
    for pat in NEG_REGEXES:
        if re.search(pat, ctx):
            return "absent"

    # token ベースでもチェック
    left_tok = 0
    right_tok = len(tokens)
    for i, (s, e) in enumerate(token_spans):
        if e <= left_char:
            left_tok = i + 1
        if s < right_char:
            right_tok = i + 1
    window_tokens = tokens[left_tok:right_tok]
    for t in window_tokens:
        if t in NEG_TOKEN_SET:
            return "absent"

    return "present"


# ============================
# n-gram 生成 & フィルタ
# ============================

@dataclass
class SpanCandidate:
    start: int
    end: int
    text: str
    best_hpo: str
    best_expr: str
    score: float
    assertion: str


def is_trivial_span(text: str) -> bool:
    s = text.strip()
    if not s:
        return True

    # 句読点などの左右は除去して判定（例: "。生後" -> "生後"）
    # 全角スペース/全角カンマも混ざることがあるので追加
    s = s.strip(" \t\r\n　、。.,()（）[]【】「」『』:：;；，")
    if not s:
        return True

    # 末尾の助詞・接続（「出生後に」「検査で」など）を落としてからも trivial 判定する
    # ※ span_text 自体はそのまま保持し、ここではフィルタ用途のみ。
    core = s
    for suf in (
        "について", "によって", "による", "として",
        "から", "まで", "より",
        "には", "では", "とは", "にも", "でも",
        "が", "を", "に", "へ", "と", "で", "は", "も", "の", "や",
    ):
        if core.endswith(suf) and len(core) > len(suf):
            core = core[: -len(suf)]
            break
    core = core.strip(" \t\r\n　、。.,()（）[]【】「」『』:：;；，")
    if core:
        s_for_check = core
    else:
        s_for_check = s

    # 1文字は特殊な漢字だけ許す（例外でホワイトリスト化）
    if len(s_for_check) == 1:
        if s_for_check not in {"熱", "咳"}:
            return True

    # ひらがなだけ & 3文字以下は落とす
    if re.fullmatch(r"[ぁ-ゖー]+", s_for_check) and len(s_for_check) <= 3:
        return True

    # 非症状の文脈語・メタ語（単体で出ると誤爆しやすい）
    # ※「発熱」「嘔吐」などの2文字症状はここには入れない
    TRIVIAL_SINGLE_TOKENS = {
        "生後", "出生後", "新生児期", "乳児", "在胎", "妊娠", "分娩", "周産期",
        "診断", "検査", "受診", "経過", "治療", "施行",
        "開始", "継続", "確認", "評価", "結果", "所見",
        # 検査名だけのスパン（「MRIで」などに引っ張られやすい）
        "MRI", "CT", "レントゲン", "X線", "超音波", "エコー",
        # 単体だと意味が薄い変化語（複合語「視力低下」などは別）
        "低下", "増加", "上昇", "下降", "亢進", "拡大", "縮小", "遷延",
    }
    if s_for_check in TRIVIAL_SINGLE_TOKENS:
        return True

    # 時間・周産期など「状況語」だけのスパンを落とす（「出生後に」「在胎で」など）
    if re.fullmatch(
        r"(?:生後|出生後|新生児期|乳児期|幼児期|学童期|思春期|成人期|在胎|妊娠|妊娠中|産後|周産期|分娩)(?:から|より|まで|に|で|頃|ごろ)?",
        s_for_check,
    ):
        return True

    # 文脈だけを表す頻出フレーズ（実際にノイズとして出たもの）
    TRIVIAL_PHRASES = {
        "後に", "期から", "繰り返し", "導入して", "経過観察", "持続したため",
        "と診断", "と診断新生児", "があり", "あり、", "疑いで", "を認めた",
        "を受けている", "で経過観察",
    }
    if s in TRIVIAL_PHRASES:
        return True

    return False

def greedy_non_overlapping(spans, **kwargs):
    return select_non_overlapping_dp(spans, **kwargs)


def generate_ngrams(
    tokens: List[str],
    token_allowed: Optional[List[bool]] = None,
    max_n: int = 6,
) -> List[Tuple[int, int, str]]:
    """
    トークン列から 1〜max_n トークンの n-gram を生成。
    token_allowed が指定されていれば、その領域にかかる n-gram だけを出す。
    """
    if token_allowed is None:
        token_allowed = [True] * len(tokens)

    spans: List[Tuple[int, int, str]] = []
    N = len(tokens)

    for i in range(N):
        for n in range(1, max_n + 1):
            j = i + n
            if j > N:
                break

            # この n-gram に token_allowed=True のトークンが1つもなければ無視
            if not any(token_allowed[k] for k in range(i, j)):
                continue

            span_tokens = tokens[i:j]
            span_text = "".join(span_tokens)  # 必要なら " ".join(span_tokens)

            # 極端に短い・ひらがな1〜2文字などは除外
            if is_trivial_span(span_text):
                continue

            spans.append((i, j, span_text))

    return spans

# 追加：DPで非重複スパンを選ぶ（重み付き区間スケジューリング）
import bisect
import math

def select_non_overlapping_dp(spans, *, len_bonus_alpha=0.15, len_bonus_beta=0.05, treat_adjacent_as_overlap=False):
    """
    weight = score + alpha*log1p(char_len) + beta*token_len
    treat_adjacent_as_overlap=True にすると end==start も競合扱い（複合語の分裂抑制に効く）
    """
    if not spans:
        return []

    # endでソート
    spans_sorted = sorted(spans, key=lambda s: (s.end, s.start))

    ends = [s.end for s in spans_sorted]

    def compatible_end(start):
        # 互換条件：通常は end <= start（隣接OK）
        # 隣接を競合扱い：end < start
        if treat_adjacent_as_overlap:
            return start - 1
        return start

    # p[i] = i番目と両立する直前のインデックス
    p = []
    for i, s in enumerate(spans_sorted):
        ce = compatible_end(s.start)
        j = bisect.bisect_right(ends, ce) - 1
        p.append(j)

    def weight(s):
        char_len = len(s.text or "")
        token_len = max(1, s.end - s.start)
        return float(s.score) + len_bonus_alpha * math.log1p(char_len) + len_bonus_beta * token_len

    dp = [0.0] * (len(spans_sorted) + 1)
    take = [False] * (len(spans_sorted) + 1)

    for i in range(1, len(spans_sorted) + 1):
        w = weight(spans_sorted[i - 1])
        opt1 = dp[i - 1]
        opt2 = w + dp[p[i - 1] + 1]
        if opt2 > opt1:
            dp[i] = opt2
            take[i] = True
        else:
            dp[i] = opt1

    # 復元
    res = []
    i = len(spans_sorted)
    while i > 0:
        if take[i]:
            s = spans_sorted[i - 1]
            res.append(s)
            i = p[i - 1] + 1
        else:
            i -= 1

    res.reverse()
    return res



# ============================
# メタデータ読み込み
# ============================

def load_metadata_from_jsonl(
    path: Path,
    text_key: str = "patient_expression_final",
    hpo_key: str = "HPO_ID",
) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    hpos: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = (rec.get(text_key) or "").strip()
            h = (rec.get(hpo_key) or "").strip()
            if not t:
                continue
            texts.append(t)
            hpos.append(h)
    return texts, hpos


# ============================
# main
# ============================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--emb-npy", type=str, required=True,
                    help="生成患者表現の埋め込み .npy (MedTXTNER ベース, L2 正規化済み)")
    ap.add_argument("--meta-jsonl", type=str, required=True,
                    help="埋め込みに対応する JSONL (表現 + HPO_ID)")
    ap.add_argument("--text-key", type=str, default="patient_expression_final",
                    help="JSONL 中の表現キー名")
    ap.add_argument("--hpo-key", type=str, default="HPO_ID",
                    help="JSONL 中の HPO ID キー名")

    ap.add_argument("--model-name", type=str, default="sociocom/MedTXTNER",
                    help="MedTXTNER モデル名")
    ap.add_argument("--device", type=str, default=None,
                    help="'cuda' or 'cpu'（未指定なら自動判定）")
    ap.add_argument("--max-length", type=int, default=256,
                    help="埋め込み用 max_length")

    # NER 領域利用
    ap.add_argument("--use-ner-region", action="store_true",
                    help="MedTXTNER NER で非Oラベル領域だけ n-gram を展開する")
    ap.add_argument("--ner-max-length", type=int, default=256,
                    help="NER 用 max_length")
    ap.add_argument("--ner-label-prefixes", type=str, default="",
                    help="症候として扱うラベルのプレフィックスをカンマ区切りで指定 (空なら非O全部)")

    # n-gram & FAISS
    ap.add_argument("--max-n", type=int, default=6,
                    help="n-gram の最大長 (トークン数)")
    ap.add_argument("--topk", type=int, default=3,
                    help="各 n-gram に対する FAISS 上位何件を見るか")
    ap.add_argument("--min-score", type=float, default=0.6,
                    help="この類似度以上の候補のみ採用")

    args = ap.parse_args()

    # 埋め込み & メタデータ読み込み
    emb = np.load(args.emb_npy).astype("float32")
    n_items, dim = emb.shape

    texts, hpo_ids = load_metadata_from_jsonl(
        Path(args.meta_jsonl), text_key=args.text_key, hpo_key=args.hpo_key
    )
    if len(texts) != n_items:
        print(f"[WARN] embeddings ({n_items}) != metadata ({len(texts)})")

    # FAISS Index
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    print(f"[INFO] FAISS index built: {index.ntotal} vectors, dim={dim}")

    # MedTXTNER helper & tokenizer
    helper = MedTxtNerHelper(model_name=args.model_name, device=args.device)
    tokenizer = SudachiTokenizerWrapper()

    # NER ラベル prefix 設定
    ner_label_prefixes: Optional[List[str]] = None
    if args.ner_label_prefixes:
        ner_label_prefixes = [s.strip() for s in args.ner_label_prefixes.split(",") if s.strip()]

    print("\n=== n-gram + MedTXTNER + NER領域 + Negation アノテーション ===")
    print("文章を入力してください（空行 or Ctrl-D で終了）\n")

    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            print("\n[INFO] EOF, bye.")
            break

        if not line:
            print("[INFO] empty input, bye.")
            break

        # Sudachi でトークン + offset
        tokens, token_spans = tokenizer.tokenize_with_offsets(line)
        print(f"[DEBUG] tokens: {tokens}")

        # NER char mask
        token_allowed = [True] * len(tokens)
        if args.use_ner_region:
            ner_mask = helper.get_ner_char_mask(
                line,
                max_length=args.ner_max_length,
                allowed_label_prefixes=ner_label_prefixes,
            )
            if ner_mask and any(ner_mask):
                # NER 領域にかかっているトークンだけ True
                token_allowed = []
                for (start, end) in token_spans:
                    in_region = any(ner_mask[pos] for pos in range(start, end) if 0 <= pos < len(ner_mask))
                    token_allowed.append(in_region)
            else:
                # NER が何も取れなかったときは全トークンを対象にする
                token_allowed = [True] * len(tokens)

        # n-gram 候補生成
        spans = generate_ngrams(tokens, token_allowed=token_allowed, max_n=args.max_n)
        span_texts = [t for (_, _, t) in spans]

        if not span_texts:
            print("[INFO] no n-gram candidates.")
            continue

        # まとめて埋め込み
        q_emb = helper.encode(span_texts, batch_size=32, max_length=args.max_length, normalize=True)
        q_np = q_emb.numpy().astype("float32")

        # FAISS 検索
        D, I = index.search(q_np, args.topk)  # (num_spans, topk)

        candidates: List[SpanCandidate] = []
        for idx_span, (score_vec, idx_vec) in enumerate(zip(D, I)):
            start, end, span_text = spans[idx_span]

            best_score = -1.0
            best_hpo = ""
            best_expr = ""
            for score, expr_idx in zip(score_vec, idx_vec):
                if expr_idx < 0 or expr_idx >= len(texts):
                    continue
                if score > best_score:
                    best_score = float(score)
                    best_hpo = hpo_ids[expr_idx]
                    best_expr = texts[expr_idx]

            if best_score < args.min_score:
                continue

            assertion = detect_assertion(line, tokens, token_spans, start, end)
            candidates.append(
                SpanCandidate(
                    start=start,
                    end=end,
                    text=span_text,
                    best_hpo=best_hpo,
                    best_expr=best_expr,
                    score=best_score,
                    assertion=assertion,
                )
            )

        selected = select_non_overlapping_dp(
            candidates,
            len_bonus_alpha=0.15,
            len_bonus_beta=0.05,
            treat_adjacent_as_overlap=False,  # まずFalse推奨。複合語がまだ割れるならTrueも試す
        )

        print("\n--- Annotations ---")
        for cand in selected:
            token_span = "".join(tokens[cand.start:cand.end])
            print(
                f"span='{token_span}' (tokens[{cand.start}:{cand.end}]) "
                f"-> HPO={cand.best_hpo}, score={cand.score:.3f}, assertion={cand.assertion}"
            )
            print(f"  matched_expr='{cand.best_expr}'")
        print("-------------------\n")


if __name__ == "__main__":
    main()
