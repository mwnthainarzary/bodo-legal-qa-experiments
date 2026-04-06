# =============================================================================
# Bodo QA — Fine-tuning with IndicBERT (ai4bharat/indic-bert)
# Transformers 4.46.3 compatible
#
# IndicBERT vs standard BERT — key differences handled here:
#
#   1. ARCHITECTURE: IndicBERT is built on ALBERT (not BERT).
#      It uses AutoModelForQuestionAnswering which maps to
#      AlbertForQuestionAnswering internally. This is handled
#      automatically by the Auto class — no manual change needed.
#
#   2. TOKENIZER: IndicBERT uses SentencePiece (not WordPiece).
#      Requires: pip install sentencepiece protobuf
#      use_fast=True + from_slow=True: bypasses tiktoken, supports offset_mapping
#      seen with the fast tokenizer path in transformers 4.46.x.
#
#   3. TOKEN TYPE IDS: ALBERT uses token_type_ids differently from BERT.
#      Our QADataCollator already handles the optional token_type_ids
#      field correctly — it includes them only when present.
#
#   4. MAX LENGTH: IndicBERT supports up to 512 tokens (same as BERT).
#      Default max_len is set to 512 instead of 384 to use the full
#      capacity for Bodo's longer Devanagari script passages.
#
#   5. LANGUAGE SUPPORT: IndicBERT is pretrained on 12 Indic languages
#      including Assamese (most similar to Bodo). It natively handles
#      Devanagari and other Indic scripts used in Bodo writing.
#
# REQUIRED PACKAGES:
#   pip install transformers==4.46.3 torch sentencepiece protobuf \
#               evaluate sacrebleu rouge_score bert_score \
#               scikit-learn matplotlib numpy
# =============================================================================

import argparse
import datetime
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# =============================================================================
# 0.  DEPENDENCY PRE-FLIGHT
# =============================================================================

def _check_dependencies() -> None:
    required = {
        "sentencepiece":  "sentencepiece",   # required for IndicBERT SentencePiece tokenizer
        "google.protobuf":"protobuf",         # required for SentencePiece model loading
        "matplotlib":     "matplotlib",
        "sklearn":        "scikit-learn",
        "bert_score":     "bert_score",
        "evaluate":       "evaluate",
    }
    missing = []
    for module, pkg in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(
            f"[ERROR] Missing package(s): {' '.join(missing)}\n"
            f"        Run: pip install {' '.join(missing)}\n"
        )
        sys.exit(1)

_check_dependencies()

# ── Imports ──────────────────────────────────────────────────────────────────
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
    set_seed as hf_set_seed,
)
import evaluate as hf_evaluate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score as bert_score_fn


# =============================================================================
# 1.  DATA LOADING  (identical to BodoQA_BERT.py — same Bodo JSON format)
# =============================================================================

def extract_answer_text(answer: Any) -> Tuple[str, int]:
    """Returns (answer_text, answer_start_char). Handles dict or plain string."""
    if isinstance(answer, dict):
        return str(answer.get("text", "")).strip(), int(answer.get("answer_start", 0))
    return str(answer).strip(), 0


def parse_bodo_nested_json(data: List[Any], max_context_chars: int) -> List[Dict]:
    """
    Parses Bodo nested format:
        [ [ { id, title, context, questions: [{question, answer}] } ] ]
    Returns list of dicts: {question, context, answer_text, answer_start}
    """
    records = []
    for group in data:
        docs = group if isinstance(group, list) else [group]
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            context   = str(doc.get("context", "")).strip()[:max_context_chars]
            questions = doc.get("questions", [])
            if not isinstance(questions, list):
                continue
            for qa in questions:
                if not isinstance(qa, dict):
                    continue
                question              = str(qa.get("question", "")).strip()
                answer_text, ans_start = extract_answer_text(qa.get("answer", ""))
                if not question or not answer_text:
                    continue
                if answer_text not in context:
                    idx = context.find(answer_text)
                    if idx == -1:
                        continue       # skip unanswerable — span models need valid span
                    ans_start = idx
                records.append({
                    "question":     question,
                    "context":      context,
                    "answer_text":  answer_text,
                    "answer_start": ans_start,
                })
    return records


def load_records(path: str, max_context_chars: int = 512) -> List[Dict]:
    """Load {question, context, answer_text, answer_start} dicts from JSON."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ["data", "items", "records", "examples"]:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("JSON dataset must be a non-empty list.")

    sample = data[0]
    if (
        isinstance(sample, list) and sample
        and isinstance(sample[0], dict) and "questions" in sample[0]
    ) or (isinstance(sample, dict) and "questions" in sample):
        records = parse_bodo_nested_json(data, max_context_chars)
    else:
        q_key = next((k for k in ["question","query","input","q"] if k in sample), None)
        a_key = next((k for k in ["answer","response","output","target","a"] if k in sample), None)
        c_key = next((k for k in ["context","passage","paragraph"] if k in sample), None)
        if not q_key or not a_key:
            raise ValueError("Cannot detect question/answer keys in dataset.")
        records = []
        for row in data:
            question              = str(row.get(q_key, "")).strip()
            answer_text, ans_st  = extract_answer_text(row.get(a_key, ""))
            context               = str(row.get(c_key, "")).strip()[:max_context_chars] if c_key else answer_text
            if not question or not answer_text:
                continue
            if answer_text not in context:
                idx = context.find(answer_text)
                if idx == -1:
                    continue
                ans_st = idx
            records.append({
                "question":     question,
                "context":      context,
                "answer_text":  answer_text,
                "answer_start": ans_st,
            })

    print(f"[INFO] Loaded {len(records)} QA records from {path}")
    return records


# =============================================================================
# 2.  DATASET WRAPPER
#     IndicBERT-specific notes:
#       - SentencePiece tokenizer produces different subword splits than
#         WordPiece (BERT). Bodo Devanagari characters are split into
#         smaller SentencePiece units — this is handled automatically.
#       - ALBERT uses token_type_ids=0 for BOTH segments (not 0 and 1).
#         This is handled correctly by AutoTokenizer + offset_mapping.
#       - The sliding-window / short-sequence branching logic is identical
#         to BodoQA_BERT.py but with IndicBERT's SentencePiece in mind.
# =============================================================================

class BodoQADataset(Dataset):
    """
    Tokenises records for IndicBERT span-extraction QA.

    Handles:
      - SentencePiece offset_mapping (char-to-token alignment)
      - ALBERT token_type_ids (both segments use type 0)
      - Short Bodo sequences (no sliding window needed)
      - Long contexts (sliding window with validated safe_stride)
    """

    def __init__(
        self,
        records: List[Dict],
        tokenizer,
        max_len: int = 512,
        doc_stride: int = 128,
    ) -> None:
        self.samples   = []
        self.tokenizer = tokenizer
        self._build(records, max_len, doc_stride)

    def _build(self, records: List[Dict], max_len: int, doc_stride: int) -> None:
        """
        Tokenisation strategy — safe for all sequence lengths:

        1. model_max: read from tokenizer (512 for IndicBERT / ALBERT).
           Guard against sentinel values (some tokenizers report 1e30).

        2. effective_max = min(max_len, model_max)
           Never feed more tokens than the model's position embeddings allow.

        3. safe_stride = min(doc_stride, effective_max // 2 - 2)
           The Rust tokenizer requires: stride < (effective_max - special_tokens).
           ALBERT adds 3 special tokens ([CLS] Q [SEP] C [SEP]).
           effective_max // 2 - 2 is always safely below that threshold.

        4. Probe with truncation="only_second" + max_length=model_max
           (NOT truncation=False) to avoid the '695 > 512' warning.

        5. Branch:
           real_len > effective_max  →  sliding window (return_overflowing_tokens)
           real_len <= effective_max →  single chunk (no stride)
        """
        model_max = getattr(self.tokenizer, "model_max_length", 512)
        if model_max > 100_000:      # sentinel guard (some tokenizers use 1e30)
            model_max = 512
        effective_max = min(max_len, model_max)
        safe_stride   = max(1, min(doc_stride, effective_max // 2 - 2))

        skipped = 0
        for rec in records:
            question     = rec["question"]
            context      = rec["context"]
            answer_text  = rec["answer_text"]
            answer_start = rec["answer_start"]
            answer_end   = answer_start + len(answer_text)

            # ── Probe: measure real token length without warning ───────
            _probe_max = min(effective_max * 4, model_max)
            probe = self.tokenizer(
                question, context,
                truncation="only_second",
                max_length=_probe_max,
                add_special_tokens=True,
            )
            real_len = len(probe["input_ids"])

            # ── Choose tokenisation strategy ──────────────────────────
            if real_len > effective_max:
                use_overflow = True
                encoding = self.tokenizer(
                    question,
                    context,
                    max_length=effective_max,
                    truncation="only_second",
                    stride=safe_stride,
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                    padding="max_length",
                )
            else:
                use_overflow = False
                encoding = self.tokenizer(
                    question,
                    context,
                    max_length=effective_max,
                    truncation=True,
                    return_offsets_mapping=True,
                    padding="max_length",
                )

            # Normalise single-chunk to list format
            if not use_overflow:
                encoding = {k: [v] for k, v in encoding.items()}
                n_chunks = 1
            else:
                n_chunks = len(encoding["input_ids"])

            for i in range(n_chunks):
                offsets   = encoding["offset_mapping"][i]
                input_ids = encoding["input_ids"][i]
                attn_mask = encoding["attention_mask"][i]

                # Build seq_ids: 1 = context token, 0 = question, None = special
                if use_overflow:
                    seq_ids = encoding.sequence_ids(i)
                else:
                    # IndicBERT / ALBERT: token_type_ids are ALL 0.
                    # Cannot use token_type_ids to distinguish Q from context.
                    # Use the offset heuristic: after the second (0,0) offset
                    # (the [SEP] between Q and context), tokens belong to context.
                    seen_special = 0
                    seq_ids = []
                    for off in offsets:
                        o = tuple(off) if not isinstance(off, tuple) else off
                        if o == (0, 0):
                            seq_ids.append(None)
                            seen_special += 1
                        elif seen_special < 2:
                            seq_ids.append(0)   # question tokens
                        else:
                            seq_ids.append(1)   # context tokens

                if 1 not in seq_ids:
                    continue    # chunk has no context tokens — skip

                ctx_start_idx = next(j for j, s in enumerate(seq_ids) if s == 1)
                ctx_end_idx   = len(seq_ids) - 1
                while seq_ids[ctx_end_idx] != 1:
                    ctx_end_idx -= 1

                ctx_char_start = offsets[ctx_start_idx][0]
                ctx_char_end   = offsets[ctx_end_idx][1]

                # Answer outside this chunk → label as CLS position (0, 0)
                if answer_start < ctx_char_start or answer_end > ctx_char_end:
                    start_pos, end_pos = 0, 0
                else:
                    start_pos = ctx_start_idx
                    while start_pos <= ctx_end_idx and offsets[start_pos][0] <= answer_start:
                        start_pos += 1
                    start_pos -= 1

                    end_pos = ctx_end_idx
                    while end_pos >= ctx_start_idx and offsets[end_pos][1] >= answer_end:
                        end_pos -= 1
                    end_pos += 1

                sample = {
                    "input_ids":       input_ids,
                    "attention_mask":  attn_mask,
                    "start_positions": start_pos,
                    "end_positions":   end_pos,
                }
                if "token_type_ids" in encoding:
                    sample["token_type_ids"] = encoding["token_type_ids"][i]

                self.samples.append(sample)

        if skipped:
            print(f"[WARN] Skipped {skipped} records due to tokenisation errors.")

    def __len__(self):          return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


# =============================================================================
# 3.  MODEL & TOKENIZER
#     IndicBERT-specific:
#       - use_fast=False  →  forces SentencePiece slow tokenizer path.
#         The fast (Rust/tiktoken) path in transformers 4.46.x fails with
#         'NoneType has no attribute encode' for ALBERT-based models whose
#         tiktoken URL is None.
#       - AutoModelForQuestionAnswering maps to AlbertForQuestionAnswering
#         automatically — no explicit class import needed.
# =============================================================================

def build_model_and_tokenizer(model_name: str):
    """
    Loads IndicBERT tokenizer and QA model.

    Tokenizer loading strategy for ai4bharat/indic-bert (ALBERT + SentencePiece):

    Problem 1 — use_fast=False (slow tokenizer):
      Does NOT support return_offsets_mapping, which is required for span
      extraction QA. Raises: NotImplementedError: return_offset_mapping is
      not available when using Python tokenizers.

    Problem 2 — use_fast=True (fast tokenizer, default):
      The fast tokenizer for ALBERT tries to load via tiktoken in
      transformers 4.46.x and crashes with:
      AttributeError: 'NoneType' object has no attribute 'encode'

    Solution — use_fast=True with from_slow=True:
      from_slow=True forces the fast tokenizer to be built by CONVERTING
      from the slow SentencePiece tokenizer, bypassing the tiktoken path
      entirely. The result is a PreTrainedTokenizerFast (Rust-backed) that
      correctly supports return_offsets_mapping.
      Requires: pip install sentencepiece protobuf
    """
    print(f"[INFO] Loading tokenizer : {model_name}")
    try:
        # Primary: fast tokenizer converted from SentencePiece slow tokenizer
        # from_slow=True bypasses tiktoken and builds from the .model file
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            from_slow=True,   # convert SentencePiece → fast (supports offset_mapping)
        )
        print(f"[INFO] Tokenizer type    : {tokenizer.__class__.__name__} (fast, from_slow)")
    except Exception as e:
        print(f"[WARN] from_slow=True failed ({e}), trying AlbertTokenizerFast directly ...")
        # Fallback: explicitly request AlbertTokenizerFast
        from transformers import AlbertTokenizerFast
        tokenizer = AlbertTokenizerFast.from_pretrained(model_name)
        print(f"[INFO] Tokenizer type    : AlbertTokenizerFast (fallback)")

    print(f"[INFO] Loading model     : {model_name}")
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    total_params   = model.num_parameters()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Vocab size        : {len(tokenizer)}")
    print(f"[INFO] Total params      : {total_params:,}")
    print(f"[INFO] Trainable params  : {trainable_params:,}")
    print(f"[INFO] Architecture      : {model.__class__.__name__}")

    return tokenizer, model


# =============================================================================
# 4.  INFERENCE — span extraction (identical logic to BodoQA_BERT.py)
#     IndicBERT-specific: strip token_type_ids if ALBERT doesn't use them.
# =============================================================================

def extract_answer_from_span(
    model,
    tokenizer,
    question: str,
    context:  str,
    max_len:  int = 512,
    n_best:   int = 20,
) -> str:
    """
    Predicts start/end logits, picks the best valid span, returns answer text.
    Works identically for IndicBERT (ALBERT) and standard BERT — the model
    outputs start_logits / end_logits regardless of underlying architecture.
    """
    inputs = tokenizer(
        question,
        context,
        max_length=max_len,
        truncation="only_second",
        return_tensors="pt",
        return_offsets_mapping=True,
        padding=True,
    )
    offset_mapping = inputs.pop("offset_mapping")
    seq_ids        = inputs.sequence_ids()

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0].cpu().numpy()
    end_logits   = outputs.end_logits[0].cpu().numpy()

    offsets      = offset_mapping[0].numpy()
    context_mask = np.array([s == 1 for s in seq_ids], dtype=bool)

    # Mask out non-context positions — model cannot answer outside context
    start_logits = np.where(context_mask, start_logits, -1e9)
    end_logits   = np.where(context_mask, end_logits,   -1e9)

    start_idxs = np.argsort(start_logits)[-n_best:][::-1]
    end_idxs   = np.argsort(end_logits)[-n_best:][::-1]

    best_score  = -float("inf")
    best_answer = ""

    for s in start_idxs:
        for e in end_idxs:
            if e < s or e - s + 1 > 50:
                continue
            score = float(start_logits[s] + end_logits[e])
            if score > best_score:
                best_score  = score
                char_start  = int(offsets[s][0])
                char_end    = int(offsets[e][1])
                best_answer = context[char_start:char_end].strip()

    return best_answer if best_answer else "[NO ANSWER]"


# =============================================================================
# 5.  FULL EVALUATION  (BLEU · ROUGE · BERTScore · Cosine · Exact Match)
#     IndicBERT-specific:
#       - Cosine similarity uses IndicBERT's ALBERT encoder hidden states.
#         Access via model.albert (not model.bert or model.roberta).
#         We use a generic fallback that works for any architecture.
#       - BERTScore lang="hi" — Hindi/Devanagari is the closest language
#         supported by the bert_score library to Bodo.
# =============================================================================

def _get_encoder_embeddings(model, tokenizer, text: str, device) -> np.ndarray:
    """
    Returns mean-pooled encoder hidden states for a text string.
    Works for ALBERT (IndicBERT), BERT, DistilBERT, and RoBERTa by
    iterating named children to find the base encoder module.
    """
    toks = tokenizer(
        text or " ", return_tensors="pt",
        max_length=128, truncation=True, padding=True,
    ).to(device)

    # Try known attribute names for each architecture family
    base = (
        getattr(model, "albert",     None) or   # IndicBERT / ALBERT
        getattr(model, "bert",       None) or   # BERT / DistilBERT
        getattr(model, "roberta",    None) or   # RoBERTa
        getattr(model, "distilbert", None)       # DistilBERT (alt attr)
    )
    if base is None:
        # Generic fallback: first child that is not the QA head
        base = next(
            (m for n, m in model.named_children()
             if n not in {"qa_outputs", "classifier", "dropout"}),
            model,
        )

    with torch.no_grad():
        out    = base(**toks)
        hidden = out.last_hidden_state
    return hidden.mean(dim=1).squeeze().cpu().float().numpy().reshape(1, -1)


def run_full_evaluation(
    model,
    tokenizer,
    test_records: List[Dict],
    out_dir: str,
    max_len: int = 512,
    n_best:  int = 20,
) -> None:
    """
    Runs span-extraction inference on all test records and saves:
        evaluation_scores.txt  — aggregate BLEU, ROUGE, BERTScore, Cosine, EM
        test_results.txt       — per-sample predictions + per-sample cosine
    """
    os.makedirs(out_dir, exist_ok=True)
    bleu_metric  = hf_evaluate.load("sacrebleu")
    rouge_metric = hf_evaluate.load("rouge")

    predictions: List[str] = []
    references:  List[str] = []
    questions:   List[str] = []
    contexts:    List[str] = []

    print(f"[INFO] Running inference on {len(test_records)} test samples ...")
    model.eval()

    for rec in test_records:
        predicted = extract_answer_from_span(
            model, tokenizer,
            question=rec["question"],
            context=rec["context"],
            max_len=max_len,
            n_best=n_best,
        )
        predictions.append(predicted)
        references.append(rec["answer_text"])
        questions.append(rec["question"])
        contexts.append(rec["context"])

    # ── BLEU ──────────────────────────────────────────────────────────
    bleu_result = bleu_metric.compute(
        predictions=predictions, references=[[r] for r in references]
    )
    bleu_score = round(bleu_result["score"], 4)

    # ── ROUGE ─────────────────────────────────────────────────────────
    rouge_result = rouge_metric.compute(
        predictions=predictions, references=references, use_stemmer=False
    )

    # ── BERTScore ─────────────────────────────────────────────────────
    print("[INFO] Computing BERTScore ...")
    P, R, F1 = bert_score_fn(
        predictions, references,
        lang="hi",                    # closest Devanagari language
        rescale_with_baseline=False,
        verbose=False,
    )
    bert_p  = round(P.mean().item(),  4)
    bert_r  = round(R.mean().item(),  4)
    bert_f1 = round(F1.mean().item(), 4)

    # ── Cosine Similarity (IndicBERT ALBERT encoder) ──────────────────
    print("[INFO] Computing cosine similarity ...")
    cos_scores: List[float] = []
    for pred, ref in zip(predictions, references):
        pred_vec = _get_encoder_embeddings(model, tokenizer, pred, model.device)
        ref_vec  = _get_encoder_embeddings(model, tokenizer, ref,  model.device)
        cos_scores.append(float(cosine_similarity(pred_vec, ref_vec)[0][0]))

    cos_mean = round(float(np.mean(cos_scores)), 4)
    cos_std  = round(float(np.std(cos_scores)),  4)

    # ── Exact Match ───────────────────────────────────────────────────
    exact     = sum(p.strip().lower() == r.strip().lower() for p, r in zip(predictions, references))
    exact_pct = round(exact / len(references) * 100, 2) if references else 0.0

    ts    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep_m = "=" * 64
    sep_s = "-" * 64

    # ── evaluation_scores.txt ─────────────────────────────────────────
    eval_path = os.path.join(out_dir, "evaluation_scores.txt")
    with open(eval_path, "w", encoding="utf-8") as f:
        f.write(f"{sep_m}\n")
        f.write("Bodo QA  --  IndicBERT Evaluation Scores\n")
        f.write(f"Saved        : {ts}\n")
        f.write(f"Test samples : {len(test_records)}\n")
        f.write(f"{sep_m}\n\n")

        f.write("BLEU\n")
        f.write(f"{sep_s}\n")
        f.write(f"  BLEU Score          : {bleu_score}\n\n")

        f.write("ROUGE\n")
        f.write(f"{sep_s}\n")
        f.write(f"  ROUGE-1             : {round(rouge_result['rouge1'], 4)}\n")
        f.write(f"  ROUGE-2             : {round(rouge_result['rouge2'], 4)}\n")
        f.write(f"  ROUGE-L             : {round(rouge_result['rougeL'], 4)}\n\n")

        f.write("BERTScore  (lang=hi, Devanagari proxy)\n")
        f.write(f"{sep_s}\n")
        f.write(f"  Precision           : {bert_p}\n")
        f.write(f"  Recall              : {bert_r}\n")
        f.write(f"  F1                  : {bert_f1}\n\n")

        f.write("Cosine Similarity  (IndicBERT ALBERT encoder mean-pool)\n")
        f.write(f"{sep_s}\n")
        f.write(f"  Mean                : {cos_mean}\n")
        f.write(f"  Std Dev             : {cos_std}\n\n")

        f.write("Exact Match\n")
        f.write(f"{sep_s}\n")
        f.write(f"  Exact Match         : {exact} / {len(test_records)}  ({exact_pct:.2f}%)\n\n")
        f.write(f"{sep_m}\n")

    print(f"[INFO] Evaluation scores saved   --> {eval_path}")

    # ── test_results.txt ──────────────────────────────────────────────
    test_path = os.path.join(out_dir, "test_results.txt")
    with open(test_path, "w", encoding="utf-8") as f:
        f.write(f"{sep_m}\n")
        f.write("Bodo QA  --  IndicBERT Test Results\n")
        f.write(f"Saved        : {ts}\n")
        f.write(f"Total samples: {len(test_records)}\n")
        f.write(f"{sep_m}\n\n")

        for i, (q, ctx, pred, ref, cos) in enumerate(
            zip(questions, contexts, predictions, references, cos_scores), start=1
        ):
            match    = pred.strip().lower() == ref.strip().lower()
            ctx_disp = (ctx[:120] + "...") if len(ctx) > 120 else ctx
            f.write(f"[Sample {i}]\n")
            f.write(f"  Question          : {q}\n")
            f.write(f"  Context           : {ctx_disp}\n")
            f.write(f"  Expected Answer   : {ref}\n")
            f.write(f"  Predicted Answer  : {pred}\n")
            f.write(f"  Cosine Similarity : {cos:.4f}\n")
            f.write(f"  Exact Match       : {'Yes' if match else 'No'}\n")
            f.write(f"{sep_s}\n\n")

        f.write(f"{sep_m}\n")
        f.write(f"  Summary  --  Exact Match: {exact}/{len(test_records)} ({exact_pct:.2f}%)\n")
        f.write(f"{sep_m}\n")

    print(f"[INFO] Test results saved        --> {test_path}")


# =============================================================================
# 6.  TRAIN-LOSS GRAPH
# =============================================================================

def save_loss_graph(log_history: List[Dict], out_dir: str, model_name: str) -> None:
    """Saves Loss vs Epochs PNG to <out_dir>/train_loss_graph.png."""
    os.makedirs(out_dir, exist_ok=True)

    train_by_epoch: Dict[float, List[float]] = {}
    for entry in log_history:
        if "loss" in entry and "epoch" in entry and "eval_loss" not in entry:
            ep = round(float(entry["epoch"]), 2)
            train_by_epoch.setdefault(ep, []).append(entry["loss"])

    train_epochs = sorted(train_by_epoch.keys())
    train_losses = [float(np.mean(train_by_epoch[e])) for e in train_epochs]
    eval_epochs  = [e["epoch"]     for e in log_history if "eval_loss" in e]
    eval_losses  = [e["eval_loss"] for e in log_history if "eval_loss" in e]

    if not train_epochs:
        print("[WARN] No training loss entries found — skipping graph.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_epochs, train_losses, marker="o", linewidth=2, markersize=5,
            color="#2563EB", label="Train Loss")
    if eval_epochs:
        ax.plot(eval_epochs, eval_losses, marker="s", linewidth=2, markersize=5,
                linestyle="--", color="#DC2626", label="Eval Loss")

    if train_losses:
        min_i = int(np.argmin(train_losses))
        ax.annotate(
            f"min {train_losses[min_i]:.4f}",
            xy=(train_epochs[min_i], train_losses[min_i]),
            xytext=(12, 12), textcoords="offset points",
            fontsize=9, color="#2563EB",
            arrowprops=dict(arrowstyle="->", color="#2563EB"),
        )

    short_name = model_name.split("/")[-1]
    ax.set_title(f"Training Loss vs Epochs  —  {short_name}  (Bodo QA)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss",  fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.set_xlim(left=0)
    plt.tight_layout()

    graph_path = os.path.join(out_dir, "train_loss_graph.png")
    fig.savefig(graph_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Loss graph saved          --> {graph_path}")


# =============================================================================
# 7.  HYPERPARAMETERS TXT
# =============================================================================

def save_hyperparameters(
    args, out_dir: str, vocab_size: int,
    train_size: int, test_size: int, arch_name: str,
) -> None:
    """Saves all training hyperparameters to <out_dir>/hyperparameters.txt."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "hyperparameters.txt")
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep  = "=" * 64

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{sep}\n")
        f.write("Bodo QA  --  IndicBERT Hyperparameters\n")
        f.write(f"Saved              : {ts}\n")
        f.write(f"{sep}\n\n")

        f.write("Model\n")
        f.write(f"  model_name           : {args.model_name}\n")
        f.write(f"  hf_architecture      : {arch_name}\n")
        f.write(f"  base_architecture    : ALBERT (parameter-shared encoder)\n")
        f.write(f"  head                 : AutoModelForQuestionAnswering\n")
        f.write(f"  tokenizer_type       : AlbertTokenizerFast (from_slow=True, supports offset_mapping)\n")
        f.write(f"  indic_languages      : 12 (Assamese, Hindi, Bengali, ...)\n\n")

        f.write("Data\n")
        f.write(f"  train_path           : {args.train_path}\n")
        f.write(f"  test_path            : {args.test_path}\n")
        f.write(f"  train_samples        : {train_size}\n")
        f.write(f"  test_samples         : {test_size}\n")
        f.write(f"  max_context_chars    : {args.max_context_chars}\n\n")

        f.write("Tokenisation\n")
        f.write(f"  vocab_size           : {vocab_size}\n")
        f.write(f"  max_len              : {args.max_len}\n")
        f.write(f"  doc_stride           : {args.doc_stride}\n")
        f.write(f"  truncation           : only_second (context only)\n\n")

        f.write("Optimisation\n")
        f.write(f"  epochs               : {args.epochs}\n")
        f.write(f"  batch_size           : {args.batch_size}\n")
        f.write(f"  grad_accum_steps     : {args.grad_accum}\n")
        f.write(f"  effective_batch      : {args.batch_size * args.grad_accum}\n")
        f.write(f"  learning_rate        : {args.lr}\n")
        f.write(f"  lr_scheduler         : linear\n")
        f.write(f"  warmup_ratio         : 0.1\n")
        f.write(f"  weight_decay         : 0.01\n")
        f.write(f"  gradient_clip        : 1.0 (HuggingFace default)\n\n")

        f.write("Span Extraction (inference)\n")
        f.write(f"  n_best               : {args.n_best}\n")
        f.write(f"  max_answer_length    : 50 tokens\n\n")

        f.write("Runtime\n")
        f.write(f"  seed                 : {args.seed}\n")
        f.write(f"  fp16                 : {torch.cuda.is_available()}\n")
        f.write(f"  device               : {'cuda' if torch.cuda.is_available() else 'cpu'}\n")
        f.write(f"  output_dir           : {args.output_dir}\n\n")
        f.write(f"{sep}\n")

    print(f"[INFO] Hyperparameters saved     --> {path}")


# =============================================================================
# 8.  PER-EPOCH EVAL LOG  (incremental callback)
# =============================================================================

def save_epoch_eval_log(log_history: List[Dict], out_dir: str) -> None:
    """Writes per-epoch loss + span_acc to epoch_eval_log.txt."""
    os.makedirs(out_dir, exist_ok=True)
    eval_entries = [e for e in log_history if "eval_loss" in e]
    sep = "=" * 64
    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(os.path.join(out_dir, "epoch_eval_log.txt"), "w", encoding="utf-8") as f:
        f.write(f"{sep}\n")
        f.write("Bodo QA  --  IndicBERT Per-Epoch Evaluation Log\n")
        f.write(f"Last updated : {ts}\n")
        f.write(f"{sep}\n\n")

        best_epoch, best_val = None, None
        for entry in eval_entries:
            epoch = entry.get("epoch", "?")
            f.write(f"Epoch {epoch}\n")
            for key in sorted(entry.keys()):
                if key == "epoch":
                    continue
                f.write(f"  {key.replace('eval_',''):<24}: {entry[key]}\n")
            f.write("\n")
            mv = entry.get("eval_loss")
            if mv is not None:
                if best_val is None or mv < best_val:
                    best_epoch, best_val = epoch, mv

        f.write(f"{sep}\n")
        if best_epoch is not None:
            f.write(f"Best epoch : {best_epoch}  (eval_loss = {best_val:.4f}, lower is better)\n")
        f.write(f"{sep}\n")


class EpochEvalCallback(TrainerCallback):
    """Saves epoch_eval_log.txt incrementally after each eval — safe on crash."""
    def __init__(self, out_dir: str) -> None:
        self.out_dir = out_dir

    def on_evaluate(self, args, state, control, **kwargs):
        if state.log_history:
            save_epoch_eval_log(state.log_history, self.out_dir)


# =============================================================================
# 9.  TRAINER METRICS  (span accuracy proxy — full metrics done post-training)
# =============================================================================

def compute_qa_metrics(eval_pred):
    """
    Token-level start/end accuracy — proxy metric used during training.
    Full text metrics (BLEU, ROUGE, BERTScore, Cosine) computed post-training.

    Receives (after Trainer gathers across batches and converts to numpy):
      logits : tuple of two numpy arrays
               logits[0] : start_logits  shape (total_samples, seq_len)
               logits[1] : end_logits    shape (total_samples, seq_len)
      labels : tuple of two numpy arrays
               labels[0] : start_positions  shape (total_samples,)
               labels[1] : end_positions    shape (total_samples,)

    Shape contract:
      prediction_step returns (start_tensor, end_tensor) as a tuple.
      The Trainer concatenates each element independently along dim=0,
      so total_samples = sum of all eval batch sizes.
      argmax over seq_len axis gives predicted token positions, same
      shape as the label arrays — no broadcast error.
    """
    logits, labels = eval_pred

    # Unpack tuple — each is (total_samples, seq_len)
    start_logits = logits[0]
    end_logits   = logits[1]

    # Unpack label tuple — each is (total_samples,)
    start_labels = labels[0]
    end_labels   = labels[1]

    # argmax over seq_len → (total_samples,) predicted positions
    start_preds = np.argmax(start_logits, axis=-1)
    end_preds   = np.argmax(end_logits,   axis=-1)

    return {
        "start_acc": round(float(np.mean(start_preds == start_labels)), 4),
        "end_acc":   round(float(np.mean(end_preds   == end_labels)),   4),
        "span_acc":  round(float(np.mean(
            (start_preds == start_labels) & (end_preds == end_labels)
        )), 4),
    }


class QADataCollator:
    """
    Collates span-QA samples into batches for AlbertForQuestionAnswering.

    ROOT CAUSE of "unexpected keyword argument 'labels'":
      AlbertForQuestionAnswering.forward() accepts start_positions and
      end_positions but NOT a generic 'labels' argument. The previous
      version included a 'labels' key for compute_qa_metrics, which the
      HuggingFace Trainer passed directly to model.forward() — crash.

    FIX: Remove 'labels' from the batch entirely. The custom QATrainer
    below overrides compute_loss() to extract start/end positions from
    the batch and call the model correctly, and overrides
    prediction_step() to capture logits for compute_qa_metrics without
    needing a 'labels' key in the batch.
    """
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids       = torch.tensor([f["input_ids"]       for f in features], dtype=torch.long)
        attention_mask  = torch.tensor([f["attention_mask"]   for f in features], dtype=torch.long)
        start_positions = torch.tensor([f["start_positions"]  for f in features], dtype=torch.long)
        end_positions   = torch.tensor([f["end_positions"]    for f in features], dtype=torch.long)

        batch = {
            "input_ids":        input_ids,
            "attention_mask":   attention_mask,
            "start_positions":  start_positions,
            "end_positions":    end_positions,
            # NO "labels" key — AlbertForQuestionAnswering does not accept it
        }
        if "token_type_ids" in features[0]:
            batch["token_type_ids"] = torch.tensor(
                [f["token_type_ids"] for f in features], dtype=torch.long
            )
        return batch


class QATrainer(Trainer):
    """
    Custom Trainer for AlbertForQuestionAnswering (IndicBERT).

    Overrides two methods:

    1. compute_loss():
       Passes start_positions and end_positions to model.forward()
       explicitly, which is the correct API for all QA models.
       This avoids the 'labels' key being forwarded to the model.

    2. prediction_step():
       Returns (loss, (start_logits, end_logits), (start_pos, end_pos))
       so compute_qa_metrics receives the right structure without needing
       a 'labels' field in the batch.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Pop span labels — pass them as explicit kwargs, not via **inputs
        start_positions = inputs.pop("start_positions", None)
        end_positions   = inputs.pop("end_positions",   None)

        outputs = model(
            **inputs,
            start_positions=start_positions,
            end_positions=end_positions,
        )
        loss = outputs.loss

        # Restore so the batch is not mutated for other uses
        if start_positions is not None:
            inputs["start_positions"] = start_positions
        if end_positions is not None:
            inputs["end_positions"] = end_positions

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        FIX — TypeError: Unsupported types (numpy.ndarray) passed to
        `_pad_across_processes`.

        Root cause: the Trainer's gather/pad utilities require all returned
        values to be torch.Tensors (or None). Returning numpy arrays from
        prediction_step causes _pad_across_processes to crash because it
        only knows how to handle tensors, not ndarrays.

        Fix: keep everything as torch.Tensor until the Trainer has finished
        its internal gather step. compute_qa_metrics receives numpy arrays
        automatically — the Trainer converts tensors to numpy before calling
        compute_metrics, so we must NOT do that conversion here.

        Return signature expected by Trainer:
            (loss_scalar_tensor_or_None,
             logits_tensor_or_tuple_of_tensors,
             labels_tensor_or_None)
        """
        inputs = self._prepare_inputs(inputs)

        start_positions = inputs.get("start_positions")
        end_positions   = inputs.get("end_positions")

        # Model input — exclude span label keys
        model_inputs = {
            k: v for k, v in inputs.items()
            if k not in {"start_positions", "end_positions"}
        }

        with torch.no_grad():
            outputs = model(**model_inputs)

        # ── Loss ────────────────────────────────────────────────────
        loss = None
        if start_positions is not None and end_positions is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = (
                loss_fn(outputs.start_logits, start_positions) +
                loss_fn(outputs.end_logits,   end_positions)
            ) / 2

        if prediction_loss_only:
            return (loss, None, None)

        # ── Logits — return as TUPLE of two tensors ─────────────────
        # Root cause of "shapes (8,) vs (668,)":
        #   torch.stack([s, e], dim=0) creates shape (2, B, L). When the
        #   Trainer concatenates across eval batches along dim=0 it gets
        #   (2*n_batches, B, L) — completely wrong shape.
        #
        # Correct approach: return a TUPLE (start_logits, end_logits).
        # The Trainer handles tuples by concatenating each element
        # independently along dim=0, giving:
        #   logits[0]: (total_samples, seq_len)  — start logits
        #   logits[1]: (total_samples, seq_len)  — end logits
        # compute_qa_metrics then receives this tuple as a numpy array
        # pair and can argmax correctly.
        logits = (
            outputs.start_logits.detach(),   # torch.Tensor (B, seq_len)
            outputs.end_logits.detach(),     # torch.Tensor (B, seq_len)
        )

        # ── Labels — two separate tensors as a tuple ─────────────────
        # Same reason: use a tuple so each is concatenated independently.
        #   labels[0]: (total_samples,)  — start positions
        #   labels[1]: (total_samples,)  — end positions
        # compute_qa_metrics unpacks with labels[0], labels[1].
        labels = None
        if start_positions is not None and end_positions is not None:
            labels = (
                start_positions.detach(),   # torch.Tensor (B,)
                end_positions.detach(),     # torch.Tensor (B,)
            )

        return (loss, logits, labels)


# =============================================================================
# 10.  MAIN TRAINING PIPELINE
# =============================================================================

def train(args) -> None:
    hf_set_seed(args.seed)

    # ── Load train / test files ───────────────────────────────────────
    train_records = load_records(args.train_path, args.max_context_chars)
    test_records  = load_records(args.test_path,  args.max_context_chars)

    if len(train_records) < 2:
        raise ValueError("Training file must have at least 2 valid QA records.")
    if len(test_records)  < 1:
        raise ValueError("Test file must have at least 1 valid QA record.")

    print(f"[INFO] Train: {len(train_records)} | Test: {len(test_records)}")

    # ── Model + tokenizer ─────────────────────────────────────────────
    tokenizer, model = build_model_and_tokenizer(args.model_name)
    arch_name = model.__class__.__name__

    # ── Save hyperparameters before training starts ───────────────────
    save_hyperparameters(
        args, args.output_dir,
        vocab_size=len(tokenizer),
        train_size=len(train_records),
        test_size=len(test_records),
        arch_name=arch_name,
    )

    # ── Datasets ──────────────────────────────────────────────────────
    train_ds = BodoQADataset(train_records, tokenizer, args.max_len, args.doc_stride)
    test_ds  = BodoQADataset(test_records,  tokenizer, args.max_len, args.doc_stride)
    print(f"[INFO] Train token samples: {len(train_ds)} | Test token samples: {len(test_ds)}")

    # ── Training arguments ────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        logging_steps=5,
        save_total_limit=2,
        report_to="none",
    )

    # ── Trainer ───────────────────────────────────────────────────────
    # QATrainer overrides compute_loss and prediction_step to prevent
    # 'labels' from being forwarded to AlbertForQuestionAnswering.forward()
    trainer = QATrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=QADataCollator(),
        compute_metrics=compute_qa_metrics,
        callbacks=[
            EpochEvalCallback(args.output_dir),
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
    )

    print("[INFO] Starting training ...")
    trainer.train()

    # ── Save final model ──────────────────────────────────────────────
    final_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[INFO] Model saved --> {final_dir}")

    # ── Train-loss graph ──────────────────────────────────────────────
    save_loss_graph(trainer.state.log_history, args.output_dir, args.model_name)

    # ── Final epoch eval log ──────────────────────────────────────────
    save_epoch_eval_log(trainer.state.log_history, args.output_dir)

    # ── Full text-quality evaluation on test set ──────────────────────
    run_full_evaluation(
        model=model,
        tokenizer=tokenizer,
        test_records=test_records,
        out_dir=args.output_dir,
        max_len=args.max_len,
        n_best=args.n_best,
    )

    print(f"\n[INFO] All outputs saved to: {args.output_dir}/")
    print("       ├── hyperparameters.txt")
    print("       ├── epoch_eval_log.txt")
    print("       ├── train_loss_graph.png")
    print("       ├── evaluation_scores.txt")
    print("       ├── test_results.txt")
    print("       └── final_model/")


# =============================================================================
# 11.  CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune IndicBERT (ai4bharat/indic-bert) on Bodo QA. Transformers 4.46.3."
    )
    parser.add_argument("--train_path",        type=str,   required=True,
                        help="Path to training JSON file.")
    parser.add_argument("--test_path",         type=str,   required=True,
                        help="Path to test JSON file.")
    parser.add_argument("--output_dir",        type=str,   default="indicbert_output",
                        help="Directory for all outputs.")
    parser.add_argument(
        "--model_name", type=str, default="ai4bharat/indic-bert",
        help=(
            "IndicBERT model variants:\n"
            "  ai4bharat/indic-bert          (default — 12 Indic languages)\n"
            "  ai4bharat/IndicBERTv2-SS      (v2 IndicBERT, SentencePiece)\n"
            "  ai4bharat/IndicBERTv2-MLM-Sam (v2 with sampling, best for QA)\n"
        ),
    )
    parser.add_argument("--epochs",            type=int,   default=10)
    parser.add_argument("--batch_size",        type=int,   default=8)
    parser.add_argument("--grad_accum",        type=int,   default=4)
    parser.add_argument("--lr",                type=float, default=2e-5,
                        help="IndicBERT/ALBERT recommended LR is 2e-5.")
    parser.add_argument("--max_len",           type=int,   default=512,
                        help="Max tokens. IndicBERT supports up to 512.")
    parser.add_argument("--doc_stride",        type=int,   default=128,
                        help="Stride for sliding window over long contexts.")
    parser.add_argument("--n_best",            type=int,   default=20,
                        help="Top-N start/end candidates at inference.")
    parser.add_argument("--max_context_chars", type=int,   default=512)
    parser.add_argument("--seed",              type=int,   default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()


# =============================================================================
# USAGE
# =============================================================================
#
# Install:
#   pip install transformers==4.46.3 torch sentencepiece protobuf \
#               evaluate sacrebleu rouge_score bert_score \
#               scikit-learn matplotlib numpy
#
# Run (default IndicBERT v1):
#   python BodoQA_IndicBERT.py \
#       --train_path bodo_train.json \
#       --test_path  bodo_test.json  \
#       --output_dir ./indicbert_output
#
# Run (IndicBERT v2 — best quality):
#   python BodoQA_IndicBERT.py \
#       --train_path bodo_train.json \
#       --test_path  bodo_test.json  \
#       --output_dir ./indicbert_v2_output \
#       --model_name ai4bharat/IndicBERTv2-MLM-Sam \
#       --lr 2e-5
#
# Output files in output_dir/:
#   hyperparameters.txt    model config, data paths, all training settings
#   epoch_eval_log.txt     eval_loss + span_acc per epoch (live updates)
#   train_loss_graph.png   Train Loss vs Epochs graph (PNG)
#   evaluation_scores.txt  BLEU, ROUGE, BERTScore, Cosine Sim, Exact Match
#   test_results.txt       per-sample: question, expected, predicted, cosine
#   final_model/           saved IndicBERT weights + SentencePiece tokenizer
#
# IndicBERT vs other models for Bodo:
#   ai4bharat/indic-bert    — trained on 12 Indic langs, Devanagari native
#   bert-base-multilingual  — 104 langs, less Indic-specific
#   xlm-roberta-base        — strong multilingual but less Indic-focused
# =============================================================================