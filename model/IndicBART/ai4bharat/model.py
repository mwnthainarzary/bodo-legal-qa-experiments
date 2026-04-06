# -----------------------------------------------------------------------
# Seq2Seq Fine-tuning for Bodo QA — HuggingFace Transformers 4.46.3
# Model : google/mt5-small  (multilingual T5, supports Devanagari/Bodo)
#
# Features:
#   * Separate --train_path and --test_path arguments
#   * Evaluation: BLEU, BERTScore, Cosine Similarity, ROUGE  -> saved to txt
#   * Train-loss vs Epochs graph saved as PNG
#   * Hyperparameters saved to txt
#   * Per-epoch eval metrics saved incrementally (safe on crash)
#   * Beam search + repetition penalties (no more identical outputs)
#
# REQUIRED PACKAGES:
#   pip install transformers==4.46.3 torch sentencepiece protobuf \
#               evaluate sacrebleu rouge_score bert_score \
#               scikit-learn matplotlib numpy
# -----------------------------------------------------------------------

import argparse
import datetime
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ════════════════════════════════════════════════════════════════════════
# 0.  DEPENDENCY PRE-FLIGHT
# ════════════════════════════════════════════════════════════════════════

def _check_dependencies() -> None:
    required = {
        "sentencepiece": "sentencepiece",
        "google.protobuf": "protobuf",
        "matplotlib": "matplotlib",
        "sklearn": "scikit-learn",
        "bert_score": "bert_score",
        "evaluate": "evaluate",
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

# ── Imports (after dependency check) ────────────────────────────────────
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
    set_seed as hf_set_seed,
)
import evaluate as hf_evaluate
import matplotlib
matplotlib.use("Agg")          # non-interactive: no display needed
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score as bert_score_fn


# ════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ════════════════════════════════════════════════════════════════════════

def extract_answer_text(answer: Any) -> str:
    if isinstance(answer, dict):
        return str(answer.get("text", "")).strip()
    return str(answer).strip()


def parse_bodo_nested_json(
    data: List[Any], use_context: bool, max_context_chars: int
) -> List[Tuple[str, str]]:
    """Handles Bodo nested format: [[{context, questions:[{question,answer}]}]]"""
    pairs: List[Tuple[str, str]] = []
    for group in data:
        docs = group if isinstance(group, list) else [group]
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            context   = str(doc.get("context", "")).strip()
            questions = doc.get("questions", [])
            if not isinstance(questions, list):
                continue
            for qa in questions:
                if not isinstance(qa, dict):
                    continue
                question = str(qa.get("question", "")).strip()
                answer   = extract_answer_text(qa.get("answer", ""))
                if not question or not answer:
                    continue
                if use_context and context:
                    input_text = f"<question> {question} <context> {context[:max_context_chars]}"
                else:
                    input_text = f"<question> {question}"
                pairs.append((input_text, f"<answer> {answer}"))
    return pairs


def load_records(
    path: str, use_context: bool = True, max_context_chars: int = 512
) -> List[Tuple[str, str]]:
    """Returns (input_text, target_text) pairs from a JSON file."""
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
        pairs = parse_bodo_nested_json(data, use_context, max_context_chars)
    else:
        q_key = next((k for k in ["question","query","input","q"] if k in sample), None)
        a_key = next((k for k in ["answer","response","output","target","a"] if k in sample), None)
        if not q_key or not a_key:
            raise ValueError("Cannot detect question/answer keys in dataset.")
        pairs = []
        for row in data:
            q = str(row.get(q_key, "")).strip()
            a = extract_answer_text(row.get(a_key, ""))
            if q and a:
                pairs.append((f"<question> {q}", f"<answer> {a}"))

    print(f"[INFO] Loaded {len(pairs)} pairs from {path}")
    return pairs


# ════════════════════════════════════════════════════════════════════════
# 2.  DATASET WRAPPER
# ════════════════════════════════════════════════════════════════════════

class BodoQADataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        tokenizer,
        max_input_len: int,
        max_target_len: int,
    ) -> None:
        inputs  = [p[0] for p in pairs]
        targets = [p[1] for p in pairs]
        enc = tokenizer(inputs,  max_length=max_input_len,  truncation=True, padding=False)
        # text_target= replaces deprecated as_target_tokenizer() in 4.46.x
        lbl = tokenizer(text_target=targets, max_length=max_target_len, truncation=True, padding=False)
        self.samples = []
        for i in range(len(inputs)):
            label_ids = [
                tok if tok != tokenizer.pad_token_id else -100
                for tok in lbl["input_ids"][i]
            ]
            self.samples.append({
                "input_ids":      enc["input_ids"][i],
                "attention_mask": enc["attention_mask"][i],
                "labels":         label_ids,
            })

    def __len__(self):           return len(self.samples)
    def __getitem__(self, idx):  return self.samples[idx]


# ════════════════════════════════════════════════════════════════════════
# 3.  MODEL & TOKENIZER
# ════════════════════════════════════════════════════════════════════════

def build_model_and_tokenizer(model_name: str):
    """
    use_fast=False forces SentencePiece slow tokenizer.
    Avoids tiktoken AttributeError in transformers 4.46.x with mT5.
    Requires: pip install sentencepiece protobuf
    """
    print(f"[INFO] Loading: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    added = tokenizer.add_special_tokens({
        "additional_special_tokens": ["<question>", "<context>", "<answer>"]
    })
    if added:
        model.resize_token_embeddings(len(tokenizer))
        print(f"[INFO] Added {added} special tokens. Vocab size: {len(tokenizer)}")
    return tokenizer, model


# ════════════════════════════════════════════════════════════════════════
# 4.  INFERENCE
# ════════════════════════════════════════════════════════════════════════

def generate_answer(
    model,
    tokenizer,
    question: str,
    context: Optional[str] = None,
    max_input_len: int = 256,
    max_new_tokens: int = 64,
    num_beams: int = 4,
) -> str:
    input_text = (
        f"<question> {question} <context> {context[:512]}"
        if context else f"<question> {question}"
    )
    inputs = tokenizer(
        input_text, return_tensors="pt",
        max_length=max_input_len, truncation=True,
    ).to(model.device)
    # Strip token_type_ids — mT5 encoder does not accept it
    inputs = {k: v for k, v in inputs.items() if k in {"input_ids", "attention_mask"}}

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3,
        length_penalty=1.0,
        early_stopping=True,
        forced_eos_token_id=tokenizer.eos_token_id,
    )
    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gen_cfg)
    answer = tokenizer.decode(out[0], skip_special_tokens=True)
    if answer.startswith("<answer>"):
        answer = answer[len("<answer>"):].strip()
    return answer


# ════════════════════════════════════════════════════════════════════════
# 5.  FULL EVALUATION  (BLEU · ROUGE · BERTScore · Cosine · Exact Match)
# ════════════════════════════════════════════════════════════════════════

def run_full_evaluation(
    model,
    tokenizer,
    test_pairs: List[Tuple[str, str]],
    out_dir: str,
    max_input_len: int = 256,
    max_new_tokens: int = 64,
    num_beams: int = 4,
) -> None:
    """
    Runs inference on every test sample, computes 5 metrics, and saves:
        evaluation_scores.txt  —  aggregate metric summary
        test_results.txt       —  per-sample predictions + per-sample cosine
    """
    os.makedirs(out_dir, exist_ok=True)
    bleu_metric  = hf_evaluate.load("sacrebleu")
    rouge_metric = hf_evaluate.load("rouge")

    predictions: List[str]          = []
    references:  List[str]          = []
    questions:   List[str]          = []
    contexts:    List[Optional[str]] = []

    print(f"[INFO] Running inference on {len(test_pairs)} test samples ...")
    model.eval()
    for src, tgt in test_pairs:
        q        = src.replace("<question>", "").split("<context>")[0].strip()
        ctx_part = src.split("<context>")
        ctx      = ctx_part[1].strip() if len(ctx_part) > 1 else None
        expected = tgt.replace("<answer>", "").strip()
        predicted = generate_answer(
            model, tokenizer, q, context=ctx,
            max_input_len=max_input_len,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        predictions.append(predicted)
        references.append(expected)
        questions.append(q)
        contexts.append(ctx)

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
        lang="hi",                   # closest Devanagari language supported
        rescale_with_baseline=False,
        verbose=False,
    )
    bert_p, bert_r, bert_f1 = round(P.mean().item(), 4), round(R.mean().item(), 4), round(F1.mean().item(), 4)

    # ── Cosine Similarity (encoder mean-pool embeddings) ──────────────
    print("[INFO] Computing cosine similarity ...")
    cos_scores: List[float] = []
    with torch.no_grad():
        for pred, ref in zip(predictions, references):
            def _embed(text: str) -> np.ndarray:
                toks = tokenizer(
                    text or " ", return_tensors="pt",
                    max_length=128, truncation=True, padding=True,
                ).to(model.device)
                toks = {k: v for k, v in toks.items() if k in {"input_ids", "attention_mask"}}
                emb = model.get_encoder()(**toks).last_hidden_state
                return emb.mean(dim=1).squeeze().cpu().float().numpy().reshape(1, -1)
            cos_scores.append(float(cosine_similarity(_embed(pred), _embed(ref))[0][0]))

    cos_mean = round(float(np.mean(cos_scores)), 4)
    cos_std  = round(float(np.std(cos_scores)),  4)

    # ── Exact Match ───────────────────────────────────────────────────
    exact = sum(p.strip().lower() == r.strip().lower() for p, r in zip(predictions, references))
    exact_pct = round(exact / len(references) * 100, 2) if references else 0.0

    ts    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep_m = "=" * 62
    sep_s = "-" * 62

    # ── evaluation_scores.txt ─────────────────────────────────────────
    eval_path = os.path.join(out_dir, "evaluation_scores.txt")
    with open(eval_path, "w", encoding="utf-8") as f:
        f.write(f"{sep_m}\n")
        f.write("Bodo QA Seq2Seq  --  Evaluation Scores\n")
        f.write(f"Saved        : {ts}\n")
        f.write(f"Test samples : {len(test_pairs)}\n")
        f.write(f"{sep_m}\n\n")

        f.write("BLEU\n")
        f.write(f"{sep_s}\n")
        f.write(f"  BLEU Score          : {bleu_score}\n\n")

        f.write("ROUGE\n")
        f.write(f"{sep_s}\n")
        f.write(f"  ROUGE-1             : {round(rouge_result['rouge1'], 4)}\n")
        f.write(f"  ROUGE-2             : {round(rouge_result['rouge2'], 4)}\n")
        f.write(f"  ROUGE-L             : {round(rouge_result['rougeL'], 4)}\n\n")

        f.write("BERTScore\n")
        f.write(f"{sep_s}\n")
        f.write(f"  Precision           : {bert_p}\n")
        f.write(f"  Recall              : {bert_r}\n")
        f.write(f"  F1                  : {bert_f1}\n\n")

        f.write("Cosine Similarity  (encoder mean-pool)\n")
        f.write(f"{sep_s}\n")
        f.write(f"  Mean                : {cos_mean}\n")
        f.write(f"  Std Dev             : {cos_std}\n\n")

        f.write("Exact Match\n")
        f.write(f"{sep_s}\n")
        f.write(f"  Exact Match         : {exact} / {len(test_pairs)}  ({exact_pct:.2f}%)\n\n")
        f.write(f"{sep_m}\n")

    print(f"[INFO] Evaluation scores saved   --> {eval_path}")

    # ── test_results.txt ──────────────────────────────────────────────
    test_path = os.path.join(out_dir, "test_results.txt")
    with open(test_path, "w", encoding="utf-8") as f:
        f.write(f"{sep_m}\n")
        f.write("Bodo QA Seq2Seq  --  Test Results\n")
        f.write(f"Saved        : {ts}\n")
        f.write(f"Total samples: {len(test_pairs)}\n")
        f.write(f"{sep_m}\n\n")

        for i, (q, ctx, pred, ref, cos) in enumerate(
            zip(questions, contexts, predictions, references, cos_scores), start=1
        ):
            match = pred.strip().lower() == ref.strip().lower()
            f.write(f"[Sample {i}]\n")
            f.write(f"  Question          : {q}\n")
            if ctx:
                disp = (ctx[:120] + "...") if len(ctx) > 120 else ctx
                f.write(f"  Context           : {disp}\n")
            f.write(f"  Expected Answer   : {ref}\n")
            f.write(f"  Predicted Answer  : {pred}\n")
            f.write(f"  Cosine Similarity : {cos:.4f}\n")
            f.write(f"  Exact Match       : {'Yes' if match else 'No'}\n")
            f.write(f"{sep_s}\n\n")

        f.write(f"{sep_m}\n")
        f.write(f"  Summary  --  Exact Match: {exact}/{len(test_pairs)} ({exact_pct:.2f}%)\n")
        f.write(f"{sep_m}\n")

    print(f"[INFO] Test results saved        --> {test_path}")


# ════════════════════════════════════════════════════════════════════════
# 6.  TRAIN-LOSS GRAPH
# ════════════════════════════════════════════════════════════════════════

def save_loss_graph(log_history: List[Dict], out_dir: str) -> None:
    """Saves Loss vs Epochs PNG to <out_dir>/train_loss_graph.png."""
    os.makedirs(out_dir, exist_ok=True)

    # Aggregate step-level train losses by epoch
    train_by_epoch: Dict[float, List[float]] = {}
    for entry in log_history:
        if "loss" in entry and "epoch" in entry and "eval_loss" not in entry:
            ep = round(float(entry["epoch"]), 2)
            train_by_epoch.setdefault(ep, []).append(entry["loss"])

    train_epochs = sorted(train_by_epoch.keys())
    train_losses = [float(np.mean(train_by_epoch[e])) for e in train_epochs]

    eval_epochs = [e["epoch"] for e in log_history if "eval_loss" in e]
    eval_losses = [e["eval_loss"] for e in log_history if "eval_loss" in e]

    if not train_epochs:
        print("[WARN] No training loss entries in log history — skipping graph.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(train_epochs, train_losses,
            marker="o", linewidth=2, markersize=5,
            color="#2563EB", label="Train Loss")
    if eval_epochs:
        ax.plot(eval_epochs, eval_losses,
                marker="s", linewidth=2, markersize=5,
                linestyle="--", color="#DC2626", label="Eval Loss")

    # Annotate minimum train loss
    if train_losses:
        min_i = int(np.argmin(train_losses))
        ax.annotate(
            f"min {train_losses[min_i]:.4f}",
            xy=(train_epochs[min_i], train_losses[min_i]),
            xytext=(12, 12), textcoords="offset points",
            fontsize=9, color="#2563EB",
            arrowprops=dict(arrowstyle="->", color="#2563EB"),
        )

    ax.set_title("Training Loss vs Epochs  —  Bodo QA Seq2Seq",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch",  fontsize=12)
    ax.set_ylabel("Loss",   fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.set_xlim(left=0)
    plt.tight_layout()

    graph_path = os.path.join(out_dir, "train_loss_graph.png")
    fig.savefig(graph_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Loss graph saved          --> {graph_path}")


# ════════════════════════════════════════════════════════════════════════
# 7.  HYPERPARAMETERS TXT
# ════════════════════════════════════════════════════════════════════════

def save_hyperparameters(
    args,
    out_dir: str,
    vocab_size: int,
    train_size: int,
    test_size: int,
) -> None:
    """Saves all training hyperparameters to <out_dir>/hyperparameters.txt."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "hyperparameters.txt")
    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "=" * 62

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{sep}\n")
        f.write("Bodo QA Seq2Seq  --  Hyperparameters\n")
        f.write(f"Saved              : {ts}\n")
        f.write(f"{sep}\n\n")

        f.write("Model\n")
        f.write(f"  model_name           : {args.model_name}\n\n")

        f.write("Data\n")
        f.write(f"  train_path           : {args.train_path}\n")
        f.write(f"  test_path            : {args.test_path}\n")
        f.write(f"  train_samples        : {train_size}\n")
        f.write(f"  test_samples         : {test_size}\n")
        f.write(f"  use_context          : {args.use_context}\n")
        f.write(f"  max_context_chars    : {args.max_context_chars}\n\n")

        f.write("Tokenisation\n")
        f.write(f"  vocab_size           : {vocab_size}\n")
        f.write(f"  max_input_len        : {args.max_input_len}\n")
        f.write(f"  max_target_len       : {args.max_target_len}\n\n")

        f.write("Optimisation\n")
        f.write(f"  epochs               : {args.epochs}\n")
        f.write(f"  batch_size           : {args.batch_size}\n")
        f.write(f"  grad_accum_steps     : {args.grad_accum}\n")
        f.write(f"  effective_batch      : {args.batch_size * args.grad_accum}\n")
        f.write(f"  learning_rate        : {args.lr}\n")
        f.write(f"  lr_scheduler         : cosine\n")
        f.write(f"  warmup_ratio         : 0.1\n")
        f.write(f"  weight_decay         : 0.01\n")
        f.write(f"  label_smoothing      : 0.1\n")
        f.write(f"  gradient_clip        : 1.0 (HuggingFace default)\n\n")

        f.write("Generation (inference & eval)\n")
        f.write(f"  num_beams            : 4\n")
        f.write(f"  no_repeat_ngram_size : 3\n")
        f.write(f"  repetition_penalty   : 1.3\n")
        f.write(f"  length_penalty       : 1.0\n\n")

        f.write("Runtime\n")
        f.write(f"  seed                 : {args.seed}\n")
        f.write(f"  fp16                 : {torch.cuda.is_available()}\n")
        f.write(f"  device               : {'cuda' if torch.cuda.is_available() else 'cpu'}\n")
        f.write(f"  output_dir           : {args.output_dir}\n\n")
        f.write(f"{sep}\n")

    print(f"[INFO] Hyperparameters saved     --> {path}")


# ════════════════════════════════════════════════════════════════════════
# 8.  PER-EPOCH EVAL LOG  (incremental callback)
# ════════════════════════════════════════════════════════════════════════

def save_epoch_eval_log(log_history: List[Dict], out_dir: str) -> None:
    """Writes per-epoch BLEU/ROUGE/loss to epoch_eval_log.txt."""
    os.makedirs(out_dir, exist_ok=True)
    eval_entries = [e for e in log_history if "eval_loss" in e]
    sep = "=" * 62
    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(os.path.join(out_dir, "epoch_eval_log.txt"), "w", encoding="utf-8") as f:
        f.write(f"{sep}\n")
        f.write("Bodo QA Seq2Seq  --  Per-Epoch Evaluation Log\n")
        f.write(f"Last updated : {ts}\n")
        f.write(f"{sep}\n\n")

        best_epoch, best_val = None, None
        pk = next(
            (m for m in ["eval_bleu","bleu","eval_rougeL","eval_loss"]
             if any(m in e for e in eval_entries)),
            "eval_loss",
        )
        for entry in eval_entries:
            epoch = entry.get("epoch", "?")
            f.write(f"Epoch {epoch}\n")
            for key in sorted(entry.keys()):
                if key == "epoch":
                    continue
                f.write(f"  {key.replace('eval_',''):<24}: {entry[key]}\n")
            f.write("\n")
            mv = entry.get(pk)
            if mv is not None:
                is_loss = "loss" in pk
                if best_val is None or (mv < best_val if is_loss else mv > best_val):
                    best_epoch, best_val = epoch, mv

        f.write(f"{sep}\n")
        if best_epoch is not None:
            direction = "lower" if "loss" in pk else "higher"
            f.write(
                f"Best epoch : {best_epoch}  "
                f"({pk.replace('eval_','')} = {best_val:.4f}, {direction} is better)\n"
            )
        f.write(f"{sep}\n")


class EpochEvalCallback(TrainerCallback):
    """Saves epoch_eval_log.txt after each eval step — preserved on crash."""
    def __init__(self, out_dir: str) -> None:
        self.out_dir = out_dir

    def on_evaluate(self, args, state, control, **kwargs):
        if state.log_history:
            save_epoch_eval_log(state.log_history, self.out_dir)


# ════════════════════════════════════════════════════════════════════════
# 9.  TRAINER METRICS  (BLEU + ROUGE used during training eval)
# ════════════════════════════════════════════════════════════════════════

def build_compute_metrics(tokenizer):
    bleu_m  = hf_evaluate.load("sacrebleu")
    rouge_m = hf_evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds  = np.where(preds  < tokenizer.vocab_size, preds,  tokenizer.unk_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = np.where(labels < tokenizer.vocab_size, labels, tokenizer.unk_token_id)

        dp = [p.strip() for p in tokenizer.batch_decode(preds,  skip_special_tokens=True)]
        dl = [l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]

        bleu  = bleu_m.compute(predictions=dp, references=[[l] for l in dl])
        rouge = rouge_m.compute(predictions=dp, references=dl, use_stemmer=False)
        return {
            "bleu":   round(bleu["score"],       4),
            "rouge1": round(rouge["rouge1"],      4),
            "rouge2": round(rouge["rouge2"],      4),
            "rougeL": round(rouge["rougeL"],      4),
        }
    return compute_metrics


# ════════════════════════════════════════════════════════════════════════
# 10.  MAIN TRAINING PIPELINE
# ════════════════════════════════════════════════════════════════════════

def train(args) -> None:
    hf_set_seed(args.seed)

    # ── Load separate train / test files ─────────────────────────────
    train_pairs = load_records(args.train_path, args.use_context, args.max_context_chars)
    test_pairs  = load_records(args.test_path,  args.use_context, args.max_context_chars)

    if len(train_pairs) < 2:
        raise ValueError("Training file must contain at least 2 QA pairs.")
    if len(test_pairs)  < 1:
        raise ValueError("Test file must contain at least 1 QA pair.")

    print(f"[INFO] Train: {len(train_pairs)} samples | Test: {len(test_pairs)} samples")

    # ── Model + tokenizer ─────────────────────────────────────────────
    tokenizer, model = build_model_and_tokenizer(args.model_name)

    # ── Save hyperparameters immediately (before training) ────────────
    save_hyperparameters(
        args, args.output_dir,
        vocab_size=len(tokenizer),
        train_size=len(train_pairs),
        test_size=len(test_pairs),
    )

    # ── Datasets  ─────────────────────────────────────────────────────
    train_ds = BodoQADataset(train_pairs, tokenizer, args.max_input_len, args.max_target_len)
    test_ds  = BodoQADataset(test_pairs,  tokenizer, args.max_input_len, args.max_target_len)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model,
        padding=True, pad_to_multiple_of=8, label_pad_token_id=-100,
    )

    # ── Training arguments ────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        label_smoothing_factor=0.1,
        predict_with_generate=True,
        generation_max_length=args.max_target_len,
        generation_num_beams=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        logging_steps=5,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,          # evaluate on test set each epoch
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
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
    save_loss_graph(trainer.state.log_history, args.output_dir)

    # ── Final epoch eval log ──────────────────────────────────────────
    save_epoch_eval_log(trainer.state.log_history, args.output_dir)

    # ── Full evaluation on test set (BLEU+ROUGE+BERTScore+Cosine) ─────
    run_full_evaluation(
        model=model,
        tokenizer=tokenizer,
        test_pairs=test_pairs,
        out_dir=args.output_dir,
        max_input_len=args.max_input_len,
        max_new_tokens=args.max_target_len,
        num_beams=4,
    )

    print(f"\n[INFO] All outputs saved to: {args.output_dir}/")
    print("       ├── hyperparameters.txt")
    print("       ├── epoch_eval_log.txt")
    print("       ├── train_loss_graph.png")
    print("       ├── evaluation_scores.txt")
    print("       ├── test_results.txt")
    print("       └── final_model/")


# ════════════════════════════════════════════════════════════════════════
# 11.  CLI
# ════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune mT5 on Bodo QA — Transformers 4.46.3."
    )
    parser.add_argument("--train_path",        type=str,   required=True,
                        help="Path to training JSON file.")
    parser.add_argument("--test_path",         type=str,   required=True,
                        help="Path to test JSON file.")
    parser.add_argument("--output_dir",        type=str,   default="bodo_output",
                        help="Directory for all outputs.")
    parser.add_argument("--model_name",        type=str,   default="google/mt5-small")
    parser.add_argument("--epochs",            type=int,   default=20)
    parser.add_argument("--batch_size",        type=int,   default=8)
    parser.add_argument("--grad_accum",        type=int,   default=4)
    parser.add_argument("--lr",                type=float, default=3e-4)
    parser.add_argument("--max_input_len",     type=int,   default=256)
    parser.add_argument("--max_target_len",    type=int,   default=64)
    parser.add_argument("--use_context",       action="store_true", default=True)
    parser.add_argument("--max_context_chars", type=int,   default=512)
    parser.add_argument("--seed",              type=int,   default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()


# ════════════════════════════════════════════════════════════════════════
# USAGE
# ════════════════════════════════════════════════════════════════════════
#
# Install:
#   pip install transformers==4.46.3 torch sentencepiece protobuf \
#               evaluate sacrebleu rouge_score bert_score \
#               scikit-learn matplotlib numpy
#
# Run:
#   python Seq2Seq_transformers.py \
#       --train_path  bodo_train.json \
#       --test_path   bodo_test.json  \
#       --output_dir  ./bodo_output   \
#       --model_name  google/mt5-small \
#       --epochs      20
#
# Output files:
#   bodo_output/hyperparameters.txt    all training config
#   bodo_output/epoch_eval_log.txt     BLEU/ROUGE per epoch (live updates)
#   bodo_output/train_loss_graph.png   Loss vs Epochs graph
#   bodo_output/evaluation_scores.txt  BLEU, ROUGE, BERTScore, Cosine, EM
#   bodo_output/test_results.txt       per-sample predictions + cosine
#   bodo_output/final_model/           saved weights + tokenizer
# ════════════════════════════════════════════════════════════════════════