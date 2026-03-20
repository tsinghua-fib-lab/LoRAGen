# -*- coding: utf-8 -*-

def choose_main_metric(perf_metrics, task_type=None):
    """
    Pick a primary metric according to task_type, with a reasonable fallback order.
    """
    print(f"[DEBUG] choose_main_metric: task_type={task_type}, perf_metrics={perf_metrics}")

    if task_type in ["classification", "multiple_choice"]:
        acc = perf_metrics.get("accuracy")
        if isinstance(acc, (int, float)):
            return float(acc), "accuracy"

    elif task_type == "question_answering":
        f1 = perf_metrics.get("f1")
        if isinstance(f1, (int, float)):
            return float(f1), "f1"

    elif task_type == "text_generation":
        rouge = perf_metrics.get("rougeL")
        if isinstance(rouge, (int, float)):
            return float(rouge), "rougeL"

    elif task_type == "unknown":
        # Default to F1 if task is unknown
        f1 = perf_metrics.get("f1", 0.0)
        return float(f1), "f1"

    # Fallback priority when task_type is missing/unrecognized
    priority = ["f1", "accuracy", "exact_match", "rougeL", "bleu", "bertscore"]
    for key in priority:
        val = perf_metrics.get(key)
        if isinstance(val, (int, float)):
            return float(val), key

    return 0.0, "unknown"


# ---- BERTScore offline helper (lazy init) ----
import os, logging, torch
from bert_score import BERTScorer

logger = logging.getLogger(__name__)

# Use local cache if provided; set by env: export BERTSCORE_MODEL_DIR=/path/to/roberta-large
_BERT_MODEL_PATH = os.environ.get("BERTSCORE_MODEL_DIR", "roberta-large")

_BERTSCORER = None
def _get_bertscorer():
    """
    Lazily create a single BERTScorer instance.
    If model_type is a local directory, it will not attempt to fetch from internet.
    """
    global _BERTSCORER
    if _BERTSCORER is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _BERTSCORER = BERTScorer(
            model_type=_BERT_MODEL_PATH,
            device=device,
            rescale_with_baseline=False
        )
    return _BERTSCORER


def compute_metrics(task_type, predictions, references):
    """
    Compute task-aware metrics. Returns a dict with keys depending on task_type.
    For text generation, includes ROUGE-1/L, SacreBLEU and BERTScore-F1 (offline-friendly).
    """
    from sklearn.metrics import accuracy_score, f1_score
    from evaluate import load as load_metric
    import numpy as np
    import string

    def normalize(text):
        if text is None:
            return ""
        return str(text).lower().strip().translate(str.maketrans('', '', string.punctuation))

    def squad_exact_match(prediction, reference):
        return int(normalize(prediction) == normalize(reference))

    def squad_f1_score(prediction, reference):
        # Simple token-level overlap F1 for QA-style answers
        pred_tokens = normalize(prediction).split()
        ref_tokens = normalize(reference).split()
        common = set(pred_tokens) & set(ref_tokens)
        if len(common) == 0:
            return 0.0
        precision = len(common) / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
        recall = len(common) / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _sanitize_seq(seq):
        """
        Replace empty/None with a placeholder to avoid warnings in metric libs.
        """
        out = []
        for x in seq:
            s = "" if x is None else str(x)
            s = s.strip()
            out.append(s if s else "[EMPTY]")
        return out

    def compute_nlg_metrics(predictions, references):
        preds = _sanitize_seq(predictions)
        refs  = _sanitize_seq(references)

        rouge = load_metric("rouge")
        rouge_result = rouge.compute(predictions=preds, references=refs)
        rouge1 = float(round(rouge_result.get("rouge1", 0.0) * 100, 4))
        rougeL = float(round(rouge_result.get("rougeL", 0.0) * 100, 4))

        try:
            scorer = _get_bertscorer()
            P, R, F1 = scorer.score(preds, refs)
            bertscore_f1 = float(round(float(F1.mean().item()) * 100, 4))
        except Exception as e:
            logger.warning("BERTScore failed (offline-safe): %s; set to NaN", e)
            bertscore_f1 = float("nan")

        bleu_score = 0.0
        try:
            bleu = load_metric("sacrebleu")
            bleu_result = bleu.compute(predictions=preds, references=[[r] for r in refs])
            bleu_score = float(round(bleu_result.get("bleu", 0.0) * 100, 4))
        except Exception as e:
            logger.warning("SacreBLEU failed: %s; set to 0.0", e)
            bleu_score = 0.0

        return {
            "rouge1": rouge1,
            "rougeL": rougeL,
            "bertscore": bertscore_f1,
            "bleu": bleu_score
        }

    if not predictions or not references:
        return {
            "accuracy": 0.0,
            "f1": 0.0,
            "exact_match": 0.0,
            "rouge1": 0.0,
            "rougeL": 0.0,
            "bleu": 0.0,
            "bertscore": 0.0,
            "note": "[WARN] Empty predictions or references"
        }

    # normalize for classification-like metrics
    preds_norm = [normalize(p) for p in predictions]
    refs_norm = [normalize(r) for r in references]

    if task_type in ['classification', 'multiple_choice']:
        try:
            acc = accuracy_score(refs_norm, preds_norm)
            f1 = f1_score(refs_norm, preds_norm, average='macro', zero_division=0)
        except Exception:
            acc, f1 = 0.0, 0.0
        return {
            "accuracy": float(round(acc * 100, 4)),
            "f1": float(round(f1 * 100, 4))
        }

    elif task_type == 'question_answering':
        exact_scores = [squad_exact_match(p, r) for p, r in zip(predictions, references)]
        f1_scores = [squad_f1_score(p, r) for p, r in zip(predictions, references)]
        nlg_metrics = compute_nlg_metrics(predictions, references)
        return {
            "exact_match": float(round(np.mean(exact_scores) * 100, 4)),
            "f1": float(round(np.mean(f1_scores) * 100, 4)),
            **nlg_metrics
        }

    elif task_type == 'text_generation':
        nlg_metrics = compute_nlg_metrics(predictions, references)
        return {
            "rouge1": nlg_metrics.get("rouge1", 0.0),
            "rougeL": nlg_metrics.get("rougeL", 0.0),
            "bertscore": nlg_metrics.get("bertscore", 0.0),
            "bleu": nlg_metrics.get("bleu", 0.0)
        }

    elif task_type == "unknown":
        acc = accuracy_score(refs_norm, preds_norm)
        f1 = f1_score(refs_norm, preds_norm, average='macro', zero_division=0)
        nlg_metrics = compute_nlg_metrics(predictions, references)
        return {
            "accuracy": float(round(acc * 100, 4)),
            "f1": float(round(f1 * 100, 4)),
            **nlg_metrics,
            "note": "[WARN] Task type 'unknown', fallback values used."
        }

    else:
        return {
            "accuracy": 0.0,
            "f1": 0.0,
            "note": f"[WARN] Unknown task type '{task_type}', fallback values used."
        }
