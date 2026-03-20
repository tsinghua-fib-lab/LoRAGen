# LoRAGen/stage2/train/evaluation/evalLora.py
import os
import torch
from typing import List
from evaluation.lora_utils import apply_weights_to_model
from evaluation.dataset_utils import get_examples_for_inference
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from evaluation.metric_eval_logic import compute_metrics, choose_main_metric

def evalStructuredLora(decoded_weights, task_names):
    def infer_task_type(task_name):
        t = task_name.lower().replace("lorahub_flan_t5_large-", "")
        rules = {
            "classification": [
                "amazon_polarity","yelp_polarity","app_reviews","glue_sst2","qqp","mrpc","stsb","qnli","wnli",
                "paws_wiki","anli","glue_cola","super_glue_wic","super_glue_wsc","dbpedia","trec","ag_news"
            ],
            "multiple_choice": [
                "race","dream","quail","cosmos_qa","quartz","ropes","social_i_qa","piqa","cos_e","siqa",
                "quarel","arc_challenge","wiqa","sciq_multiple_choice","qasc","proofwriter"
            ],
            "question_answering": [
                "squad","quoref","duorc","record","wiki_qa","wiki_hop","hotpotqa","kilt_tasks_hotpotqa",
                "coqa","quac","nq","trivia_qa","web_questions","sciq_direct_question","adversarial_qa","drop"
            ],
            "text_generation": [
                "cnn_dailymail","xsum","gem_xsum","newsroom","web_nlg","wiki_bio","gem_web_nlg","gem_e2e_nlg",
                "para_crawl","opus100","wmt","para_crawl_enes","title_generation","question_generation",
                "word_segment","fix_punct","true_case"
            ]
        }
        for tp, kws in rules.items():
            if any(k in t for k in kws):
                return tp
        return "unknown"

    # 1) Load base model (env var for anonymity; fallback to HF id)
    base_model_name = os.environ.get("LORAGEN_T5_BASE", "google/flan-t5-large")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Inject LoRA deltas
    model = apply_weights_to_model(model, decoded_weights)
    model.eval()

    # 2) Load eval data
    all_examples = get_examples_for_inference(task_names)
    module_inputs  = [ex["input"] for ex in all_examples]
    module_outputs = [ex["output"] for ex in all_examples]
    module_task_names = [ex["task_name"] for ex in all_examples]

    # 3) Inference
    def batched_infer(example_inputs: List[str], batch_size: int = 10):
        predictions = []
        for i in range(0, len(example_inputs), batch_size):
            batch = example_inputs[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = model.generate(**inputs, max_new_tokens=128)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(decoded)
        return predictions

    predictions = batched_infer(module_inputs)

    # 4) Per-task metrics
    task_metrics = {}
    main_scores = {}
    all_metric_keys = set()
    for task in task_names:
        task_indices = [i for i, t in enumerate(module_task_names) if t == task]
        if not task_indices:
            print(f"[WARN] No examples found for task: {task}")
            continue
        preds = [predictions[i] for i in task_indices]
        gts   = [module_outputs[i] for i in task_indices]

        task_type = infer_task_type(task)
        metrics = compute_metrics(task_type, preds, gts)
        all_metric_keys.update(metrics.keys())
        score_val, score_key = choose_main_metric(metrics, task_type)

        task_metrics[task] = {
            "metrics": metrics,
            "score_key": score_key,
            "score_value": float(round(score_val, 4)),
            "task_type": task_type
        }
        main_scores[task] = score_val

    # 5) Overall score
    score_key_set = {task_metrics[t]["score_key"] for t in task_metrics}
    if len(score_key_set) == 1 and len(task_metrics) > 0:
        score_key = list(score_key_set)[0]
        overall_score = float(round(sum(main_scores.values()) / len(main_scores), 4))
    else:
        score_key = "mixed"
        overall_score = 0.0

    # 6) Average all metrics (optional)
    average_metrics = {}
    for key in all_metric_keys:
        vals = [t["metrics"].get(key, 0.0) for t in task_metrics.values()]
        vals = [float(v) if isinstance(v, str) else v for v in vals]
        average_metrics[key] = float(round(sum(vals) / len(vals), 4)) if len(vals) > 0 else 0.0

    overall_metrics = {
        "score_key": score_key,
        "score_value": overall_score,
        "task_count": len(task_metrics),
        "average_metrics": average_metrics
    }

    return overall_metrics, task_metrics, task_names
