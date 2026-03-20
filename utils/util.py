# ================= Utility Functions =================
# ================= stage1 =================
import importlib

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


# ================= stage2 =================
# LoRAGen/utils/util.py
# Utilities shared by stage-2 training / inference.
import os
import json
import yaml
from datetime import datetime

def make_run_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_parent_output_dir(args, run_tag: str) -> str:
    """
    Reproduce the hierarchical output directory layout used elsewhere:
      <output_dir>/<targetTask>/<denoise>/modeldim_<dim>/<run_tag>/
    """
    parent_output_dir = os.path.join(
        args.output_dir,
        str(args.targetTask),
        str(args.denoise),
        f"modeldim_{int(args.modeldim)}",
        run_tag
    )
    os.makedirs(parent_output_dir, exist_ok=True)
    return parent_output_dir

def humanize_seconds(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def count_params(module):
    """Return (total_params, trainable_params)"""
    total_params = 0
    trainable_params = 0
    for p in module.parameters():
        n = p.numel()
        total_params += n
        if p.requires_grad:
            trainable_params += n
    return total_params, trainable_params

def is_dec_only_from_cfg(vae_config_path: str) -> bool:
    try:
        with open(vae_config_path, "r") as f:
            y = yaml.safe_load(f) or {}
        enc = (((y.get("model") or {}).get("params") or {}).get("ddconfig") or {}).get("encoder") or {}
        enc_params = (enc.get("params") or {})
        return bool(enc_params.get("dec_only", False))
    except Exception:
        return False

def _extract_short_name(task_name: str) -> str:
    return task_name.split("lorahub_flan_t5_large-")[-1]

def read_split_keys(vae_config_path: str):
    try:
        with open(vae_config_path, "r") as f:
            y = yaml.safe_load(f) or {}
        data = (y.get("data") or {}).get("params") or {}
        split_file = data.get("split_file", None)
        if not split_file or not os.path.isfile(split_file):
            return [], []
        with open(split_file, "r") as f:
            sp = json.load(f) or {}
        train_keys = list(sp.get("train", []) or [])
        val_keys   = list(sp.get("val",   []) or [])
        return [ _extract_short_name(n) for n in train_keys ], [ _extract_short_name(n) for n in val_keys ]
    except Exception:
        return [], []

def task_name_wo_prefix(task: str) -> str:
    base = task
    prefix = "lorahub_flan_t5_large-"
    if base.startswith(prefix):
        base = base[len(prefix):]
    base = base.replace('/', '_').replace(os.sep, '_').replace(' ', '_')
    return base

def infer_task_type(task_name: str) -> str:
    t = task_name.lower().replace("lorahub_flan_t5_large-", "")
    rules = {
        "classification": ["amazon_polarity","yelp_polarity","app_reviews","glue_sst2","qqp","mrpc","stsb","qnli",
                           "wnli","paws_wiki","anli","glue_cola","super_glue_wic","super_glue_wsc","dbpedia","trec","ag_news"],
        "multiple_choice": ["race","dream","quail","cosmos_qa","quartz","ropes","social_i_qa","piqa","cos_e","siqa",
                            "quarel","arc_challenge","wiqa","sciq_multiple_choice","qasc","proofwriter"],
        "question_answering": ["squad","quoref","duorc","record","wiki_qa","wiki_hop","hotpotqa","kilt_tasks_hotpotqa",
                               "coqa","quac","nq","trivia_qa","web_questions","sciq_direct_question","adversarial_qa","drop"],
        "text_generation": ["cnn_dailymail","xsum","gem_xsum","newsroom","web_nlg","wiki_bio","gem_web_nlg","gem_e2e_nlg",
                            "para_crawl","opus100","wmt","title_generation","question_generation","word_segment","fix_punct","true_case"]
    }
    for tp, keys in rules.items():
        if any(k in t for k in keys):
            return tp
    return "unknown"
