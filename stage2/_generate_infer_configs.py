#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import glob
import yaml
from typing import Optional

# --- Paths ---
CURRENT_DIR = os.getcwd()
CONFIG_DIR = os.path.join(CURRENT_DIR, "configs/infer")
print("Working directory:", CURRENT_DIR)
os.makedirs(CONFIG_DIR, exist_ok=True)

# Root path used to locate training outputs and configs
TRAIN_REPO_BASE = "LoRAGen/stage2"

# --- Load stage1 experiment registry ---
LATEST_YAML = "LoRAGen/stage1/checkpoints/latest_epochs.yaml"
if not os.path.isfile(LATEST_YAML):
    raise FileNotFoundError(f"latest_epochs.yaml not found: {LATEST_YAML}")

with open(LATEST_YAML, "r") as f:
    EXPERIMENT_CONFIGS = yaml.safe_load(f)

if not isinstance(EXPERIMENT_CONFIGS, dict) or not EXPERIMENT_CONFIGS:
    raise ValueError(f"latest_epochs.yaml is empty or invalid: {LATEST_YAML}")

# --- Domain extraction (kept compatible) ---
OPTIONAL_DOMAIN_MAP = {}
DEFAULT_DOMAIN = "flanv2_sub"

def extract_domain(exp_name: str) -> str:
    if exp_name in OPTIONAL_DOMAIN_MAP:
        return OPTIONAL_DOMAIN_MAP[exp_name]
    pats = [
        r'(flanv2_zero_shot_ex)',
        r'(flanv2_sub)',
        r'(bench_tasks)',
    ]
    for pat in pats:
        m = re.search(pat, exp_name)
        if m:
            return m.group(1)
    print(f"[WARN] Failed to parse domain; use default: {exp_name} -> {DEFAULT_DOMAIN}")
    return DEFAULT_DOMAIN

# --- Read dec_only and decoder layer count from VAE config ---
def load_dec_only_and_layers(vae_config_path: str) -> tuple[bool, int]:
    dec_only = False
    num_layers = 24
    try:
        with open(vae_config_path, "r") as f:
            y = yaml.safe_load(f) or {}
        enc = (((y.get("model") or {}).get("params") or {}).get("ddconfig") or {}).get("encoder") or {}
        dec = (((y.get("model") or {}).get("params") or {}).get("ddconfig") or {}).get("decoder") or {}
        enc_params = (enc.get("params") or {})
        dec_params = (dec.get("params") or {})
        dec_only = bool(enc_params.get("dec_only", False))
        num_layers = int(dec_params.get("num_layers", num_layers))
    except Exception as e:
        print(f"[WARN] Failed to parse VAE config; using defaults dec_only={dec_only}, num_layers={num_layers} | {vae_config_path}\n{e}")
    return dec_only, num_layers

def guess_lora_data_path(lora_base: str, epoch_str: str, latent_dim: int, dec_only: bool, num_layers: int) -> str:
    candidates = []
    if dec_only:
        n = 4 * num_layers
        candidates.append(os.path.join(lora_base, f"e_{epoch_str}_vae_task_latent_{n}_{latent_dim}_embed_1024.pt"))
        candidates += sorted(glob.glob(os.path.join(lora_base, f"e_{epoch_str}_vae_task_latent_*_{latent_dim}_embed_1024.pt")))
    else:
        n = 12 * num_layers
        candidates.append(os.path.join(
            lora_base,
            f"e_{epoch_str}_with_task_name_vae_task_172_latent_{n}_{latent_dim}_embed_1024.pt"
        ))
        candidates.append(os.path.join(
            lora_base,
            f"e_{epoch_str}_with_task_name_vae_task_172_latent_288_{latent_dim}_embed_1024.pt"
        ))
        candidates.append(os.path.join(
            lora_base,
            f"e_{epoch_str}_vae_task_latent_{n}_{latent_dim}_embed_1024.pt"
        ))
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]

def read_train_cfg(exp_name: str, epoch: int) -> dict:
    train_yaml = os.path.join(
        TRAIN_REPO_BASE, "configs/train", f"{exp_name}_vae_{epoch}.yaml"
    )
    epoch_str = str(epoch).zfill(6)
    if not os.path.isfile(train_yaml):
        print(f"[WARN] train.yaml not found: {train_yaml}; fallback to defaults")
        return {
            "denoise": "LoRATrans",
            "modeldim": 64,
            "diffusionstep": 500,
            "epochs": 4010,
            "batch_size": 64,
            "lr": "1e-4",
            "output_dir": f"logs/stage2/train/{exp_name}/vae_{epoch_str}",
            "lora_data_path": "",
            "vae_config_path": f"{TRAIN_REPO_BASE}/train/denoising_diffusion_pytorch/configs/{exp_name}.yaml",
            "vae_ckpt_path": f"LoRAGen/stage1/checkpoints/stage1/{exp_name}/checkpoints/epochepoch={epoch_str}-aelosstrain/",
            "targetTaskList": None,
        }

    with open(train_yaml, "r") as f:
        y = yaml.safe_load(f) or {}

    cfg = {}
    cfg["denoise"]        = y.get("denoise", "LoRATrans")
    cfg["modeldim"]       = int(y.get("modeldim", 64))
    cfg["diffusionstep"]  = int(y.get("diffusionstep", 500))
    cfg["epochs"]         = int(y.get("epochs", 4010))
    cfg["batch_size"]     = int(y.get("batch_size", 64))
    cfg["lr"]             = y.get("lr", "1e-4")
    cfg["basemodel"]      = y.get("basemodel", "flan-t5-large")
    cfg["output_dir"]     = y.get("output_dir", f"logs/stage2/train/{exp_name}/vae_{epoch_str}")
    cfg["lora_data_path"] = y.get("lora_data_path", "")
    cfg["vae_config_path"]= y.get("vae_config_path", f"{TRAIN_REPO_BASE}/train/denoising_diffusion_pytorch/configs/{exp_name}.yaml")
    cfg["vae_ckpt_path"]  = y.get("vae_ckpt_path", f"LoRAGen/stage1/checkpoints/stage1/{exp_name}/checkpoints/epochepoch={epoch_str}-aelosstrain/")
    cfg["targetTaskList"] = y.get("targetTaskList", None)
    return cfg

_MODEL_STEP_RE = re.compile(r"model-(\d+)\.pt$")

def _parse_step_from_model(path: str) -> Optional[int]:
    m = _MODEL_STEP_RE.search(path)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def find_best_resume_path(train_output_dir_rel: str, target_task: str, denoise_name: str, modeldim: int) -> Optional[str]:
    run_base = os.path.join(
        TRAIN_REPO_BASE,
        train_output_dir_rel,
        target_task,
        denoise_name,
        f"modeldim_{modeldim}",
    )
    model_glob = os.path.join(run_base, "*", "ModelSave", "model-*.pt")
    model_files = glob.glob(model_glob)

    best_step, best_file = -1, None
    for mp in model_files:
        s = _parse_step_from_model(mp)
        if s is None:
            continue
        if s > best_step:
            best_step, best_file = s, mp

    if best_file:
        return best_file

    print(f"[ERROR] No checkpoint found matching: {model_glob}")
    return None

def fmt_lr(v):
    if isinstance(v, (int, float)):
        return f"{v:g}"
    return str(v)

# --- Generate per-experiment inference YAMLs ---
target_tasks_manifest = []

for exp_name, cfg in EXPERIMENT_CONFIGS.items():
    if not isinstance(cfg, dict):
        print(f"[WARN] Skip invalid registry item: {exp_name} -> {cfg}")
        continue

    try:
        epoch = int(cfg["epoch"])
        latent_dim = int(cfg["latent_dim"])
        hidden_dim = int(cfg["hidden_dim"])
    except Exception as e:
        raise ValueError(f"Experiment fields missing or invalid (need epoch/latent_dim/hidden_dim): {exp_name} -> {cfg}") from e

    epoch_str = str(epoch).zfill(6)
    target_task = f"{exp_name}_vae_{epoch}"
    target_tasks_manifest.append(target_task)

    # Load aligned params from corresponding train.yaml
    tcfg = read_train_cfg(exp_name, epoch)
    denoise_name   = tcfg["denoise"]
    modeldim       = tcfg["modeldim"]
    diffusionstep  = tcfg["diffusionstep"]
    epochs_train   = tcfg["epochs"]
    batch_size     = tcfg["batch_size"]
    lr_val         = tcfg["lr"]
    basemodel      = tcfg["basemodel"]
    train_out_rel  = tcfg["output_dir"]
    lora_data_path = tcfg["lora_data_path"]
    vae_config_path= tcfg["vae_config_path"]
    vae_ckpt_path  = tcfg["vae_ckpt_path"]
    tlist_from_train = tcfg["targetTaskList"]

    # If lora_data_path missing in train.yaml, infer it via dec_only + num_layers
    if not lora_data_path:
        lora_base = f"LoRAGen/stage1/checkpoints/stage1/{exp_name}"
        dec_only, num_layers = load_dec_only_and_layers(vae_config_path)
        lora_data_path = guess_lora_data_path(lora_base, epoch_str, latent_dim, dec_only, num_layers)

    # targetTaskList
    if tlist_from_train and isinstance(tlist_from_train, list) and len(tlist_from_train) > 0:
        task_list_block = [f'  - "{k}"' for k in tlist_from_train]
    else:
        domain = extract_domain(exp_name)
        split_json_path = f"LoRAGen/stage1/data_utils/split/{domain}.json"
        try:
            with open(split_json_path, "r") as f:
                task_keys = json.load(f).get("train", [])
            task_list_block = [f'  - "{k}"' for k in task_keys]
        except Exception as e:
            print(f"[ERROR] Failed to read split file: {split_json_path}\n{e}")
            task_list_block = []

    # checkpoint to resume from
    resume_path = find_best_resume_path(train_out_rel, target_task, denoise_name, modeldim)
    if resume_path is None:
        resume_path = "/path/to/checkpoint.pt"  # placeholder to be replaced manually if needed
        print(f"[WARN] {exp_name}: resume_path not determined; placeholder kept. Check your training output tree.")

    # infer outputs
    output_dir_infer = f"logs/stage2/infer/{exp_name}"

    # write infer.yaml
    filename = f"{target_task}.yaml"
    yaml_path = os.path.join(CONFIG_DIR, filename)
    print(f"Generating config: {yaml_path}")
    with open(yaml_path, "w") as f:
        f.write("# infer.yaml\n\n")
        f.write("# model config\n")
        f.write(f'targetTask: "{target_task}"\n\n')

        f.write("targetTaskList:\n")
        if task_list_block:
            f.write("\n".join(task_list_block) + "\n\n")
        else:
            f.write("  # (empty)\n\n")

        f.write(f'denoise: "{denoise_name}"\n')
        f.write(f"modeldim: {modeldim}\n\n")
        f.write(f"diffusionstep: {diffusionstep}\n\n")
        f.write(f"epochs: {epochs_train}\n\n")

        f.write("# infer parameters\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"lr: {fmt_lr(lr_val)}\n")
        f.write('mode: "infer"\n\n')

        f.write("# diffusion checkpoint\n")
        f.write(f'resume_path: "{resume_path}"\n\n')

        f.write("# path config\n")
        f.write(f'output_dir: "{output_dir_infer}/vae_{epoch_str}_gt_latent"\n')
        f.write(f'lora_data_path: "{lora_data_path}"\n')
        f.write(f'vae_config_path: "{vae_config_path}"\n')
        f.write(f'vae_ckpt_path: "{vae_ckpt_path}"\n')

# manifest for shell launcher
with open(os.path.join(CURRENT_DIR, "configs/infer/infer_target_tasks.json"), "w") as f:
    json.dump(target_tasks_manifest, f, indent=2)

print(f"Generated {len(target_tasks_manifest)} YAML files under {CONFIG_DIR}")
