import os
import re
import json
import glob
import yaml

# --- Paths ---
CURRENT_DIR = os.getcwd()
CONFIG_DIR = os.path.join(CURRENT_DIR, "configs/train")
print("Working directory:", CURRENT_DIR)
os.makedirs(CONFIG_DIR, exist_ok=True)

# --- Load stage1 experiment registry ---
LATEST_YAML = "LoRAGen/stage1/checkpoints/latest_epochs.yaml"
if not os.path.isfile(LATEST_YAML):
    raise FileNotFoundError(f"latest_epochs.yaml not found: {LATEST_YAML}")

with open(LATEST_YAML, "r") as f:
    EXPERIMENT_CONFIGS = yaml.safe_load(f)

if not isinstance(EXPERIMENT_CONFIGS, dict) or not EXPERIMENT_CONFIGS:
    raise ValueError(f"latest_epochs.yaml is empty or invalid: {LATEST_YAML}")

# --- Domain extraction (kept compatible with your existing conventions) ---
OPTIONAL_DOMAIN_MAP = {}
DEFAULT_DOMAIN = "flanv2_sub"

def extract_domain(exp_name: str) -> str:
    if exp_name in OPTIONAL_DOMAIN_MAP:
        return OPTIONAL_DOMAIN_MAP[exp_name]
    patterns = [
        r'(flanv2_zero_shot_ex)',
        r'(flanv2_sub)',
        r'(bench_tasks)',
    ]
    for pat in patterns:
        m = re.search(pat, exp_name)
        if m:
            return m.group(1)
    print(f"[WARN] Failed to parse domain from exp name; use default: {exp_name} -> {DEFAULT_DOMAIN}")
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
    """
    Returns an existing file if found; otherwise returns a canonical candidate path.

    T5 (encoder-decoder):
      e_<epoch>_with_task_name_vae_task_172_latent_<12*num_layers>_<latent_dim>_embed_1024.pt
      (compat: also try ..._latent_288_...)

    Decoder-only:
      e_<epoch>_vae_task_latent_<4*num_layers>_<latent_dim>_embed_1024.pt
    """
    candidates = []
    if dec_only:
        n = 4 * num_layers
        candidates.append(os.path.join(lora_base, f"e_{epoch_str}_vae_task_latent_{n}_{latent_dim}_embed_1024.pt"))
        # fallback: try any N for the same latent_dim
        candidates += sorted(glob.glob(os.path.join(lora_base, f"e_{epoch_str}_vae_task_latent_*_{latent_dim}_embed_1024.pt")))
    else:
        n = 12 * num_layers  # 3 modules × 4 paths × L
        # new naming
        candidates.append(os.path.join(
            lora_base,
            f"e_{epoch_str}_with_task_name_vae_task_172_latent_{n}_{latent_dim}_embed_1024.pt"
        ))
        candidates.append(os.path.join(
            lora_base,
            f"e_{epoch_str}_with_task_name_vae_task_172_latent_288_{latent_dim}_embed_1024.pt"
        ))
        # unified naming without "with_task_name"
        candidates.append(os.path.join(
            lora_base,
            f"e_{epoch_str}_vae_task_latent_{n}_{latent_dim}_embed_1024.pt"
        ))

    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]

target_tasks = []

for exp_name, cfg in EXPERIMENT_CONFIGS.items():
    try:
        epoch = int(cfg["epoch"])
        latent_dim = int(cfg["latent_dim"])
        hidden_dim = int(cfg["hidden_dim"])
    except Exception as e:
        raise ValueError(f"Experiment fields missing or invalid (need epoch/latent_dim/hidden_dim): {exp_name} -> {cfg}") from e

    epoch_str = str(epoch).zfill(6)
    target_task = f"{exp_name}_vae_{epoch}"
    target_tasks.append(target_task)

    # read split list (default to 'train' split)
    domain = extract_domain(exp_name)
    split_json_path = f"LoRAGen/stage1/data_utils/split/{domain}.json"
    try:
        with open(split_json_path, "r") as f:
            val_keys = json.load(f).get("train", [])
        task_list_block = [f'  - "{k}"' for k in val_keys]
    except Exception as e:
        print(f"[ERROR] Failed to read split file: {split_json_path}\n{e}")
        task_list_block = []

    # paths
    output_dir = f"logs/stage2/train/{exp_name}"
    ckpt_base  = f"LoRAGen/stage1/checkpoints/stage1/checkpoints/stage1/{exp_name}/checkpoints"
    lora_base  = f"LoRAGen/stage1/checkpoints/stage1/checkpoints/stage1/{exp_name}"
    vae_config_path = f"LoRAGen/stage2/train/denoising_diffusion_pytorch/configs/{exp_name}.yaml"

    # dec_only + layers -> auto infer latent file path
    dec_only, num_layers = load_dec_only_and_layers(vae_config_path)
    lora_data_path = guess_lora_data_path(lora_base, epoch_str, latent_dim, dec_only, num_layers)

    filename = f"{target_task}.yaml"
    yaml_path = os.path.join(CONFIG_DIR, filename)
    print(f"Generating config: {yaml_path}")
    with open(yaml_path, "w") as f:
        f.write("# train.yaml\n\n")
        f.write("# model config\n")
        f.write(f'targetTask: "{target_task}"\n\n')

        f.write("targetTaskList:\n")
        if task_list_block:
            f.write("\n".join(task_list_block) + "\n\n")
        else:
            f.write("  # (empty)\n\n")

        f.write('denoise: "LoRATrans"\n')
        f.write("modeldim: 64\n\n")
        f.write("diffusionstep: 500\n\n")
        f.write("epochs: 10010\n\n")

        f.write("# training parameters\n")
        f.write("batch_size: 64\n")
        f.write("lr: 1e-4\n")
        f.write('mode: "train"\n\n')

        f.write("# diffusion checkpoint\n")
        f.write('# resume_path: "/path/to/checkpoint.pt"\n\n')

        f.write("# path config\n")
        f.write(f'output_dir: "{output_dir}/vae_{epoch_str}"\n')
        f.write(f'lora_data_path: "{lora_data_path}"\n')
        f.write(f'vae_config_path: "{vae_config_path}"\n')
        f.write(f'vae_ckpt_path: "{ckpt_base}/epochepoch={epoch_str}-aelosstrain/"\n')

with open(os.path.join(CURRENT_DIR, "configs/train/train_target_tasks.json"), "w") as f:
    json.dump(target_tasks, f, indent=2)

print(f"Generated {len(target_tasks)} YAML files under {CONFIG_DIR}")
