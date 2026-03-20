import os, json, yaml
import torch

# ---------- name helpers ----------

def extract_short_name(task_name: str):
    return task_name.split("lorahub_flan_t5_large-")[-1]

def _read_split_from_vae_cfg(vae_config_path: str):
    try:
        with open(vae_config_path, "r") as f:
            y = yaml.safe_load(f) or {}
        data = (y.get("data") or {}).get("params") or {}
        split_file = data.get("split_file", None)
        if not split_file or not os.path.isfile(split_file):
            print(f"[WARN] split_file not found: {split_file} (returning empty sets)")
            return [], [], split_file
        with open(split_file, "r") as f:
            sp = json.load(f) or {}
        train_keys = [extract_short_name(s) for s in list(sp.get("train", []) or [])]
        val_keys   = [extract_short_name(s) for s in list(sp.get("val",   []) or [])]
        return train_keys, val_keys, split_file
    except Exception as e:
        print(f"[WARN] failed to read split_file from VAE config: {vae_config_path}\n{e}")
        return [], [], None

def _build_short_to_full_map(latent_dict, embed_dict):
    short_to_full = {}
    common_keys = set(latent_dict.keys()) & set(embed_dict.keys())
    for k in common_keys:
        short = extract_short_name(k)
        if short not in short_to_full:
            short_to_full[short] = k
    return short_to_full

def _to_tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


# ---------- dataset builders ----------

def prepare_lora_dataset(
    latent_dict,                 # {task_key: Tensor[layers, dz]}
    embed_dict,                  # {task_key: Tensor[cond_dim]}
    vae_config_path: str,
    repeat_num: int = 120,
    cond_dim: int = 1024,
):
    train_keys, val_keys, _ = _read_split_from_vae_cfg(vae_config_path)
    if not train_keys and not val_keys:
        print("[WARN] empty split; using all keys as both train/gen (in-domain).")
        all_keys = list(set(latent_dict.keys()) & set(embed_dict.keys()))
        train_keys, val_keys = all_keys, []

    gen_keys = train_keys if len(val_keys) == 0 else val_keys
    short2full = _build_short_to_full_map(latent_dict, embed_dict)

    train_latents, train_embeddings = [], []
    gen_latents, gen_embeddings = [], []

    # ---------- train set ----------
    for short in train_keys:
        orig = short2full.get(short, short)
        if orig not in latent_dict or orig not in embed_dict:
            print(f"[WARN] missing latent or embedding for train key: {short} (orig={orig}); skip.")
            continue
        z = _to_tensor(latent_dict[orig])          # [layers, dz]
        e = _to_tensor(embed_dict[orig])           # [cond_dim]
        if z.dim() != 2:
            raise ValueError(f"[ERR] latent has shape {tuple(z.shape)} for {orig}; expected [layers, dz].")
        if e.dim() != 1:
            raise ValueError(f"[ERR] embedding has shape {tuple(e.shape)} for {orig}; expected [cond_dim].")
        z = z.T.unsqueeze(0)                       # -> [1, dz, layers]
        e = e.unsqueeze(0)                         # -> [1, cond_dim]
        train_latents.append(z.repeat(repeat_num, 1, 1))
        train_embeddings.append(e.repeat(repeat_num, 1))

    # ---------- generation targets ----------
    for short in gen_keys:
        orig = short2full.get(short, short)
        if orig not in latent_dict or orig not in embed_dict:
            print(f"[WARN] missing latent or embedding for gen key: {short} (orig={orig}); skip.")
            continue
        z = _to_tensor(latent_dict[orig]).T.unsqueeze(0)  # [1, dz, layers]
        e = _to_tensor(embed_dict[orig]).unsqueeze(0)     # [1, cond_dim]
        gen_latents.append(z)
        gen_embeddings.append(e)

    # ---------- stack + normalize ----------
    if train_latents:
        training_seq  = torch.cat(train_latents, dim=0).float()  # [N, dz, layers]
        condTrainEmb  = torch.cat(train_embeddings, dim=0).float()
        stepTrainEmb  = torch.ones_like(condTrainEmb)
        scale = max(training_seq.abs().max(), torch.tensor(1.0))
        training_seq /= scale
    else:
        print("[WARN] empty training set; returning zero placeholders with scale=1.")
        training_seq = torch.zeros((0, 32, next(iter(latent_dict.values())).shape[0]))
        condTrainEmb = torch.zeros((0, cond_dim))
        stepTrainEmb = torch.zeros((0, cond_dim))
        scale = torch.tensor(1.0)

    if gen_latents:
        genTarget    = torch.cat(gen_latents, dim=0).float()
        condGenEmb   = torch.cat(gen_embeddings, dim=0).float()
        genTarget   /= scale
    else:
        print("[WARN] empty gen set; returning zero placeholders.")
        genTarget  = torch.zeros((0, training_seq.shape[1],
                                  training_seq.shape[2] if training_seq.shape[0]
                                  else next(iter(latent_dict.values())).shape[0]))
        condGenEmb = torch.zeros((0, cond_dim))

    stepGenEmb = torch.ones_like(condGenEmb)

    return (
        training_seq.cpu().numpy(),
        float(scale.item()),
        condTrainEmb.cpu().numpy(),
        condGenEmb.cpu().numpy(),
        stepTrainEmb.cpu().numpy(),
        stepGenEmb.cpu().numpy(),
        genTarget.cpu().numpy(),
    )


def prepare_lora_dataset_for_infer(
    latent_dict,
    embed_dict,
    vae_config_path: str,
    repeat_num: int = 120,
    cond_dim: int = 1024,
):
    train_keys, val_keys, _ = _read_split_from_vae_cfg(vae_config_path)
    if not train_keys and not val_keys:
        print("[WARN] empty split; using all keys as both train/gen (in-domain).")
        all_keys = list(set(latent_dict.keys()) & set(embed_dict.keys()))
        train_keys, val_keys = all_keys, []
    gen_keys = train_keys if len(val_keys) == 0 else val_keys
    short2full = _build_short_to_full_map(latent_dict, embed_dict)

    train_latents, train_embeddings = [], []
    gen_latents, gen_embeddings = [], []

    for short in train_keys:
        orig = short2full.get(short, short)
        if orig not in latent_dict or orig not in embed_dict:
            print(f"[WARN] missing latent or embedding for train key: {short} (orig={orig}); skip.")
            continue
        z = _to_tensor(latent_dict[orig]).T.unsqueeze(0)
        e = _to_tensor(embed_dict[orig]).unsqueeze(0)
        train_latents.append(z.repeat(repeat_num, 1, 1))
        train_embeddings.append(e.repeat(repeat_num, 1))

    for short in gen_keys:
        orig = short2full.get(short, short)
        if orig not in latent_dict or orig not in embed_dict:
            print(f"[WARN] missing latent or embedding for gen key: {short} (orig={orig}); skip.")
            continue
        z = _to_tensor(latent_dict[orig]).T.unsqueeze(0)
        e = _to_tensor(embed_dict[orig]).unsqueeze(0)
        gen_latents.append(z)
        gen_embeddings.append(e)

    if train_latents:
        training_seq = torch.cat(train_latents, dim=0).float()
        condTrainEmb = torch.cat(train_embeddings, dim=0).float()
        stepTrainEmb = torch.ones_like(condTrainEmb)
    else:
        print("[WARN] empty train placeholder for infer.")
        training_seq = torch.zeros((0, 32, next(iter(latent_dict.values())).shape[0]))
        condTrainEmb = torch.zeros((0, cond_dim))
        stepTrainEmb = torch.zeros((0, cond_dim))

    if gen_latents:
        genTarget  = torch.cat(gen_latents, dim=0).float()
        condGenEmb = torch.cat(gen_embeddings, dim=0).float()
    else:
        print("[WARN] empty gen set for infer.")
        genTarget  = torch.zeros((0, training_seq.shape[1],
                                  training_seq.shape[2] if training_seq.shape[0]
                                  else next(iter(latent_dict.values())).shape[0]))
        condGenEmb = torch.zeros((0, cond_dim))

    stepGenEmb = torch.ones_like(condGenEmb)
    scale = torch.tensor(1.0)

    return (
        training_seq.cpu().numpy(),
        float(scale.item()),
        condTrainEmb.cpu().numpy(),
        condGenEmb.cpu().numpy(),
        stepTrainEmb.cpu().numpy(),
        stepGenEmb.cpu().numpy(),
        genTarget.cpu().numpy(),
    )
