#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(watch + embed) — unified (T5 + decoder-only)
"""

import os
import re
import glob
import time
import yaml
import shutil
import argparse
from typing import Dict, List, Tuple, Optional

# ---------- shared utils ----------

_TS_RE = re.compile(r"_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})$")
_last_mtime: Dict[str, int] = {}

def strip_timestamp(name: str) -> str:
    m = _TS_RE.search(name)
    return name[:m.start()] if m else name

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def copy_if_changed(src_file: str, tgt_file: str):
    try:
        cur_mtime = os.stat(src_file).st_mtime_ns
    except FileNotFoundError:
        return
    prev_mtime = _last_mtime.get(src_file)
    need_copy = (not os.path.exists(tgt_file)) or (prev_mtime is None) or (cur_mtime > prev_mtime)
    if need_copy:
        ensure_dir(os.path.dirname(tgt_file))
        shutil.copy2(src_file, tgt_file)
        _last_mtime[src_file] = cur_mtime
        print(f"[Copied] {src_file} -> {tgt_file}")

def add_prefix_to_targets(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "target" and isinstance(v, str) and not v.startswith("denoising_diffusion_pytorch."):
                out[k] = "denoising_diffusion_pytorch." + v
            else:
                out[k] = add_prefix_to_targets(v)
        return out
    if isinstance(obj, list):
        return [add_prefix_to_targets(x) for x in obj]
    return obj

def build_watch_pairs(base_stage1_log_dir: str, experiment_names: List[str]) -> List[Dict[str, str]]:
    pairs = []
    for exp_name in experiment_names:
        exp_dir = os.path.join(base_stage1_log_dir, exp_name)
        watch_dir = os.path.join(exp_dir, "checkpoints")
        exp_no_ts = strip_timestamp(exp_name)
        stage1_root = os.path.dirname(base_stage1_log_dir)  # .../logs
        project_root = os.path.dirname(stage1_root)         # .../<project>
        checkpoints_stage1 = os.path.join(project_root, "checkpoints", "stage1")
        target_root = os.path.join(checkpoints_stage1, exp_no_ts, "checkpoints")
        pairs.append({"exp_name": exp_name, "exp_no_ts": exp_no_ts,
                      "watch_dir": watch_dir, "target_root": target_root})
    return pairs

_EPOCH_DIR_RE = re.compile(r"^epochepoch=(\d{6})-aelosstrain")

def find_latest_epoch_dir(checkpoints_dir: str) -> Optional[Tuple[str, str]]:
    if not os.path.isdir(checkpoints_dir):
        return None
    latest = None
    latest_ep = None
    for name in os.listdir(checkpoints_dir):
        m = _EPOCH_DIR_RE.match(name)
        if not m:
            continue
        ep = m.group(1)
        if (latest_ep is None) or (int(ep) > int(latest_ep)):
            latest_ep = ep
            latest = os.path.join(checkpoints_dir, name)
    if latest is None:
        return None
    return latest, latest_ep

def copy_latest_ckpt(pair: Dict[str, str]) -> Optional[str]:
    found = find_latest_epoch_dir(pair["watch_dir"])
    if not found:
        print(f"[Warn] No epoch dir found under {pair['watch_dir']}")
        return None
    latest_dir, epoch_str = found
    rel_epoch_dir = os.path.relpath(latest_dir, pair["watch_dir"])
    for f in os.listdir(latest_dir):
        if f.endswith(".ckpt"):
            src = os.path.join(latest_dir, f)
            dst = os.path.join(pair["target_root"], rel_epoch_dir, f)
            copy_if_changed(src, dst)
    return epoch_str

def copy_version0(pair: Dict[str, str]):
    exp_root = os.path.dirname(pair["watch_dir"])
    src_version = os.path.join(exp_root, "version_0")
    if not os.path.isdir(src_version):
        return
    tgt_version = os.path.join(os.path.dirname(pair["target_root"]), "version_0")
    for root, _, files in os.walk(src_version):
        rel = os.path.relpath(root, src_version)
        for f in files:
            copy_if_changed(os.path.join(root, f), os.path.join(tgt_version, rel, f))

def copy_config_yaml_model_only(pair: Dict[str, str], config_yaml_target_dir: str):
    exp_root = os.path.dirname(pair["watch_dir"])
    src_yaml = os.path.join(exp_root, "config.yaml")
    if not os.path.isfile(src_yaml):
        return
    base_name = strip_timestamp(os.path.basename(exp_root))
    ensure_dir(config_yaml_target_dir)
    tgt_yaml = os.path.join(config_yaml_target_dir, f"{base_name}.yaml")
    try:
        with open(src_yaml, "r") as f:
            cfg = yaml.safe_load(f) or {}
        if "model" not in cfg:
            print(f"[Warn] No 'model' in {src_yaml}")
            return
        model_cfg = add_prefix_to_targets(cfg["model"])
        out_txt = yaml.safe_dump({"model": model_cfg, "data": cfg["data"]}, sort_keys=False, allow_unicode=True)
        need_write = True
        if os.path.exists(tgt_yaml):
            with open(tgt_yaml, "r") as rf:
                need_write = (rf.read() != out_txt)
        if need_write:
            with open(tgt_yaml, "w") as wf:
                wf.write(out_txt)
            print(f"[Written] {tgt_yaml}")
    except Exception as e:
        print(f"[Error] Processing {src_yaml}: {e}")

def record_latest_epoch(pair: Dict[str, str], epoch_str: str, latest_map_path: Optional[str],
                        latent_dim: int, hidden_dim: int):
    per_exp_root = os.path.dirname(pair["target_root"])
    ensure_dir(per_exp_root)
    with open(os.path.join(per_exp_root, "LATEST_EPOCH.txt"), "w") as f:
        f.write(str(int(epoch_str)))

    if latest_map_path:
        ensure_dir(os.path.dirname(latest_map_path))
        data = {}
        if os.path.isfile(latest_map_path):
            try:
                with open(latest_map_path, "r") as f:
                    data = yaml.safe_load(f) or {}
            except Exception:
                data = {}
        data[pair["exp_no_ts"]] = {
            "epoch": int(epoch_str),
            "latent_dim": int(latent_dim),
            "hidden_dim": int(hidden_dim),
        }
        with open(latest_map_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=True)
        print(f"[Updated] {latest_map_path}: {pair['exp_no_ts']} -> "
              f"epoch={int(epoch_str)}, latent_dim={int(latent_dim)}, hidden_dim={int(hidden_dim)}")

# ---------- subcommand: watch ----------

def cmd_watch(args: argparse.Namespace):
    pairs = build_watch_pairs(args.base_stage1_log_dir, args.experiments)
    print("==== Monitor (latest-epoch only) ====")
    while True:
        for pair in pairs:
            latest_ep = copy_latest_ckpt(pair)
            if latest_ep:
                copy_version0(pair)
                copy_config_yaml_model_only(pair, args.config_out_dir)
                record_latest_epoch(pair, latest_ep, args.latest_map,
                                    latent_dim=args.latent_dim,
                                    hidden_dim=args.hidden_dim)
        time.sleep(args.interval)

# ---------- embed helpers ----------

_GENERIC_PREFIX_RE = re.compile(
    r"^\s*(the\s+task\s+(?:is|involves|requires|asks|aims|entails)\b(?:[^:,.]*[:,-])?\s*)",
    flags=re.IGNORECASE
)
def maybe_strip_generic(s: str, enable: bool) -> str:
    if not enable:
        return s.strip()
    return _GENERIC_PREFIX_RE.sub("", s.strip()).strip()

def load_task_descs_from_yaml_root(yaml_root: str, strip_generic: bool = False) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    if not os.path.isdir(yaml_root):
        raise FileNotFoundError(f"YAML root not found: {yaml_root}")
    for entry in sorted(os.listdir(yaml_root)):
        subdir = os.path.join(yaml_root, entry)
        if not os.path.isdir(subdir):
            continue
        meta_path = os.path.join(subdir, "metadata.yaml")
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = yaml.safe_load(f)
        except Exception as e:
            print(f"[Warn] Failed to read {meta_path}: {e}")
            continue
        descs = meta.get("descriptions", None)
        if not descs or not isinstance(descs, list):
            print(f"[Warn] No valid 'descriptions' in {meta_path}")
            continue
        cleaned = [maybe_strip_generic(d, strip_generic) for d in descs if isinstance(d, str) and d.strip()]
        cleaned = cleaned[:20]  # limit to 20
        if not cleaned:
            continue
        for key in [entry, f"lorahub_flan_t5_large-{entry}", f"lorahub_flan_t5_large_{entry}"]:
            mapping[key] = cleaned
    return mapping

def encode_texts(texts, tokenizer, text_encoder, device, pooling="mean", max_length=512):
    import torch
    assert pooling in ("mean", "first")
    vecs = []
    for t in texts:
        inputs = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            hidden = text_encoder(**inputs).last_hidden_state
            if pooling == "first":
                v = hidden[:, 0, :]
            else:
                attn = inputs["attention_mask"].unsqueeze(-1)
                v = (hidden * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1)
        vecs.append(v.squeeze(0).cpu())
    return __import__("torch").stack(vecs, dim=0).mean(dim=0)

def find_ckpt_dirs(checkpoints_root: str,
                   experiment_configs: Optional[Dict[str, Dict]] = None) -> List[Tuple[str, str, Dict]]:
    results: List[Tuple[str, str, Dict]] = []
    stage1_root = os.path.join(checkpoints_root, "stage1")
    if not os.path.isdir(stage1_root):
        print(f"[Error] Not found: {stage1_root}")
        return results
    if not experiment_configs:
        print("[Error] experiment_configs is required for embed (use --experiments-config or --latest-map).")
        return results
    for exp_name, cfg in experiment_configs.items():
        epoch = int(cfg["epoch"])
        pat = os.path.join(stage1_root, exp_name, "checkpoints", f"epochepoch={epoch:06d}-aelosstrain*")
        matches = sorted(glob.glob(pat))
        if matches:
            results.append((exp_name, matches[0], cfg))
        else:
            print(f"[Warn] Missing ckpt dir for {exp_name} @ epoch {epoch:06d}")
    return results


def _safe_load_file(path: str) -> Dict[str, "torch.Tensor"]:
    import torch
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file as _load
        return dict(_load(path))
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        return sd["state_dict"]
    if isinstance(sd, dict):
        return sd
    raise ValueError(f"Unrecognized state_dict in {path}")

T5_PAT = re.compile(
    r"^base_model\.model\.(encoder|decoder)\.block\.(\d+)\.layer\.(\d+)\."
    r"(SelfAttention|EncDecAttention)\.(q|v)\.lora_(A|B)\.weight$"
)

DEC_PAT = re.compile(
    r"^base_model\.model\.model\.layers\.(\d+)\.self_attn\."
    r"(q_proj|v_proj)\.lora_(A|B)\.weight$"
)

def _detect_mode_and_dims_from_dataset(first_sd: Dict[str, "torch.Tensor"]) -> Tuple[str, Dict[str, int]]:
    import torch
    # decoder-only
    dec_hits = [(k, DEC_PAT.match(k)) for k in first_sd.keys()]
    dec_hits = [(k, m) for (k, m) in dec_hits if m]
    if dec_hits:
        def _pick(lidx: int, proj: str, ab: str) -> "torch.Tensor":
            k = f"base_model.model.model.layers.{lidx}.self_attn.{proj}.lora_{ab}.weight"
            return first_sd[k]
        Ls = sorted({int(m.group(1)) for _, m in dec_hits})
        l0 = Ls[0]
        Aq, Bq = _pick(l0, "q_proj", "A"), _pick(l0, "q_proj", "B")
        Av, Bv = _pick(l0, "v_proj", "A"), _pick(l0, "v_proj", "B")

        def _norm_A(t):  # (r, in)
            a, b = t.shape
            return t if a < b else t.t()
        def _norm_B(t):  # (out, r)
            a, b = t.shape
            return t if a > b else t.t()

        Aq, Bq = _norm_A(Aq), _norm_B(Bq)
        Av, Bv = _norm_A(Av), _norm_B(Bv)

        r     = Aq.shape[0]
        d_in  = Aq.shape[1]
        d_out_q = Bq.shape[0]
        d_out_v = Bv.shape[0]
        return "decoder_only", dict(rank=r, d_in=d_in, d_out_q=d_out_q, d_out_v=d_out_v)

    # T5?
    t5_hits = any(T5_PAT.match(k) for k in first_sd.keys())
    if t5_hits:
        any_k = next(k for k in first_sd.keys() if T5_PAT.match(k))
        t = first_sd[any_k]
        d_model, r = (t.shape if t.shape[0] > t.shape[1] else (t.shape[1], t.shape[0]))
        return "t5", dict(rank=r, d_model=d_model)

    raise RuntimeError("LoRA keys look neither decoder-only nor T5; please check dataset naming.")

def _parse_t5_as_tensor(lora_sd: Dict[str, "torch.Tensor"], d_model: int, r: int) -> "torch.Tensor":
    import torch
    groups = {"encoder": [], "decoder.decoder_attn": [], "decoder.cross_attn": []}
    for k, t in lora_sd.items():
        m = T5_PAT.match(k)
        if not m: 
            continue
        module_type, layer, lidx, attn_type, qv, AB = m.groups()
        module = "encoder" if module_type == "encoder" else ("decoder.decoder_attn" if attn_type == "SelfAttention" else "decoder.cross_attn")
        param  = f"lora_{qv}{'a' if AB=='A' else 'b'}"
        if tuple(t.shape) != (d_model, r):
            t = t.t()
        groups[module].append((int(layer), param, t))

    modules = []
    for module in ["encoder", "decoder.decoder_attn", "decoder.cross_attn"]:
        by_layer = {i: {"lora_qa":None,"lora_qb":None,"lora_va":None,"lora_vb":None} for i in range(24)}
        for layer, param, t in groups[module]:
            by_layer[layer][param] = t
        for i in range(24):
            ent = by_layer[i]
            for p in ["lora_qa","lora_qb","lora_va","lora_vb"]:
                assert ent[p] is not None, f"T5 missing {module} layer={i} {p}"
                modules.append(ent[p])
    return torch.stack(modules, dim=0).unsqueeze(0).contiguous()   # [1,288,d_model,r]

def _parse_dec_as_dict(lora_sd: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
    import torch
    layers = sorted({int(DEC_PAT.match(k).group(1)) for k in lora_sd.keys() if DEC_PAT.match(k)})
    assert layers, "No decoder-only LoRA keys matched."
    A_q, B_q, A_v, B_v = [], [], [], []

    def _norm_A(t):  # (r,in)
        a,b = t.shape
        return t if a < b else t.t()
    def _norm_B(t):  # (out,r)
        a,b = t.shape
        return t if a > b else t.t()

    for i in layers:
        Aq = lora_sd[f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight"]
        Bq = lora_sd[f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_B.weight"]
        Av = lora_sd[f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_A.weight"]
        Bv = lora_sd[f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_B.weight"]
        A_q.append(_norm_A(Aq));  B_q.append(_norm_B(Bq))
        A_v.append(_norm_A(Av));  B_v.append(_norm_B(Bv))

    A_q = torch.stack(A_q, 0).unsqueeze(0).contiguous()   # [1,L,r,in]
    B_q = torch.stack(B_q, 0).unsqueeze(0).contiguous()   # [1,L,out_q,r]
    A_v = torch.stack(A_v, 0).unsqueeze(0).contiguous()   # [1,L,r,in]
    B_v = torch.stack(B_v, 0).unsqueeze(0).contiguous()   # [1,L,out_v,r]
    return {"A_q":A_q, "B_q":B_q, "A_v":A_v, "B_v":B_v}

def _collect_adapter_paths_from_glob(glob_pat: str) -> List[str]:
    return sorted(glob.glob(glob_pat))

def _filter_encoder_sd_for_model(encoder, full_state: dict) -> dict:
    enc_raw = {k.replace("encoder.", ""): v for k, v in full_state.items() if k.startswith("encoder.")}
    if getattr(encoder, "dec_only", False):
        enc_raw = {k: v for k, v in enc_raw.items()
                   if (k.startswith("net_A.") or k.startswith("net_Bq.") or k.startswith("net_Bv."))}
    model_sd = encoder.state_dict()
    enc_sd = {k: v for k, v in enc_raw.items() if (k in model_sd) and (tuple(model_sd[k].shape) == tuple(v.shape))}
    dropped = sorted(set(enc_raw) - set(enc_sd))
    if dropped:
        print(f"[encoder] drop {len(dropped)} keys (mismatch/unused):", dropped[:8], "...")
    return enc_sd

def _load_encoder_weights_safe(encoder, ckpt_path: str):
    import torch
    vae_full = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = vae_full["state_dict"] if isinstance(vae_full, dict) and "state_dict" in vae_full else vae_full
    enc_sd = _filter_encoder_sd_for_model(encoder, state)
    miss_unexp = encoder.load_state_dict(enc_sd, strict=False)
    print("[encoder] loaded keys:", len(enc_sd),
          "| missing:", len(miss_unexp.missing_keys),
          "| unexpected:", len(miss_unexp.unexpected_keys))

# ---------- subcommand: embed (unified) ----------

def cmd_embed(args: argparse.Namespace):
    import torch
    from transformers import T5Tokenizer, T5EncoderModel
    from stage1.modules.lora_modules import LoRAEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    text_encoder = T5EncoderModel.from_pretrained(args.model_name).eval().to(device)
    yaml_desc_map = load_task_descs_from_yaml_root(args.yaml_root, strip_generic=args.strip_generic)

    # Load experiment configs
    exp_cfg: Optional[Dict[str, Dict]] = None
    if args.experiments_config:
        with open(args.experiments_config, "r") as f:
            exp_cfg = yaml.safe_load(f)
    elif args.latest_map:
        with open(args.latest_map, "r") as f:
            latest = yaml.safe_load(f) or {}
        exp_cfg = {
            k: {
                "epoch": int(v["epoch"]),
                "latent_dim": int(v.get("latent_dim", args.latent_dim)),
                "hidden_dim": int(v.get("hidden_dim", args.hidden_dim)),
            }
            for k, v in latest.items()
        }
    else:
        print("[Error] Provide --experiments-config or --latest-map.")
        return

    ckpt_info_list = find_ckpt_dirs(args.checkpoints_root, exp_cfg)
    if not ckpt_info_list:
        print("[Error] No checkpoint directories found.")
        return

    bin_paths = _collect_adapter_paths_from_glob(args.lora_bin_glob)
    if not bin_paths:
        print(f"[Error] No adapter_model.(bin|safetensors) matched: {args.lora_bin_glob}")
        return

    first_sd = _safe_load_file(bin_paths[0])
    mode, dims = _detect_mode_and_dims_from_dataset(first_sd)
    print(f"▶︎ dataset-mode: {mode}, dims={dims}")

    for exp_name, ckpt_dir, cfg in ckpt_info_list:
        latent_dim = int(cfg.get("latent_dim", args.latent_dim))
        hidden_dim = int(cfg.get("hidden_dim", args.hidden_dim))
        epoch = int(cfg["epoch"])

        if mode == "t5":
            encoder = LoRAEncoder(d_model=dims["d_model"], rank=dims["rank"],
                                  latent_dim=latent_dim, hidden_dim=hidden_dim,
                                  dec_only=False).to(device).eval()
        else:
            encoder = LoRAEncoder(d_model=1024, rank=dims["rank"], latent_dim=latent_dim,
                                  hidden_dim=hidden_dim, dec_only=True,
                                  d_in=dims["d_in"], d_out_q=dims["d_out_q"], d_out_v=dims["d_out_v"]
                                  ).to(device).eval()

        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
        if not ckpts:
            print(f"[Error] No .ckpt under {ckpt_dir}")
            continue
        _load_encoder_weights_safe(encoder, os.path.join(ckpt_dir, ckpts[0]))

        structured, missing = {}, []

        for p in bin_paths:
            task_key = os.path.basename(os.path.dirname(p))
            lora_sd  = _safe_load_file(p)

            if mode == "t5":
                x = _parse_t5_as_tensor(lora_sd, dims["d_model"], dims["rank"]).to(device)   # [1,288,d,r]
            else:
                x = {k:v.to(device) for k,v in _parse_dec_as_dict(lora_sd).items()}          # dict(A_q,B_q,A_v,B_v)

            with torch.no_grad():
                stats = encoder(x)                    # [1,N,2*latent]
                mu = stats[:, :, :latent_dim]
                latent = mu.squeeze(0).cpu()          # [N, latent]

            # 文本描述
            candidates = [
                task_key,
                re.sub(r"^lorahub_flan_t5_large[-_]*", "", task_key),
                task_key.replace("-", "_"),
                task_key.replace("_", "-"),
            ]
            desc = None
            for c in candidates:
                if c in yaml_desc_map:
                    desc = yaml_desc_map[c]
                    break
            if desc is None:
                simple = re.sub(r"^lorahub_flan_t5_large[-_]*", "", task_key)
                for c in [simple, f"lorahub_flan_t5_large-{simple}", f"lorahub_flan_t5_large_{simple}"]:
                    if c in yaml_desc_map:
                        desc = yaml_desc_map[c]; break
            if desc is None:
                missing.append(task_key)
                text_vec = encode_texts([task_key], tokenizer, text_encoder, device, pooling=args.text_pooling)
            else:
                text_vec = encode_texts(desc, tokenizer, text_encoder, device, pooling=args.text_pooling)

            simple_key = re.sub(r"^lorahub_flan_t5_large[-_]*", "", task_key)
            structured[simple_key] = {"task_name": simple_key, "latent": latent, "text_embedding": text_vec}

        exp_dir = os.path.dirname(os.path.dirname(ckpt_dir))
        N = next(iter(structured.values()))["latent"].shape[0]
        out_path = os.path.join(
            exp_dir, f"e_{epoch:06d}_vae_task_latent_{N}_{latent_dim}_embed_1024.pt"
        )
        __import__("torch").save(structured, out_path)
        print(f"[Saved] {out_path}")
        if missing:
            print(f"[Warn] missing YAML for {len(missing)} tasks; e.g., {missing[:8]}")

# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ICLR26 Anonymous Pipeline (T5 + Decoder-only)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # watch
    p_w = sub.add_parser("watch", help="Pick latest-epoch ckpt only; copy artifacts; record epoch.")
    p_w.add_argument("--base-stage1-log-dir", type=str, required=True)
    p_w.add_argument("--experiments", type=str, nargs="+", required=True)
    p_w.add_argument("--config-out-dir", type=str, required=True)
    p_w.add_argument("--latest-map", type=str, default=None)
    p_w.add_argument("--interval", type=int, default=10)
    p_w.add_argument("--latent-dim", type=int, default=64)
    p_w.add_argument("--hidden-dim", type=int, default=128)
    p_w.set_defaults(func=cmd_watch)

    # embed
    p_e = sub.add_parser("embed", help="Generate latents + text embeddings (auto-detect T5 or decoder-only).")
    p_e.add_argument("--checkpoints-root", type=str, required=True,
                     help="e.g., /path/to/project/checkpoints")
    p_e.add_argument("--yaml-root", type=str, required=True,
                     help="Root of YAML task descriptions.")
    p_e.add_argument("--lora-bin-glob", type=str, required=True,
                     help='e.g., "/.../flan_t5_large_lora/*/adapter_model.bin" or "/.../t2l_bench_tasks/*/adapter_model.safetensors"')
    p_e.add_argument("--model_name", type=str, default="/path/to/flan-t5-large")
    p_e.add_argument("--text_pooling", type=str, choices=["mean", "first"], default="mean")
    p_e.add_argument("--strip_generic", action="store_true")

    p_e.add_argument("--experiments-config", type=str, default=None,
                     help="YAML/JSON: name -> {epoch, latent_dim, hidden_dim}.")
    p_e.add_argument("--latest-map", type=str, default=None,
                     help="Use YAML produced by watch to get {name: {epoch}}.")
    p_e.add_argument("--latent-dim", type=int, default=64,
                     help="Default latent_dim if latest-map has no dims.")
    p_e.add_argument("--hidden-dim", type=int, default=128,
                     help="Default hidden_dim if latest-map has no dims.")

    p_e.set_defaults(func=cmd_embed)
    return parser.parse_args()

def main():
    args = parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

'''
# ============= flanv2 =============
python emb_generate.py watch \
  --base-stage1-log-dir LoRAGen/stage1/logs/stage1 \
  --experiments <exp_name> \
  --config-out-dir LoRAGen/stage2/train/denoising_diffusion_pytorch/configs \
  --latest-map LoRAGen/stage1/checkpoints/latest_epochs.yaml \
  --latent-dim 64 \
  --hidden-dim 128 \
  --interval 10

  
python emb_generate.py embed \
  --checkpoints-root LoRAGen/stage1/checkpoints \
  --yaml-root LoRAGen/stage1/emb_generator/task_descriptions/task_descriptions_flanv2 \
  --lora-bin-glob "/path/to/flan_t5_large_lora/*/adapter_model.bin" \
  --latest-map LoRAGen/stage1/checkpoints/latest_epochs.yaml \
  --model_name /path/to/flan-t5-large \
  --text_pooling mean \
  --strip_generic

# ============= bench_tasks =============  

python emb_generate.py watch \
  --base-stage1-log-dir LoRAGen/stage1/logs/stage1 \
  --experiments <exp_name> \
  --config-out-dir stage2/train/denoising_diffusion_pytorch/configs \
  --latest-map LoRAGen/stage1/checkpoints/latest_epochs.yaml \
  --latent-dim 64 \
  --hidden-dim 128 \
  --interval 10

  
python emb_generate.py embed \
  --checkpoints-root LoRAGen/stage1/checkpoints \
  --yaml-root LoRAGen/stage1/emb_generator/task_descriptions/task_descriptions_bench_tasks \
  --lora-bin-glob "/path/to/t2l_bench_tasks/*/adapter_model.safetensors" \
  --latest-map LoRAGen/stage1/checkpoints/latest_epochs.yaml \
  --model_name /path/to/flan-t5-large \
  --text_pooling mean \
  --strip_generic

'''