# LoRAGen/stage2/train/main_stage2.py
import os
import sys
import glob
import json
import yaml
import argparse
import logging
from datetime import datetime

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# --- repo root to import utilities placed at LoRAGen/utils ---
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from utils.util import (
    make_run_tag, get_parent_output_dir, humanize_seconds,
    count_params, is_dec_only_from_cfg, read_split_keys,
    task_name_wo_prefix, infer_task_type
)

from _datasets.lora_dataset import TrainDataset
from _datasets.dataprepraring import (
    prepare_lora_dataset, prepare_lora_dataset_for_infer
)

from denoising_diffusion_pytorch.denoising_diffusion_lora import GaussianDiffusion1D, Trainer1D
from Transformer import LoRATransformer

from omegaconf import OmegaConf
from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType
from peft.utils import get_peft_model_state_dict
from denoising_diffusion_pytorch.stage1.models.lora_autoencoder import LoRAVAEModel_MoE


torch.set_num_threads(20)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RUN_TAG = make_run_tag()


def train_model(args, targetTaskList, train_dataset, training_seq, scale,
                condTrainEmb, condGenEmb, stepTrainEmb, stepGenEmb, genTarget):

    t_start = datetime.now()
    denoisingNetworkChoose = args.denoise
    diffusion_steps = args.diffusionstep
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    numEpoch = args.epochs
    batchsize = args.batch_size
    d_model = args.modeldim
    logger.info(f"denoiser: {denoisingNetworkChoose}")

    if denoisingNetworkChoose == 'LoRATrans':
        N = 4
        d_cond = condTrainEmb.shape[1]
        d_aux = stepTrainEmb.shape[1]
        dropout = 0.1
        pe = 'original'
        d_input = training_seq.shape[1]      
        d_output = d_input
        layernum = training_seq.shape[2]    

        denoiser = LoRATransformer(
            d_input, d_model, d_output, d_cond, d_aux, N,
            layernum=layernum, dropout=dropout, pe=pe
        ).to(device)
    else:
        raise ValueError(f"Unknown denoiser: {denoisingNetworkChoose}")

    diffusion = GaussianDiffusion1D(
        denoiser,
        seq_length=training_seq.shape[2],
        timesteps=diffusion_steps,
        loss_type='l2',
        objective='pred_v',
        auto_normalize=False,
        beta_schedule='linear',
    ).to(device)

    tot_params, trn_params = count_params(diffusion)
    tr_tot, tr_trn = count_params(denoiser)

    if args.resume_path and os.path.exists(args.resume_path):
        state_dict = torch.load(args.resume_path, map_location=device)
        if isinstance(state_dict, dict) and "model" in state_dict:
            diffusion.load_state_dict(state_dict["model"])
        else:
            diffusion.load_state_dict(state_dict)
        logger.info(f"[✓] Resumed from checkpoint: {args.resume_path}")

    parent_output_dir = get_parent_output_dir(args, RUN_TAG)
    outputpath = os.path.join(parent_output_dir, 'Output'); os.makedirs(outputpath, exist_ok=True)
    modelsavepath = os.path.join(parent_output_dir, 'ModelSave'); os.makedirs(modelsavepath, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(parent_output_dir, 'TensorBoardLogs'))

    param_report = {
        "when": t_start.isoformat(sep=" ", timespec="seconds"),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "exp": {
            "targetTask": args.targetTask,
            "num_tasks": len(targetTaskList) if targetTaskList is not None else 0,
        },
        "counts": {
            "total_params": int(tot_params),
            "trainable_params": int(trn_params),
            "frozen_params": int(tot_params - trn_params),
            "transformer_total_params": int(tr_tot),
            "transformer_trainable_params": int(tr_trn),
        }
    }
    with open(os.path.join(parent_output_dir, "param_report.json"), "w", encoding="utf-8") as f:
        json.dump(param_report, f, indent=2, ensure_ascii=False)
    logger.info(f"[✓] Saved parameter report -> {os.path.join(parent_output_dir, 'param_report.json')}")

    dec_only = is_dec_only_from_cfg(args.vae_config_path)

    trainer = Trainer1D(
        diffusion,
        dataset=train_dataset,
        train_batch_size=batchsize,
        train_lr=8e-5,
        train_num_steps=numEpoch,
        gradient_accumulate_every=1,
        save_and_sample_every=10,
        results_folder=modelsavepath,
        ema_decay=0.995,
        amp=False,
        logger=logger,

        # renamed conditioning banks
        cond_embedding_bank=condGenEmb,
        aux_embedding_bank=stepGenEmb,
        target_latents=genTarget,

        task_names=targetTaskList,
        scale=scale,
        tbwriter=writer,
        outputpath=outputpath,
        num_samples=1,
        vae_config_path=args.vae_config_path,
        vae_ckpt_path=args.vae_ckpt_path,

        # decoder-only eval options
        dec_only=dec_only,
        dec_adapter_config=getattr(args, "dec_adapter_config", None),
        dec_eval_model_dir=getattr(args, "dec_eval_model_dir", None),
        dec_eval_repo_dir=getattr(args, "dec_eval_repo_dir", None),
        dec_eval_script_rel=getattr(args, "dec_eval_script_rel", "scripts/run_eval.py"),
        dec_save_dtype=getattr(args, "dec_save_dtype", "fp32"),
    )

    logger.info("[✓] Start training...")
    trainer.train()

    # timing
    elapsed = (datetime.now() - t_start).total_seconds()
    time_report = {
        "start_time": t_start.isoformat(sep=" ", timespec="seconds"),
        "end_time": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "total_seconds": round(float(elapsed), 2),
        "total_hms": humanize_seconds(elapsed),
    }
    with open(os.path.join(parent_output_dir, "time_report.json"), "w", encoding="utf-8") as f:
        json.dump(time_report, f, indent=2, ensure_ascii=False)
    logger.info(f"[✓] Saved time report -> {os.path.join(parent_output_dir, 'time_report.json')}")


# ---------- T5 (encoder–decoder) infer helpers ----------

def _infer_lora_rank(decoded_weights) -> int:
    qa = decoded_weights["encoder"]["lora_qa"]
    t = torch.as_tensor(qa)
    if t.dim() == 4:
        return int(t.size(2))
    return int(t[0, 0].size(0))

@torch.no_grad()
def _load_structured_lora_to_peft_model_cpu(peft_model, decoded_weights):
    def _cpu(t): return t.detach().to("cpu") if torch.is_tensor(t) else torch.as_tensor(t, device="cpu")
    n_enc = len(peft_model.base_model.model.encoder.block)
    n_dec = len(peft_model.base_model.model.decoder.block)
    n = min(24, n_enc, n_dec)

    for i in range(n):
        blk = peft_model.base_model.model.encoder.block[i].layer[0].SelfAttention
        blk.q.lora_A.default.weight.copy_(_cpu(decoded_weights["encoder"]["lora_qa"][0, i]))
        blk.q.lora_B.default.weight.copy_(_cpu(decoded_weights["encoder"]["lora_qb"][0, i]))
        blk.v.lora_A.default.weight.copy_(_cpu(decoded_weights["encoder"]["lora_va"][0, i]))
        blk.v.lora_B.default.weight.copy_(_cpu(decoded_weights["encoder"]["lora_vb"][0, i]))

        blk = peft_model.base_model.model.decoder.block[i].layer[0].SelfAttention
        blk.q.lora_A.default.weight.copy_(_cpu(decoded_weights["decoder"]["decoder_attn"]["lora_qa"][0, i]))
        blk.q.lora_B.default.weight.copy_(_cpu(decoded_weights["decoder"]["decoder_attn"]["lora_qb"][0, i]))
        blk.v.lora_A.default.weight.copy_(_cpu(decoded_weights["decoder"]["decoder_attn"]["lora_va"][0, i]))
        blk.v.lora_B.default.weight.copy_(_cpu(decoded_weights["decoder"]["decoder_attn"]["lora_vb"][0, i]))

        blk = peft_model.base_model.model.decoder.block[i].layer[1].EncDecAttention
        blk.q.lora_A.default.weight.copy_(_cpu(decoded_weights["decoder"]["cross_attn"]["lora_qa"][0, i]))
        blk.q.lora_B.default.weight.copy_(_cpu(decoded_weights["decoder"]["cross_attn"]["lora_qb"][0, i]))
        blk.v.lora_A.default.weight.copy_(_cpu(decoded_weights["decoder"]["cross_attn"]["lora_va"][0, i]))
        blk.v.lora_B.default.weight.copy_(_cpu(decoded_weights["decoder"]["cross_attn"]["lora_vb"][0, i]))

def _save_decoded_as_peft_bin(decoded_weights, basemodel_path_or_id: str, save_path_bin: str):
    r = _infer_lora_rank(decoded_weights)
    lora_cfg = LoraConfig(
        r=r, lora_alpha=r, lora_dropout=0.0, bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM, target_modules=["q", "v"],
    )
    base_model = AutoModelForSeq2SeqLM.from_pretrained(basemodel_path_or_id)
    peft_model = get_peft_model(base_model, lora_cfg).eval()
    _load_structured_lora_to_peft_model_cpu(peft_model, decoded_weights)
    only_lora = get_peft_model_state_dict(peft_model)
    os.makedirs(os.path.dirname(save_path_bin), exist_ok=True)
    torch.save(only_lora, save_path_bin)
    del peft_model, base_model, only_lora


def infer_model_t5_3times(args, training_seq, condGenEmb, stepGenEmb, device, repeats: int = 3):
    from evaluation.evalLora import evalStructuredLora

    # VAE
    ckpt_files = glob.glob(os.path.join(args.vae_ckpt_path, "*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No .ckpt file found in: {args.vae_ckpt_path}")
    vae_ckpt_path = ckpt_files[0]
    with open(args.vae_config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    config = OmegaConf.create(raw_config)
    vae_model = LoRAVAEModel_MoE(
        ddconfig=config.model.params.ddconfig,
        lossconfig=config.model.params.lossconfig,
        embed_dim=config.model.params.embed_dim,
        learning_rate=config.model.params.learning_rate,
        ckpt_path=vae_ckpt_path,
        device=device
    ).to(device).eval()

    # Diffusion
    diffusion = GaussianDiffusion1D(
        LoRATransformer(
            d_input=training_seq.shape[1],
            d_model=args.modeldim,
            d_output=training_seq.shape[1],
            d_cond=condGenEmb.shape[1],   # was d_kgEmb
            d_aux=stepGenEmb.shape[1],    # was d_timeEmb
            N=4,
            layernum=training_seq.shape[2],
            dropout=0.1,
            pe='original'
        ).to(device),
        seq_length=training_seq.shape[2],
        timesteps=args.diffusionstep,
        loss_type='l2',
        objective='pred_v',
        auto_normalize=False,
        beta_schedule='linear'
    ).to(device)

    diff_ckpt = torch.load(args.resume_path, map_location=device)
    diffusion.load_state_dict(diff_ckpt['model'])
    diffusion.eval()

    parent_output_dir = get_parent_output_dir(args, RUN_TAG)
    os.makedirs(parent_output_dir, exist_ok=True)

    best_metrics_all = {"task_metrics": {}}
    from statistics import mean

    for i, task in enumerate(args.targetTaskList):
        task_embed = torch.tensor(condGenEmb[i:i+1]).to(device)
        time_embed = torch.tensor(stepGenEmb[i:i+1]).to(device)
        per_scores = []
        score_key_seen, task_type_seen = None, None

        for k in range(repeats):
            with torch.no_grad():
                latent = diffusion.sample(task_embed, time_embed, 1).transpose(1, 2).to(device)
                decoded = vae_model.decode(latent)

            overall_metrics, task_metrics, _ = evalStructuredLora(decoded, [task])
            sc = float(task_metrics[task]["score_value"])
            per_scores.append(sc)
            if score_key_seen is None: score_key_seen = task_metrics[task]["score_key"]
            if task_type_seen is None: task_type_seen = infer_task_type(task)

            bin_dir = os.path.join(parent_output_dir, task.replace('/', '_'), f"eval_{k}")
            os.makedirs(bin_dir, exist_ok=True)
            _save_decoded_as_peft_bin(decoded, args.basemodel, os.path.join(bin_dir, f"{task_name_wo_prefix(task)}.bin"))
            with open(os.path.join(bin_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(task_metrics[task], f, indent=2)

            del decoded, latent
            torch.cuda.empty_cache()

        best_metrics_all["task_metrics"][task] = {
            "score_key": score_key_seen,
            "score_value": float(mean(per_scores)),
            "scores": per_scores,
            "task_type": task_type_seen
        }

    with open(os.path.join(parent_output_dir, "best_scores_summary.json"), "w", encoding="utf-8") as f:
        json.dump(best_metrics_all, f, indent=2)


# ---------- decoder-only: external eval helpers ----------

def _try_read_dec_score(lora_dir: str):
    import json, glob, os
    cand = sorted(glob.glob(os.path.join(lora_dir, "*.json")))
    cand += sorted(glob.glob(os.path.join(lora_dir, "**/*.json"), recursive=True))
    for fp in cand:
        try:
            with open(fp, "r") as f:
                j = json.load(f)
            for k in ["accuracy", "acc", "exact_match", "score", "eval_accuracy"]:
                if k in j and isinstance(j[k], (int, float)):
                    return float(j[k])

            def _scan(obj):
                if isinstance(obj, dict):
                    for kk, vv in obj.items():
                        if kk in ["accuracy", "acc", "exact_match", "score", "eval_accuracy"] and isinstance(vv, (int, float)):
                            return float(vv)
                        r = _scan(vv)
                        if r is not None: return r
                elif isinstance(obj, list):
                    for it in obj:
                        r = _scan(it)
                        if r is not None: return r
                return None
            r = _scan(j)
            if r is not None:
                return r
        except Exception:
            continue
    return None

def _run_external_dec_eval(lora_dir: str,
                           task: str,
                           repo_dir: str,
                           model_dir: str,
                           script_rel: str = "scripts/run_eval.py"):
    """
    Invoke external eval script as a subprocess. lora_dir must be absolute.
    """
    import subprocess

    abs_lora_dir = os.path.abspath(lora_dir)

    if not os.path.isdir(abs_lora_dir):
        logger.warning(f"[dec-eval] LORA dir not found: {abs_lora_dir}")
        return None
    if not os.path.isfile(os.path.join(abs_lora_dir, "adapter_model.safetensors")):
        logger.warning(f"[dec-eval] Missing adapter_model.safetensors: {abs_lora_dir}")
        return None
    if not os.path.isfile(os.path.join(abs_lora_dir, "adapter_config.json")) and \
       not os.path.isfile(os.path.join(abs_lora_dir, "config.json")):
        logger.warning(f"[dec-eval] Missing adapter_config.json/config.json: {abs_lora_dir}")
        return None

    if not repo_dir or not os.path.isdir(repo_dir):
        logger.warning("[dec-eval] repo_dir not set or not found; skip.")
        return None
    if not model_dir or not os.path.isdir(model_dir):
        logger.warning("[dec-eval] model_dir not set or not found; skip.")
        return None

    script_path = os.path.join(repo_dir, script_rel)
    if not os.path.isfile(script_path):
        logger.warning(f"[dec-eval] script not found: {script_path}; skip.")
        return None

    env = os.environ.copy()
    src_dir = os.path.join(repo_dir, "src")
    env["PYTHONPATH"] = src_dir + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    cmd = [
        sys.executable, script_path,
        "--model-dir", model_dir,
        "--lora-dirs", abs_lora_dir,
        "--tasks", task,
        "--save-results"
    ]
    logger.info(f"[dec-eval] Run: {' '.join(cmd)}")
    try:
        out = subprocess.run(cmd, cwd=repo_dir, env=env, capture_output=True, text=True, check=False)
        logger.info(f"[dec-eval stdout]\n{out.stdout[-2000:]}")
        if out.returncode != 0:
            logger.warning(f"[dec-eval] non-zero return code {out.returncode}\n{out.stderr[-1000:]}")
            return None
    except Exception as e:
        logger.error(f"[dec-eval] failed: {e}")
        return None

    return _try_read_dec_score(abs_lora_dir)

def _export_decoder_only_to_dir(decoded_weights: dict,
                                out_dir: str,
                                adapter_config_path: str,
                                dtype: torch.dtype = torch.float32):
    """
    Export decoder-only LoRA as PEFT-compatible safetensors under out_dir:
        out_dir/adapter_model.safetensors
        out_dir/adapter_config.json (copied if exists)
        out_dir/config.json          (compat copy)
    """
    import shutil
    from safetensors.torch import save_file as save_safetensors

    os.makedirs(out_dir, exist_ok=True)
    d = decoded_weights.get("decoder_only", decoded_weights)
    Aq, Bq, Av, Bv = d["lora_qa"], d["lora_qb"], d["lora_va"], d["lora_vb"]

    def _prep(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 4 and t.size(0) == 1:
            t = t[0]
        return t.detach().to("cpu", dtype=dtype).contiguous()

    Aq, Bq, Av, Bv = map(_prep, (Aq, Bq, Av, Bv))  # [L,*,*]

    def _norm_A(t):  # -> [L, r, in]
        _, x, y = t.shape
        return t if x < y else t.transpose(-1, -2).contiguous()

    def _norm_B(t):  # -> [L, out, r]
        _, x, y = t.shape
        return t if y < x else t.transpose(-1, -2).contiguous()

    Aq, Av = _norm_A(Aq), _norm_A(Av)
    Bq, Bv = _norm_B(Bq), _norm_B(Bv)

    sd = {}
    L = Aq.shape[0]
    for i in range(L):
        sd[f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight"] = Aq[i]
        sd[f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_B.weight"] = Bq[i]
        sd[f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_A.weight"] = Av[i]
        sd[f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_B.weight"] = Bv[i]

    st_path = os.path.join(out_dir, "adapter_model.safetensors")
    save_safetensors(sd, st_path)

    if adapter_config_path and os.path.isfile(adapter_config_path):
        base_name = os.path.basename(adapter_config_path)
        shutil.copy2(adapter_config_path, os.path.join(out_dir, base_name))
        shutil.copy2(adapter_config_path, os.path.join(out_dir, "config.json"))
    else:
        print(f"[WARN] adapter_config not found: {adapter_config_path}")

    print(f"[✓] Saved: {st_path}  (+ adapter_config.json & config.json)")
    return st_path

def infer_model_decoder_only(args, training_seq, condGenEmb, stepGenEmb, device):
    """
    Decode decoder-only LoRA and export to:
        <parent_output_dir>/<safe_task>/eval_k/
    Then run external eval per directory if configured.
    """
    from statistics import mean

    # VAE
    ckpt_files = glob.glob(os.path.join(args.vae_ckpt_path, "*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No .ckpt file found in: {args.vae_ckpt_path}")
    vae_ckpt_path = ckpt_files[0]
    with open(args.vae_config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)
    cfg = OmegaConf.create(raw_cfg)
    vae = LoRAVAEModel_MoE(
        ddconfig=cfg.model.params.ddconfig,
        lossconfig=cfg.model.params.lossconfig,
        embed_dim=cfg.model.params.embed_dim,
        learning_rate=cfg.model.params.learning_rate,
        ckpt_path=vae_ckpt_path,
        device=device
    ).to(device).eval()

    # Diffusion
    diffusion = GaussianDiffusion1D(
        LoRATransformer(
            d_input=training_seq.shape[1],
            d_model=args.modeldim,
            d_output=training_seq.shape[1],
            d_cond=condGenEmb.shape[1],   # was d_kgEmb
            d_aux=stepGenEmb.shape[1],    # was d_timeEmb
            N=4,
            layernum=training_seq.shape[2],
            dropout=0.1,
            pe='original'
        ).to(device),
        seq_length=training_seq.shape[2],
        timesteps=args.diffusionstep,
        loss_type='l2',
        objective='pred_v',
        auto_normalize=False,
        beta_schedule='linear'
    ).to(device)

    diff_ckpt = torch.load(args.resume_path, map_location=device)
    diffusion.load_state_dict(diff_ckpt['model'])
    diffusion.eval()

    parent_output_dir = os.path.abspath(get_parent_output_dir(args, RUN_TAG))
    os.makedirs(parent_output_dir, exist_ok=True)
    logger.info(f"[✓] decoder-only export/eval dir: {parent_output_dir}")

    dtype = torch.float16 if getattr(args, "dec_save_dtype", "fp32") == "fp16" else torch.float32
    adapter_cfg = getattr(args, "dec_adapter_config", "")
    repeats = int(getattr(args, "infer_repeats", 3))
    best_metrics_all = {"task_metrics": {}}

    for i, task in enumerate(args.targetTaskList):
        safe_task = task.replace('/', '_')
        logger.info(f"[dec-only] Task: {task}")
        task_embed = torch.tensor(condGenEmb[i:i+1]).to(device)
        time_embed = torch.tensor(stepGenEmb[i:i+1]).to(device)

        per_scores = []
        for k in range(repeats):
            with torch.no_grad():
                latent = diffusion.sample(task_embed, time_embed, 1).transpose(1, 2).to(device)
                decoded = vae.decode(latent)

            eval_dir = os.path.join(parent_output_dir, safe_task, f"eval_{k}")
            _export_decoder_only_to_dir(decoded, eval_dir, adapter_cfg, dtype=dtype)

            score = _run_external_dec_eval(
                lora_dir=eval_dir,
                task=task,
                repo_dir=getattr(args, "dec_eval_repo_dir", None),
                model_dir=getattr(args, "dec_eval_model_dir", None),
                script_rel=getattr(args, "dec_eval_script_rel", "scripts/run_eval.py")
            )
            per_scores.append(None if score is None else float(score))

            del decoded, latent
            torch.cuda.empty_cache()

        valid_scores = [s for s in per_scores if isinstance(s, (int, float))]
        score_avg = float(mean(valid_scores)) if len(valid_scores) > 0 else None

        best_metrics_all["task_metrics"][task] = {
            "score_key": "external_score",
            "score_value": score_avg,
            "scores": per_scores,
            "task_type": infer_task_type(task)
        }
        logger.info(f"[dec-only] Task {task} | external_score (avg over {repeats}): {score_avg}")

    summary_path = os.path.join(parent_output_dir, "best_scores_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(best_metrics_all, f, indent=2, ensure_ascii=False)
    logger.info(f"[✓] Saved decoder-only infer summary -> {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./train_outputs/stage2",
                        help="Root directory to store outputs/checkpoints.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["train", "infer"], default="train")

    parser.add_argument("--modeldim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100001)
    parser.add_argument("--diffusionstep", type=int, default=500)
    parser.add_argument("--denoise", type=str, default='LoRATrans')
    parser.add_argument("--targetTask", type=str, default="lorahub_flan_t5_large-race_middle_Taking_a_test")
    parser.add_argument("--basemodel", type=str, default="google/flan-t5-large",
                        help="HF model id or local path for T5 eval decoding.")
    parser.add_argument("--lora_data_path", type=str, default="./artifacts/stage1_latents/latent_embed.pt")
    parser.add_argument("--vae_config_path", type=str, default="./configs/stage1_vae.yaml")
    parser.add_argument("--vae_ckpt_path", type=str, default="./checkpoints/stage1/")
    parser.add_argument("--targetTaskList", nargs='+', default=None)

    # decoder-only export/eval config 
    parser.add_argument("--dec_adapter_config", type=str, default="./external_eval/adapter_config.json",
                        help="Path to adapter_config.json used by external decoder-only eval.")
    parser.add_argument("--dec_eval_model_dir", type=str, default="./external_eval/base_model",
                        help="Directory of the base decoder-only model for evaluation.")
    parser.add_argument("--dec_eval_repo_dir", type=str, default="./external_eval/repo",
                        help="Repository directory that contains the eval script.")
    parser.add_argument("--dec_eval_script_rel", type=str, default="text-to-lora/scripts/run_eval.py",
                        help="Eval script path relative to --dec_eval_repo_dir")
    parser.add_argument("--dec_save_dtype", type=str, choices=["fp32", "fp16"], default="fp32",
                        help="Precision to export safetensors for decoder-only LoRA.")
    parser.add_argument("--infer_repeats", type=int, default=3)

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
        for key, val in config_dict.items():
            if not hasattr(args, key) or getattr(args, key) == parser.get_default(key):
                setattr(args, key, val)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"[✓] Created output_dir: {args.output_dir}")

    # load {task -> {latent, text_embedding}}
    try:
        data = torch.load(args.lora_data_path, map_location="cpu", weights_only=True)  
    except TypeError:
        data = torch.load(args.lora_data_path, map_location="cpu")
    latent_dict = {k: v["latent"] for k, v in data.items()}
    embed_dict  = {k: v["text_embedding"] for k, v in data.items()}

    if args.targetTaskList is None:
        tr, va = read_split_keys(args.vae_config_path)
        args.targetTaskList = va if len(va) > 0 else tr
        if not args.targetTaskList:
            args.targetTaskList = list(set(latent_dict.keys()) & set(embed_dict.keys()))
        logger.info(f"[✓] targetTaskList inferred from split: {len(args.targetTaskList)} items")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        training_seq, scale, condTrainEmb, condGenEmb, stepTrainEmb, stepGenEmb, genTarget = \
            prepare_lora_dataset(latent_dict, embed_dict, args.vae_config_path)

        train_dataset = TrainDataset(training_seq, condTrainEmb, stepTrainEmb)
        logger.info(f"[✓] Train samples: {len(train_dataset)}")
        logger.info(f"[✓] Output directory: {args.output_dir}")

        train_model(args, args.targetTaskList, train_dataset, training_seq, scale,
                    condTrainEmb, condGenEmb, stepTrainEmb, stepGenEmb, genTarget)

    elif args.mode == "infer":
        training_seq, scale, condTrainEmb, condGenEmb, stepTrainEmb, stepGenEmb, genTarget = \
            prepare_lora_dataset_for_infer(latent_dict, embed_dict, args.vae_config_path)

        dec_only = is_dec_only_from_cfg(args.vae_config_path)
        if dec_only:
            logger.info("[✓] Detected dec_only=True -> decoder-only export/infer")
            infer_model_decoder_only(args, training_seq, condGenEmb, stepGenEmb, device)
            logger.info("[✓] External eval can now consume the exported LoRA dirs under --output_dir.")
        else:
            logger.info("[✓] Detected dec_only=False -> encoder-decoder (T5) infer/eval")
            infer_model_t5_3times(args, training_seq, condGenEmb, stepGenEmb, device, repeats=args.infer_repeats)


if __name__ == '__main__':
    main()
