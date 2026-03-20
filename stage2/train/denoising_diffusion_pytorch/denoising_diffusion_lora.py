# -*- coding: utf-8 -*-

import math
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple

import os
import yaml
import numpy as np
from tqdm.auto import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ema_pytorch import EMA
from einops import reduce
from torch.optim import Adam
from sklearn import metrics

from denoising_diffusion_pytorch.version import __version__

# -----------------------------------------------------------------------------
# Constants & small helpers
# -----------------------------------------------------------------------------

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def exists(x):
    return x is not None


def default(val, d):
    return val if exists(val) else (d() if callable(d) else d)


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    """Infinite dataloader."""
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


# -----------------------------------------------------------------------------
# Normalization helpers
# -----------------------------------------------------------------------------

def normalize_to_neg_one_to_one(x):
    return x * 2 - 1


def unnormalize_to_zero_to_one(x):
    return (x + 1) * 0.5


# -----------------------------------------------------------------------------
# Diffusion scheduling
# -----------------------------------------------------------------------------

def extract(a, t, x_shape):
    """Helper to gather a[t] and reshape to broadcast over x."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    # A simple linear schedule, lightly scaled.
    scale = 1.0
    beta_start = scale * 1e-5
    beta_end = scale * 2e-2
    betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    # Cosine schedule commonly used for improved stability.
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# -----------------------------------------------------------------------------
# 1D Gaussian Diffusion (LoRA parameter generator)
# -----------------------------------------------------------------------------

class GaussianDiffusion1D(nn.Module):
    """
    A 1D diffusion wrapper that calls a denoising network with optional conditioning.
    Inputs are shaped as (B, C, N), where N is sequence length.
    """

    def __init__(
        self,
        denoising_model,
        *,
        seq_length,
        timesteps=1000,
        sampling_timesteps=None,
        loss_type='l1',
        objective='pred_noise',
        beta_schedule='cosine',
        ddim_sampling_eta=0.0,
        auto_normalize=False
    ):
        super().__init__()
        self.model = denoising_model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.seq_length = seq_length
        self.objective = objective
        assert objective in {
            'pred_noise', 'pred_x0', 'pred_v'
        }, "objective must be one of {'pred_noise','pred_x0','pred_v'}"

        # betas
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'Unknown beta schedule: {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling config
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # register diffusion buffers as float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # posterior terms q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # loss re-weighting by SNR
        snr = alphas_cumprod / (1 - alphas_cumprod)
        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif objective == 'pred_x0':
            loss_weight = snr
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)
        register_buffer('loss_weight', loss_weight)

        # optional auto (un)normalize
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    # --- prediction conversions ---

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # --- core model calls ---

    def model_predictions(self, x, t, cond_emb, aux_emb, x_self_cond=None,
                          clip_x_start=False, rederive_pred_noise=False):
        """
        Call underlying denoising model with both conditioning embeddings.
        """
        model_output = self.model(x, t, cond_emb, aux_emb, x_self_cond)

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = maybe_clip(model_output)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = maybe_clip(self.predict_start_from_v(x, t, v))
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, cond_emb, aux_emb, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions(x, t, cond_emb, aux_emb, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, cond_emb, aux_emb, x_self_cond=None, clip_denoised=True):
        """
        One reverse diffusion step at (integer) time t.
        """
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)

        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, cond_emb=cond_emb, aux_emb=aux_emb,
            x_self_cond=x_self_cond, clip_denoised=clip_denoised
        )
        noise = torch.randn_like(x) if t > 0 else 0.
        pred = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, cond_emb, aux_emb):
        """
        Full ancestral sampling loop.
        """
        _, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, cond_emb, aux_emb, self_cond, clip_denoised=True)

        img = self.unnormalize(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, shape, cond_emb, aux_emb, clip_denoised=True):
        """
        DDIM sampling (when sampling_timesteps < train timesteps).
        """
        batch, device = shape[0], self.betas.device
        total_timesteps = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        eta = self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=device)
        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(
                img, time_cond, cond_emb, aux_emb, self_cond, clip_x_start=clip_denoised
            )

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        img = self.unnormalize(img)
        return img

    @torch.no_grad()
    def sample(self, cond_emb, aux_emb, batch_size=16):
        """
        Public sampling API. Chooses DDPM/DDIM based on configuration.
        """
        seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, seq_length), cond_emb, aux_emb)

    @torch.no_grad()
    def interpolate(self, x1, x2, cond_emb, aux_emb, t=None, lam=0.5):
        """
        Interpolate between two inputs in noisy space and denoise back.
        """
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)
        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device=device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))
        img = (1 - lam) * xt1 + lam * xt2

        x_start = None
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, cond_emb, aux_emb, self_cond)

        return img

    # --- training loss ---

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'Invalid loss type: {self.loss_type}')

    def p_losses(self, x_start, t, cond_emb, aux_emb, noise=None):
        """
        Standard diffusion training objective with optional self-conditioning.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t, cond_emb, aux_emb).pred_x_start
                x_self_cond.detach_()

        model_out = self.model(x, t, cond_emb, aux_emb, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            target = self.predict_v(x_start, t, noise)
        else:
            raise ValueError(f'Unknown objective: {self.objective}')

        loss = self.loss_fn(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, input_signal, cond_emb, aux_emb, *args, **kwargs):
        """
        Training forward: sample a diffusion step t and compute loss.
        """
        b, c, n = input_signal.shape
        device = input_signal.device
        assert n == self.seq_length, f'Seq length {n} must equal {self.seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        input_signal = self.normalize(input_signal)
        return self.p_losses(input_signal, t, cond_emb, aux_emb, *args, **kwargs)


# -----------------------------------------------------------------------------
# Trainer (includes optional decoder-only export + external eval hooks)
# -----------------------------------------------------------------------------

class Trainer1D(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=25,
        results_folder='./ModelSave/exp777',
        amp=False,
        fp16=False,
        split_batches=True,
        logger=None,

        # conditioning banks
        cond_embedding_bank=None,
        aux_embedding_bank=None,
        target_latents=None,       # ground truth latent vectors for sanity-check MAE

        task_names=None,
        scale=None,
        tbwriter=None,
        outputpath=None,
        sampleTimes=None,
        vae_config_path=None,
        vae_ckpt_path=None,

        # decoder-only export & external eval
        dec_only=False,
        dec_adapter_config=None,
        dec_eval_model_dir=None,
        dec_eval_repo_dir=None,
        dec_eval_script_rel="scripts/run_eval.py",
        dec_save_dtype="fp32",
    ):
        super().__init__()

        from accelerate import Accelerator
        from ema_pytorch import EMA
        from einops import repeat  # may be used by downstream models

        self.Accelerator = Accelerator
        self.EMA = EMA
        self.Path = Path
        self.DataLoader = DataLoader
        self.Adam = Adam

        self.accelerator = self.Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )
        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert has_int_squareroot(num_samples), 'num_samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        # dataloader (dataset expected to return (input_signal, cond_emb, aux_emb))
        dl = self.DataLoader(
            dataset,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=os.cpu_count(),
            drop_last=True
        )
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        self.opt = self.Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        if self.accelerator.is_main_process:
            self.ema = self.EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        self.results_folder = self.Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.step = 0
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        self.logger = logger

        # external condition banks / eval metas
        self.cond_embedding_bank = cond_embedding_bank
        self.aux_embedding_bank = aux_embedding_bank
        self.target_latents = target_latents
        self.task_names = task_names or []
        self.scale = scale

        # TensorBoard
        self.outputpath = os.path.abspath(outputpath) if outputpath else os.path.abspath("./outputs")
        os.makedirs(self.outputpath, exist_ok=True)
        tb_logdir = os.path.join(self.outputpath, "TensorBoardLogs")
        os.makedirs(tb_logdir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tb_logdir)

        self.sampleTimes = sampleTimes
        self.vae_config_path = vae_config_path
        self.vae_ckpt_path = vae_ckpt_path

        # decoder-only related
        self.dec_only = bool(dec_only)
        self.dec_adapter_config = dec_adapter_config
        self.dec_eval_model_dir = dec_eval_model_dir
        self.dec_eval_repo_dir = dec_eval_repo_dir
        self.dec_eval_script_rel = dec_eval_script_rel
        self.dec_save_dtype = str(dec_save_dtype).lower()

    @property
    def device(self):
        return self.accelerator.device

    # --- checkpointing ---

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        scaler_state = None
        if hasattr(self.accelerator, 'scaler') and self.accelerator.scaler is not None:
            try:
                scaler_state = self.accelerator.scaler.state_dict()
            except Exception as e:
                print(f"[WARN] Failed to save scaler state: {e}")
                scaler_state = None

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': scaler_state,
            'version': '1.0'
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
        print(f"[INFO] Model checkpoint saved to {self.results_folder}/model-{milestone}.pt")

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=self.device)
        self.accelerator.unwrap_model(self.model).load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])

        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data['ema'])

        if hasattr(self.accelerator, 'scaler') and self.accelerator.scaler is not None and data.get('scaler') is not None:
            try:
                self.accelerator.scaler.load_state_dict(data['scaler'])
            except Exception as e:
                print(f"[WARN] Failed to load scaler state: {e}")

    # --- eval/save cadence ---

    def should_eval_step(self, step: int) -> bool:
        if self.dec_only:
            if step <= 1000:
                return step % 200 == 0
            elif step <= 2000:
                return step % 500 == 0
            elif step <= 10000:
                return step % 1000 == 0
            return False
        else:
            if step <= 100:
                return step % 10 == 0
            elif step <= 1000:
                return step % 100 == 0
            elif step <= 4000:
                return step % 500 == 0
            return False

    def should_save_step(self, step):
        return (step % 100 == 0) if step <= 1000 else (step % 500 == 0)

    # --- decoder-only export & external eval ---

    @torch.no_grad()
    def _export_decoder_only(self, decoded_weights: dict, task_name: str, out_root: str):
        """
        Save decoder-only LoRA weights in PEFT-compatible shape:
          <out_root>/<task>/adapter_model.safetensors + adapter_config.json
        """
        from safetensors.torch import save_file as save_safetensors
        import shutil

        os.makedirs(out_root, exist_ok=True)
        out_dir = os.path.join(out_root, task_name)
        os.makedirs(out_dir, exist_ok=True)

        d = decoded_weights.get("decoder_only", decoded_weights)
        Aq, Bq, Av, Bv = d["lora_qa"], d["lora_qb"], d["lora_va"], d["lora_vb"]

        def _prep(t: torch.Tensor) -> torch.Tensor:
            # Squeeze optional leading batch dim and cast dtype for export
            if t.dim() == 4 and t.size(0) == 1:
                t = t[0]
            dtype = torch.float16 if self.dec_save_dtype == "fp16" else torch.float32
            return t.detach().to("cpu", dtype=dtype).contiguous()

        Aq, Bq, Av, Bv = map(_prep, (Aq, Bq, Av, Bv))

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

        if self.dec_adapter_config and os.path.isfile(self.dec_adapter_config):
            base_name = os.path.basename(self.dec_adapter_config)
            shutil.copy2(self.dec_adapter_config, os.path.join(out_dir, base_name))
            shutil.copy2(self.dec_adapter_config, os.path.join(out_dir, "config.json"))
        else:
            print(f"[WARN] adapter_config not found: {self.dec_adapter_config}")

        return out_dir

    def _run_external_dec_eval(self, lora_dir: str, task: str):
        """
        Spawn an external process to evaluate exported LoRA using a separate repo.
        """
        import subprocess, sys

        if not self.dec_eval_repo_dir or not os.path.isdir(self.dec_eval_repo_dir):
            self.logger.warning("[dec-eval] repo_dir not set or missing; skip external eval.")
            return None
        if not self.dec_eval_model_dir or not os.path.isdir(self.dec_eval_model_dir):
            self.logger.warning("[dec-eval] model_dir not set or missing; skip external eval.")
            return None

        script_path = os.path.join(self.dec_eval_repo_dir, self.dec_eval_script_rel)
        if not os.path.isfile(script_path):
            self.logger.warning(f"[dec-eval] script not found: {script_path}; skip external eval.")
            return None

        env = os.environ.copy()
        src_dir = os.path.join(self.dec_eval_repo_dir, "src")
        env["PYTHONPATH"] = (src_dir + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else ""))

        cmd = [
            sys.executable, script_path,
            "--model-dir", self.dec_eval_model_dir,
            "--lora-dirs", lora_dir,
            "--tasks", task,
            "--save-results"
        ]

        self.logger.info(f"[dec-eval] Run: {' '.join(cmd)}")
        try:
            out = subprocess.run(cmd, cwd=self.dec_eval_repo_dir, env=env,
                                 capture_output=True, text=True, check=False)
            self.logger.info(f"[dec-eval stdout]\n{out.stdout[-2000:]}")
            if out.returncode != 0:
                self.logger.warning(f"[dec-eval] nonzero exit {out.returncode}\n{out.stderr[-1000:]}")
        except Exception as e:
            self.logger.error(f"[dec-eval] failed to invoke: {e}")
            return None

        score = self._try_read_dec_score(lora_dir)
        return score

    def _try_read_dec_score(self, lora_dir: str):
        """
        Try to parse a JSON file in lora_dir for a scalar score (accuracy/acc/exact_match/score...).
        """
        import json, glob
        cand = sorted(glob.glob(os.path.join(lora_dir, "*.json")))
        cand += sorted(glob.glob(os.path.join(lora_dir, "**/*.json"), recursive=True))
        for fp in cand:
            try:
                with open(fp, "r") as f:
                    j = json.load(f)
                for k in ["accuracy", "acc", "exact_match", "score", "eval_accuracy"]:
                    if k in j and isinstance(j[k], (int, float)):
                        return float(j[k])

                # fallback: recursively search for a numeric field with one of those keys
                def _scan(obj):
                    if isinstance(obj, dict):
                        for kk, vv in obj.items():
                            if kk in ["accuracy", "acc", "exact_match", "score", "eval_accuracy"] and isinstance(vv, (int, float)):
                                return float(vv)
                            r = _scan(vv)
                            if r is not None:
                                return r
                    elif isinstance(obj, list):
                        for it in obj:
                            r = _scan(it)
                            if r is not None:
                                return r
                    return None

                r = _scan(j)
                if r is not None:
                    return r
            except Exception:
                continue
        return None

    # --- training loop with periodic evaluation ---

    def train(self):
        """
        Main training loop. Periodically:
          1) samples latent(s) with EMA model,
          2) decodes them via VAE to structured LoRA weights,
          3) runs either internal (T5) or external (decoder-only) evaluation,
          4) tracks best scores and saves checkpoints.
        """
        import glob, json
        from omegaconf import OmegaConf
        from denoising_diffusion_pytorch.stage1.models.autoencoder_lora import LoRAVAEModel_MoE

        # build / load VAE (for decoding structured LoRA)
        with open(self.vae_config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        config = OmegaConf.create(raw_config)
        vae_ckpt_dir = self.vae_ckpt_path
        ckpt_files = glob.glob(os.path.join(vae_ckpt_dir, "*.ckpt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No .ckpt file found in directory: {vae_ckpt_dir}")
        vae_ckpt_path = ckpt_files[0]
        print(f"Using VAE checkpoint: {vae_ckpt_path}")

        vae_model = LoRAVAEModel_MoE(
            ddconfig=config.model.params.ddconfig,
            lossconfig=config.model.params.lossconfig,
            embed_dim=config.model.params.embed_dim,
            learning_rate=config.model.params.learning_rate,
            ckpt_path=vae_ckpt_path,
            device=self.device
        ).to(self.device).eval()

        # internal T5 evaluation only if not in decoder-only mode
        if not self.dec_only:
            from evaluation.evalLora import evalStructuredLora

        best_score_per_task = {}
        best_task_metrics = {}
        task_idx = 0

        with tqdm(initial=self.step, total=self.train_num_steps,
                  disable=not self.accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                total_loss = 0.0

                # gradient accumulation
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    batch_signal = data[0].to(self.device)
                    batch_cond = data[1].to(self.device)
                    batch_aux = data[2].to(self.device)

                    with self.accelerator.autocast():
                        loss = self.model(batch_signal, batch_cond, batch_aux) / self.gradient_accumulate_every
                        total_loss += loss.item()
                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')
                self.writer.add_scalar('Diffusion Loss', total_loss, self.step)
                self.opt.step()
                self.opt.zero_grad()
                self.accelerator.wait_for_everyone()
                self.step += 1

                # EMA & periodic eval
                if self.accelerator.is_main_process:
                    self.ema.update()

                    if self.should_eval_step(self.step):
                        self.ema.ema_model.eval()
                        self.logger.info('\n---------------- eval ----------------')
                        try:
                            task_name = self.task_names[task_idx % max(1, len(self.task_names))]
                            task_cond = torch.tensor(self.cond_embedding_bank[task_idx % len(self.cond_embedding_bank)]).unsqueeze(0).to(self.device)
                            task_target = torch.tensor(self.target_latents[task_idx % len(self.target_latents)]).unsqueeze(0).to(self.device)
                            repeated_aux = torch.tensor(self.aux_embedding_bank[0:1]).repeat(task_cond.shape[0], 1).to(self.device)
                            task_idx += 1

                            with torch.no_grad():
                                latent = self.ema.ema_model.sample(task_cond, repeated_aux, len(task_cond)).to(self.device)
                                if latent.isnan().any():
                                    self.logger.warning(f"[WARN] latent for task {task_name} has NaNs, skip.")
                                    continue

                                decoded_weights = vae_model.decode(latent.transpose(1, 2))
                                mae = metrics.mean_absolute_error(
                                    latent.cpu().numpy().flatten(),
                                    task_target.cpu().numpy().flatten()
                                )
                                self.writer.add_scalar('latent_mae', mae, self.step)
                                self.logger.info(f"Step: {self.step} | Evaluating task: {task_name}")

                                if not self.dec_only:
                                    # internal T5 evaluation on structured LoRA
                                    overall_metrics, task_metrics, _ = evalStructuredLora(decoded_weights, [task_name])
                                    main_score = task_metrics[task_name]["score_value"]
                                    score_key = task_metrics[task_name]["score_key"]
                                else:
                                    # decoder-only route: export and run external eval
                                    export_root = os.path.join(self.outputpath, "dec_eval", f"step_{self.step}")
                                    lora_dir = self._export_decoder_only(decoded_weights, task_name, export_root)
                                    score = self._run_external_dec_eval(lora_dir=lora_dir, task=task_name)
                                    main_score = float(score) if isinstance(score, (int, float)) else float('nan')
                                    score_key = "external_score"

                                prev_best = best_score_per_task.get(task_name, -1.0)
                                improved = (np.isnan(prev_best) and not np.isnan(main_score)) or (main_score > prev_best)

                                if improved or self.should_save_step(self.step):
                                    best_score_per_task[task_name] = float(main_score) if not np.isnan(main_score) else float(prev_best)
                                    best_task_metrics[task_name] = {
                                        "step": self.step,
                                        "score_value": None if np.isnan(main_score) else float(main_score),
                                        "score_key": score_key,
                                        "mae": float(mae)
                                    }

                                    self.logger.info(f'Best so far @ step {self.step}')
                                    self.logger.info(f'Task: {task_name}, {score_key}: {main_score}')
                                    np.save(os.path.join(self.outputpath, f'sampleRes_{self.step}.npy'), latent.cpu().numpy())
                                    self.save(self.step)

                                    with open(os.path.join(self.outputpath, f"best_metrics_{task_name}_{self.step}.json".replace("/", "_")), "w", encoding="utf-8") as f:
                                        json.dump(best_task_metrics[task_name], f, indent=2)

                                    with open(os.path.join(self.outputpath, "best_scores_summary.json"), "w", encoding="utf-8") as f:
                                        json.dump(best_task_metrics, f, indent=2)

                        except Exception as e:
                            self.logger.error(f"[ERROR] Task {task_name} failed during evaluation: {e}")

                pbar.update(1)

        self.accelerator.print('training complete')
