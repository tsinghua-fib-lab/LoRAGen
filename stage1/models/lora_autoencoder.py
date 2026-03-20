import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from stage1.modules.distributions import DiagonalGaussianDistribution

from utils.util import instantiate_from_config
import os

try:
    from safetensors.torch import load_file as _safe_load
except Exception:
    _safe_load = None


def _load_state_any(path: str):
    if path.endswith(".safetensors"):
        assert _safe_load is not None, "Please `pip install safetensors` to load .safetensors files."
        return dict(_safe_load(path))
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        return sd["state_dict"]
    if isinstance(sd, dict):
        return sd
    raise ValueError(f"Unrecognized state_dict in {path}")


def _move_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_move_to_device(v, device) for v in x)
    return x


def _infer_batch_size(x):
    if torch.is_tensor(x):
        return x.size(0)
    if isinstance(x, dict):
        for v in x.values():
            bs = _infer_batch_size(v)
            if bs is not None:
                return bs
        return None
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return _infer_batch_size(x[0])
    return None


class LoRAVAEModel_MoE(pl.LightningModule):

    def __init__(
        self,
        ddconfig,
        lossconfig,
        embed_dim,
        learning_rate,
        ckpt_path=None,
        ignore_keys=[],
        input_key="weight",
        cond_key="dataset",
        device="cuda",
        monitor=None,
        moe_aux_weight: float = 0.0,
        moe_aux_warmup_steps: int = 0,
        moe_aux_in_val: bool = False,
        moe_router_tau: float = 1.5,
        moe_router_tau_final: float = 1.5,
        moe_router_tau_decay_steps: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.devices = device
        self.cond_key = cond_key
        self.learning_rate = learning_rate
        self.input_key = input_key

        self.encoder = instantiate_from_config(ddconfig["encoder"])
        self.decoder = instantiate_from_config(ddconfig["decoder"])
        self.loss = instantiate_from_config(lossconfig)

        self.quant_conv = nn.Identity()
        self.post_quant_conv = nn.Identity()
        self.embed_dim = embed_dim

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.moe_aux_weight = moe_aux_weight
        self.moe_aux_warmup_steps = moe_aux_warmup_steps
        self.moe_aux_in_val = moe_aux_in_val
        self.moe_router_tau = moe_router_tau
        self.moe_router_tau_final = moe_router_tau_final
        self.moe_router_tau_decay_steps = moe_router_tau_decay_steps

        try:
            self.save_hyperparameters(ignore=["encoder", "decoder", "loss"])
        except Exception:
            pass

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    # ---------------- VAE encode/decode ----------------
    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)

        if z.dim() == 3:
            return self.decoder(z)  # [B, N, latent_dim]
        elif z.dim() == 4:
            return self.decoder(z)
        elif z.dim() == 5 and z.size(1) == 1:
            return self.decoder(z.squeeze(1))
        elif z.dim() == 5:
            b, v, n, d = z.shape
            z = z.view(b * v, n, d)
            return self.decoder(z)
        else:
            raise ValueError(f"Unsupported shape for z: {z.shape}")

    def forward(self, batch, sample_posterior=True):
        """
        Returns: inputs (moved to device), reconstructions, posterior.
        """
        x = batch[self.input_key]
        x = _move_to_device(x, self.device)
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        reconstructed = self.decode(z)
        return x, reconstructed, posterior

    # ---------------- train/val step ----------------
    def training_step(self, batch, batch_idx):
        structured_gt = _load_state_any(batch["path"][0])

        inputs, reconstructions, posterior = self(batch)
        bs = _infer_batch_size(inputs) or 1

        aeloss, log_dict_ae = self.loss(structured_gt, reconstructions, posterior, split="train")

        if hasattr(self.decoder, "set_tau") and self.moe_router_tau_decay_steps > 0:
            t = min(1.0, float(self.global_step) / max(1, self.moe_router_tau_decay_steps))
            tau = self.moe_router_tau * (1 - t) + self.moe_router_tau_final * t
            self.decoder.set_tau(tau)

        # MoE auxiliary loss with linear warmup
        warm_ratio = (
            min(1.0, float(self.global_step) / max(1, self.moe_aux_warmup_steps))
            if self.moe_aux_warmup_steps > 0
            else 1.0
        )
        aux_w = self.moe_aux_weight * warm_ratio
        aux = torch.tensor(0.0, device=self.device)
        if hasattr(self.decoder, "moe_aux_loss"):
            aux = self.decoder.moe_aux_loss()

        total = aeloss + aux_w * aux

        self.log("train/aeloss", aeloss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log("train/moe_aux_loss", aux.detach(), on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log(
            "train/aux_weight_eff",
            torch.tensor(aux_w, device=self.device),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=bs,
        )
        self.log("train/total_loss", total, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)

        for k in ["train/aeloss", "train/total_loss", "train/moe_aux_loss", "train/aux_weight_eff"]:
            log_dict_ae.pop(k, None)
        if log_dict_ae:
            self.log_dict(log_dict_ae, on_step=True, on_epoch=False, sync_dist=True, batch_size=bs)

        return total

    def validation_step(self, batch, batch_idx):
        structured_gt = _load_state_any(batch["path"][0])
        inputs, reconstructions, posterior = self(batch)
        bs = _infer_batch_size(inputs) or 1

        aeloss, log_dict_ae = self.loss(structured_gt, reconstructions, posterior, split="val")

        aux = torch.tensor(0.0, device=self.device)
        if hasattr(self.decoder, "moe_aux_loss"):
            aux = self.decoder.moe_aux_loss()

        val_total = aeloss + (self.moe_aux_weight * aux if self.moe_aux_in_val else 0.0)

        self.log("val/aeloss", aeloss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log("val/moe_aux_loss", aux.detach(), on_step=False, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log("val/total_loss", val_total, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=bs)

        for k in ["val/aeloss", "val/total_loss", "val/moe_aux_loss", "val/rec_loss"]:
            log_dict_ae.pop(k, None)
        if log_dict_ae:
            self.log_dict(log_dict_ae, on_step=False, on_epoch=True, sync_dist=True, batch_size=bs)

        return None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters()),
            lr=self.learning_rate,
            betas=(0.5, 0.9),
        )
        return optimizer
