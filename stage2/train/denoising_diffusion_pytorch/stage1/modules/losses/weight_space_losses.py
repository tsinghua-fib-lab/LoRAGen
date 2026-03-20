

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any

Tensor = torch.Tensor


def _flatten_lora_tree(x: Dict[str, Any]) -> Dict[str, Tensor]:
    """
    Flattens a nested LoRA dict into {path: tensor}.
    Expected input patterns:
      - T5-style:
        {
          "encoder": {"lora_qa": ..., "lora_qb": ..., "lora_va": ..., "lora_vb": ...},
          "decoder": {
            "decoder_attn": {... four keys ...},
            "cross_attn":   {... four keys ...}
          }
        }
      - Decoder-only:
        { "decoder_only": {"lora_qa": ..., "lora_qb": ..., "lora_va": ..., "lora_vb": ...} }
    """
    out: Dict[str, Tensor] = {}

    def visit(prefix: str, node: Any):
        if isinstance(node, dict):
            for k, v in node.items():
                new_prefix = f"{prefix}.{k}" if prefix else k
                visit(new_prefix, v)
        elif torch.is_tensor(node):
            out[prefix] = node
        else:
            raise ValueError(f"Unsupported node type at {prefix}: {type(node)}")

    visit("", x)
    return out


def _safe_cosine(a: Tensor, b: Tensor, eps: float = 1e-12) -> Tensor:
    a = a.reshape(a.size(0), -1)
    b = b.reshape(b.size(0), -1)
    an = a.norm(dim=1, keepdim=True).clamp_min(eps)
    bn = b.norm(dim=1, keepdim=True).clamp_min(eps)
    return (a * b).sum(dim=1, keepdim=True) / (an * bn)


def _fft_1d(x: Tensor) -> Tensor:
    """
    Computes magnitude spectrum along the last dimension.
    Works for real tensors of any rank >= 1.
    """
    # Move last dim to FFT; keep dtype for gradients
    spec = torch.fft.rfft(x.float(), dim=-1)
    mag = spec.abs()
    # Cast back to original dtype if needed
    return mag.to(dtype=x.dtype)


def _reduce_loss(v: Tensor) -> Tensor:
    # Mean over all elements and batch if present
    return v.mean()


class LoRAloss(nn.Module):
    """
    Composite loss for LoRA-structured autoencoders.

    Components:
      - Reconstruction loss (L1/L2) on structured weights.
      - Direction loss: 1 - cosine(pred, gt).
      - Spectral loss: p-norm between rFFT magnitudes.
      - KL divergence for VAE posteriors.

    All components are optional and weighted.
    """

    def __init__(
        self,
        rec_metric: str = "l2",
        rec_core_weight: float = 1.0,
        kl_weight: float = 0.0,
        use_dir: bool = False,
        w_dir: float = 1.0,
        use_spec: bool = False,
        w_spec: float = 1.0,
        spec_p: int = 2,
    ):
        super().__init__()
        rec_metric = rec_metric.lower()
        assert rec_metric in {"l1", "l2"}
        assert spec_p in {1, 2}

        self.rec_metric = rec_metric
        self.rec_core_weight = float(rec_core_weight)
        self.kl_weight = float(kl_weight)

        self.use_dir = bool(use_dir)
        self.w_dir = float(w_dir)

        self.use_spec = bool(use_spec)
        self.w_spec = float(w_spec)
        self.spec_p = int(spec_p)

    def _pairwise_terms(self, gt: Dict[str, Tensor], pred: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Computes per-key losses and aggregates them.
        Returns a dict of scalar tensors for each enabled component.
        """
        rec_terms = []
        dir_terms = []
        spec_terms = []

        keys = sorted(set(gt.keys()) & set(pred.keys()))
        if not keys:
            raise ValueError("Empty intersection between GT and prediction keys.")

        for k in keys:
            g = gt[k].to(pred[k].dtype).to(pred[k].device)
            p = pred[k]

            if self.rec_core_weight > 0.0:
                if self.rec_metric == "l1":
                    rec = (p - g).abs()
                else:
                    rec = (p - g) ** 2
                rec_terms.append(_reduce_loss(rec))

            if self.use_dir:
                if g.dim() == 4:
                    B = g.size(0)
                else:
                    B = g.size(0) if g.size(0) == p.size(0) else 1
                    if B == 1 and g.dim() >= 1 and p.dim() >= 1:
                        g = g.unsqueeze(0)
                        p = p.unsqueeze(0)
                cos = _safe_cosine(p, g)  # [B,1]
                dir_terms.append(_reduce_loss(1.0 - cos))

            if self.use_spec:
                if g.size(-1) >= 2 and p.size(-1) >= 2:
                    G = _fft_1d(g)
                    P = _fft_1d(p)
                    if self.spec_p == 1:
                        spec = (P - G).abs()
                    else:
                        spec = (P - G) ** 2
                    spec_terms.append(_reduce_loss(spec))

        out: Dict[str, Tensor] = {}
        if rec_terms:
            out["rec_loss"] = torch.stack(rec_terms).mean()
        if dir_terms:
            out["dir_loss"] = torch.stack(dir_terms).mean()
        if spec_terms:
            out["spec_loss"] = torch.stack(spec_terms).mean()
        return out

    def _kl_term(self, posterior) -> Tensor:
        if posterior is None:
            return torch.tensor(0.0)
        if hasattr(posterior, "kl"):
            kl = posterior.kl()
            if isinstance(kl, torch.Tensor):
                return kl.mean()
        mu = getattr(posterior, "mean", None)
        logvar = getattr(posterior, "logvar", None)
        if isinstance(mu, torch.Tensor) and isinstance(logvar, torch.Tensor):
            # KL(q||p), p ~ N(0, I)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            return kl.mean()
        return torch.tensor(0.0)

    def _extract_structured(self, x: Any) -> Dict[str, Tensor]:
        if not isinstance(x, dict):
            raise ValueError("Expected structured dict for LoRA weights.")
        return _flatten_lora_tree(x)

    def forward(
        self,
        structured_gt: Dict[str, Any],
        reconstructions: Dict[str, Any],
        posterior: Any = None,
        split: str = "train",
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Returns
        -------
        total_loss : Tensor
        log_dict   : Dict[str, Tensor]
        """
        gt_flat = self._extract_structured(structured_gt)
        pr_flat = self._extract_structured(reconstructions)

        parts = self._pairwise_terms(gt_flat, pr_flat)

        total = torch.tensor(0.0, device=next(iter(pr_flat.values())).device)
        log: Dict[str, Tensor] = {}

        if "rec_loss" in parts and self.rec_core_weight > 0.0:
            rec_loss = self.rec_core_weight * parts["rec_loss"]
            total = total + rec_loss
            log[f"{split}/rec_loss"] = parts["rec_loss"].detach()

        if self.use_dir and "dir_loss" in parts:
            dir_loss = self.w_dir * parts["dir_loss"]
            total = total + dir_loss
            log[f"{split}/dir_loss"] = parts["dir_loss"].detach()

        if self.use_spec and "spec_loss" in parts:
            spec_loss = self.w_spec * parts["spec_loss"]
            total = total + spec_loss
            log[f"{split}/spec_loss"] = parts["spec_loss"].detach()

        if self.kl_weight > 0.0:
            kl = self._kl_term(posterior).to(total.device)
            total = total + self.kl_weight * kl
            log[f"{split}/kl_loss"] = kl.detach()

        log[f"{split}/aeloss"] = total.detach()

        return total, log

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
