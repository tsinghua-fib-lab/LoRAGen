

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


def _canon_AB(A: Tensor, B: Tensor) -> Tuple[Tensor, Tensor]:
    """Canonicalize A, B so that the tall dimension is dim -2."""
    if A.shape[-2] < A.shape[-1]:
        A = A.transpose(-1, -2).contiguous()
    if B.shape[-2] < B.shape[-1]:
        B = B.transpose(-1, -2).contiguous()
    return A, B


def _qr_svd_vals(A: Tensor, B: Tensor) -> Tensor:
    """
    Efficiently compute singular values of ΔW = A @ B^T via QR + small SVD
    (Appendix A.4). Avoids forming the full d×d matrix.

    A: [..., d, r],  B: [..., d, r]
    Returns: singular values [..., r] in descending order.
    """
    QA, RA = torch.linalg.qr(A.float(), mode='reduced')
    QB, RB = torch.linalg.qr(B.float(), mode='reduced')
    K = torch.matmul(RA, RB.transpose(-1, -2))
    S = torch.linalg.svdvals(K)
    return S


@torch.no_grad()
def _select_topk_energy(S: Tensor, keep: float = 0.85, min_k: int = 1) -> int:
    """
    Determine the minimal k such that the top-k singular values explain
    at least a fraction `keep` (ρ) of the squared Frobenius norm.
    """
    energy = S ** 2
    cum = energy.cumsum(dim=-1)
    tot = energy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    ratio = cum / tot
    ks = (ratio >= keep).to(torch.int64).argmax(dim=-1) + 1
    return int(max(min_k, ks.max().item()))


def _reduce_loss(v: Tensor) -> Tensor:
    return v.mean()


class LoRAloss(nn.Module):
    """
    Composite loss for LoRA-structured autoencoders.

    Components:
      - Reconstruction loss (L1/L2) on structured weights.
      - Direction loss: 1 - cosine(pred, gt) on the full adaptation matrix ΔW.
      - Spectral loss (Eq. 3): weighted ℓp norm between leading singular values
        of ΔW, computed efficiently via QR decomposition.  The spectral-energy
        threshold ρ (`spec_energy_keep`) controls how many singular values to
        compare.
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
        spec_energy_keep: float = 0.85,
        **kwargs,
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
        self.spec_energy_keep = float(spec_energy_keep)

    def _find_ab_pairs(self, flat: Dict[str, Tensor]):
        """
        Discover (A, B) pairs from flattened keys.
        E.g. "encoder.lora_qa" + "encoder.lora_qb" -> pair for encoder-q.
        Returns list of (key_a, key_b) tuples.
        """
        pairs = []
        a_keys = {k for k in flat if k.endswith("_qa") or k.endswith("_va")}
        for ka in sorted(a_keys):
            kb = ka[:-1] + "b"
            if kb in flat:
                pairs.append((ka, kb))
        return pairs

    def _pairwise_terms(self, gt: Dict[str, Tensor], pred: Dict[str, Tensor]) -> Dict[str, Tensor]:
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
                cos = _safe_cosine(p, g)
                dir_terms.append(_reduce_loss(1.0 - cos))

        if self.use_spec:
            ab_pairs = self._find_ab_pairs(gt)
            if ab_pairs:
                for ka, kb in ab_pairs:
                    if ka not in pred or kb not in pred:
                        continue
                    ga, gb = gt[ka], gt[kb]
                    pa, pb = pred[ka], pred[kb]
                    ga = ga.to(pa.dtype).to(pa.device)
                    gb = gb.to(pb.dtype).to(pb.device)
                    ga, gb = _canon_AB(ga, gb)
                    pa, pb = _canon_AB(pa, pb)

                    S_gt = _qr_svd_vals(ga, gb)
                    S_pred = _qr_svd_vals(pa, pb)
                    k = _select_topk_energy(S_gt, keep=self.spec_energy_keep)
                    S_gt_k = S_gt[..., :k]
                    S_pred_k = S_pred[..., :k]
                    omega = (S_gt_k / S_gt_k.sum(dim=-1, keepdim=True).clamp_min(1e-12)).detach()

                    diff = S_pred_k - S_gt_k
                    if self.spec_p == 1:
                        spec = (omega * diff.abs()).mean()
                    else:
                        spec = (omega * diff ** 2).mean()
                    spec_terms.append(spec)

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
