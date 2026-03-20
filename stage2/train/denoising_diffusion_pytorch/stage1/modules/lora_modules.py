import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




# =========================
# LoRA Encoder (supports T5-style and decoder-only)
# =========================
class LoRAEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        rank: int,
        latent_dim: int,
        hidden_dim: int = 0,
        dec_only: bool = False,
        d_in: int = None,
        d_out_q: int = None,
        d_out_v: int = None,
    ):
        super().__init__()
        self.d = d_model
        self.r = rank
        self.latent_dim = latent_dim
        self.dec_only = dec_only

        # T5 path
        in_dim = d_model * rank
        if not dec_only:
            if hidden_dim and hidden_dim > 0:
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 2 * latent_dim),
                )
            else:
                self.net = nn.Linear(in_dim, 2 * latent_dim)
        else:
            self.net = nn.Identity()

        # Decoder-only path (four heads)
        if dec_only:
            assert all(v is not None for v in [d_in, d_out_q, d_out_v]), "decoder-only requires d_in/d_out_q/d_out_v"
            def make(head_in):
                if hidden_dim and hidden_dim > 0:
                    return nn.Sequential(
                        nn.Linear(head_in, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, 2 * latent_dim),
                    )
                else:
                    return nn.Linear(head_in, 2 * latent_dim)

            self.net_A = make(d_in * rank)          # shared for A_q/A_v
            self.net_Bq = make(d_out_q * rank)      # B_q
            self.net_Bv = make(d_out_v * rank)      # B_v
            self._d_in, self._dq, self._dv = d_in, d_out_q, d_out_v

    def forward(self, x):
        # T5: x is Tensor [B, N, d, r]
        if torch.is_tensor(x):
            B, N, d, r = x.shape
            assert d == self.d and r == self.r, f"Shape mismatch: got d={d}, r={r}, expected {self.d}, {self.r}"
            flat = x.view(B * N, d * r)
            stats = self.net(flat).view(B, N, 2 * self.latent_dim)
            return stats

        # Decoder-only: x is dict{A_q, B_q, A_v, B_v}
        assert isinstance(x, dict) and self.dec_only, "decoder-only input requires dec_only=True"
        Aq, Bq, Av, Bv = x["A_q"], x["B_q"], x["A_v"], x["B_v"]
        if Aq.dim() == 3:  # single sample
            Aq, Bq, Av, Bv = Aq.unsqueeze(0), Bq.unsqueeze(0), Av.unsqueeze(0), Bv.unsqueeze(0)
        Bsz, L = Aq.shape[:2]

        fa_q = Aq.reshape(Bsz * L, self.r * self._d_in)
        fa_v = Av.reshape(Bsz * L, self.r * self._d_in)
        fb_q = Bq.reshape(Bsz * L, self._dq * self.r)
        fb_v = Bv.reshape(Bsz * L, self._dv * self.r)

        s_qa = self.net_A(fa_q).view(Bsz, L, 2 * self.latent_dim)
        s_qb = self.net_Bq(fb_q).view(Bsz, L, 2 * self.latent_dim)
        s_va = self.net_A(fa_v).view(Bsz, L, 2 * self.latent_dim)
        s_vb = self.net_Bv(fb_v).view(Bsz, L, 2 * self.latent_dim)

        # Interleave in [qA, qB, vA, vB] order → N = 4L
        stats = torch.stack([s_qa, s_qb, s_va, s_vb], dim=2).reshape(Bsz, 4 * L, 2 * self.latent_dim)
        return stats


# =========================
# MoE Head with load-balancing aux loss
# =========================
class MoEHead(nn.Module):
    """
    Input h: [B, N, in_dim] → output y: [B, N, out_dim].
    Returns y, aux, and router statistics. `aux` is a normalized load-balancing loss (≈0 when balanced).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_experts: int = 8,
        top_k: int = 1,
        hidden_dim: int = 1024,
        noisy_gating: bool = False,
        tau: float = 1.5,
    ):
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.E = num_experts
        self.K = top_k
        self.noisy = noisy_gating
        self.tau = tau

        self.router = nn.Linear(in_dim, self.E)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, out_dim),
                )
                for _ in range(self.E)
            ]
        )

    def set_tau(self, tau: float):
        self.tau = tau

    def _compute_aux(self, logits: torch.Tensor, topk_idx: torch.Tensor, gate: torch.Tensor):
        """
        Normalized aux:
          aux_raw = E * sum_e p_e * f_e
          aux = max(aux_raw - 1, 0)
        """
        B, N, E = logits.shape
        probs = logits.softmax(dim=-1)
        p_e = probs.mean(dim=(0, 1))  # [E]

        load = torch.zeros(E, device=logits.device)
        idx_flat = topk_idx.reshape(-1)
        w_flat = gate.reshape(-1)
        load = load.scatter_add(0, idx_flat, w_flat)
        f_e = load / (B * N)  # [E]

        aux_raw = E * (p_e * f_e.detach()).sum()
        aux = (aux_raw - 1.0).clamp_min(0.0)
        return aux, p_e.detach(), f_e.detach()

    def forward(self, h: torch.Tensor):
        B, N, D = h.shape
        logits = self.router(h) / self.tau
        if self.noisy:
            logits = logits + torch.randn_like(logits) / math.sqrt(D)

        topk_val, topk_idx = torch.topk(logits, k=self.K, dim=-1)
        gate = F.softmax(topk_val, dim=-1)

        h_flat = h.reshape(B * N, D)
        out_dim = self.experts[0][-1].out_features
        out_flat = torch.zeros(B * N, out_dim, device=h.device)

        for j in range(self.K):
            idx_j = topk_idx[..., j].reshape(-1)
            w_j = gate[..., j].reshape(-1, 1)
            for e, expert in enumerate(self.experts):
                mask = idx_j == e
                if not mask.any():
                    continue
                ye = expert(h_flat[mask])
                out_flat[mask] += ye * w_j[mask]

        y = out_flat.reshape(B, N, out_dim)
        aux, p_e, f_e = self._compute_aux(logits, topk_idx, gate)

        with torch.no_grad():
            router_entropy = (-F.softmax(logits, -1) * F.log_softmax(logits, -1)).sum(-1).mean()

        stats = {"p_e": p_e, "f_e": f_e, "entropy_router": router_entropy.item()}
        return y, aux, stats


# =========================
# Structure_Aware LoRA Decoder with MoE
# Supports T5-style and decoder-only shapes
# =========================
class StructureAware_LoRADecoder_MoE(nn.Module):
    def __init__(
        self,
        latent_dim,
        lora_rank,
        plm_hidden_size,  # for T5-style
        hidden_dim=128,
        num_layers=24,
        num_experts=8,
        top_k=2,
        noisy_gating=True,
        tau: float = 1.5,
        shared_expert_pool: bool = True,
        max_blocks: int = 288,
        # decoder-only dims
        dec_only: bool = False,
        dec_d_in: int = None,
        dec_dq_out: int = None,
        dec_dv_out: int = None,
        **kwargs,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.lora_rank = lora_rank
        self.plm_hidden_size = plm_hidden_size
        self.num_layers_t5 = num_layers
        self.max_blocks = max_blocks

        in_dim = latent_dim * 2
        out_dim = lora_rank * plm_hidden_size
        if shared_expert_pool:
            shared = MoEHead(in_dim, out_dim, num_experts, top_k, hidden_dim, noisy_gating, tau)
            self.encoder_head = self.decoder_attn_head = self.cross_attn_head = shared
        else:
            self.encoder_head = MoEHead(in_dim, out_dim, num_experts, top_k, hidden_dim, noisy_gating, tau)
            self.decoder_attn_head = MoEHead(in_dim, out_dim, num_experts, top_k, hidden_dim, noisy_gating, tau)
            self.cross_attn_head = MoEHead(in_dim, out_dim, num_experts, top_k, hidden_dim, noisy_gating, tau)

        self.blocks_per_layer_t5 = 3
        self.weights_per_block = 4
        self.total_blocks_t5 = self.num_layers_t5 * self.blocks_per_layer_t5 * self.weights_per_block
        self.blocks_per_type = self.num_layers_t5 * self.weights_per_block

        # decoder-only heads
        self.dec_only = dec_only
        if dec_only:
            assert all(v is not None for v in [dec_d_in, dec_dq_out, dec_dv_out]), "decoder-only requires dims"
            self.dec_d_in, self.dec_dq_out, self.dec_dv_out = dec_d_in, dec_dq_out, dec_dv_out
            self.dec_head_qa = MoEHead(in_dim, lora_rank * dec_d_in, num_experts, top_k, hidden_dim, noisy_gating, tau)
            self.dec_head_qb = MoEHead(in_dim, dec_dq_out * lora_rank, num_experts, top_k, hidden_dim, noisy_gating, tau)
            self.dec_head_va = MoEHead(in_dim, lora_rank * dec_d_in, num_experts, top_k, hidden_dim, noisy_gating, tau)
            self.dec_head_vb = MoEHead(in_dim, dec_dv_out * lora_rank, num_experts, top_k, hidden_dim, noisy_gating, tau)

        self._last_aux = None
        self.latest_router_stats = {"encoder": None, "decoder_attn": None, "cross_attn": None}
        self.module_id_embedding = nn.Embedding(self.max_blocks, latent_dim)

    def set_tau(self, tau: float):
        for m in [
            self.encoder_head,
            self.decoder_attn_head,
            self.cross_attn_head,
            getattr(self, "dec_head_qa", None),
            getattr(self, "dec_head_qb", None),
            getattr(self, "dec_head_va", None),
            getattr(self, "dec_head_vb", None),
        ]:
            if m is not None and hasattr(m, "set_tau"):
                m.set_tau(tau)

    def moe_aux_loss(self):
        return (
            torch.tensor(0.0, device=self.module_id_embedding.weight.device)
            if self._last_aux is None
            else self._last_aux
        )

    def forward(self, z: torch.Tensor):
        """
        z: [B, N, latent_dim]
           - N == 288 → T5-style path
           - N == 4*L → decoder-only path
        """
        B, N, _ = z.shape
        ids = torch.arange(N, device=z.device).unsqueeze(0).expand(B, -1)
        z_in = torch.cat([z, self.module_id_embedding(ids)], dim=-1)  # [B, N, 2*latent]

        if N == self.total_blocks_t5:
            # T5-style: split into encoder/self-attn/cross-attn groups
            s0, e0 = 0, self.blocks_per_type
            s1, e1 = e0, e0 + self.blocks_per_type
            s2, e2 = e1, e1 + self.blocks_per_type
            z_enc, z_dsa, z_dca = z_in[:, s0:e0], z_in[:, s1:e1], z_in[:, s2:e2]

            y_enc, aux_enc, stats_enc = self.encoder_head(z_enc)
            y_dsa, aux_dsa, stats_dsa = self.decoder_attn_head(z_dsa)
            y_dca, aux_dca, stats_dca = self.cross_attn_head(z_dca)

            self._last_aux = aux_enc + aux_dsa + aux_dca
            self.latest_router_stats = {"encoder": stats_enc, "decoder_attn": stats_dsa, "cross_attn": stats_dca}

            out = (
                torch.cat([y_enc, y_dsa, y_dca], dim=1)
                .view(B, 288, self.lora_rank, self.plm_hidden_size)
                .view(B, 3, self.num_layers_t5, 4, self.lora_rank, self.plm_hidden_size)
            )

            enc, dec_self, dec_cross = out[:, 0], out[:, 1], out[:, 2]

            def pick(t, slot, T=False):
                w = t[:, :, slot]
                return w.permute(0, 1, 3, 2).contiguous() if T else w

            return {
                "encoder": {
                    "lora_qa": pick(enc, 0),
                    "lora_qb": pick(enc, 1, T=True),
                    "lora_va": pick(enc, 2),
                    "lora_vb": pick(enc, 3, T=True),
                },
                "decoder": {
                    "decoder_attn": {
                        "lora_qa": pick(dec_self, 0),
                        "lora_qb": pick(dec_self, 1, T=True),
                        "lora_va": pick(dec_self, 2),
                        "lora_vb": pick(dec_self, 3, T=True),
                    },
                    "cross_attn": {
                        "lora_qa": pick(dec_cross, 0),
                        "lora_qb": pick(dec_cross, 1, T=True),
                        "lora_va": pick(dec_cross, 2),
                        "lora_vb": pick(dec_cross, 3, T=True),
                    },
                },
            }

        # decoder-only: N = 4L
        assert self.dec_only, "Got decoder-only input but `dec_only=True` is not set."
        L = N // 4
        z_qa = z_in[:, 0::4]
        z_qb = z_in[:, 1::4]
        z_va = z_in[:, 2::4]
        z_vb = z_in[:, 3::4]

        y_qa, aux_qa, st_qa = self.dec_head_qa(z_qa)
        y_qb, aux_qb, st_qb = self.dec_head_qb(z_qb)
        y_va, aux_va, st_va = self.dec_head_va(z_va)
        y_vb, aux_vb, st_vb = self.dec_head_vb(z_vb)

        self._last_aux = aux_qa + aux_qb + aux_va + aux_vb
        self.latest_router_stats = {"encoder": None, "decoder_attn": st_qa, "cross_attn": st_qb}

        qa = y_qa.view(B, L, self.lora_rank, self.dec_d_in)        # [B, L, r, in]
        qb = y_qb.view(B, L, self.dec_dq_out, self.lora_rank)      # [B, L, out_q, r]
        va = y_va.view(B, L, self.lora_rank, self.dec_d_in)        # [B, L, r, in]
        vb = y_vb.view(B, L, self.dec_dv_out, self.lora_rank)      # [B, L, out_v, r]

        return {
            "decoder_only": {
                "lora_qa": qa,
                "lora_qb": qb,
                "lora_va": va,
                "lora_vb": vb,
            }
        }
