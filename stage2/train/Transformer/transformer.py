import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from Transformer.utils import generate_original_PE, generate_regular_PE

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class LoRATransformer(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_model: int,
        d_output: int,
        d_cond: int,
        d_aux: int,
        N: int,
        layernum: int = 0,
        dropout: float = 0.1,
        pe: str = None,
        pe_period: int = None,
        learned_sinusoidal_cond: bool = False,
        random_fourier_features: bool = False,
        learned_sinusoidal_dim: int = 16,
    ):
        super().__init__()

        self.channels = d_input
        self.self_condition = False 

        self._d_model = d_model
        self.layernum = layernum
        self.cond_dim = d_cond
        self.aux_dim = d_aux

        self.layers_encoding = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=dropout, batch_first=True)
            for _ in range(N)
        ])
        self._embedding = nn.Linear(d_input, d_model)   # project tokens (C) -> hidden
        self._linear    = nn.Linear(d_model, d_output)  # project hidden -> tokens (C)

        pe_functions = {
            'original': generate_original_PE,
            'regular':  generate_regular_PE,
        }
        if pe in pe_functions:
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
            self._pe_period = None
        else:
            raise NameError(f'Unknown PE "{pe}". Use one of {", ".join(pe_functions.keys())} or None.')

        # diffusion timestep embedding φ(t)
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(d_model)
            fourier_dim = d_model

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        self.cond_proj = nn.Linear(self.cond_dim, d_model)
        self.aux_proj  = nn.Linear(self.aux_dim,  d_model)

        self.query_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )

    def forward(
        self,
        x: torch.Tensor,                 # [B, C, L]
        t: torch.Tensor,                 # [B] diffusion step indices
        cond_emb: torch.Tensor,          # [B, d_cond]
        aux_emb: torch.Tensor,           # [B, d_aux]
        x_self_cond: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Returns:
            Tensor of shape [B, C, L]
        """
        B, C, L = x.shape

        x_hidden = self._embedding(x.permute(0, 2, 1))

        cond_h = self.cond_proj(cond_emb).unsqueeze(2)
        aux_h  = self.aux_proj(aux_emb).unsqueeze(2)
        kv_bank = torch.cat((cond_h, aux_h), dim=2)  

        q = self.query_mlp(x_hidden)                 # [B, L, d_model]
        logits = torch.bmm(q, kv_bank)               # [B, L, 2]
        weights = F.softmax(logits, dim=2)           # [B, L, 2]

        blended = torch.bmm(weights, kv_bank.transpose(1, 2))

        t_h = self.time_mlp(t).unsqueeze(1)
        t_h = t_h.expand(-1, self.layernum if self.layernum > 0 else L, -1)

        encoding = x_hidden + t_h

        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            pos_enc = self._generate_PE(self.layernum if self.layernum > 0 else L, self._d_model, **pe_params)
            pos_enc = pos_enc.to(encoding.device)     # [L, d_model]
            encoding = encoding + pos_enc.unsqueeze(0)  # broadcast to [B, L, d_model]

        for layer in self.layers_encoding:
            encoding = encoding + blended
            encoding = layer(encoding)  # batch_first=True

        out = self._linear(encoding)     # [B, L, C]
        return out.permute(0, 2, 1)      # [B, C, L]
