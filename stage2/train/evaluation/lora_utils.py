# -*- coding: utf-8 -*-
import re
import torch


def apply_weights_to_model(model, lora_weights):
    """
    Dispatch to flat dict or structured dict injection.
    """
    if isinstance(lora_weights, dict) and "encoder" in lora_weights:
        return _apply_structured_weights(model, lora_weights)
    else:
        return _apply_flat_state_dict(model, lora_weights)


def _inject_delta(layer_module, delta):
    """
    Add delta (ΔW) to layer_module.weight with automatic dtype/device alignment.
    Tries a transpose if shapes do not match.
    """
    W = layer_module.weight
    delta = delta.to(W.device, dtype=W.dtype)
    if delta.shape != W.shape:
        if delta.T.shape == W.shape:
            delta = delta.T
        else:
            raise RuntimeError(f"[LoRA] shape mismatch: delta {tuple(delta.shape)} vs weight {tuple(W.shape)}")
    W.data.add_(delta)


def _A_B_to_delta(B, A):
    """
    By convention: A: [r, d_in], B: [d_out, r]  => ΔW = B @ A  (shape [d_out, d_in])
    Some exports may flip dims; try a few common transpositions for robustness.
    """
    try:
        return B @ A
    except Exception:
        candidates = []
        for Bb in (B, B.T):
            for Aa in (A, A.T):
                try:
                    candidates.append(Bb @ Aa)
                except Exception:
                    pass
        if not candidates:
            raise
        return candidates[0]


_PAT = re.compile(
    r"^base_model\.model\."
    r"(?P<encdec>encoder|decoder)\.block\.(?P<layer>\d+)\.layer\.(?P<layeridx>\d+)\."
    r"(?P<attn>SelfAttention|EncDecAttention)\.(?P<proj>[qv])\.lora_(?P<side>[AB])\.weight$"
)


def _apply_flat_state_dict(model, state_dict):
    store = {}  # key: (encdec, layer, layeridx, attn, proj) -> {"A": tensor, "B": tensor}
    for k, v in state_dict.items():
        if not isinstance(k, str):
            continue
        m = _PAT.match(k)
        if not m:
            continue
        gd = m.groupdict()
        key = (gd["encdec"], int(gd["layer"]), int(gd["layeridx"]), gd["attn"], gd["proj"])
        side = gd["side"]  # 'A' or 'B'
        if key not in store:
            store[key] = {}
        store[key][side] = v

    device = next(model.parameters()).device
    for (encdec, layer, layeridx, attn, proj), sides in store.items():
        if "A" not in sides or "B" not in sides:
            # skip incomplete pairs
            continue
        A = sides["A"].to(device)
        B = sides["B"].to(device)
        delta = _A_B_to_delta(B, A)

        if encdec == "encoder":
            blk = model.encoder.block[layer]
            if attn == "SelfAttention":
                attn_mod = blk.layer[layeridx].SelfAttention
            else:
                continue
        else:
            blk = model.decoder.block[layer]
            if attn == "SelfAttention":
                attn_mod = blk.layer[layeridx].SelfAttention
            else:  # EncDecAttention
                attn_mod = blk.layer[layeridx].EncDecAttention

        tgt = attn_mod.q if proj == "q" else attn_mod.v
        _inject_delta(tgt, delta)

    return model


def _apply_structured_weights(model, lora_weights):
    """
    Inject deltas from the legacy structured format used in this repo.
    """
    device = next(model.parameters()).device

    def delta_from_struct(A, B):
        # A: [r, d_in], B: [d_out, r]
        return (B @ A)

    # encoder self-attention
    for i in range(24):
        blk = model.encoder.block[i].layer[0].SelfAttention
        _inject_delta(blk.q, delta_from_struct(
            lora_weights["encoder"]["lora_qa"][0, i].to(device),
            lora_weights["encoder"]["lora_qb"][0, i].to(device),
        ))
        _inject_delta(blk.v, delta_from_struct(
            lora_weights["encoder"]["lora_va"][0, i].to(device),
            lora_weights["encoder"]["lora_vb"][0, i].to(device),
        ))

    # decoder self-attention
    for i in range(24):
        blk = model.decoder.block[i].layer[0].SelfAttention
        _inject_delta(blk.q, delta_from_struct(
            lora_weights["decoder"]["decoder_attn"]["lora_qa"][0, i].to(device),
            lora_weights["decoder"]["decoder_attn"]["lora_qb"][0, i].to(device),
        ))
        _inject_delta(blk.v, delta_from_struct(
            lora_weights["decoder"]["decoder_attn"]["lora_va"][0, i].to(device),
            lora_weights["decoder"]["decoder_attn"]["lora_vb"][0, i].to(device),
        ))

    # decoder cross-attention
    for i in range(24):
        blk = model.decoder.block[i].layer[1].EncDecAttention
        _inject_delta(blk.q, delta_from_struct(
            lora_weights["decoder"]["cross_attn"]["lora_qa"][0, i].to(device),
            lora_weights["decoder"]["cross_attn"]["lora_qb"][0, i].to(device),
        ))
        _inject_delta(blk.v, delta_from_struct(
            lora_weights["decoder"]["cross_attn"]["lora_va"][0, i].to(device),
            lora_weights["decoder"]["cross_attn"]["lora_vb"][0, i].to(device),
        ))

    return model
