import os
import random
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from pytorch_lightning.utilities.rank_zero import rank_zero_only

# Optional: safetensors loader
try:
    from safetensors.torch import load_file as _safe_load
except Exception:
    _safe_load = None


def _load_state_any(path: str):
    if path.endswith(".safetensors"):
        assert _safe_load is not None, "Please `pip install safetensors` to read .safetensors files."
        return dict(_safe_load(path))
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        return sd["state_dict"]
    if isinstance(sd, dict):
        return sd
    raise ValueError(f"Unrecognized state_dict format in: {path}")


def _state_to_tensor_and_schema(sd: dict):
    """
    Normalize LoRA weights into a tensor or a structured dict and return (weight, meta).

    - T5 schema:
        Returns weight: Tensor [N=288, d, r]
        meta: {"schema": "t5", "num_modules": N}
    - Decoder-only schema:
        Returns weight: dict{"A_q":[L,r,in], "B_q":[L,out_q,r], "A_v":[L,r,in], "B_v":[L,out_v,r]}
        meta: {
            "schema": "dec_only", "num_layers": L, "r": r,
            "d_in": in, "d_out_q": out_q, "d_out_v": out_v, "num_modules": 4*L
        }
    """
    keys = list(sd.keys())
    is_t5 = any("encoder.block" in k or "EncDecAttention" in k for k in keys)
    is_dec = any(".model.layers." in k and ".self_attn." in k for k in keys)

    if is_t5:
        # Order: enc -> dec self-attn -> dec cross-attn; layers 0..23; proj q,v; parts A,B
        def _key(prefix, proj, ab):
            return f"{prefix}.{proj}.lora_{ab}.weight"

        segments = [
            ("base_model.model.encoder.block.{i}.layer.0.SelfAttention", 24),
            ("base_model.model.decoder.block.{i}.layer.0.SelfAttention", 24),
            ("base_model.model.decoder.block.{i}.layer.1.EncDecAttention", 24),
        ]

        mats = []
        d = r = None
        for pref, L in segments:
            for i in range(L):
                for proj in ("q", "v"):
                    for ab in ("A", "B"):
                        k = _key(pref.format(i=i), proj, ab)
                        t = sd[k]
                        if t.dim() != 2:
                            raise ValueError("Expected 2D LoRA weight for T5.")
                        a, b = t.shape
                        t = t if a >= b else t.t()
                        if d is None:
                            d, r = t.shape
                        mats.append(t)

        weight = torch.stack(mats, dim=0).contiguous()  # [288, d, r]
        return weight, dict(schema="t5", num_modules=weight.size(0))

    if is_dec:
        import re
        pat = re.compile(
            r"^base_model\.model\.model\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)\.lora_(A|B)\.weight$"
        )

        layer_ids = set()
        for k in keys:
            m = pat.match(k)
            if m:
                layer_ids.add(int(m.group(1)))
        if not layer_ids:
            raise ValueError("Decoder-only schema detection failed: no matching keys found.")
        L = max(layer_ids) + 1

        A_q_list, B_q_list, A_v_list, B_v_list = [], [], [], []
        r_seen = in_seen = outq_seen = outv_seen = None

        def to_Arin(t):  # LoRA A: [r, in]
            a, b = t.shape
            r_, in_ = (a, b) if a < b else (b, a)
            return (t if a < b else t.t()), r_, in_

        def to_Boutr(t):  # LoRA B: [out, r]
            a, b = t.shape
            out_, r_ = (a, b) if a > b else (b, a)
            return (t if a > b else t.t()), out_, r_

        for i in range(L):
            Aq, rA, inA = to_Arin(sd[f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight"])
            Bq, outQ, rB = to_Boutr(sd[f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_B.weight"])
            Av, rA2, inA2 = to_Arin(sd[f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_A.weight"])
            Bv, outV, rB2 = to_Boutr(sd[f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_B.weight"])

            if r_seen is None:
                r_seen = rA
            assert r_seen == rA == rB == rA2 == rB2, f"Inconsistent LoRA rank across layers: {r_seen} vs {rA,rB,rA2,rB2}"
            if in_seen is None:
                in_seen = inA
            assert in_seen == inA == inA2, f"Inconsistent input dim across layers: {in_seen} vs {inA,inA2}"
            if outq_seen is None:
                outq_seen = outQ
            assert outq_seen == outQ, f"Inconsistent q out dim across layers: {outq_seen} vs {outQ}"
            if outv_seen is None:
                outv_seen = outV
            assert outv_seen == outV, f"Inconsistent v out dim across layers: {outv_seen} vs {outV}"

            A_q_list.append(Aq)  # [L_i: r, in]
            B_q_list.append(Bq)  # [L_i: out_q, r]
            A_v_list.append(Av)  # [L_i: r, in]
            B_v_list.append(Bv)  # [L_i: out_v, r]

        A_q = torch.stack(A_q_list, dim=0).contiguous()
        B_q = torch.stack(B_q_list, dim=0).contiguous()
        A_v = torch.stack(A_v_list, dim=0).contiguous()
        B_v = torch.stack(B_v_list, dim=0).contiguous()

        weight = dict(A_q=A_q, B_q=B_q, A_v=A_v, B_v=B_v)
        meta = dict(
            schema="dec_only",
            num_layers=L,
            r=r_seen,
            d_in=in_seen,
            d_out_q=outq_seen,
            d_out_v=outv_seen,
            num_modules=4 * L,
        )
        return weight, meta

    raise RuntimeError("Failed to identify schema: not T5 nor decoder-only.")


class LoRAMultiDataset(Dataset):

    def __init__(
        self,
        root_dir,
        split: str = "train",
        val_ratio: float = 0.1,
        max_blocks: int = 288,
        seed: int = 42,
        record_split_path: str | None = None,
        split_file: str | None = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.val_ratio = val_ratio
        self.max_blocks = max_blocks
        self.seed = seed
        self.record_split_path = record_split_path
        self.split_file = split_file

        all_dirs = sorted([
            d for d in os.listdir(root_dir)
            if (
                os.path.exists(os.path.join(root_dir, d, "adapter_model.bin")) or
                os.path.exists(os.path.join(root_dir, d, "adapter_model.safetensors"))
            )
        ])

        if split_file is not None:
            import json
            with open(split_file, "r") as f:
                split_dict = json.load(f)
            assert split in split_dict, f"Split '{split}' not found in split file."

            def _pick_file(t):
                p_bin = os.path.join(root_dir, t, "adapter_model.bin")
                p_st = os.path.join(root_dir, t, "adapter_model.safetensors")
                return p_bin if os.path.exists(p_bin) else p_st

            self.files = [
                _pick_file(t) for t in split_dict[split]
                if (
                    os.path.exists(os.path.join(root_dir, t, "adapter_model.bin")) or
                    os.path.exists(os.path.join(root_dir, t, "adapter_model.safetensors"))
                )
            ]
            self.train_files = split_dict.get("train", [])
            self.val_files = split_dict.get("val", [])
        else:
            self.all_files = []
            for d in all_dirs:
                p_bin = os.path.join(root_dir, d, "adapter_model.bin")
                p_st = os.path.join(root_dir, d, "adapter_model.safetensors")
                if os.path.exists(p_bin):
                    self.all_files.append(p_bin)
                elif os.path.exists(p_st):
                    self.all_files.append(p_st)
            self.train_files, self.val_files = self._split_files_by_task()
            self.files = self.train_files if split == "train" else self.val_files

    @rank_zero_only
    def _save_split_record(self, train, val):
        if self.record_split_path:
            os.makedirs(os.path.dirname(self.record_split_path), exist_ok=True)
            with open(self.record_split_path, "w") as f:
                f.write("# Train files\n")
                for p in train:
                    f.write(f"{p}\n")
                f.write("\n# Val files\n")
                for p in val:
                    f.write(f"{p}\n")

    def _split_files_by_task(self):
        """
        Split by coarse task groups to avoid leaking similar tasks across splits.
        """
        random.seed(self.seed)

        keyword2group = {
            "sentiment": ["amazon_polarity", "yelp_polarity", "app_reviews"],
            "nli": ["glue_", "super_glue_", "anli"],
            "qa_reading": [
                "race", "quoref", "ropes", "duorc", "dream", "quartz", "quail", "quac",
                "hotpotqa", "wiqa", "wiki_hop", "web_questions",
                "wiki_qa", "sciq", "social_i_qa",
            ],
            "logical_reasoning": ["quarel"],
            "extraction": ["dbpedia", "wiki_bio", "qasc", "fix_punct", "word_segment"],
            "generation": ["gem_", "newsroom", "para_crawl", "true_case"],
            "adversarial": ["adversarial_qa_"],
        }

        grouped = defaultdict(list)
        for path in self.all_files:
            task_dir = os.path.basename(os.path.dirname(path))
            assigned = False
            for group, keywords in keyword2group.items():
                if any(k in task_dir for k in keywords):
                    grouped[group].append(path)
                    assigned = True
                    break
            if not assigned:
                grouped["other"].append(path)

        train, val = [], []
        for _, paths in grouped.items():
            random.shuffle(paths)
            split_idx = int(len(paths) * self.val_ratio)
            val.extend(paths[:split_idx])
            train.extend(paths[split_idx:])

        self._save_split_record(train, val)
        return train, val

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Returns a dict with:
            - weight: Tensor or dict of Tensors
            - task_name: str
            - path: str
            - schema: "t5" | "dec_only"
            - num_modules: int
          Optional (decoder-only):
            - num_layers, r, d_in, d_out_q, d_out_v
        """
        path = self.files[idx]
        task_name = os.path.basename(os.path.dirname(path))
        sd = _load_state_any(path)
        weight, meta = _state_to_tensor_and_schema(sd)

        sample = {
            "weight": weight,
            "task_name": task_name,
            "path": path,
            "schema": meta["schema"],
            "num_modules": meta["num_modules"],
        }
        if meta["schema"] == "dec_only":
            sample.update(
                num_layers=meta["num_layers"],
                r=meta["r"],
                d_in=meta["d_in"],
                d_out_q=meta["d_out_q"],
                d_out_v=meta["d_out_v"],
            )
        return sample
