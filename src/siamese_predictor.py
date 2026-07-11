"""Siamese variant-effect scorer — direct ref/alt skew prediction (Enhancement #1).

THEORY (plain English — docs/07_enhancement_design.md #1):
The `ActivityPredictor` regresses a single sequence's activity, then `score_variant` takes
the DIFFERENCE activity(alt) - activity(ref). That indirect Δ is the suspected cause of the
0.70 -> 0.15 gap between element-activity accuracy and variant-effect accuracy: the model was
never optimised on the difference, only on the two endpoints.

This module scores the difference DIRECTLY. A shared-weight ("siamese") backbone encodes both
alleles; a small head reads the *difference embedding* (h_alt - h_ref) and predicts the measured
allelic skew (logFC). Training happens in `train/finetune_siamese.py`; this file is the inference
counterpart and the single source of truth for the head architecture (imported by the trainer so
train- and score-time graphs are byte-identical — the same discipline ActivityPredictor follows).

ADDITIVE: this is a NEW module. `src/predictor.py` is untouched; the activity path and every
existing calibrated number stay exactly as they were. A siamese checkpoint is opt-in.

CHECKPOINT FORMAT (produced by finetune_siamese.py) — same container as ActivityPredictor's:
    OUT/model.pt        {"backbone": state_dict, "head": state_dict, "meta": {objective:"siamese", ...}}
    OUT/provenance.json
"""
from __future__ import annotations

import os

# reuse the exact tokenizer + pooling the activity model uses, so encoding is identical
from src.predictor import DEFAULT_CHECKPOINT, build_tokenizer, pool_hidden


def build_siamese_head(hidden: int):
    """The variant-effect head: MLP over the (h_alt - h_ref) difference embedding -> scalar Δ.

    Defined here (not inline in the trainer) so training and inference construct byte-identical
    modules. Keep small — the signal is thin and the training set is ~10k pairs.
    """
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(hidden, hidden // 2),
        nn.GELU(),
        nn.Linear(hidden // 2, 1),
    )


class SiameseSkewRegressor:
    """Container building a shared backbone + difference head. Used by BOTH the trainer
    (as an nn.Module via `.module`) and the scorer. Kept framework-light so importing this
    file never forces a torch import until a model is actually built."""

    def __init__(self, checkpoint: str, pool: str = "mean"):
        self.checkpoint = checkpoint
        self.pool = pool
        self.module = None
        self.hidden = None

    def build(self):
        import torch.nn as nn
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained(self.checkpoint, trust_remote_code=True)
        hidden = _infer_hidden(backbone)
        head = build_siamese_head(hidden)
        pool_mode = self.pool

        class _Siamese(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.head = head

            def _encode(self, input_ids, attention_mask):
                out = self.backbone(input_ids)
                last = getattr(out, "last_hidden_state", None)
                if last is None:
                    last = out[0]
                return pool_hidden(last, attention_mask, pool_mode)

            def forward(self, ref_ids, ref_mask, alt_ids, alt_mask):
                h_ref = self._encode(ref_ids, ref_mask)
                h_alt = self._encode(alt_ids, alt_mask)
                return self.head(h_alt - h_ref).squeeze(-1)

        self.module = _Siamese()
        self.hidden = hidden
        return self.module


def _infer_hidden(backbone) -> int:
    cfg = getattr(backbone, "config", None)
    for attr in ("hidden_size", "d_model", "n_embd"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int):
            return v
    raise RuntimeError("could not infer backbone hidden size from config")


class SiameseVariantScorer:
    """Load a fine-tuned siamese checkpoint and score (seq_ref, seq_alt) -> predicted Δ skew.

    Mirrors ActivityPredictor.load: always rebuilds the base backbone from the checkpoint id
    saved in model.pt's meta, then overlays the fine-tuned state_dicts.
    """

    def __init__(self, weights: str, checkpoint: str = DEFAULT_CHECKPOINT,
                 device: str | None = None, pool: str = "mean", batch_size: int = 64):
        self.weights = weights
        self.checkpoint = checkpoint
        self.device = device
        self.pool = pool
        self.batch_size = batch_size
        self._module = None
        self._tokenizer = None
        self.backbone_type = None
        self.meta = {}
        # duck-type parity with ActivityPredictor so interpret_variant / app.py accept a siamese
        # scorer wherever they accept a predictor (both expose score_variant + these attrs).
        self.is_finetuned = False
        self.seq_len = 270

    def load(self) -> "SiameseVariantScorer":
        if self._module is not None:
            return self
        import torch

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_path = os.path.join(self.weights, "model.pt")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"no siamese checkpoint at {model_path}")
        blob = torch.load(model_path, map_location="cpu")
        self.meta = blob.get("meta", {}) if isinstance(blob, dict) else {}
        if self.meta.get("objective") != "siamese":
            raise ValueError(f"{model_path} is not a siamese checkpoint "
                             f"(objective={self.meta.get('objective')!r}); use ActivityPredictor")
        saved_ckpt = self.meta.get("checkpoint")
        if saved_ckpt:
            self.checkpoint = saved_ckpt
        self.pool = self.meta.get("pool", self.pool)

        builder = SiameseSkewRegressor(self.checkpoint, self.pool)
        self._module = builder.build()
        self._module.backbone.load_state_dict(blob["backbone"])
        self._module.head.load_state_dict(blob["head"])
        self.backbone_type = getattr(getattr(self._module.backbone, "config", None),
                                     "model_type", "hyenadna")
        self._tokenizer = build_tokenizer(self.backbone_type)
        self._module.to(self.device).eval()
        self.is_finetuned = True                 # a siamese checkpoint is fine-tuned by construction
        self.seq_len = int(self.meta.get("seq_len", self.seq_len))
        return self

    def _require(self):
        if self._module is None:
            raise RuntimeError("call .load() before scoring")

    def score_variants_batch(self, seq_refs: list[str], seq_alts: list[str]) -> list[float]:
        import torch
        self._require()
        out: list[float] = []
        for i in range(0, len(seq_refs), self.batch_size):
            refs = seq_refs[i:i + self.batch_size]
            alts = seq_alts[i:i + self.batch_size]
            ref_ids, ref_mask = self._tokenizer.encode(refs)
            alt_ids, alt_mask = self._tokenizer.encode(alts)
            ref_ids, ref_mask = ref_ids.to(self.device), ref_mask.to(self.device)
            alt_ids, alt_mask = alt_ids.to(self.device), alt_mask.to(self.device)
            with torch.no_grad():
                d = self._module(ref_ids, ref_mask, alt_ids, alt_mask)
            out.extend(d.float().cpu().reshape(-1).tolist())
        return out

    def score_variant(self, seq_ref: str, seq_alt: str) -> float:
        return self.score_variants_batch([seq_ref], [seq_alt])[0]

    def provenance(self) -> dict:
        return {"kind": "siamese", "checkpoint": self.checkpoint,
                "backbone": self.backbone_type, **self.meta}
