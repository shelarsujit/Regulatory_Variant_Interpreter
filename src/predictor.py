"""Load the fine-tuned HyenaDNA checkpoint and score variants by saturation mutagenesis.

THEORY (plain English — see docs/00_overview_for_non_biologists.md §3.5):
This is the one module that produces the model's raw prediction; everything else in the
system exists to make that prediction trustworthy.

  * We fine-tuned a DNA language model (HyenaDNA) to map a ~200 bp sequence -> regulatory
    activity (see train/finetune_hyenadna.py). HyenaDNA reads DNA one letter at a time
    (single-nucleotide resolution), which is essential here: our whole method changes ONE
    letter and needs that change to be crisply visible.
  * "In-silico saturation mutagenesis" = ask the model, for a sequence, what happens to
    predicted activity if we change each position to each other base. For a real variant
    we only need one entry of that map: activity(alt) - activity(ref) = Δactivity.
  * Two separately fine-tuned models exist (decision D4): `context="primary"` is the call;
    `context="organoid"` is an INDEPENDENT second opinion consumed by the trust layer.

ARCHITECTURE (matches train/finetune_hyenadna.py):
    backbone = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)   # HyenaDNA
    head     = Linear(hidden, 1)                                               # -> activity
    activity(seq) = head( pool( backbone(tokens).last_hidden_state ) )

A fine-tuned checkpoint is a directory written by the trainer:
    weights/<context>/model.pt        # {"backbone": state_dict, "head": state_dict, "meta": {...}}
    weights/<context>/provenance.json # checkpoint id, git commit, hyperparams, best val metrics
`load()` works WITHOUT that directory too — it loads the base backbone and a randomly
initialized head, so the whole pipeline runs end-to-end before Phase-2 training finishes.
In that pre-trained state the Δactivity numbers are NOT meaningful; `is_finetuned` is False
and the trust layer should treat the primary model as uncalibrated.

IMPLEMENTATION NOTE: torch/transformers are imported lazily inside methods so that
`import predictor` stays cheap and download-free (tests import this module but never
call load()).
"""
from __future__ import annotations

import hashlib
import os
from typing import Optional

# Charter pins the checkpoint `LongSafari/hyenadna-tiny-1k-seqlen`; that base repo ships no
# HF `auto_map`/`model_type`, so `AutoModel` can't load it. `-hf` is LongSafari's official
# HF-loadable mirror of the SAME weights (adds configuration_hyena.py / modeling_hyena.py).
DEFAULT_CHECKPOINT = "LongSafari/hyenadna-tiny-1k-seqlen-hf"
_BASES = ("A", "C", "G", "T")
_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _revcomp(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]

# HyenaDNA's character-level vocabulary (from its standalone CharacterTokenizer). We build
# input_ids from this map directly instead of AutoTokenizer, because HyenaDNA ships no fast
# tokenizer and transformers' slow->fast conversion needs sentencepiece it doesn't have.
# Defining it here makes tokenization identical between this scorer and the trainer.
_HYENA_VOCAB = {"[CLS]": 0, "[SEP]": 1, "[BOS]": 2, "[MASK]": 3, "[PAD]": 4,
                "[RESERVED]": 5, "[UNK]": 6, "A": 7, "C": 8, "G": 9, "T": 10, "N": 11}
_PAD_ID = _HYENA_VOCAB["[PAD]"]
_UNK_ID = _HYENA_VOCAB["[UNK]"]


def pool_hidden(last_hidden, attention_mask=None, mode: str = "mean"):
    """Reduce (B, L, hidden) -> (B, hidden). SHARED by the scorer and the trainer so their
    sequence->activity function is byte-identical (train/inference parity). mode: "mean"|"last".
    """
    import torch
    if mode == "last":
        if attention_mask is not None:
            idx = (attention_mask.sum(dim=1) - 1).clamp(min=0).long()   # last real token
            return last_hidden[torch.arange(last_hidden.size(0)), idx]
        return last_hidden[:, -1]
    # mean pool over real (unpadded) tokens
    if attention_mask is not None:
        m = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        return (last_hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)
    return last_hidden.mean(dim=1)


class HyenaDNACharTokenizer:
    """Minimal char-level tokenizer matching HyenaDNA's vocab (A=7,C=8,G=9,T=10,N=11).

    Returns padded `input_ids` and an `attention_mask` (1=real token, 0=pad). No special
    BOS/SEP tokens are added — the regression head pools over real tokens only, so they add
    nothing and only complicate ref/alt alignment.
    """

    vocab = dict(_HYENA_VOCAB)
    pad_id = _PAD_ID

    def encode(self, sequences: list[str]):
        import torch
        ids = [[self.vocab.get(c, _UNK_ID) for c in s.upper()] for s in sequences]
        n = max(len(x) for x in ids)
        input_ids = torch.full((len(ids), n), _PAD_ID, dtype=torch.long)
        attn = torch.zeros((len(ids), n), dtype=torch.long)
        for i, row in enumerate(ids):
            input_ids[i, :len(row)] = torch.tensor(row, dtype=torch.long)
            attn[i, :len(row)] = 1
        return input_ids, attn


class CaduceusCharTokenizer(HyenaDNACharTokenizer):
    """Char tokenizer for the Caduceus backbone (decision D2 upgrade path / D16).

    Caduceus (kuleshov-group) descends from the SAME `CharacterTokenizer` as HyenaDNA — its
    vocabulary is byte-identical (verified against caduceus/tokenization_caduceus.py:
    [CLS]0 [SEP]1 [BOS]2 [MASK]3 [PAD]4 [RESERVED]5 [UNK]6 A7 C8 G9 T10 N11), so the same encoder
    works. Kept as a distinct class so the backbone type is explicit in provenance and so any
    future SEP/reverse-complement handling can diverge here without touching the HyenaDNA path.
    """
    pass


# Backbone registry: model_type -> default checkpoint + char tokenizer. New DNA-LM backbones plug
# in here; the rest of predictor.py / finetune is backbone-agnostic because we tokenize ourselves
# and persist raw state_dicts (not save_pretrained), so no per-backbone serialization quirks leak.
# NOTE (Caduceus): needs `mamba-ssm` + `causal-conv1d`, which require CUDA to build — it loads and
# trains on a GPU box only, NOT on CPU/Windows. The HyenaDNA path stays fully CPU-testable.
# The `-ps` default below is the RC-EQUIVARIANT variant. CAVEAT (docs/03_results.md §6): test-time
# RC averaging HURT variant-effect on this MPRA (activity is orientation-specific), which argues
# against forcing `f(seq)=f(rc(seq))` — consider a bidirectional non-RC-equivariant / `-ph` variant
# instead, and test both on GPU. Also verify mean-pool + Linear head behaves for `-ps` (its hidden
# channels are RC-structured), or use CaduceusForSequenceClassification's pooling.
BACKBONES = {
    "hyenadna": {"default_checkpoint": DEFAULT_CHECKPOINT,
                 "tokenizer": HyenaDNACharTokenizer},
    "caduceus": {"default_checkpoint":
                 "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
                 "tokenizer": CaduceusCharTokenizer},
}


def build_tokenizer(model_type: str | None):
    """Return the char tokenizer for a backbone `model_type` (from `config.model_type`).

    Unknown types fall back to the HyenaDNA-lineage char vocab (shared across the DNA LMs we
    target) rather than failing — the vocab is very likely identical; provenance records which
    backbone was actually detected.
    """
    spec = BACKBONES.get((model_type or "").lower())
    return spec["tokenizer"]() if spec else HyenaDNACharTokenizer()


def default_checkpoint_for(backbone: str) -> str:
    """Map a backbone name ('hyenadna' | 'caduceus') to its default checkpoint."""
    spec = BACKBONES.get((backbone or "").lower())
    if spec is None:
        raise ValueError(f"unknown backbone {backbone!r}; known: {sorted(BACKBONES)}")
    return spec["default_checkpoint"]


class ActivityPredictor:
    """Wraps one fine-tuned single-context model (primary cortex OR organoid).

    Args:
        checkpoint: HF hub id (or local path) of the HyenaDNA backbone.
        context:    "primary" | "organoid" — which fine-tuned model this instance is.
        device:     "cuda" | "cpu" | None (auto-select).
        seq_len:    modeled element length (bp); sequences are used as-is, this is metadata.
        weights:    optional path to a fine-tuned checkpoint dir (see module docstring).
        pool:       "mean" (default) or "last" — how to reduce (L x hidden) -> (hidden).
        batch_size: forward-pass batch size for saturation mutagenesis.
    """

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, context: str = "primary",
                 device: str | None = None, seq_len: int = 200,
                 weights: str | None = None, pool: str = "mean", batch_size: int = 64,
                 rc_average: bool = False):
        self.checkpoint = checkpoint
        self.context = context          # "primary" | "organoid"
        self.device = device
        self.seq_len = seq_len
        self.weights = weights
        self.pool = pool
        self.batch_size = batch_size
        # rc_average: test-time reverse-complement augmentation — score forward AND the RC strand
        # and average. HyenaDNA is not RC-aware, so this can steady variant Δ; default OFF keeps
        # behavior identical. (Caduceus is RC-equivariant, so it needs this less.)
        self.rc_average = rc_average
        self._model = None              # lazily loaded torch module (backbone)
        self._head = None               # torch.nn.Linear(hidden, 1)
        self._tokenizer = None
        self.backbone_type = None       # detected from config.model_type at load ("hyenadna"|"caduceus")
        self.is_finetuned = False       # True once fine-tuned head weights are loaded

    # ------------------------------------------------------------------ loading
    def load(self) -> "ActivityPredictor":
        """Load backbone + head + tokenizer (transformers, trust_remote_code=True).

        Always builds the base backbone from `checkpoint`; if `weights` points at a
        fine-tuned dir containing `model.pt` ({"backbone":..., "head":...}), overlays those
        state_dicts and sets `is_finetuned`. (We store the fine-tuned backbone as a torch
        state_dict, not save_pretrained, because HyenaDNA shares one `freq` buffer across
        layers and transformers' save_pretrained rejects shared tensors — decision D12.)
        Idempotent — returns self.
        """
        if self._model is not None:
            return self
        import torch
        from transformers import AutoModel

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Fine-tuned weights KNOW their own backbone (the trainer saves `checkpoint` in model.pt's
        # meta). Read it and build the matching backbone — otherwise loading e.g. small-32k (d=256)
        # weights into the default tiny-1k (d=128) backbone crashes on shape mismatch.
        model_path = os.path.join(self.weights, "model.pt") if self.weights else None
        blob = None
        if model_path and os.path.isfile(model_path):
            blob = torch.load(model_path, map_location="cpu")
            saved_ckpt = blob.get("meta", {}).get("checkpoint") if isinstance(blob, dict) else None
            if saved_ckpt and saved_ckpt != self.checkpoint:
                if self.checkpoint != DEFAULT_CHECKPOINT:
                    print(f"[predictor] note: weights were trained on {saved_ckpt!r}; using that "
                          f"backbone (ignoring passed checkpoint {self.checkpoint!r})")
                self.checkpoint = saved_ckpt          # weights dictate the architecture

        self._model = AutoModel.from_pretrained(self.checkpoint, trust_remote_code=True)
        # pick the tokenizer from the loaded backbone's type (HyenaDNA -> identical to before)
        self.backbone_type = getattr(getattr(self._model, "config", None), "model_type", "hyenadna")
        self._tokenizer = build_tokenizer(self.backbone_type)
        hidden = self._infer_hidden_size()
        self._head = torch.nn.Linear(hidden, 1)

        if blob is not None:
            if isinstance(blob, dict) and "backbone" in blob:
                self._model.load_state_dict(blob["backbone"])
                self._head.load_state_dict(blob["head"])
            else:                                   # a bare head state_dict (head-only ckpt)
                self._head.load_state_dict(blob["head"] if "head" in blob else blob)
            self.is_finetuned = True

        self._model.to(self.device).eval()
        self._head.to(self.device).eval()
        return self

    def _infer_hidden_size(self) -> int:
        cfg = getattr(self._model, "config", None)
        for attr in ("hidden_size", "d_model", "n_embd"):
            v = getattr(cfg, attr, None)
            if isinstance(v, int):
                return v
        raise RuntimeError("could not infer backbone hidden size from config; set head dim explicitly")

    def _require_loaded(self):
        if self._model is None:
            raise RuntimeError("call .load() before predicting")

    # ------------------------------------------------------------------ forward
    def _pool(self, last_hidden, attention_mask=None):
        return pool_hidden(last_hidden, attention_mask, self.pool)

    def _forward_batch(self, sequences: list[str]):
        """Return a 1-D tensor of predicted activities for a list of sequences."""
        import torch
        self._require_loaded()
        input_ids, attn = self._tokenizer.encode(sequences)
        input_ids = input_ids.to(self.device)
        attn = attn.to(self.device)
        with torch.no_grad():
            out = self._model(input_ids)
            last_hidden = getattr(out, "last_hidden_state", None)
            if last_hidden is None:                                # some remote heads return a tuple
                last_hidden = out[0]
            pooled = self._pool(last_hidden, attn)
            activity = self._head(pooled).squeeze(-1)
        return activity.float().cpu()

    def predict_activity(self, sequence: str) -> float:
        """Predicted regulatory activity (normalized log2 RNA/DNA) for one sequence."""
        return float(self._forward_batch([sequence])[0])

    def predict_activity_batch(self, sequences: list[str]) -> list[float]:
        """Batched `predict_activity` (chunked by `self.batch_size`)."""
        out: list[float] = []
        for i in range(0, len(sequences), self.batch_size):
            chunk = sequences[i:i + self.batch_size]
            out.extend(v.item() for v in self._forward_batch(chunk))
        return out

    # ------------------------------------------------------------------ ISM
    def saturation_mutagenesis(self, sequence: str):
        """Return a (len(sequence) x 4) array of Δactivity for every single-base substitution.

        Column order is `_BASES` = (A, C, G, T). Entry [i, b] = activity(seq with position i
        set to base b) - activity(ref). The reference base's own column is 0 by construction.
        """
        import numpy as np
        self._require_loaded()
        seq = sequence.upper()
        ref_activity = self.predict_activity(seq)

        # build every single-base variant, skipping no-op (same base) substitutions
        variants: list[str] = []
        coords: list[tuple[int, int]] = []
        for i, ref_base in enumerate(seq):
            for b, alt_base in enumerate(_BASES):
                if alt_base == ref_base:
                    continue
                variants.append(seq[:i] + alt_base + seq[i + 1:])
                coords.append((i, b))

        acts = self.predict_activity_batch(variants) if variants else []
        ism = np.zeros((len(seq), 4), dtype=np.float32)
        for (i, b), a in zip(coords, acts):
            ism[i, b] = a - ref_activity
        return ism

    def score_variant(self, seq_ref: str, seq_alt: str) -> float:
        """Predicted Δactivity for a variant = activity(alt) - activity(ref).

        With `rc_average=True`, averages the forward-strand Δ with the reverse-complement-strand
        Δ (test-time RC augmentation).
        """
        return self.score_variants_batch([seq_ref], [seq_alt])[0]

    def score_variants_batch(self, seq_refs: list[str], seq_alts: list[str]) -> list[float]:
        """Batched Δactivity for many variants. Honors `rc_average` (adds the RC-strand Δ).

        Efficient: one batched forward pass per allele set (and two more for the RC strand when
        `rc_average` is on), rather than per-variant calls.
        """
        ref = self.predict_activity_batch(seq_refs)
        alt = self.predict_activity_batch(seq_alts)
        fwd = [a - r for r, a in zip(ref, alt)]
        if not self.rc_average:
            return fwd
        rc_ref = self.predict_activity_batch([_revcomp(s) for s in seq_refs])
        rc_alt = self.predict_activity_batch([_revcomp(s) for s in seq_alts])
        rc = [a - r for r, a in zip(rc_ref, rc_alt)]
        return [0.5 * (f + b) for f, b in zip(fwd, rc)]

    # ------------------------------------------------------------------ provenance
    def provenance(self) -> dict:
        """Auditable tags for the Interpretation.provenance field."""
        return {
            "checkpoint": self.checkpoint,
            "backbone_type": self.backbone_type,
            "context": self.context,
            "device": self.device,
            "pool": self.pool,
            "rc_average": self.rc_average,
            "is_finetuned": self.is_finetuned,
            "weights": self.weights,
            "weights_hash": _hash_dir(self.weights) if self.weights else None,
        }


class EnsemblePredictor:
    """N independently-seeded ActivityPredictors of ONE context.

    THEORY (decision D17): the mean over members is a slightly better point prediction; the
    **std** over members is a per-variant MODEL uncertainty — a variant the ensemble disagrees on
    should get lower trust. This is duck-compatible with ActivityPredictor (same `score_variant` /
    `score_variants_batch` / `provenance` / `seq_len` / `is_finetuned`), so `interpret_variant`
    accepts either. The extra `*_with_uncertainty` methods expose the disagreement for the trust
    layer.
    """

    def __init__(self, members: list["ActivityPredictor"]):
        if not members:
            raise ValueError("EnsemblePredictor needs >=1 member")
        self.members = members
        self.context = members[0].context

    @classmethod
    def from_dir(cls, root: str, *, checkpoint: str = DEFAULT_CHECKPOINT, context: str = "primary",
                 device: str | None = None, seq_len: int = 270, pool: str = "mean",
                 batch_size: int = 64, rc_average: bool = False) -> "EnsemblePredictor":
        """Build from a dir of `seed*/` member checkpoints (falls back to `root` itself as N=1)."""
        import glob
        dirs = sorted(d for d in glob.glob(os.path.join(root, "seed*")) if os.path.isdir(d))
        if not dirs:
            dirs = [root]                                   # a single-model dir = ensemble of 1
        members = [ActivityPredictor(checkpoint, context, device, seq_len, w, pool, batch_size,
                                     rc_average) for w in dirs]
        return cls(members)

    def load(self) -> "EnsemblePredictor":
        for m in self.members:
            m.load()
        return self

    @property
    def is_finetuned(self) -> bool:
        return all(m.is_finetuned for m in self.members)

    @property
    def seq_len(self) -> int:
        return self.members[0].seq_len

    def predict_activity(self, sequence: str) -> float:
        import numpy as np
        return float(np.mean([m.predict_activity(sequence) for m in self.members]))

    def score_variants_batch(self, seq_refs, seq_alts) -> list[float]:
        mean, _ = self.score_variants_batch_with_uncertainty(seq_refs, seq_alts)
        return list(mean)

    def score_variants_batch_with_uncertainty(self, seq_refs, seq_alts):
        """Return (mean Δ, std Δ) across members, per variant."""
        import numpy as np
        per = np.array([m.score_variants_batch(seq_refs, seq_alts) for m in self.members])  # (N, B)
        return per.mean(axis=0), per.std(axis=0)

    def score_variant(self, seq_ref: str, seq_alt: str) -> float:
        return self.score_variants_batch([seq_ref], [seq_alt])[0]

    def score_variant_with_uncertainty(self, seq_ref: str, seq_alt: str) -> tuple[float, float]:
        mean, std = self.score_variants_batch_with_uncertainty([seq_ref], [seq_alt])
        return float(mean[0]), float(std[0])

    def provenance(self) -> dict:
        return {"ensemble_n": len(self.members), "context": self.context,
                "is_finetuned": self.is_finetuned,
                "members": [m.provenance() for m in self.members]}


def _hash_dir(path: str | None) -> Optional[str]:
    """Stable short hash of a checkpoint dir's file contents (for provenance)."""
    if not path or not os.path.isdir(path):
        return None
    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(path)):
        for name in sorted(files):
            fp = os.path.join(root, name)
            h.update(name.encode())
            with open(fp, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
    return h.hexdigest()[:16]
