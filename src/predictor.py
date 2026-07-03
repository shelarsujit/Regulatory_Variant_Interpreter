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

STATUS: stub. Signatures are frozen; bodies land in Phase 2.
"""
from __future__ import annotations

DEFAULT_CHECKPOINT = "LongSafari/hyenadna-tiny-1k-seqlen"


class ActivityPredictor:
    """Wraps one fine-tuned single-context model (primary cortex OR organoid)."""

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, context: str = "primary",
                 device: str | None = None, seq_len: int = 200):
        self.checkpoint = checkpoint
        self.context = context          # "primary" | "organoid"
        self.device = device
        self.seq_len = seq_len
        self._model = None              # lazily loaded torch module

    def load(self) -> "ActivityPredictor":
        """Load weights + tokenizer (transformers, trust_remote_code=True)."""
        raise NotImplementedError("Phase 2: load HyenaDNA checkpoint")

    def predict_activity(self, sequence: str) -> float:
        """Predicted regulatory activity (normalized log2 RNA/DNA) for one sequence."""
        raise NotImplementedError("Phase 2: forward pass -> scalar activity")

    def saturation_mutagenesis(self, sequence: str):
        """Return a (len(sequence) x 4) array of Δactivity for every single-base substitution."""
        raise NotImplementedError("Phase 2: full ISM map")

    def score_variant(self, seq_ref: str, seq_alt: str) -> float:
        """Predicted Δactivity for a variant = activity(alt) - activity(ref)."""
        return self.predict_activity(seq_alt) - self.predict_activity(seq_ref)
