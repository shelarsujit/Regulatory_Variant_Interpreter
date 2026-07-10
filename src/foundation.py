"""Frozen foundation-model zero-shot Δ — the meta-learner's independent feature (docs/08).

THEORY (plain English — docs/07 #2, docs/08):
Our fine-tuned model reads a 270 bp MPRA oligo. A frozen big model (Enformer) reads ~196 kb of
real genomic context and predicts assay tracks (CAGE / DNase / ChIP). It makes DIFFERENT errors —
that independence is the entire value as a stacking feature. We run it FROZEN, zero-shot: score the
reference and alternate 196 kb windows, take the difference in a brain track = a signed Δ that the
meta-learner consumes as `frozen_delta`. The big model is a FEATURE, never fine-tuned (CLAUDE.md §3).

WHY A FACTORY: the model is ~250 MB and GPU-scale; `enformer_pytorch` is imported lazily INSIDE
`make_enformer_fn` so importing this module (or the app) never pulls the heavy dep. Build the fn
once, reuse it across variants. Contract matches the seam in `src/evidence.py`:
    foundation_fn(chrom, pos, ref, alt, build="hg38") -> signed float Δ  (or None on a bad window)

USAGE (GPU, e.g. Colab — see eval/precompute_frozen.py):
    from data.genome import Genome
    from src.foundation import make_enformer_fn
    fn = make_enformer_fn(Genome("data/raw/genome/hg38.fa"), device="cuda")
    delta = fn("chr11", 64443180, "G", "C")
"""
from __future__ import annotations

SEQ_LEN = 196_608          # Enformer receptive field
N_BINS = 896               # output bins (128 bp each) at the default target length
CENTER = N_BINS // 2       # the variant sits in the center output bin

# Curated developmental/cortical CAGE tracks (Enformer `targets_human.txt`) — averaged. The Deng
# MPRA is MID-GESTATION human cortex + organoids, so the readout should emphasize FETAL brain,
# neural progenitors, cortex, and neurons — NOT adult whole-brain (the single track 4980 tried
# first, a NEGATIVE; docs/07 §2 Third result). False positives excluded (smooth-muscle/brain-
# vascular 4768, renal-cortical 4882, adrenal-cortex 5051). CAGE (transcription initiation) is the
# closest proxy to an enhancer's MPRA activity; averaging a curated set is steadier than one track.
DEFAULT_TRACKS = (
    4680,   # CAGE:brain, adult, pool1
    4769,   # CAGE:Astrocyte - cerebral cortex
    4798,   # CAGE:Neural stem cells        (progenitors — closest to the MPRA cell state)
    4981,   # CAGE:brain, fetal, pool1      (developmental — matches mid-gestation cortex)
    5103,   # CAGE:occipital cortex, adult
    5112,   # CAGE:Neurons
    5263,   # CAGE:occipital cortex - adult
    5282,   # CAGE:occipital cortex, newborn (developmental cortex)
    5305,   # CAGE:parietal cortex, adult
)


def make_enformer_fn(genome, *, device: str = "cuda",
                     checkpoint: str = "EleutherAI/enformer-official-rough",
                     tracks=DEFAULT_TRACKS, center_bins: int = 3,
                     require_ref_match: bool = False):
    """Return a `foundation_fn(chrom,pos,ref,alt,build)` computing a frozen-Enformer zero-shot Δ.

    `genome`: a data.genome.Genome (or anything with `window_centered(chrom,pos,length)->str`).
    `tracks`: output-track index or iterable of indices to average.
    `center_bins`: how many center output bins (128 bp each) to average around the variant.
    `require_ref_match`: if True, return None when the hg38 center base != `ref` (strand/coord issue).
    """
    import torch
    from enformer_pytorch import from_pretrained
    from enformer_pytorch.data import str_to_one_hot

    model = from_pretrained(checkpoint).to(device).eval()
    track_idx = list(tracks) if hasattr(tracks, "__iter__") else [int(tracks)]
    lo = max(0, CENTER - center_bins // 2)
    hi = min(N_BINS, CENTER + center_bins // 2 + 1)

    def _human(out):
        # enformer-pytorch returns either a dict {'human':..,'mouse':..} or a bare tensor
        t = out["human"] if isinstance(out, dict) else out
        return t[0] if t.dim() == 3 else t          # (bins, tracks)

    def _score(seq: str) -> float:
        oh = str_to_one_hot(seq).unsqueeze(0).to(device)     # (1, SEQ_LEN, 4)
        with torch.no_grad():
            pred = _human(model(oh))                          # (bins, tracks)
        return float(pred[lo:hi][:, track_idx].mean().item())

    def fn(chrom, pos, ref, alt, build="hg38"):
        seq_ref = genome.window_centered(str(chrom), int(pos), SEQ_LEN)
        if len(seq_ref) != SEQ_LEN:
            return None
        c = SEQ_LEN // 2
        if require_ref_match and seq_ref[c].upper() != str(ref).upper():
            return None
        seq_alt = seq_ref[:c] + str(alt).upper() + seq_ref[c + 1:]
        return _score(seq_alt) - _score(seq_ref)              # signed Δ = alt - ref

    fn.model = model                                          # expose for reuse / introspection
    return fn
