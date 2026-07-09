# Enhancement #2 completion plan — wire a frozen foundation model into the meta-learner

> **Status: PLAN ONLY (not implemented here).** A parallel session owns the meta-learner code
> (`src/meta.py`, `eval/fit_meta.py`, `src/evidence.py`). This doc is the *process spec* for the one
> remaining piece — a real `foundation_fn` — so it can be picked up without re-deriving anything.
> Nothing below has been coded or run yet.

## 0. Where things stand (why this is the last mile)

Enhancement #2 (`docs/07` §2) is **fully built except one input**. The stacking `MetaCombiner`,
its fit/grade harness (`eval/fit_meta.py`), and the evidence seams
(`evidence.frozen_foundation_delta`, `evidence.from_frozen_foundation_model`) all exist and run
clean. The first real run returned a **TIE** (meta AUC == baseline 0.6096) — *by design*: the only
dense feature was `dna_lm_delta`; `frozen_delta` was `0/2273` (unwired), so the linear model
correctly collapsed to the single informative feature = the calibrator.

**The payoff is gated on one thing: a genuinely independent, informative feature.** The cleanest is
a frozen big-model zero-shot Δ. The code contract is already there — this plan fills it.

The exact seam (already in `src/evidence.py`):
```
foundation_fn(chrom, pos, ref, alt, build="hg38") -> signed float Δ   # None if unwired
```
`frozen_foundation_delta(...)` wraps it into the scalar the meta-learner reads as `frozen_delta`.
`fit_meta.py._signals_for` does NOT yet call it — that wiring is Step 4 below.

## 1. Model choice — Enformer (default), Borzoi (upgrade)

Pick a model that makes **different errors** than our 270 bp single-nt DNA-LM — independence is the
entire value; a correlated feature adds nothing to the stack.

| | Enformer (recommended first) | Borzoi (upgrade) |
|---|---|---|
| context | 196,608 bp | 524,288 bp |
| tracks | CAGE / DNase / ChIP (5,313 human) | RNA-seq (closer to MPRA expression readout) |
| weight | ~250 MB, `enformer-pytorch` HF | ~2 GB, heavier deps |
| speed | ~1 s/variant on GPU | ~2–3× slower |
| why | best effort/payoff; standard | may carry stronger expression signal |

Start Enformer. If its brain track shows a nonzero fitted weight but a weak win, try Borzoi's
RNA-seq tracks (a closer readout to allelic skew). Design the fn behind a backend flag so the swap
is one argument.

## 2. The key subtlety — coordinates, not the 270 bp element

The siamese/activity path scores the 270 bp MPRA oligo. **Enformer cannot** — it needs ~196 kb of
genomic context centered on the variant. That context is NOT in `*_variants.parquet` (only
`seq_ref/seq_alt` 270 bp). It must come from:

- **hg38 FASTA** — already local at `data/raw/genome/hg38.fa` (`data/genome.py::Genome` reads it).
- **`chrom`, `pos`** — carried by `calibration_variants.parquet` and inherited by
  `eval_variants_siamese.parquet` / `train_variants.parquet` / `val_variants.parquet`.

This is exactly why the seam is `foundation_fn(chrom, pos, ref, alt, build)` and not
`fn(seq_ref, seq_alt)`. The fn pulls its own window from hg38.

## 3. New file `src/foundation.py` — the `foundation_fn` factory

Additive, opt-in, GPU-only. Skeleton (to be implemented + tested by the meta session):

```python
"""Frozen foundation-model zero-shot Δ (Enhancement #2 independent feature).
CHARTER: the big model stays FROZEN, zero-shot — a FEATURE, never fine-tuned (CLAUDE.md §3)."""
from enformer_pytorch import from_pretrained

SEQ_LEN   = 196_608          # Enformer receptive field
N_BINS    = 896             # output bins (128 bp each)
CENTER    = N_BINS // 2     # variant sits in the center bin
BRAIN_TRK = 4980            # a brain CAGE track (resolve from targets_human.txt; may use a set)

def make_enformer_fn(genome, device="cuda", track=BRAIN_TRK, center_bins=3):
    model = from_pretrained("EleutherAI/enformer-official-rough").to(device).eval()
    def fn(chrom, pos, ref, alt, build="hg38"):
        # 1-based pos -> centered 196kb window from hg38; assert ref matches the reference base
        ref_seq = genome.window_centered(chrom, pos, SEQ_LEN)      # str, len SEQ_LEN
        ref_ids = encode_acgt(ref_seq)                             # LongTensor (SEQ_LEN,)
        alt_ids = ref_ids.clone(); alt_ids[SEQ_LEN // 2] = base_idx(alt)
        import torch
        with torch.no_grad():
            lo, hi = CENTER - center_bins // 2, CENTER + center_bins // 2 + 1
            r = model(ref_ids.to(device))["human"][lo:hi, track].mean()
            a = model(alt_ids.to(device))["human"][lo:hi, track].mean()
        return float(a - r)                                        # signed Δ (alt - ref)
    return fn
```

Implementation notes:
- **Reference-base check** — assert the hg38 window's center base == `ref`; if not, the variant's
  strand/coord is off. Reuse the same `genome_ref_warning` discipline `interpret.py` already has.
- **Track selection** — a single brain CAGE track is the MVP. Better: mean over a curated brain
  CAGE+DNase track set (resolve indices once from Enformer's `targets_human.txt`, hard-code the
  list with a comment). Keep the Δ **signed** (meta builds `concordance_dna_frozen` from the sign).
- **`genome.window_centered`** — if `data/genome.py` lacks a centered-window helper, add a tiny one
  (it already has windowing for `interpret.py`); do not duplicate FASTA logic.

## 4. Precompute + cache (Enformer is expensive — score once, reuse forever)

~1 s/variant × (10,659 train + 2,341 val + 2,273 eval ≈ 15k) ≈ **hours on GPU**. Do it **once**,
persist, then every refit is CPU-only (same pattern just used for `siamese_eval_predictions.parquet`).

- New script `eval/precompute_frozen.py` (or a `--dump` flag on a small runner):
  reads the three variant tables, runs `foundation_fn` over unique `(chrom,pos,ref,alt)`, writes
  **`data/processed/frozen_delta_cache.parquet`** with columns `chrom,pos,ref,alt,frozen_delta`.
- Cache is keyed by variant identity so train/val/eval all draw from one file. Gitignored (derived).
- Log coverage + any ref-mismatch drops honestly.

## 5. Wire `fit_meta.py` (the only change to existing code — coordinate with the meta session)

Minimal, additive:
- Add `--frozen-cache PATH`.
- Load it into a dict `{(chrom,pos,ref,alt): frozen_delta}`.
- In `_signals_for`, after setting `dna_lm_delta`, inject:
  ```python
  fd = frozen_lookup.get((row.chrom, row.pos, row.ref, row.alt))
  if fd is not None: sig["frozen_delta"] = float(fd)
  ```
That's it — `assemble_features` already turns `frozen_delta` into `abs_frozen_delta` +
`concordance_dna_frozen`; `MetaCombiner` already imputes/standardizes/fits.

## 6. Run + verdict

```bash
# one-time GPU precompute:
python eval/precompute_frozen.py --genome data/raw/genome/hg38.fa --backend enformer
# CPU refit + grade (uses the siamese Δ as the DNA-LM signal — our best):
python eval/fit_meta.py --siamese-weights weights/siamese_cad \
       --frozen-cache data/processed/frozen_delta_cache.parquet
```

**Success gate:** `META AUC > baseline` on the held-out eval slice (single-feature |Δ| ≈ 0.61–0.67,
same slice as the siamese eval so it's comparable + leakage-safe). Read `feature_weights`:
- `abs_frozen_delta` and `concordance_dna_frozen` **nonzero** → the independent signal is doing work
  (the intended win). These weights ARE the "why we trust this call" the product promises.
- both ≈ 0 → Enformer's chosen track carries no emVar signal on this MPRA set → **honest negative**;
  log it (try Borzoi RNA-seq tracks or a different track set before rejecting).

## 7. Guardrails (do not regress what already works)

- **Charter:** frozen model stays frozen/zero-shot — a feature, never fine-tuned (`CLAUDE.md §3`).
- **Leakage:** meta fits on train+val loci, grades on the disjoint eval slice — the harness already
  asserts this (`fit_meta.py` locus check). The frozen cache is just a feature; it does not change
  the split.
- **Graceful degrade:** no `--frozen-cache` → `frozen_delta` imputes → today's TIE behavior, unchanged.
  CPU deploys never need Enformer.
- **Deps:** `enformer-pytorch` (+ its torch/einops) are GPU-only extras. Keep them OUT of the base
  `requirements.txt`; add a commented `# frozen-model (Enhancement #2, GPU): enformer-pytorch` line,
  same discipline as `mamba-ssm`.

## 8. Effort / sequencing

| Step | File | Effort | Gate |
|---|---|---|---|
| 3. `foundation_fn` (Enformer) | `src/foundation.py` (NEW) | M | smoke: returns finite Δ, ref-base matches |
| 4. precompute cache | `eval/precompute_frozen.py` (NEW) | S | cache parquet covers all variants |
| 5. wire feature | `eval/fit_meta.py` (edit) | S | `frozen_delta` coverage 2273/2273 on eval |
| 6. run + interpret | — | S (GPU time) | meta AUC vs baseline; frozen weight nonzero |

MVP = Enformer + single brain track. If it ties, escalate track set → Borzoi before calling it a
negative. Whole thing is ~half a day of GPU work on top of the finished machinery.
