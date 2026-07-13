# Submission — Researcher Track ("Build From the Bench")

> Paste-ready framing for the *Researcher Track*. The track asks: start from a biological
> question, use **Claude Science** to find the datasets and tools to answer it, and submit
> something **discrete** — a finding, a trained model, or a reproducible analysis — showing how
> Claude Science got you there. This project answers the track's own example #2:
> *"Predict what a noncoding variant does … in a cell type you care about."*

---

## 1. The biological question

**Does a given noncoding, regulatory variant change how much a gene is expressed — and can we
say so with a *calibrated, auditable* confidence rather than an opaque score?**

Coding-variant interpretation is mature. A variant in regulatory DNA usually gets **no**
principled call and is filed as a VUS, because its effect is not "changes a protein" but
"changes *how much* a gene is transcribed." Answering it requires a model that reads raw DNA.

Cell type of interest: **developing human cortex** (and organoid as an independent second
context) — the tissue behind the psychiatric-disorder variants in our training assay.

## 2. The finding (the discrete deliverable)

A single-context DNA language model, fine-tuned on cortex MPRA and evaluated **leakage-safe** on
**15,273 held-out variants**, produces a real, calibrated variant-effect signal — and a
**direct-skew siamese objective on an RC-equivariant backbone (Caduceus) lifts variant-effect
Δ-Pearson from 0.19 → 0.28** over the naïve activity-difference baseline.

| Result | Value | Honest read |
|---|---|---|
| Element-activity generalization (primary cortex) | val Pearson **0.70** | strong; in line with MPRAnn/Sei-family CNN baselines |
| Independent organoid model (2nd context) | val Pearson **0.69** | near-independent second opinion |
| Variant effect — activity-difference baseline (HyenaDNA) | Δ-Pearson **0.149** | weak but real |
| **Variant effect — direct-skew siamese (Caduceus)** | **Δ-Pearson 0.28**, emVar AUC **0.67** | the win: optimize the *difference*, not the endpoints |
| Held-out emVar recovery (strict, active-gated) | **163/164** significant emVars retained through the leakage-safe split | ground-truth set is intact |
| Calibration | isotonic, monotone; confidence is honest by construction | a well-calibrated "uncertain" is a success, not a failure |

**Why 0.70 → 0.15 → 0.28 is the real story.** Predicting a 270 bp element's activity is far
easier than predicting the effect of flipping **one base** — that gap (0.70 → 0.15) is expected
and matches the field. The contribution is *closing part of that gap the right way*: the siamese
objective trains on the ref/alt **difference** directly instead of subtracting two independently
regressed endpoints, recovering 0.15 → 0.28 with no new data.

## 3. How Claude Science got us there

Full narrative: [`05_how_claude_science_built_this.md`](05_how_claude_science_built_this.md) and
[`06_claude_science_toolchain.md`](06_claude_science_toolchain.md). In brief, Claude Science was
the co-developer across four bench steps:

1. **Found the data.** Traced the assay from the Deng et al. 2024 *Science* paper (`adh0559`) to
   its processed supplement, and the raw PsychENCODE Synapse deposit (`syn51090452`) behind a
   Data-Use Agreement — and made the provenance decision to build on the DUA-free processed
   tables. (Synapse + Hugging Face connectors.)
2. **Chose the tools.** Scouted DNA-LM backbones on the HF Hub and picked **HyenaDNA**
   (single-nucleotide resolution — required for single-base saturation mutagenesis), with
   **Caduceus** (RC-equivariant, bidirectional) as a one-flag upgrade for the variant-effect head.
3. **Made the leakage-safe call.** Designed a **locus-grouped split** so no training sequence
   overlaps a calibration variant, and asserted it in `data/prepare_data.py` — the difference
   between an honest held-out number and a leaked one.
4. **Ran and grounded the analysis.** Fine-tuned both context models, ran in-silico saturation
   mutagenesis, fit the isotonic calibrator, and built an **independent-evidence trust layer**
   (held-out MPRA, GTEx eQTLs, ClinVar, TSS, a frozen foundation model, and — track example #3 —
   **Zoonomia 241-mammal constraint / Human Accelerated Regions**) that surfaces model↔evidence
   conflicts instead of averaging them away.

## 4. Reproducibility (others can rerun it)

- **Leakage-safe eval:** `python eval/calibrate.py --weights weights/primary --s2 <DataS2.xlsx>`
  reproduces the variant-effect Pearson/AUC and refits the calibrator.
- **Siamese refit without a GPU:** the per-variant Δ dump
  (`weights/siamese_eval_predictions.parquet`) refits the calibrator on CPU —
  `python eval/fit_calibrator_from_dump.py --dump weights/siamese_eval_predictions.parquet`
  reproduces the 0.28 / 0.67 diagnostics offline.
- **Provenance:** data `mode=deng`, `n=15,273`, coordinate concordance **99.875%**; every
  prediction carries a checkpoint hash + data versions in its evidence chain.
- **Tests:** `python tests/test_core.py` (leakage split, motif gain/loss, calibration+conflict,
  conservation evidence, end-to-end) — all green.

## 5. Alignment to the track

| Track ask | This submission |
|---|---|
| Start from a biological question | Regulatory VUS have no principled call — a real bench question |
| Use Claude Science to find datasets + tools | Deng MPRA (Synapse/HF), HyenaDNA→Caduceus, JASPAR, GTEx, ClinVar, Zoonomia — all sourced via Claude Science |
| Submit a **finding / trained model / reproducible analysis** | A fine-tuned DNA LM + a reproducible leakage-safe variant-effect result (0.19→0.28) + calibrator |
| Example #2: noncoding variant → cell-type effect | Exactly this — cortex + organoid contexts |
| Example #3: Zoonomia constraint / HARs | Wired as an independent evidence source in the trust layer |

**One honest caveat to state up front:** example #2 names the *ChromBPNet / ATAC-seq* route
(variant → chromatin accessibility). We do the sibling readout — variant → **MPRA regulatory
activity** (expression) — not accessibility. Same question, complementary measurement.
