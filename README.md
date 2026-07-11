# Regulatory Variant Interpreter

**A trust-first tool for interpreting non-coding, regulatory DNA variants.**

Most variant-interpretation tools work on *coding* variants — changes that alter a
protein. But a huge fraction of disease-associated genetic variation sits in
**regulatory** DNA: sequence that doesn't code for a protein but controls **how much**
a gene is switched on. For those, a curator today usually gets *no principled call at
all* — the variant is filed as "of uncertain significance" (a VUS).

This project reads raw DNA with a fine-tuned **DNA language model**, predicts what a
single-base change does to regulatory activity (**in-silico saturation mutagenesis**),
explains the likely mechanism (**transcription-factor motif** gain/loss), and — the
core of the system — **grounds every prediction in independent evidence** to return a
**calibrated confidence** with an **auditable evidence chain**. When the model and the
independent evidence agree, confidence is high; when they conflict, the tool says so
honestly.

> New here, or not a biologist? Read
> **[`docs/00_overview_for_non_biologists.md`](docs/00_overview_for_non_biologists.md)** —
> it explains the whole subject from first principles (DNA, genes, regulation, variants,
> MPRA, and "reading DNA with a model") before any code. Built with **Claude Science** —
> see **[`docs/05_how_claude_science_built_this.md`](docs/05_how_claude_science_built_this.md)**.

## Status — working end to end
Not a skeleton: two DNA-LMs fine-tuned, a variant-effect model trained, the trust layer
calibrated, and a two-tab demo app that returns auditable, calibrated calls today.

| Component | State |
|---|---|
| Leakage-safe data prep → train/val + held-out variant calibration set | ✅ `data/prepare_data.py`, `data/splits.py`, `data/load_deng.py` |
| DNA-LM fine-tune (HyenaDNA + Caduceus-ph, backbone-swappable) | ✅ `train/finetune_hyenadna.py` — trained |
| Saturation-mutagenesis variant scorer | ✅ `src/predictor.py` — trained |
| **Direct-skew siamese variant-effect model** | ✅ `train/finetune_siamese.py`, `src/siamese_predictor.py` — trained (best variant-effect) |
| TF-motif gain/loss mechanism | ✅ `src/motifs.py` |
| Independent evidence (held-out MPRA, GTEx, ClinVar, TSS, frozen model) | ✅ `src/evidence.py` |
| Isotonic calibration + agreement/conflict trust layer | ✅ `src/trust.py` |
| **Stacking meta-learner** (fuses DNA-LM + organoid + motif + frozen) | ✅ `src/meta.py`, `eval/fit_meta.py` |
| `interpret_variant()` orchestrator | ✅ `src/interpret.py` |
| Gradio demo app (coordinates + paste-sequence tabs) | ✅ `app/app.py` — narrated demo videos included |
| Tests (leakage, motif, calibration+conflict, siamese, meta, end-to-end) | ✅ `tests/` — all green |

## Headline results (held-out, leakage-safe)
Full detail + every tried-and-rejected experiment in
[`docs/03_results.md`](docs/03_results.md) and [`docs/07_enhancement_design.md`](docs/07_enhancement_design.md).

| What | Metric | Result |
|---|---|---|
| Element activity (the easy metric) | val Pearson | **≈0.72** (HyenaDNA) → **0.78** (Caduceus-ph) |
| Variant effect — subtract-endpoints Δ | Δ-Pearson / emVar AUC | 0.19 / 0.61 (Caduceus) |
| **Variant effect — direct-skew siamese** | **Δ-Pearson / emVar AUC** | **0.28 / 0.67** (Caduceus-ph, +47% over subtract) |
| Trust layer — stacking meta-learner | emVar AUC vs single-feature | **0.623 vs 0.610** (independent organoid model is the lever) |
| Pipeline validation | reconstructed emVar count | **163 ≈ 164** (reproduces Deng's headline set) |
| Calibration | reliability on the bulk bin | predicts 1.0%, observes 1.0% — honest |

**Honest negatives are documented, not hidden** (the whole point of a trust-first tool):
reverse-complement averaging, lower calibration τ, bigger HyenaDNA, and a **frozen Enformer
CAGE feature** (tested single-track → curated developmental set → full 15k coverage) all failed
to help — each logged with the reason. This is the trust thesis applied to our own work.

## What a call looks like
`chr1:41041729 G>A` → **likely benign · confidence 29% ⚠ CONFLICT**
- primary-cortex model Δ = +0.052 (UP); independent organoid model Δ = +0.062 (UP)
- mechanism: ALT disrupts a predicted CCAAT motif (Δscore −5.20)
- **conflicting evidence:** held-out MPRA measured skew −1.174 (significant emVar) — the wet lab says strong DOWN
- verdict: model leaned UP at 72%, but the independent measurement disagrees → **surfaced, not averaged away** → 29%

A confident wrong call is dangerous; a calibrated "uncertain — here's the conflict" is the product.

## Quickstart

**Run the demo app** (interprets variants with calibrated trust + the meta-learner):
```bash
pip install -r requirements.txt
# with a local hg38 FASTA, the coordinates tab works for any variant; without one, a
# calibration-backed shim serves the curated demo variants offline.
RVI_META=1 python app/app.py           # http://127.0.0.1:7860
```
Two input modes: **variant coordinates** (chrom/pos/ref/alt) and **paste sequences** (ref/alt
elements, no genome needed). Narrated walkthroughs are generated by `demo/make_demo_video.py`.

**Reproduce the models** (data → train → evaluate → calibrate):
```bash
# 1. leakage-safe data prep (needs the Deng supplement + hg38 FASTA; --synthetic to smoke-test):
python data/prepare_data.py --deng-dir data/raw/science.adh0559_... --genome data/raw/genome/hg38.fa
python data/make_variant_pairs.py               # carves the siamese variant train/val/eval split

# 2. fine-tune (Colab A100/T4 — see notebooks/):
python train/finetune_hyenadna.py --context primary  --backbone caduceus --amp
python train/finetune_siamese.py   --backbone caduceus --init-from weights/primary_cad --amp

# 3. evaluate + calibrate:
python eval/calibrate.py    --weights weights/primary_cad --s2 <DataS2.xlsx>
python eval/eval_siamese.py --siamese-weights weights/siamese_cad --fit-calibrator
python eval/fit_meta.py     --activity-weights weights/primary --organoid-weights weights/organoid
```
GPU recipes (Caduceus, siamese, calibrator, frozen-model precompute) are in
[`notebooks/`](notebooks/). Every run writes a `provenance.json` (checkpoint id, git commit,
data versions, metrics).

## Pipeline
```
fine-tune DNA-LM (HyenaDNA | Caduceus)   →   siamese variant-effect model (direct allelic-skew)
        │                                            │
        ▼                                            ▼
saturation mutagenesis  →  motif gain/loss  →  independent evidence  →  calibrated trust + meta-learner
   (predictor.py)           (motifs.py)          (evidence.py)          (trust.py, meta.py)
                                                                              │
                                                                              ▼
                                                            interpret_variant() → Interpretation
```

## Repo map
```
data/prepare_data.py    supplement tables -> train/val + held-out variant calibration set
data/make_variant_pairs.py  leakage-safe siamese variant train/val/eval split
data/splits.py          leakage-safe, locus-grouped splitting
data/genome.py          pyfaidx hg38 window reader (+ window_centered for Enformer)
data/calib_genome.py    calibration-backed genome shim (coords tab works offline)
src/schema.py           the Interpretation contract (dataclasses)
src/predictor.py        checkpoint loader + saturation-mutagenesis scorer (backbone-swappable)
src/siamese_predictor.py  direct ref/alt skew scorer (best variant-effect)
src/motifs.py           JASPAR PWM gain/loss
src/evidence.py         held-out MPRA, GTEx, ClinVar, TSS, frozen foundation model
src/trust.py            isotonic calibrator + agreement/conflict aggregation
src/meta.py             stacking meta-learner (multi-feature calibrated confidence)
src/foundation.py       frozen-Enformer zero-shot Δ (independent feature)
src/interpret.py        interpret_variant() orchestrator
train/                  HyenaDNA + siamese fine-tunes
eval/                   calibrate, eval_siamese, fit_meta, precompute_frozen
app/app.py              Gradio demo (coordinates + paste-sequence)
tests/                  leakage, motif, calibration+conflict, siamese, meta, end-to-end
docs/                   plain-English theory, provenance, decision log, results, method
notebooks/              Colab GPU recipes (Caduceus, siamese, calibrator, frozen precompute)
```

## Design principles
1. **Trust > accuracy** — a calibrated "uncertain" beats a confident wrong call.
2. **Always surface conflict** — never hide model-vs-evidence disagreement.
3. **Never train on calibration variants** — enforced by a locus-grouped split and asserted in code.
4. **Everything is auditable** — evidence chain + provenance (checkpoint hash, data versions) on every call.
5. **Report negatives honestly** — every failed experiment is logged with its reason.

See [`CLAUDE.md`](CLAUDE.md) for the full charter,
[`docs/02_decision_log.md`](docs/02_decision_log.md) for the reasoning behind every choice, and
[`docs/04_prior_art.md`](docs/04_prior_art.md) for how this sits against existing tools.

## License
Code in this repository: **Apache License 2.0** — see [`LICENSE`](LICENSE).

Third-party assets are used under their own licenses and are **not** redistributed here:

| Asset | Use | License |
|---|---|---|
| **Deng et al. 2024** lentiMPRA supplement (`adh0559`) | training + calibration data | **CC-BY-NC** — downloaded by you |
| **HyenaDNA** (`LongSafari/hyenadna-*`) | fine-tune backbone | **BSD-3-Clause** |
| **Caduceus** (`kuleshov-group/caduceus-ph_*`) | bidirectional backbone | **Apache-2.0** |
| **Enformer** (`EleutherAI/enformer-official-rough`) | frozen independent feature | **CC-BY-4.0** |

## Acknowledgements & citations
- **Deng, C., Whalen, S., Steyert, M., … Ahituv, N., Pollard, K. S.** (2024). *Massively
  parallel characterization of regulatory elements in the developing human cortex.*
  **Science** 384(6698):eadh0559. https://doi.org/10.1126/science.adh0559
- **Nguyen, E., Poli, M., … Ré, C., Baccus, S.** (2023). *HyenaDNA: Long-Range Genomic
  Sequence Modeling at Single Nucleotide Resolution.* arXiv:2306.15794.
- **Schiff, Y., Kao, C.-H., … Kuleshov, V.** (2024). *Caduceus: Bi-Directional Equivariant
  Long-Range DNA Sequence Modeling.* arXiv:2403.03234.
- **Avsec, Ž., Agarwal, V., … Kelley, D. R.** (2021). *Effective gene expression prediction
  from sequence by integrating long-range interactions* (Enformer). **Nature Methods** 18:1196–1203.
