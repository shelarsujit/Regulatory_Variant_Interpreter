# Results — first real fine-tune + calibration (Deng cortical MPRA)

First end-to-end run on the **real** Deng *Science* adh0559 data (not the synthetic fixture).
Two HyenaDNA models fine-tuned (primary + organoid), then the primary model evaluated on the
held-out calibration variants and an isotonic trust-calibrator fit. This file is the honest
record of what the numbers are and what they mean.

Provenance: data `mode=deng`, `n_calibration=15,273`, coordinate concordance **99.875%**
(manifest `git 33d471d`); backbone **HyenaDNA** `hyenadna-tiny-1k-seqlen-hf`; CPU fine-tune.
Reproduce the eval with:
`python eval/calibrate.py --weights weights/primary --s2 <DataS2.xlsx>`.

---

## 1. Fine-tuning — element activity (the easy metric)

Target = normalized RNA/DNA ratio per 270 bp element. Val split is locus-disjoint (D5), so val
Pearson is an honest generalization number. Best epoch auto-saved (val-Pearson selection).

| Model | Target | Best epoch | **val Pearson** | val MSE |
|---|---|---|---|---|
| **Primary cortex** (the call) | `activity_primary` | 5 | **0.7026** | 0.069 |
| **Organoid** (independent 2nd, D4) | `activity_organoid` | 4 | **0.694** | 0.080 |

Both climbed cleanly then overfit (train MSE kept falling while val Pearson rolled over) — the
best epoch, not the last, is the checkpoint. **r ≈ 0.70** is strong for MPRA element activity and
in line with the paper's CNN baselines (MPRAnn/Sei family).

## 2. Variant effect — the hard, real metric

Predicted Δactivity = activity(alt) − activity(ref) on all **15,273** held-out variants vs the
wet-lab measured allelic skew (`logFC`). The model never trained on these (locus split, D5).

| Metric | Value | Read |
|---|---|---|
| Pearson (Δ vs measured skew) | **0.149** | weak but real |
| Spearman | 0.104 | weak but real |
| emVar AUC — loose (FDR≤0.10, 596 pos) | 0.611 | > chance |
| **emVar AUC — strict (active-gated, 163 pos)** | **0.615** | > chance |

**Why the gap from 0.70 → 0.15 is expected.** Predicting a 270 bp element's activity is far
easier than predicting the effect of flipping **one base**. Tiny, noisy single-base effects are
the open hard problem of the field; r≈0.15 / AUC≈0.61 sits at the low end of the realistic range
(literature ~0.1–0.35 / 0.6–0.75) for a small model.

## 2b. Backbone A/B — tiny-1k vs small-32k (bigger HyenaDNA)

Swapped the backbone to `hyenadna-small-32k-seqlen-hf` (d=256, 3.3M params vs tiny's d=128,
1.6M) — a `--checkpoint` swap, same pipeline. Trained on T4 (~55 s/epoch, batch 128, AMP).

| Metric | tiny-1k | small-32k | Δ |
|---|---|---|---|
| element val Pearson (primary) | 0.7026 | 0.7194 | +0.017 |
| element val Pearson (organoid) | 0.694 | 0.7156 | +0.022 |
| variant Pearson (Δ vs skew) | **0.149** | 0.1445 | −0.005 |
| variant Spearman | 0.104 | 0.091 | −0.013 |
| emVar AUC (loose) | 0.611 | 0.596 | −0.015 |
| **emVar AUC (strict, the 164)** | 0.615 | **0.628** | **+0.013** |

**Verdict: marginal.** 2× capacity improved *element activity* (+0.02) and nudged the strict-emVar
classification AUC (+0.013 on the paper's significant set), but did **nothing** for the continuous
variant-effect correlation (flat/slightly worse). This is the **data ceiling**: the model can fit
element activity better, but the single-base variant signal is limited by the assay's noise, not
by model capacity. Bigger HyenaDNA is a small, honest win on classification — not the lever that
unlocks variant-effect. That lever is **architecture** (bidirectional Caduceus, §8), not size.
Logged: `weights/results_s32.json`.

## 3. Pipeline validation — 163 ≈ 164

The strict, active-gated emVar count from our reconstructed data was **163**, essentially the
paper's **164**. That reproduces Deng's headline emVar set end-to-end and independently validates
the whole data pipeline: hg38 sequence reconstruction, dbSNP allele resolution, and the
FDR<10% ∧ ≥1-allele-active definition (docs/01 §4). The foundation is correct.

## 4. Calibration — is the confidence honest?

Isotonic calibrator (trust.Calibrator, D11) fit on |Δ| → P(real effect), strict emVar as the
classification label. Reliability (calibrated P vs observed emVar rate):

| Calibrated P bin | n | mean P | observed emVar rate |
|---|---|---|---|
| [0.0, 0.2) | 15,248 | 0.010 | 0.010 |
| [0.2, 0.4) | 23 | 0.304 | 0.174 |
| [0.8, 1.0) | 2 | 0.999 | 0.000 (n=2, noise) |

The bulk bin is **honest** — it predicts 1.0% and observes 1.0%; the model does **not** claim
confidence it lacks. Mid bin is roughly ordered on tiny n. This is the trust thesis working (D1):
most variants correctly return "uncertain", and the AUC>0.5 means the few it flags are enriched
for true movers. Saved: `weights/calibrator_primary.json`.

## 5. Honest verdict

- **Element predictor: strong** (r≈0.70).
- **Variant-effect signal: modest but real** (r≈0.15, AUC≈0.62) — the genuinely hard task.
- **Data pipeline: validated** (163≈164 emVars).
- **Calibration: honest** (reliability diagonal on the bulk).

For a *trust-first* tool this is the right kind of result: it is honest about modest performance
and calibrated so a curator knows when to trust a call. A confident r=0.15 predictor would be
dangerous; a calibrated one that says "uncertain — here is the evidence" is the product.

## 6. Tried: test-time reverse-complement averaging — NEGATIVE result

Averaging the forward-strand Δ with the reverse-complement-strand Δ (`--rc`) was tested on the
full 15,273 variants. It **hurt** across the board:

| | Pearson | Spearman | emVar AUC (loose) | emVar AUC (strict) |
|---|---|---|---|---|
| Baseline (no-RC) | **0.149** | **0.104** | **0.611** | **0.615** |
| RC-averaged | 0.113 | 0.082 | 0.575 | 0.583 |

**Why it fails (and why it matters):** the MPRA tests a *designed oligo in one orientation*; the
model learned forward-strand → activity. The reverse complement is a sequence the assay never
measured — out-of-distribution — so averaging it in dilutes the real signal. **MPRA activity is
orientation-specific**, unlike arbitrary genomic strand. RC-averaging is kept implemented
(`rc_average`, default OFF) but is **not** used. Logged: `weights/results_primary_rc.json`.

**Consequence for the Caduceus plan:** this result argues *against* the RC-**equivariant**
`caduceus-ps` variant (which forces `f(seq)=f(rc(seq))` — a constraint mismatched to an
orientation-specific assay). Prefer a **bidirectional, non-RC-equivariant** config (e.g. a
`caduceus-ph`/standard variant): bidirectionality reads both flanks of the variant *on the same
forward strand* (the useful part), without imposing RC-invariance (the harmful part).

## 7. Tried: lower calibration τ — NEGATIVE result

Lowering the "real effect" threshold τ (|skew|≥τ) to give the calibrator more positive labels
did **not** help. emVar AUC is **identical at every τ** (0.6153) — τ only rescales the probability,
it does not change ranking/discrimination. Worse, lower τ *de-aligns* the calibrated P from the
observed emVar rate:

| τ | frac positive | bulk mean P | bulk observed emVar |
|---|---|---|---|
| **0.5** (kept) | 1.1% | 0.010 | 0.010 (aligned) |
| 0.3 | 4.1% | 0.040 | 0.010 (overstates) |
| 0.2 | 10.9% | 0.106 | 0.010 (overstates) |

**Keep τ=0.5** — its label rate (1.1%) nearly reproduces the paper's emVar rate (1.07%), so its
probabilities stay honest against the significance definition. Logged: `weights/results_tau*.json`.

## 8. Roadmap to lift variant-effect performance

Ranked by leverage (raw accuracy is the tunable knob; the trust thesis already holds):

1. **Caduceus backbone — bidirectional, NOT the RC-equivariant `-ps`** (see §6). HyenaDNA is
   *causal*, so a mid-sequence variant suffers left-context bias; a bidirectional model sees both
   flanks. Highest-leverage. Plumbing ready (`--backbone caduceus`, D16); GPU-only.
2. **Deep ensemble** (N seeds) — plumbing **implemented + tested** (D18): `--n-seeds N` trains
   members, `EnsemblePredictor` gives mean Δ + per-variant σ that shrinks trust confidence. Awaits
   a real multi-seed training run to quantify the accuracy bump.
3. **Bigger HyenaDNA** (`small-32k` d=256) — easy `--checkpoint` swap, more capacity, CPU-testable.

*Tried and rejected this session: reverse-complement averaging (§6), lower calibration τ (§7).*

## 7. Reproduce

```
# data (needs hg38 FASTA + myvariant.info):
python data/prepare_data.py --deng-dir data/raw/science.adh0559_data_s1_to_s3 --genome data/raw/genome/hg38.fa
# fine-tune (Colab A100 or CPU):
python train/finetune_hyenadna.py --context primary  --data data/processed --out weights/primary  --amp
python train/finetune_hyenadna.py --context organoid --data data/processed --out weights/organoid --amp
# eval + calibrator:
python eval/calibrate.py --weights weights/primary --s2 data/raw/.../DataS2-Variant-library-ratios.xlsx
```
Exact per-run provenance: `weights/<ctx>/provenance.json`, `data/processed/manifest.json`,
`weights/calibrator_primary.json`.
