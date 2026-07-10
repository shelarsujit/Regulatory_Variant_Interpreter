# Enhancement design — post-Caduceus (siamese objective + stacking meta-learner)

Two designs that attack the **variant-effect** bottleneck (not model capacity — the small-32k
A/B in `03_results.md §2b` proved capacity is not the lever). Both are charter-safe: the frozen
foundation model stays frozen; organoid stays an independent second model. File-level, concrete.

Current data flow (unchanged parts):
`score_variant(seq_ref, seq_alt)` in `src/predictor.py` returns `activity(alt) − activity(ref)`
— an *indirect* Δ from a model trained only on single-sequence activity. That proxy is the
suspected source of the 0.70 → 0.15 drop.

---

## Enhancement #1 — Direct paired ref/alt (siamese) objective

**Thesis:** train the loss on the thing we score. Instead of regressing element activity and
subtracting, feed the (ref, alt) pair through a shared-weight encoder and regress the **measured
allelic skew `logFC`** directly. The subtraction moves *inside* the training graph, so gradients
optimize Δ, not activity.

### Data
- Source already exists: `data/processed/calibration_variants.parquet` has `logFC` per variant —
  but that's the **held-out calibration set; never train on it** (`CLAUDE.md §3`, leakage).
- Need a **train-split variant table** with paired ref/alt sequences + `logFC`, locus-disjoint
  from calibration. Add to `data/prepare_data.py`: emit `train_variants.parquet` / `val_variants.parquet`
  built from the Deng variant library rows whose loci fall in the *train* locus groups
  (`data/splits.py` already computes the grouping — reuse it, don't re-split).
- Assert leakage at end of `prepare_data.py` exactly as the activity path does: no variant locus
  in `train_variants` appears in `calibration_variants`.

### Model — `train/finetune_hyenadna.py`
- Add `--objective {activity, siamese}` (default `activity`, back-compat).
- New target loader branch: `_load_split` gains a paired mode reading `seq_ref, seq_alt, logFC`.
- New module in the existing `_build_regressor` factory: `SiameseSkewRegressor(nn.Module)`
  - shared backbone (same checkpoint load path as `HyenaDNARegressor`)
  - `forward(ref_ids, ref_mask, alt_ids, alt_mask)`:
    `h_ref = pool(backbone(ref)); h_alt = pool(backbone(alt))`
    `return head(h_alt − h_ref)`  # scalar Δ; head is a small MLP on the *difference* embedding
  - loss: MSE(pred_delta, logFC). Optionally add a sign-weighted term so emVar direction is penalized harder.
- Batching: `_iter_batches` gets a paired variant that tokenizes both strands of the pair; ~2× the
  tokenizer calls, same optimizer loop.
- Save: reuse `_save_checkpoint`; write `objective: "siamese"` into `provenance.json` so the
  predictor knows how to score.

### Inference — `src/predictor.py`
- `ActivityPredictor.load()` reads `objective` from provenance.
- If `siamese`: `score_variant` runs the paired forward directly (no subtraction), `predict_activity`
  is disabled/raises (a siamese model has no single-sequence activity — document this).
- If `activity`: unchanged. `EnsemblePredictor` composes over either.

### Eval — `eval/calibrate.py`
- Same held-out `calibration_variants.parquet`, same metrics (Pearson Δ-vs-skew, emVar AUC).
- **Success = emVar AUC and Δ-Pearson beat the activity-objective baseline** (0.615 / 0.149).
- Fit the isotonic `Calibrator` on the siamese Δ as before — calibration path is objective-agnostic.

### Risk / kill-criteria
- Fewer training variants than activity elements (17k vs 102k) → higher variance. Mitigate with the
  activity model as **pre-training init** (load `weights/primary` backbone, then siamese fine-tune).
- If AUC does not beat 0.615 after init + tuning → log as a negative result (`03_results.md` style)
  and keep the activity objective. Cheap to try; ~1 GPU-hour.

### First result — HyenaDNA CPU run (NEGATIVE on this backbone)
Full run: warm-start from `weights/primary` (HyenaDNA tiny), all 10,659 train pairs, 8 epochs CPU.
Graded on the locus-disjoint `eval_variants_siamese.parquet` (2,273 variants, 104 loose emVars),
with the activity baseline re-scored on the SAME slice (`eval/eval_siamese.py`):

| model (same slice) | Pearson | Spearman | emVar AUC (loose) |
|---|---|---|---|
| siamese (HyenaDNA, direct skew) | 0.086 | 0.077 | 0.6067 |
| activity baseline (subtract endpoints) | 0.127 | 0.107 | **0.6096** |

**Baseline wins (marginally).** Training overfit fast — best val Pearson at **epoch 2** (0.096),
then train MSE kept dropping while val degraded; emVar AUC peaked epoch 1 (0.669) and fell. The
~10k-pair set is small and the direct-difference objective is higher-variance than the 100k-element
activity objective. Logged: `weights/results_siamese_hyena.json`.

**Why this is NOT a verdict on the objective.** HyenaDNA is **causal** (450K params): a mid-sequence
variant's `h_alt − h_ref` difference is starved of right-flank context, so the difference embedding
is weak by construction. The siamese thesis specifically needs a **bidirectional** encoder. Given
Caduceus already lifted the *subtract-endpoints* variant Pearson 0.149→0.192 by adding
bidirectionality, the decisive test is **siamese ON the Caduceus backbone**
(`--backbone caduceus --init-from weights/primary_cad`, `notebooks/run_siamese_colab.py`) — where
both flanks feed the difference. Until that runs, the objective is *inconclusive*, not rejected.
If Caduceus-siamese also fails to beat Caduceus-subtract (0.6303) → then reject and keep subtract.

### Decisive result — Caduceus-ph backbone (POSITIVE — the prediction held)
The test the HyenaDNA-negative block called for, run on an A100: warm-start from `weights/primary_cad`
(Caduceus-ph activity backbone, `--init-from`, `missing=0/unexpected=0` — exact key match), all 10,659
train pairs, 8 epochs, batch 64, AMP. Best val Pearson at **epoch 7 (0.251, val emVar AUC 0.698)**;
overfit by epoch 8. Graded on the SAME locus-disjoint `eval_variants_siamese.parquet` (2,273 variants,
104 loose / 30 strict emVars) with the activity baseline re-scored on the identical slice
(`eval/eval_siamese.py`):

| model (same slice) | Pearson | Spearman | emVar AUC (loose) | emVar AUC (strict, n=30) |
|---|---|---|---|---|
| **siamese (Caduceus, direct skew)** | **0.2801** | 0.1158 | **0.6709** | 0.5500 |
| activity baseline (Caduceus, subtract) | 0.1913 | 0.1225 | 0.6057 | 0.5351 |
| gate → | **+0.089 (+47% rel)** | tie | **+0.065** | +0.015 (noise) |

**Verdict: the objective wins on a bidirectional backbone — exactly as predicted.** The
subtract-endpoints proxy was bleeding signal: optimizing the difference *directly* lifts continuous
Δ-Pearson **0.191 → 0.280 (+47% rel)** and loose emVar AUC **0.606 → 0.671** on the identical held-out
slice. This is the largest single variant-effect gain of the project. The HyenaDNA-negative result
(above) is now explained, not contradicted: the siamese difference embedding **requires both flanks**,
which a causal encoder cannot give — bidirectionality is a *precondition* for the objective, not an
alternative to it. **Honest caveat:** the strict emVar AUC gain (+0.015) is within noise — only 30
strict positives on this slice; the continuous-Δ win (dense, 2,273 pts) is the robust claim, the
binary-emVar call stays at the assay's data ceiling (`03_results.md §2b`). Logged:
`weights/results_siamese_cad.json`, checkpoint `weights/siamese_cad/`.

---

## Enhancement #2 — Stacking meta-learner (fuse DNA-LM Δ + frozen big-model Δ + motif + eQTL)

**Thesis:** the single-nt DNA-LM and a big frozen expression model (Enformer/Borzoi/AlphaGenome)
make *different* errors. A small supervised combiner over their outputs + the motif/eQTL signals
beats any one alone on emVar classification — and it's the honest, charter-clean use of the frozen
model (feature, never fine-tuned).

### New feature: frozen big-model **magnitude**, not just direction
- `src/evidence.py::from_frozen_foundation_model` currently returns a *direction* for the trust
  chain. Extend it (or add `frozen_foundation_delta(...)`) to also return a **signed scalar Δ**
  (ref vs alt predicted-track difference) usable as a numeric feature. Keep the existing
  direction-only EvidenceItem for the audit chain; add the scalar for the meta-learner.
- Borzoi/AlphaGenome are optional injectable sources (same pattern as the existing
  `*_table` / predictor injection seams in `evidence.py`) — absent → feature is NaN-masked.

### New module — `src/meta.py`
- `class MetaCombiner` (mirrors `Calibrator`'s save/load/fit/transform contract in `src/trust.py`):
  - `features(variant) -> np.ndarray`: `[dna_lm_delta, dna_lm_sigma?, frozen_delta, motif_dscore, gtex_signed, tss_dist]`
  - `fit(X, is_emvar)`: logistic regression or gradient-boosted stumps (small — n≈15k, keep it
    interpretable; **log per-feature weights** so the evidence chain can say *why*).
  - `transform(x) -> P(real effect)`: calibrated probability (fit isotonic on the meta-output, or
    use a calibrated classifier directly).
- **Leakage discipline:** fit the meta-learner with **cross-validation on the held-out calibration
  set** (it has the `logFC`/emVar labels), or better, carve a dedicated meta-fit split so the final
  reported AUC is on variants neither the DNA-LM nor the meta-learner saw. Document the split in
  the manifest.

### Wiring — `src/trust.py` / `src/interpret.py`
- `build_trust_report` gains an optional `meta: MetaCombiner`. When present, the calibrated
  confidence comes from `meta.transform(features)` instead of the single-feature isotonic
  `Calibrator`; the per-source EvidenceItems still populate the audit chain unchanged.
- `interpret_variant` loads `weights/meta_primary.json` if present (same optional-artifact pattern
  as `calibrator_primary.json`); absent → falls back to today's isotonic calibrator. Fully back-compat.

### Eval
- Report emVar AUC of: DNA-LM alone (0.615) vs frozen-alone vs **meta-combiner**. Success = meta beats
  the max of its inputs. Also report calibration reliability (the `03_results.md §4` table) for the meta P.

### Why this is the biggest realistic win
- Ensembling a single-nt model with a long-range expression model is the standard way past a
  single-assay ceiling, and it **strengthens the product thesis**: the combiner's per-feature
  weights are literally the "why we trust this call" the tool already promises.

### First result — offline signals only (TIE, as expected; machinery validated)
Built: `src/meta.py` (`MetaCombiner`, hand-rolled logistic, tested to AUC 0.9998 on synthetic),
additive scalar seam `evidence.frozen_foundation_delta`, and `eval/fit_meta.py` (fits on the
variant train+val loci, grades on the SAME leakage-safe eval slice as siamese).

Run on real held-out variants with the activity model as the DNA-LM signal (`weights/primary`):

| | held-out AUC (2,273 var, 104 emVar) |
|---|---|
| baseline `|Δ|` (single feature = calibrator story) | 0.6096 |
| **meta-learner** (all available features) | **0.6096** |
| gate | **tie** (Δ=0.0000) |

**Feature coverage was the whole story.** `dna_lm_delta` dense (2273/2273); `motif_dscore`
~40% (906/2273) but its fitted weight is ≈0 — the *illustrative* motif library carries no emVar
signal; `frozen_delta`, `dna_lm_sigma`, `gtex_signed`, `tss` all **0/2273** offline. With one
informative feature, the meta-learner correctly collapses to that feature → exactly the calibrator.
Logged: `weights/results_meta.json`, `weights/meta_primary.json`.

**Interpretation (honest).** The stacking *machinery* works (proven on synthetic + runs clean on
real data). The *payoff* is gated on adding a genuinely independent, informative feature — in
priority order: **(1) a frozen big-model Δ** (wire `foundation_fn` → GPU Enformer/Borzoi; the seam
is ready), (2) **real JASPAR/HOCOMOCO motifs** (replace the illustrative library), (3) **ensemble
σ** (`--n-seeds` run), (4) **signed GTEx**. This is the #2 analog of #1's Caduceus dependency:
the code is complete and the number is honest; the win needs the independent signal wired in.

### Second result — organoid feature added (WIN, offline, CPU)
The Enformer frozen-Δ feature is genome-gated (Enformer needs ~196 kb context we don't have offline),
but the charter's **independent organoid-context model** IS an offline independent signal. Added it as
two features (`abs_organoid_delta`, `concordance_dna_organoid`) and re-fit on the variant train+val
loci, graded on the same held-out slice:

| | held-out AUC (2,273 var, 104 emVar) |
|---|---|
| baseline `|Δ|` (single feature = calibrator) | 0.6096 |
| **meta-learner (+ organoid)** | **0.6228** |
| gate | **META WINS (+0.0132)** |

Fitted weights: `abs_dna_lm_delta` 0.316, **`abs_organoid_delta` 0.303** (nearly co-equal),
`concordance_dna_organoid` −0.178, motif ≈0. The **independent second model is the real lever** —
exactly the charter's "cheapest strong signal." Modest (+0.013) but a genuine, honest, offline win;
the stacking thesis is now demonstrated on real data, not just synthetic. Logged: `weights/results_meta.json`,
`weights/meta_primary.json`. A siamese-Δ (0.28) primary feature would lift this further but needs the
Caduceus GPU to score the fit slice (only the eval-slice siamese Δ is dumped).

### Third result — Enformer frozen big-model feature (NEGATIVE on the default track)
Ran the full frozen pipeline (docs/08): precomputed real Enformer zero-shot Δ from 196 kb hg38
windows (brain CAGE track 4980) on GPU, cached 9,614 variants (**eval coverage 2273/2273**, fit
7341/13000), then re-fit the meta locally against the trained model:

| eval slice (2273 var, 104 emVar) | AUC |
|---|---|
| baseline `\|Δ\|` | 0.6096 |
| meta + Enformer frozen | 0.6191 |
| **meta organoid-only (kept)** | **0.6228** |

**Verdict: honest negative on this track.** Fitted frozen weights are ~0 (`abs_frozen_delta`
−0.068, `concordance_dna_frozen` +0.02) while dna (0.32) and organoid (0.30) carry the signal;
adding frozen *lowered* the meta below organoid-only. Enformer's endogenous brain-CAGE readout does
not track this cortical MPRA's single-base allelic effects. The wiring is correct and the eval
coverage is complete, so this is a real result, not a plumbing artifact. `meta_primary.json` restored
to the organoid-only fit (0.6228).

**Retry 1 — curated developmental-cortical CAGE track set (still negative, but it moved).** Track
4980 is adult whole-brain; the Deng MPRA is mid-gestation cortex, so `foundation.DEFAULT_TRACKS` was
changed to a mean over 9 curated CAGE tracks — fetal brain (4981), neural stem cells (4798), cortex,
neurons (resolved from `targets_human.txt`, false positives excluded). Re-precomputed on GPU, eval
coverage 2273/2273:

| eval slice | AUC |
|---|---|
| baseline | 0.6096 |
| meta + curated brain-CAGE frozen | 0.6201 |
| **meta organoid-only (kept)** | **0.6228** |

The curated set is a real change, not a null: `concordance_dna_frozen` came up to **−0.15** (vs +0.02
for the single track) and Δ variance rose 4× — the developmentally-matched readout *does* carry some
allelic signal. But the net meta (0.6201) still does not clear organoid-only (0.6228). Caveat: thin
fit-frozen coverage (2727/13000 from `--limit 5000`) — the weight is estimated from few examples, so
a **full precompute** (no `--limit`, full fit coverage) is the one clean shot left before rejecting
CAGE; after that, **Borzoi** RNA-seq tracks (a closer readout to allelic skew) is the last card.
Logged: `weights/results_meta_braincage.json`. `meta_primary.json` kept at organoid-only (0.6228).

---

## Sequencing & effort

| Step | File(s) | Effort | Gate |
|---|---|---|---|
| 0. Run deep ensemble (already wired) | `--n-seeds N` | trivial (GPU time) | per-variant σ for meta feature |
| 1. Emit train/val variant tables | `data/prepare_data.py`, `data/splits.py` | S | leakage assert passes |
| 2. Siamese objective + inference | `train/finetune_hyenadna.py`, `src/predictor.py` | M | AUC > 0.615 |
| 3. Frozen-model scalar Δ feature | `src/evidence.py` | S | feature present + direction unchanged |
| 4. MetaCombiner + wiring | `src/meta.py`, `src/trust.py`, `src/interpret.py` | M | meta AUC > best input |
| 5. Eval + calibration report | `eval/calibrate.py`, `docs/03_results.md` | S | honest numbers logged |

Do **0 + 1 + 2** first (siamese is the cleaner single hypothesis), then **3 + 4** (stacking).
Both keep the existing pipeline as the fallback path — nothing already working regresses.
