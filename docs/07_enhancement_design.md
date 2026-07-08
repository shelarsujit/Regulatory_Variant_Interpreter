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
