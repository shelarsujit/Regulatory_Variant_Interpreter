# Preprint skeleton — methods note

**Working title:**
**The objective, not the architecture, gates single-base variant-effect prediction with DNA language models**

*Alt titles (pick on submission):*
- *Optimize the difference: a direct-skew siamese objective for MPRA variant-effect prediction*
- *Bidirectionality is a precondition, not a fix: what actually moves DNA-LM variant-effect prediction*

**Target venue (in order):** bioRxiv (v1, priority) → MLCB or LMRL (ICLR) 4-page findings → *Bioinformatics* Application Note / *NAR Genomics & Bioinformatics* (only after multi-dataset generalization, see §Limitations).

**Author(s):** Sujit Shelar. **Code/data:** this repo (checkpoint hashes + manifests in `weights/*/provenance.json`, `data/processed/manifest.json`).

---

## Abstract (draft, ~150 words)

DNA language models (DNA-LMs) predict regulatory-element activity well but predict the effect of a
*single-base* variant poorly — the metric that matters clinically. Using the Deng et al. (2024)
cortical lentiMPRA, we isolate *why*. On a locus-disjoint held-out set we show the element→variant
performance gap (Pearson 0.70→0.15) is **not** a capacity problem: doubling model size leaves
variant-effect correlation flat. Two levers move it. (1) **Architecture:** a bidirectional backbone
(Caduceus-ph) lifts variant-effect Pearson +33% relative over a causal one (HyenaDNA), by reading
both flanks of a mid-sequence variant on the same strand. (2) **Objective:** training the ref/alt
*difference* directly with a shared-weight siamese head — rather than subtracting two independently
predicted activities — lifts it a further +47% relative (0.19→0.28), the single largest gain. The
two compound and are *dependent*: the siamese objective **fails** on a causal backbone and **wins**
on a bidirectional one, because the difference embedding requires both flanks. Bidirectionality is a
precondition for the objective, not an alternative to it. The binary emVar-significance call, by
contrast, stays at the assay's noise ceiling under every configuration — architecture and objective
buy effect-*magnitude* fidelity, not the yes/no call. We release calibrated confidences throughout.

---

## 1. Introduction (½ page)

- Non-coding regulatory VUS are the frontier where variant interpretation fails; effect = "changes
  *how much* a gene is expressed," requiring a model that reads raw DNA. (cite: regulatory VUS burden)
- DNA-LMs (HyenaDNA, Caduceus, Nucleotide Transformer) fit element activity; single-base
  variant-effect is the open hard sub-problem. (cite backbones + Deng 2024 MPRA)
- **Gap in the literature:** most DNA-LM variant scoring uses *subtract-endpoints* Δ =
  activity(alt) − activity(ref), from a model never trained on the difference. It is unclear whether
  the weak variant-effect signal is a *capacity*, *architecture*, or *objective* limit.
- **Contribution.** A controlled 3-way ablation (capacity / architecture / objective) on one MPRA,
  with leakage-safe locus-disjoint splits, that (i) rules out capacity, (ii) quantifies the
  bidirectional-backbone lever, (iii) shows a direct-skew siamese objective is the largest lever,
  and (iv) demonstrates the objective *depends on* bidirectionality (fails causal, wins bidirectional)
  — a precondition relationship not previously isolated. (v) Shows the binary significance call is
  assay-ceiling-limited regardless.

## 2. Data & splits (½ page)

- **Source:** Deng et al. 2024 *Science* (adh0559) processed supplement; cortical lentiMPRA element
  activity + variant allelic skew (`logFC`). Provenance in `docs/01_data_provenance.md`.
- **Pipeline validation:** reconstructed strict active-gated emVar count **163 ≈ paper's 164** —
  independent end-to-end validation of hg38 reconstruction + dbSNP allele resolution + emVar
  definition (FDR<10% ∧ ≥1-allele-active). (Fig. 1a inset / Table S1)
- **Leakage discipline (central):** locus-grouped split (`data/splits.py`); no training sequence
  overlaps any calibration/eval variant locus; asserted programmatically. Variant pairs
  (`data/make_variant_pairs.py`) inherit the same grouping. State exact n: 10,659 train pairs;
  held-out eval slice `eval_variants_siamese.parquet` = 2,273 variants (104 loose / 30 strict emVars);
  activity calibration set = 15,273 variants.

## 3. Methods (¾ page)

- **Backbones.** HyenaDNA (causal, single-nt: `tiny-1k` d=128/1.6M, `small-32k` d=256/3.3M);
  Caduceus-ph (Mamba, **bidirectional, non-RC-equivariant** `-ph`, d=256/7.7M). Justify `-ph` not
  `-ps`: MPRA activity is orientation-specific, so RC-*equivariance* is a mismatched constraint
  (§ negative-result, RC-averaging hurts).
- **Activity objective (baseline).** Regress normalized RNA/DNA per element; score variant as
  Δ = activity(alt) − activity(ref). (`src/predictor.py`)
- **Siamese direct-skew objective (this work).** Shared-weight backbone encodes ref and alt; small
  MLP head reads the *difference embedding* `h_alt − h_ref` and regresses measured `logFC`; loss =
  MSE(pred_Δ, logFC). Warm-started from the activity backbone (exact key match, missing=0/unexpected=0).
  (`train/finetune_siamese.py`)
- **Calibration.** Isotonic calibrator |Δ|→P(real effect), strict emVar label. (`src/trust.py`)
- **Metrics.** Continuous: Pearson/Spearman of predicted Δ vs measured skew. Binary: emVar AUC
  (loose FDR≤0.10; strict active-gated). All on the identical held-out slice; baseline re-scored on
  the same slice so any delta is the manipulated variable, not a test-set change.

## 4. Results (1 page)

**R1 — Element activity is easy; single-base variant-effect is hard (Fig. 1).**
Primary-cortex element val Pearson **0.70** (organoid 0.69); variant-effect subtract-endpoints
Pearson **0.15**. The 0.70→0.15 gap is the phenomenon under study.

**R2 — Capacity is not the lever (Fig. 2a).**
`tiny-1k`→`small-32k` (2× params): element +0.017, **variant-effect flat/slightly worse** (0.149→0.145);
strict emVar AUC +0.013. Rules out capacity → the gap is architecture or objective.

**R3 — Bidirectionality lifts variant-effect +33% rel (Fig. 2b).**
HyenaDNA(causal)→Caduceus-ph(bidirectional): variant Pearson **0.145→0.192 (+33% rel)**, Spearman
+33% rel; strict emVar AUC flat (~0.63). Reading both flanks on the same forward strand recovers
signal a causal model structurally cannot see for a mid-sequence variant.

**R4 — Direct-skew siamese objective is the largest lever, +47% rel (Fig. 3, headline).**
On Caduceus-ph, same held-out slice: **subtract-endpoints 0.191 → siamese 0.280 Pearson (+47% rel)**;
loose emVar AUC **0.606→0.671**. Larger than the architecture jump. Optimizing the difference
directly stops the signal bleed of the indirect subtract-endpoints proxy.

**R5 — The objective DEPENDS on bidirectionality (Fig. 3, the mechanistic result).**
Same siamese objective on *causal* HyenaDNA **loses** to its own subtract baseline (0.086 vs 0.127).
On *bidirectional* Caduceus it **wins** (0.280 vs 0.191). The difference embedding `h_alt − h_ref`
requires both flanks; a causal encoder starves it. **Bidirectionality is a precondition, not an
alternative.** This is the paper's distinctive claim.

**R6 — The binary significance call is assay-ceiling-limited (Fig. 4).**
Strict emVar AUC stays ~0.63 across capacity, architecture, and objective changes; the +0.015
siamese gain is within noise (30 strict positives). Architecture and objective buy *magnitude*
fidelity, not the yes/no emVar decision — that ceiling is the assay's single-base noise, not model
capacity. (Report bootstrap CI here — see Limitations.)

**R7 — Confidences are calibrated (Fig. 5).**
Isotonic reliability: bulk bin predicts 1.0%, observes 1.0%; τ=0.5 label rate 1.1% ≈ paper emVar
rate 1.07%. The tool does not claim confidence it lacks.

**Negative results (report — they strengthen the ablation):**
- Test-time RC-averaging **hurts** (0.149→0.113) — MPRA is orientation-specific; RC is OOD.
  Argues against RC-*equivariant* `caduceus-ps`.
- Lower calibration τ does not change ranking (emVar AUC identical 0.6153 at every τ); only
  de-aligns probabilities. Keep τ=0.5.

### Figure list
1. **Fig 1** — element-activity scatter (r≈0.70) vs variant-effect scatter (r≈0.15); the gap. Inset: 163≈164 emVar reproduction.
2. **Fig 2** — bar/forest: (a) capacity A/B flat; (b) causal→bidirectional +33%. Variant Pearson, same slice.
3. **Fig 3** — headline: 2×2 {causal, bidirectional} × {subtract, siamese}. Shows siamese wins only on bidirectional (R4+R5 in one panel). **Make this the graphical abstract.**
4. **Fig 4** — emVar AUC across all configs staying flat at the ceiling, with bootstrap CIs.
5. **Fig 5** — calibration reliability diagram.
- **Table 1** — full metric matrix (all configs × {Pearson, Spearman, loose AUC, strict AUC}), from `weights/results_*.json`.

## 5. Discussion

- Practical guidance for DNA-LM practitioners: **don't scale — change the objective, on a
  bidirectional backbone.** Ordered levers: objective (siamese) > architecture (bidirectional) >
  size (flat).
- Why the binary call resists: episodic single-base MPRA noise sets a discrimination ceiling no
  reformulation of a single-assay model escapes here; the honest product move is calibration +
  ensembling across independent signals (organoid second model +0.013 AUC; frozen Enformer CAGE =
  documented null — see `docs/07 §Enh-2`).

## 6. Limitations (write honestly — this is what reviewers hit first)

1. **Single MPRA.** One dataset (Deng cortex). Generalization to lentiMPRA/SuRE/other tissues untested
   → the workshop/preprint ceiling. Multi-dataset replication is the journal gate.
2. **CIs.** Headline Δ-Pearson (0.19→0.28) needs a paired bootstrap CI / permutation test, not a point
   estimate. Strict-emVar gains already flagged as within-noise; extend rigor to the headline. **(TODO
   before submission — 1000× bootstrap, mirror ncypher-style rigor.)**
3. **Small models.** ≤7.7M params. Claims are about *objective/architecture at fixed small scale*, not
   a scaling law. State scope explicitly.
4. **Baseline scope.** Compared against own subtract-endpoints Δ. Add ≥1 external supervised VEP
   baseline (Sei/Enformer-delta/a caQTL model) so "siamese beats subtract" isn't only self-referential.

## 7. Reproducibility

Every run logs `weights/<ctx>/provenance.json` (checkpoint hash, data version), manifests, and a
`results_*.json`. Commands in `docs/03_results.md §9`. One-command re-score of any config.

---

## Pre-submission checklist (concrete, in order)

- [ ] **Bootstrap CIs** on R4 headline (1000×, paired, same slice) + R6 — `eval/eval_siamese.py` extension.
- [ ] **One external VEP baseline** on the same slice (Sei or Enformer-delta) → Table 1 row.
- [ ] Regenerate Figs 1–5 from `weights/results_*.json` (Fig 3 = graphical abstract).
- [ ] Prose pass on Abstract + R5 (the precondition claim is the novelty — lead with it).
- [ ] Author/affiliation, data-availability, code-availability (repo + commit hash), license.
- [ ] Post bioRxiv v1 → get DOI → submit same PDF to next MLCB/LMRL deadline.
- [ ] *(Journal only)* add ≥1 more MPRA dataset (Agarwal/Shendure lentiMPRA, Kircher SuRE) → R2–R5 replication.
