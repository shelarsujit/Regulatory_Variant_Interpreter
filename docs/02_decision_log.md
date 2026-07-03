# Decision log — "the theory of everything we do"

Append-only record of every non-trivial engineering/scientific decision, with the
reasoning and the alternatives we rejected. Newest decisions go at the bottom. When a
decision changes, add a new entry that supersedes the old one (don't edit history).

Format per entry: **Context → Decision → Why → Rejected alternatives → Status.**

---

## D1 — Frame the problem as *trust*, not prediction
- **Context.** Regulatory variants are mostly filed as VUS because no tool gives a
  usable call. A raw activity number from a neural net is not clinically actionable.
- **Decision.** Build a *trust-first* interpreter: the deliverable is a **calibrated
  confidence + auditable evidence chain**, not a bare score. Predicting activity is the
  means; making the prediction trustworthy is the product.
- **Why.** It's the real unmet need and the defensible differentiator; it's also what a
  curator (our named user) actually requires to act.
- **Rejected.** "Just ship the best activity predictor." — commoditized, and unusable in
  the clinic without calibration/grounding.
- **Status.** Locked (charter).

## D2 — Backbone: HyenaDNA, single-nucleotide, `hyenadna-tiny-1k-seqlen`
- **Context.** The method scores single-base changes; we have one week and one A100.
- **Decision.** Fine-tune **HyenaDNA** (single-nucleotide tokenization, sub-quadratic),
  starting from **`LongSafari/hyenadna-tiny-1k-seqlen`** (1 kb context ≫ ~200 bp
  elements). Upgrade path: **Caduceus** (RC-equivariant) if we converge early.
- **Why.** Single-base resolution is essential — saturation mutagenesis changes one
  letter, and it must be crisply visible to the model. HyenaDNA is small/fast enough to
  fine-tune in the time/hardware budget.
- **Rejected.** (a) **Nucleotide Transformer / DNABERT-2** — k-mer / byte-pair
  tokenization blurs a one-base change. (b) **Fine-tuning Enformer / AlphaGenome** —
  excellent but too heavy to responsibly fine-tune in 7 days; instead used **frozen** as
  independent evidence (see D7).
- **Status.** Locked.

## D3 — Data source of truth: the *processed* Science supplement (raw Synapse = fallback)
- **Context.** Data exists in two tiers: raw FASTQs on Synapse (behind a PsychENCODE DUA
  + auth token) and processed activity/variant tables in the *Science* supplement. See
  [`01_data_provenance.md`](01_data_provenance.md).
- **Decision.** Build from the **processed supplement**. Treat raw Synapse as a
  fallback only, to be pursued *in parallel* solely if processed granularity proves
  insufficient.
- **Why.** The raw tier needs (a) DUA approval latency and (b) a full MPRAflow
  reprocessing pipeline before you even have a label — both fatal to a one-week build.
  The processed tables hand us `(sequence, activity)` and `(variant, skew)` directly.
- **Rejected.** Starting from raw FASTQs; starting from GEO (the data isn't in GEO).
- **Status.** Locked (user-confirmed).

## D4 — Targets: two *separately* fine-tuned single-context models
- **Context.** The MPRA reports activity in **two** contexts: primary cortex and
  organoid. Options: (A) primary-only, (B) one shared-trunk multitask model, (C) two
  separate single-context models.
- **Decision.** **(C)** — a primary-cortex model is **the call**; a separately trained
  organoid model is an **independent second opinion** in the trust layer.
- **Why.** Trust is the judged axis (D1), and independence is its currency. Two disjoint
  models give a genuine model-vs-model agreement signal *inside the dataset*, forming an
  independence ladder: primary-cortex → organoid (near-independent) → frozen foundation
  model (most independent) → databases. Primary accuracy is uncompromised (single-task on
  the richest context). Cost is one extra cheap fine-tune (one context flag).
- **Rejected.** **(B) multitask** — a marginal primary-accuracy gain (the cortex label
  set is already large) bought by *destroying* organoid's independence (shared trunk →
  circular agreement). **(A) primary-only** — wastes a free independent signal.
- **Status.** Locked (user-confirmed).

## D5 — Leakage discipline: locus-grouped split, asserted in code
- **Context.** A variant's reference sequence *is* a library element with one letter
  changed; overlapping tiles share most bases. Random row splits would leak calibration
  sequences into training and inflate confidence.
- **Decision.** Split by **genomic locus** (binned region, with ±1-bin neighbor guard):
  an entire locus goes to training *or* to calibration, never both. All variant-bearing
  loci are held out of training. Disjointness is **asserted** at the end of
  `prepare_data.py`; the run aborts if it fails. Implemented in `data/splits.py`.
- **Why.** It's the only honest way to calibrate — the whole trust claim collapses if the
  model was trained on what we grade it against.
- **Rejected.** Random/stratified row splits; sequence-identity dedup alone (misses
  overlapping non-identical tiles).
- **Status.** Locked.

## D6 — Calibration strategy: continuous-first, classification as a check
- **Context.** ~17,069 measured variants, but only **164** significant (~1% positive) —
  heavy class imbalance.
- **Decision.** Calibrate primarily on the **continuous** predicted-Δ vs measured-skew
  relationship (isotonic / Platt regression); use the 164 emVars as a **classification**
  reliability check, not as the sole calibration target.
- **Why.** 1% positives make a classification-only calibration noisy and unstable; the
  continuous skew uses all 17k rows and is the richer signal.
- **Rejected.** Calibrating on the 164 emVars alone.
- **Status.** Locked (revisit once real effect-size distribution is seen).

## D7 — Frozen foundation model as an independent evidence source
- **Context.** We need an *outside* opinion uncorrelated with our fine-tune.
- **Decision.** Use **Enformer or AlphaGenome zero-shot (frozen)** in `evidence.py` as one
  grounding signal — never trained by us.
- **Why.** A different model class trained on different data is a strong independence axis
  for the trust layer (top of the D4 ladder). Fine-tuning it ourselves (D2) is off the
  table; freezing it keeps it genuinely independent.
- **Rejected.** Fine-tuning it (cost + destroys independence); omitting it (weaker
  grounding).
- **Status.** Locked in principle; exact choice (enformer-pytorch vs AlphaGenome API)
  pinned in Phase 3.

## D8 — `prepare_data.py` engineering: fuzzy resolver + synthetic self-test + manifest
- **Context.** Exact supplement column names / table numbers are unverified from the
  sandbox, and the real files can't be downloaded here. Day 1 still needs a *runnable*
  data-prep script.
- **Decision.** (1) A **fuzzy column resolver** maps logical fields to real headers by
  case-insensitive substring match, overridable, recorded in `manifest.json`. (2) A
  **`--synthetic`** mode generates a small MPRA-like fixture so the full pipeline
  (build → locus split → leakage assertion → outputs) runs *today* with zero downloads.
  (3) Every run writes a **`manifest.json`** with params, chosen mapping, row counts,
  git commit, and the leakage-check result (provenance = part of the trust story).
- **Why.** Makes the script genuinely runnable now, resilient to the ⚠️ unknowns, and
  auditable — without pretending we have data we don't.
- **Rejected.** Hard-coding column names (brittle); waiting for the download before
  writing any code (blocks Day 1).
- **Status.** Locked; implemented.

## D9 — Interface contract frozen early (`src/schema.py`)
- **Context.** Many modules (predictor, evidence, motifs, trust, app) must agree on one
  return type.
- **Decision.** Freeze `interpret_variant(chrom, pos, ref, alt, build="hg38") ->
  Interpretation` now, with `Interpretation / TrustReport / EvidenceItem / Mechanism`
  dataclasses in `src/schema.py`. Modules are stubs, but the *shape* is fixed.
- **Why.** Lets the stubs be filled independently and in parallel without churn; the
  schema encodes the trust thesis (confidence + conflicts + evidence chain + provenance
  are first-class fields).
- **Rejected.** Letting each module invent its own return shape.
- **Status.** Locked; implemented.
