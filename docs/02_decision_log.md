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

## D10 — Motif engine: offline-first, PWM math in-house
- **Context.** The mechanism layer needs JASPAR PWMs, but pyjaspar downloads a DB at
  runtime and the build sandbox blocks most egress — and we want the module testable now.
- **Decision.** Implement the PWM scoring math in `src/motifs.py` with **numpy only**, ship
  a small set of **illustrative consensus motifs** (AP-1, CRE, GC-box, TATA, E-box, CCAAT)
  for offline use, and provide `load_jaspar_pfms()` / a pyjaspar path to swap in the real
  database for production. Only motif windows that **overlap the variant** are compared
  between alleles, so a strong site elsewhere can't mask or fake the local effect.
- **Why.** Runs and is unit-tested today; production simply swaps the motif library.
- **Rejected.** Hard-depending on pyjaspar (network + heavy); scanning whole windows
  without the overlap constraint (would leak non-local sites into the Δ).
- **Status.** Implemented + tested (`tests/test_core.py`: gain and loss both detected).

## D11 — Trust calibration: isotonic (PAVA) on a continuous-derived label; log-odds aggregation
- **Context.** Need honest confidence from ~17k variants that are only ~1% "significant."
- **Decision.** `src/trust.Calibrator` fits a **self-contained isotonic regression
  (pool-adjacent-violators, numpy only)** mapping |predicted Δ| -> P(real effect), where the
  label is the **continuous** `|measured_skew| >= tau` (D6), not the sparse significance
  flag (kept only as a secondary AUC check). `build_trust_report` then combines calibrated
  model confidence with evidence **in log-odds space** — concordant sources add, conflicts
  subtract — and **always lists conflicts explicitly**.
- **Why.** Continuous label uses all rows and is stable under class imbalance; log-odds
  aggregation is monotone, bounded, and keeps conflict visible instead of averaging it away.
  No heavy dependency, so it ships in the demo and is easy to test.
- **Rejected.** sklearn IsotonicRegression (unnecessary dependency for ~15 lines of PAVA);
  calibrating on the 164 emVars alone; averaging evidence into one opaque score.
- **Status.** Implemented + tested (monotone; AUC≈0.87 on the simulated fixture; agreement
  beats conflict and conflict is surfaced). `interpret_variant` is now wired to run motifs +
  organoid-agreement + calibrated trust live, given sequences and a model Δ.

## D12 — Predictor: HyenaDNA `-hf` mirror + in-house char tokenizer + always-loadable head
- **Context.** `src/predictor.py` must load the D2 backbone and score variants by saturation
  mutagenesis, on a Windows/CPU dev box, before the Phase-2 fine-tune exists. Two blockers
  surfaced on first live load: (a) the pinned repo `LongSafari/hyenadna-tiny-1k-seqlen` ships
  **no HF `model_type`/`auto_map`**, so `AutoModel.from_pretrained` can't build it — it expects
  the repo's bundled `standalone_hyenadna.py`; (b) HyenaDNA ships **no fast tokenizer**, and
  transformers 5.x's slow→fast conversion demands `sentencepiece` it doesn't provide.
- **Decision.** (1) Load the backbone from **`LongSafari/hyenadna-tiny-1k-seqlen-hf`** —
  LongSafari's official HF-loadable mirror of the *same weights* (adds `configuration_hyena.py`
  / `modeling_hyena.py`, `model_type=hyenadna`). (2) Tokenize with an **in-house
  `HyenaDNACharTokenizer`** built on HyenaDNA's exact character vocab (A=7,C=8,G=9,T=10,N=11),
  killing the `sentencepiece`/AutoTokenizer dependency and guaranteeing identical tokenization
  between this scorer and the trainer. (3) Architecture = **backbone + `Linear(hidden,1)`
  regression head** matching `finetune_hyenadna.py`; `load()` **always succeeds** — with a
  fine-tuned `weights/<context>/model.pt` (`{"backbone":state_dict,"head":state_dict,"meta":…}`)
  it rebuilds the base backbone then `load_state_dict`s both and sets `is_finetuned=True`,
  otherwise it loads a random head so the whole pipeline runs pre-training (Δ numbers are then
  *not* meaningful and the trust layer must treat the primary model as uncalibrated).
  NOTE: the fine-tuned backbone is persisted as a **torch state_dict, not `save_pretrained`** —
  HyenaDNA reuses one `freq` buffer across filter layers (shared tensors) and transformers 5.x's
  `save_pretrained` rejects shared tensors unconditionally; `torch.save` preserves shared storage.
- **Why.** Same weights as the charter pin, just the loadable packaging; own tokenizer removes
  a fragile heavy dependency and locks train/inference parity; an always-loadable head keeps the
  end-to-end demo runnable today (matches the D8/D10 "runnable now, swap in the real thing"
  discipline). Torch/transformers are imported **lazily** inside methods so `import predictor`
  stays cheap and download-free (the test suite imports it but never calls `load()`).
- **Rejected.** (a) Patching the base repo's config to add `model_type` (brittle fork of an
  upstream checkpoint); (b) installing `sentencepiece` + `AutoTokenizer` (heavyweight, and still
  wouldn't fix the missing `model_type`); (c) making `load()` raise until Phase-2 weights exist
  (blocks the live end-to-end path the schema/trust work already unblocked).
- **Status.** Implemented; live-tested (loads on CPU, `predict_activity` / `score_variant` /
  `saturation_mutagenesis` all run, ISM ref-column is zero by construction). Head is untrained
  until Phase 2 (`train/finetune_hyenadna.py`).

## D13 — Grounding evidence: offline-first, real held-out MPRA now, external sources swappable
- **Context.** `src/evidence.py` gathers the independent signals the trust layer grounds on
  (held-out MPRA, GTEx eQTL, ClinVar, TSS, frozen foundation model, organoid). Most need
  network APIs / large annotation files unavailable in the build sandbox, but the layer must be
  runnable and testable now (same constraint as D8/D10).
- **Decision.** Implement `gather_evidence` **offline-first**: the **held-out MPRA** source is
  wired for real against `data/processed/calibration_variants.parquet` (matches the variant by
  `chrom:pos:ref>alt`, returns the measured `skew` direction + FDR, marks `concordant` vs the
  model — this is a genuine wet-lab measurement, independent of our model). Every external source
  (`GTEx`, `ClinVar`, `TSS`, `frozen_foundation_model`) takes an **injectable local table or
  callable** and returns `None` when its resource is absent — **a missing source is omitted, not
  a conflict**. Direction-less sources (TSS proximity, ClinVar class without a sign) attach as
  `concordant=None` context. Per-source **weights** encode the independence ladder (held-out MPRA
  strongest).
- **Why.** Ships a real, high-value evidence source today with zero downloads, keeps the trust
  aggregation honest (absence ≠ disagreement), and leaves a clean seam to drop in the production
  GTEx/ClinVar/Enformer resources without touching the trust layer.
- **Rejected.** Fabricating placeholder evidence values (would poison calibration); hard-wiring
  network calls (unrunnable + untestable in the sandbox); raising `NotImplementedError` (blocks
  the live `interpret_variant` path).
- **Status.** Implemented; held-out MPRA live-tested against the synthetic calibration set.

## D14 — Real Deng ingest: hg38 reconstruction + dbSNP alleles (the supplement has neither)
- **Context.** The downloaded Science adh0559 supplement (Data S1 element activity, Data S2
  variant ratios) does **not** match the generic `load_real` assumptions. Confirmed on first
  open: **(a)** there is **no `sequence` column** anywhere — only coordinates; **(b)** element
  windows are **270 bp** (`insert_end - insert_start`), not 200; **(c)** primary and organoid
  activity live in **separate sheets** (`Primary` / `Organoids`), joined on `insert_name`;
  **(d)** the variant table (S2) carries **no `ref`/`alt` alleles** — only an `rsid`,
  `variant_pos`, `logFC`, `adj.P.Val`. So `sequence`, `ref`, and `alt` must all be *derived*.
- **Decision.** Add a Deng-specific loader (`data/load_deng.py`) + two reusable helpers,
  selected by `python data/prepare_data.py --deng-dir <supplement> --genome <hg38.fa>`:
  * **`data/genome.py`** — pyfaidx wrapper; reconstructs each 270 bp element and each variant's
    ref window from a local **hg38 FASTA** (BED 0-based, half-open). Reusable by `interpret.py`
    for its still-stubbed genome-window step.
  * **`data/dbsnp.py`** — resolves each S2 `rsid` → `ref`/`alt` via **myvariant.info** (batched
    ≤1000/req, ~16 requests for the full set) and **disk-cached** (`data/raw/dbsnp_cache.json`).
    Multi-allelic sites keep all alts (first alt tested). (First tried Ensembl REST — too flaky at
    15k scale; see the addendum below.)
  * **`load_deng.py`** — joins S1 sheets, reconstructs sequences, builds `seq_alt` by swapping the
    variant base, maps `logFC → measured_skew`, `adj.P.Val → fdr`. Rescues `-`-strand variants by
    reverse-complementing when the hg38 base matches `revcomp(ref)`; **drops** rows with no allele,
    a ref/hg38 mismatch, or an out-of-window position, and reports every count. A sampled
    **ref-allele concordance check** guards against the wrong genome build/contig naming (warns if
    < 90%). Output DataFrames match the `splits`/leakage/manifest schema, so that machinery is
    reused unchanged.
- **Why.** hg38 + dbSNP is the standard, reproducible way to recover sequences the supplement
  omits; local FASTA handles the 46k elements in bulk (46k REST calls would take hours) while
  myvariant.info handles the ~15k rsIDs once (cached). The concordance guard turns a silent
  wrong-genome error into a loud one — trust discipline (D1) applied to data prep.
- **emVar caveat.** `adj.P.Val ≤ 0.10` yields **600** significant variants, not the paper's
  **164** (the paper's emVar definition is stricter). Not blocking: calibration is
  continuous-first (D6); `is_emvar` is only the secondary AUC check. Threshold is `--emvar-fdr`.
- **Rejected.** Fetching all element sequences from Ensembl (too many requests); guessing ref/alt
  from the reference base alone (can't know the tested alt); hard-coding the 0/1-based convention
  (detected/guarded instead). Raw Synapse oligo library (would give exact sequences but needs the
  PsychENCODE DUA — deferred, D3).
- **Status.** Implemented; genome + dbSNP + full `load_deng` orchestration integration-tested
  offline (synthetic FASTA + mini S1/S2 + seeded cache: element seq == FASTA slice, `seq_alt`
  differs at exactly the variant base, drops/emVar correct). Awaits a real hg38 FASTA download to
  run on the full supplement.
- **Addendum (real run — allele source switched to myvariant.info).** First attempt used the
  Ensembl REST variation endpoint; at 15k scale it is unusable — 200-id batches time out and it
  intermittently returns HTTP 500, so a run crawls in backoff (14% after ~15 min). Switched
  `dbsnp.py` to **myvariant.info** (`/v1/query`, `scopes=dbsnp.rsid`): ~1000 IDs per POST, the
  full ~15k in ~16 requests / ~1 min. It returns one row per allele, grouped back by rsID (ref
  shared, all alts collected); its `_id` is hg19-anchored but we take only the assembly-invariant
  allele LETTERS — positioning uses Deng's hg38 `variant_pos`, and the ref-vs-hg38 concordance
  check (+ revcomp rescue) catches the rare hg19/hg38 disagreement. Kept the robustness scaffolding
  (backoff on 429/5xx/timeout, multi-sweep requeue, per-batch cache checkpoint).
  **Real-run result:** 15,324/15,335 alleles resolved; **coordinate concordance 99.875%** (hg38
  build + contig naming confirmed); 15,273 calibration variants kept (drops: 51 ref-mismatch, 11
  no-allele; 10 revcomp-rescued; 0 out-of-window); train/val 32,964/3,647; leakage PASS.
  *Gotcha logged:* a background copy of the earlier Ensembl run wrote its partial output AFTER the
  good run and clobbered it (2,593 rows on disk vs 15,273 in console) — never leave a superseded
  data-prep process alive; confirm no rogue writer before trusting `data/processed/`.

## D15 — Orchestrator genome wiring + Gradio app: graceful degradation over hard requirements
- **Context.** `interpret.py` still stubbed genome-window extraction, and `app/app.py` was a
  skeleton. Both must be usable *before* the Phase-2 fine-tune and *without* forcing every user to
  have a 3 GB hg38 FASTA on hand.
- **Decision.** (1) `interpret_variant` gains a `genome=` argument that accepts **either a Genome
  instance or an hg38 FASTA path** (duck-typed on `.window()`), reusing `data/genome.py`; it
  centers a `window_len`-bp window (default = predictor `seq_len` or 270) on `pos`, builds
  `seq_alt` by swapping the variant base, and records a **`genome_ref_match`** sanity flag (+ a
  warning when the reference base ≠ `ref`) in provenance — never a hard failure, since the caller
  asserts the alleles. (2) The app degrades gracefully at every missing dependency: no weights →
  model still loads (untrained head) behind an explicit banner; no genome → the **Paste sequences**
  tab still runs the full pipeline; no calibration table → trust uses its documented uncalibrated
  heuristic. Conflicts render in their own ⚠️ section (working agreement #2: never hide them).
- **Why.** Keeps the whole tool demoable and testable today (matches D8/D10/D12/D13 discipline),
  and makes the trust story the centerpiece of the UI rather than a bare score.
- **Rejected.** Requiring a fine-tuned checkpoint or a genome before the app will start (kills the
  demo); silently substituting zeros for a missing model (dishonest — the banner says so instead).
- **Status.** Implemented; both interpret paths + both app modes headless-tested (genome window
  extraction, ref-mismatch warning, AP-1 motif-loss detection, full trust render); 6/6 core tests
  still pass.

## D16 — Backbone abstraction: Caduceus-ready without touching the HyenaDNA pipeline
- **Context.** HyenaDNA is causal (a mid-sequence variant's pooled representation is dominated by
  left context) and not reverse-complement aware. **Caduceus** (kuleshov-group; Mamba-based,
  bidirectional, RC-equivariant) is the charter's named upgrade (D2) and should help variant-effect
  scoring. Requirement: make it a drop-in `--backbone`/`--checkpoint` swap **without changing the
  working, CPU-testable HyenaDNA path**.
- **Decision.** Add a small **backbone registry** in `predictor.py` (`BACKBONES` → default
  checkpoint + char tokenizer) plus `build_tokenizer(model_type)` / `default_checkpoint_for()`.
  `ActivityPredictor.load()` now loads the backbone first, reads `config.model_type`, and selects
  the tokenizer via the factory — for HyenaDNA this returns the **same** `HyenaDNACharTokenizer`,
  so behavior is byte-identical. `finetune_hyenadna.py` gains `--backbone {hyenadna,caduceus}` (a
  bare `--checkpoint` still overrides) and tokenizes via the same factory. The rest of the
  predictor/trainer is *already* backbone-agnostic — we tokenize ourselves and persist raw
  state_dicts (not `save_pretrained`, D12) — so nothing else changed.
- **Key finding.** Caduceus's tokenizer is **byte-identical to HyenaDNA's** — both descend from the
  same `CharacterTokenizer` (verified against `caduceus/tokenization_caduceus.py`:
  `[CLS]0 [SEP]1 [BOS]2 [MASK]3 [PAD]4 [RESERVED]5 [UNK]6 A7 C8 G9 T10 N11`). So `CaduceusCharTokenizer`
  subclasses `HyenaDNACharTokenizer` unchanged — kept as a distinct, labelable type (provenance +
  a seam for future SEP/RC handling), not a re-implementation. `backbone_type` is now recorded in
  predictor + checkpoint provenance.
- **Two caveats, documented not hidden.** (1) Caduceus needs **`mamba-ssm` + `causal-conv1d`**,
  which build **CUDA-only** — it trains/loads on a GPU box only, never CPU/Windows; requirements
  lists them commented + GPU-only, and the HyenaDNA path stays fully CPU-testable. (2) The `-ps`
  (parameter-sharing, RC-equivariant) variant structures its hidden channels for RC — verify the
  mean-pool + `Linear` head on GPU (or use `CaduceusForSequenceClassification`'s pooling) before
  trusting its Δ. Both are flagged in `BACKBONES` and here.
- **Why.** A registry + config-driven tokenizer is the minimal seam that makes Caduceus a one-flag
  swap while guaranteeing the HyenaDNA pipeline is untouched (it re-selects the identical
  tokenizer). No premature Caduceus code we can't test on CPU — just the plug points, ready.
- **Rejected.** Renaming/replacing `HyenaDNACharTokenizer` (would perturb the working path);
  hardcoding a guessed Caduceus vocab (silent embedding corruption — verified from source instead);
  pulling `mamba-ssm` into base requirements (breaks every CPU/Windows install).
- **Status.** Implemented; HyenaDNA verified byte-identical (auto-detects `backbone_type=hyenadna`,
  same tokenizer, load/predict/ISM/finetune all unchanged; 6/6 tests pass). Caduceus plumbing in
  place; actual Caduceus load/train awaits a CUDA GPU (Colab) — first run must confirm the pooling
  and transformers×remote-code compatibility.
- **Addendum (variant chosen: `-ph`, not `-ps`).** The RC-averaging negative result (docs/03 §6)
  showed MPRA activity is orientation-specific, so an RC-*equivariant* backbone is mismatched.
  The `BACKBONES` default is now **`kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16`**
  — the **post-hoc (ph)** variant. Confirmed from its HF config via the Hugging Face Claude Science
  tool: **`rcps: false`, `bidirectional: true`**, `model_type: caduceus`, `d_model: 256`, 16 layers,
  7.7M params, `vocab_size: 16` (matches our char vocab), `auto_map.AutoModel → modeling_caduceus.Caduceus`.
  Because `rcps: false`, its hidden channels are standard `d_model` — the mean-pool + `Linear` head
  works directly, resolving the earlier RCPS-pooling caveat. Apache-2.0 (clean license). GPU-only
  (`mamba-ssm`); run `--backbone caduceus` on Colab.

## D17 — Motif database: keep JASPAR as primary, offer HOCOMOCO v11 as a concordance overlay
- **Context.** Confirmed from the Deng *Science* full text (PMC12085231, read 2026-07-07): the
  paper's own mechanism layer scores TFBS gain/loss with **motifbreakR + HOCOMOCO v11**
  (threshold 1e-4), and its CNN-filter interpretation matches HOCOMOCO. Our `motifs.py` (D10)
  uses **JASPAR** PWMs. Divergence surfaced while nailing data provenance — must be logged, not
  silently left inconsistent.
- **Decision.** **Keep JASPAR as the primary motif source**; treat HOCOMOCO v11 as an
  **optional second library** loadable through the same `load_jaspar_pfms()`-style seam (D10),
  used as a **concordance overlay** on the 164 emVars — where the paper already published a
  motifbreakR/HOCOMOCO call, we can check our gain/loss agrees. Not swapping the default.
- **Why.** (a) JASPAR is the broader, better-maintained, CC-licensed standard and is already
  wired + tested; (b) the paper's HOCOMOCO calls on the 164 DAVs become a **free external
  validation set** for our motif engine — an independence signal for the trust story (D1),
  not just an aesthetic match; (c) the D10 loader seam makes adding HOCOMOCO a library swap,
  not a rewrite — no reason to displace JASPAR to gain the overlay.
- **Rejected.** (a) Switch wholesale to HOCOMOCO v11 to mirror the paper — loses JASPAR's
  coverage/licensing and the value of an *independent* motif call (matching their DB by
  construction weakens it as validation). (b) Ignore the divergence — leaves an unexplained
  method mismatch in a trust-first tool. (c) Run motifbreakR itself — heavy R/Bioconductor
  dependency against the D10 offline-first, numpy-only discipline.
- **Status.** Decided; not yet implemented. HOCOMOCO-overlay loader + emVar concordance check
  are a post-freeze nice-to-have (validation, not core path) — implement only if Sat/Sun
  calibration work has slack. Primary JASPAR path unchanged.

## D18 — Deep ensemble → per-variant model uncertainty feeds the trust layer
- **Context.** A single fine-tune gives one Δ per variant with no notion of how sure the model
  is. The judged axis is trust (D1); a variant the model is internally unsure about should be
  trusted less. With a GPU (or patience on CPU) N seeded fine-tunes are cheap.
- **Decision.** Train **N seeded members** per context (`finetune --n-seeds N` → `<out>/seed{k}/`)
  and wrap them in an **`EnsemblePredictor`** (duck-compatible with `ActivityPredictor`). The
  **mean** over members is the point Δ; the **std** over members is a per-variant *model
  uncertainty*. `interpret_variant` reads it (`score_variant_with_uncertainty`), stores it in
  `Interpretation.model_delta_primary_std`, and passes it to `build_trust_report`, which **shrinks
  the model's own log-odds toward 0** (confidence toward 0.5) as σ grows — evidence terms are
  independent of the model, so they are NOT shrunk. Default off (`model_uncertainty=None`), so
  single-model behavior is unchanged.
- **Why.** Ensemble disagreement is the cheapest honest uncertainty signal, and routing it into
  confidence is exactly the trust thesis: the tool says "uncertain" when its *own* models can't
  agree, not just when the evidence conflicts. Mean also modestly improves the point estimate.
- **Rejected.** A heteroscedastic head (predict μ+σ) — more invasive to the arch and the loss for
  a similar signal; MC-dropout — HyenaDNA/Caduceus aren't dropout-heavy and it's a weaker
  uncertainty than independent seeds. Averaging members into one opaque score without exposing σ
  (throws away the trust signal).
- **Status.** Implemented + CPU-tested: `--n-seeds` trains members; `EnsemblePredictor.from_dir`
  loads them; `score_variant_with_uncertainty` returns (mean, std>0 verified); `interpret_variant`
  threads σ into schema + trust (confidence 0.924→0.723 as σ 0→0.8); `eval/calibrate.py --ensemble`
  scores the mean. 6/6 tests pass. Real multi-seed members await a training run (the plumbing is
  proven on 2 quick members). Also this session: **RC test-time averaging (docs/03 §6) and lower
  calibration τ both tried and REJECTED** — negative results, logged, not adopted.

## D19 — Conservation evidence: Zoonomia 241-mammal constraint + HARs as an independent source
- **Context.** The trust layer (D13) grounds each prediction in signals produced *independently*
  of our model. Every source so far is an expression assay (MPRA, GTEx eQTL) or a human annotation
  (ClinVar, TSS). Comparative genomics — is this base under purifying selection across mammals? —
  is a genuinely *different data modality* and one of the strongest independence signals available.
  It is also a listed Researcher-track dataset (Pollard-lab Zoonomia constraint + Human Accelerated
  Regions), so wiring it ties the project to a second track dataset at near-zero cost.
- **Decision.** Add `from_conservation` (src/evidence.py): given a table with `chrom,pos` and any
  of `phylop`/`constraint` and/or a boolean `in_har`, emit an EvidenceItem when the position is
  **constrained** (phyloP241 ≥ 2.0) **or** in a **HAR**. Direction-less but *not neutral* — the
  same logic as the boolean GTEx eQTL branch: a constrained/human-accelerated base **corroborates**
  a predicted regulatory effect (concordant) and **conflicts** with a predicted-benign call
  (`model_direction==NONE` → concordant False, surfaced). Weight 1.1 (above ClinVar, below GTEx),
  ×1.2 when in a HAR. Absent table or an unconstrained non-HAR position → returns None (absence is
  never conflict, D13). Threaded through `gather_evidence(conservation_table=…)` and loaded in
  app.py via `_conservation()` (`RVI_CONSERVATION` env → `data/processed/conservation.parquet`).
- **Why.** Constraint is independent of every expression-based source we have, so concordance with
  it is strong corroboration and a conserved-but-model-says-benign case is exactly the kind of
  honest conflict the trust thesis exists to surface. HARs add a human-lineage regulatory-candidate
  flag. Cheap, additive, and defensible.
- **Rejected.** Treating low constraint as evidence *for* benign (too weak / noisy to assert a
  negative). A signed conservation "direction" (constraint has magnitude, not up/down expression
  direction) — kept it directionless-but-corroborating. Fetching phyloP live from the UCSC 241-way
  bigwig at query time (adds pyBigWig + network to the hot path; a pre-joined local table is the
  offline-first choice per D13). Real Zoonomia table ingest is a documented next step — the code
  path is wired, tested, and degrades gracefully until the table is dropped in.
- **Status.** Implemented + tested: `test_conservation_evidence` covers corroborate / benign-conflict
  / HAR-aware / silent-when-absent / gather_evidence pass-through. 7/7 core tests pass. Data ingest
  (join phyloP241 + HAR BED onto the 15,273 calibration variants) pending.
