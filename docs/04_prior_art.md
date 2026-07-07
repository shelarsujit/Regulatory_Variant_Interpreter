# Prior art — has this been done already?

> Scan date: **2026-07-07** (kickoff). Purpose: locate this project against existing work,
> confirm the novelty claim, and pre-load the answer to the judge's inevitable question —
> *"why not just use AlphaGenome?"* Sources are primary papers / tools found via PubMed +
> web search. See [`02_decision_log.md`](02_decision_log.md) **D1** for the framing this
> defends.

## Verdict
**The pieces exist; this exact tool does not.** Regulatory variant *prediction* is a mature,
crowded field. Non-coding variant *annotation aggregation* is an older, shallower one.
**Nobody ships a trust-first interpreter** that returns a **calibrated confidence + an
auditable, multi-source evidence chain that surfaces model-vs-evidence conflict** for a
regulatory VUS, built around a named curator. That is the whitespace this project occupies.

---

## The landscape, in four buckets

### 1. Sequence → activity variant predictors (the crowded frontier)
Enformer, Basenji, Borzoi, **AlphaGenome** (DeepMind, 2025 — current SOTA), Sei. Take DNA
in, predict regulatory tracks / variant effect scores out. **State, not our competitor:**
they output a *number*. No calibrated clinical confidence, no evidence chain, no conflict
surfacing. This project **deliberately does not compete here** — it uses one such model
**frozen, zero-shot** as an independent evidence source (decision **D7**).

### 2. Zero-shot DNA language models
GPN (Benegas et al., *"DNA language models are powerful zero-shot predictors of non-coding
variant effects"*), Nucleotide Transformer, species-aware DLMs. Score variants without task
training. Still **prediction**, not trust; still a bare score.

### 3. Machine learning on MPRA (closest to our *training* approach)
Deng et al.'s own **MPRAnn** / **SeiMPRA**; *"Identifying non-coding variant effects at scale
via ML models of cis-regulatory reporter assays"* (bioRxiv 2025). Same data family, same
sequence-to-activity idea we fine-tune. **These are predictors** — none add a calibration /
trust layer or an evidence chain.

### 4. Clinical annotation aggregators (the older attempt at "interpretation")
**RegulationSpotter**, **Revana**, plus CADD/ncER-style scores, under the **ACMG/AMP
non-coding interpretation guidelines**. These annotate a variant with 100+ genome-wide
features for a clinician. **State:** feature dumps, not a model-driven calibrated call; no
notion of "my model and the independent evidence disagree."

---

## The whitespace this project occupies
1. **A trust layer, not a score.** Calibrated confidence + explicit **model-vs-evidence
   conflict** + auditable evidence chain. Predictors give scores; annotators give feature
   dumps. Neither says: *"here is the call, here is my calibrated confidence, and here is
   where my model and the independent evidence DISAGREE."* (Working agreement #2; **D1**.)
2. **An independence ladder as calibration substrate** — two separately fine-tuned MPRA
   models (primary cortex + organoid, **D4**) → frozen foundation model (**D7**) → GTEx /
   ClinVar / held-out MPRA / TSS databases. Agreement across independent axes is the
   confidence signal.
3. **Deployable software for a named user** — a variant-curation scientist holding a
   regulatory VUS — not a benchmark model or a research notebook.

---

## The one real threat: AlphaGenome (2025) — and why it *strengthens* the thesis
AlphaGenome (DeepMind, 2025) is the elephant: a unified 1 Mb-context DNA model predicting
thousands of tracks at single-bp resolution, with an API, beating Enformer/Borzoi on
fine-mapped GTEx eQTLs. A judge **will** ask *"why not just use AlphaGenome?"*

The answer is on our side:
- **AlphaGenome is a better *predictor* — so use it.** It is exactly the kind of frozen
  independent evidence source **D7** already names as an option. It is the engine, not the
  product; our product is the trust wrapper around it. The two are orthogonal.
- **Even SOTA prediction is not a trustable clinical call.** AlphaGenome **underperforms on
  deep-intronic and synonymous ClinVar variants** and on some splicing benchmarks. The
  frontier model exists, is excellent, and *still* leaves the curator without a calibrated,
  conflict-aware call. **That gap is the entire product.** The strongest possible predictor
  landing and not closing the trust gap is the best available evidence that the gap is real.

> **One-liner for the video / summary:** *"AlphaGenome made prediction better. It did not
> make interpretation trustable. We build the trust layer — and AlphaGenome plugs into it as
> one more independent witness."*

---

## Source index
- AlphaGenome (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC12851941/
- AlphaGenome (bioRxiv preprint): https://www.biorxiv.org/content/10.1101/2025.06.25.661532v1.full.pdf
- Leveraging genomic deep learning for non-coding variant effects (review): https://arxiv.org/html/2411.11158v2
- ML models of cis-regulatory reporter assays (2025): https://www.biorxiv.org/content/10.1101/2025.04.16.648420.full.pdf
- GPN — DNA language models as zero-shot non-coding variant predictors: https://www.researchgate.net/publication/362879910
- RegulationSpotter: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6602480/
- Revana (regulatory variant analysis): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9822537/
- Exploration of tools for non-coding variant interpretation (review): https://pmc.ncbi.nlm.nih.gov/articles/PMC9654743/
- ACMG/AMP non-coding clinical interpretation recommendations: https://pubmed.ncbi.nlm.nih.gov/35850704/
