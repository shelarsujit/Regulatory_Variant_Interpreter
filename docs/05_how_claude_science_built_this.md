# How Claude Science built this

> The research-track ask: *"submit something discrete — a finding, a trained model, an analysis
> others can reproduce — and show us how Claude Science got you there."* This file is the "how."

**The discrete deliverables:** two fine-tuned DNA language models (a Caduceus-ph element-activity
model and a direct-skew **siamese variant-effect model**, the project's best result at
Δ-Pearson 0.28 / emVar AUC 0.67), a calibrated **trust layer** with a stacking meta-learner that
beats a single-feature baseline (0.623 vs 0.610), a reproducible leakage-safe pipeline with
per-run provenance, and an honest map of where single-base prediction hits the assay's data
ceiling — including a fully-tested **negative** for a frozen Enformer feature.

**The differentiator:** not the raw predictor (single-base MPRA effect is an open hard problem) but
the **trust/calibration layer** — every call carries a calibrated confidence and an auditable
evidence chain, and disagreements are surfaced, not hidden. Nothing in the track's own examples has it.

---

## Claude Science as the co-developer

Claude Science here means **Claude Code driving the whole engineering loop** — reading the biology,
writing every module, running experiments, and reporting results honestly — plus the **connectors**
that ground the work in real scientific sources. Claude did not just autocomplete; it framed the
thesis, made and documented architectural decisions, ran the honest-negative experiments, and caught
its own mistakes (e.g. a small-sample artifact in the frozen-feature fit, §below).

### Connectors used (and for what)
Full map in [`docs/06_claude_science_toolchain.md`](06_claude_science_toolchain.md); the load-bearing ones:

| Connector | Concrete use in this project |
|---|---|
| **PubMed** | Verified the Deng et al. 2024 anchor (PMID 38781390, DOI 10.1126/science.adh0559) and its numbers (102,767 elements; 17,069 variants; 164 emVars at 10% FDR) before they shipped in docs — and caught that the paper's term is "DAV," not "emVar." |
| **Synapse.org** | Inspected the raw deposit `syn51090452` (`MPRA_CapstoneII`, PsychENCODE) and confirmed it is **DUA-gated** (`can_download: false`, certification required) — which is *why* the processed *Science* supplement is the source of truth, not the raw FASTQs. |
| **Hugging Face Hub** | Verified backbone configs before wiring them: HyenaDNA (tiny/small), and **Caduceus-ph** (`rcps: false`, bidirectional, Apache-2.0) — confirming it is the correct non-RC-equivariant variant for an orientation-specific MPRA. Resolved the curated brain-CAGE Enformer track indices from `targets_human.txt`. |

Three read-only project agents (`.claude/agents/`) wrap these — a literature-grounder, a
checkpoint-scout, and a Synapse data-scout — so provenance and citation checks are repeatable.

---

## The build, phase by phase (what Claude actually did)

1. **Framed the trust-first thesis.** The insight that the *hard, valuable* problem is trust, not
   raw accuracy — a calibrated "uncertain" beats a confident wrong call — shaped every module. It is
   why the schema (`src/schema.py`) makes every field exist to make the model number *trustable*.

2. **Decoded the data and enforced leakage discipline.** Mapped the *Science* supplement schema,
   built the hg38 + dbSNP/myvariant sequence reconstruction (`data/load_deng.py`), and made the
   non-negotiable **locus-grouped split** (`data/splits.py`) that guarantees no training sequence
   overlaps a calibration variant — asserted in code, not just intended. Independently reproduced
   Deng's headline **163 ≈ 164 emVar** count end-to-end, validating the whole pipeline.

3. **Built a backbone-swappable predictor and fine-tuned it.** HyenaDNA first (CPU-testable), then
   **Caduceus-ph** as a one-flag swap — and showed, with an honest A/B, that *bidirectionality*
   (not model size) is what moves variant-effect: element r 0.70 → 0.78, variant Δ-Pearson 0.15 → 0.19.

4. **Invented and trained the direct-skew siamese model.** Reasoning that the subtract-endpoints Δ
   was a lossy proxy, Claude built a shared-weight encoder trained *directly* on measured allelic
   skew (`src/siamese_predictor.py`, `train/finetune_siamese.py`). On a *causal* backbone it lost
   (a documented negative) — then, exactly as predicted, **won on the bidirectional Caduceus backbone:
   Δ-Pearson 0.19 → 0.28 (+47%)**, the largest single variant-effect gain of the project.

5. **Built the trust layer and the stacking meta-learner.** An isotonic calibrator
   (`src/trust.py`) whose reliability diagonal is honest on the bulk bin, plus a hand-rolled logistic
   **meta-learner** (`src/meta.py`) that fuses the primary model, the independent organoid model, and
   the motif signal — beating the single-feature calibrator (0.623 vs 0.610), with the organoid model
   the real lever. Per-feature weights *are* the "why we trust this call."

6. **Ran the honest-negative experiments — including on its own hypotheses.** Reverse-complement
   averaging, lower calibration τ, bigger HyenaDNA, and a **frozen Enformer CAGE feature** (single
   adult-brain track → curated developmental-cortical set → full 15,273-variant coverage) all failed
   to help. On the frozen feature Claude even **caught its own artifact**: a promising `−0.15` weight
   on a thin-coverage run collapsed to `0.004` under full coverage — small-sample overfit, not signal.
   Every negative is logged with its reason ([`docs/07`](07_enhancement_design.md), [`docs/08`](08_frozen_model_wiring_plan.md)).

7. **Shipped a usable product.** A two-tab Gradio app (`app/app.py`) that serves the best model with
   its matched calibrator, degrades gracefully (CPU fallback, offline calibration-backed genome shim),
   and exposes the meta-learner; plus an automated narrated **demo-video pipeline**
   (`demo/make_demo_video.py`: TTS + Playwright + ffmpeg) and reproducible Colab GPU recipes.

---

## Why this reads as good science

- **Reproducible:** leakage-safe splits, per-run `provenance.json`, deterministic data derivation,
  and Colab recipes; a stranger can regenerate the numbers.
- **Honest:** the headline is a *modest* variant-effect result (single-base effect is the field's
  open hard problem) made *trustworthy* by calibration — and every failed idea, including Claude's
  own, is on the record with its reason. The trust-first thesis is applied to our own work, not just
  the variants.
- **Grounded:** the connectors tie the data provenance, the citations, and the model choices to real,
  verifiable sources rather than assertion.

The engine predicts; the contribution is making that prediction *trustable* — and Claude Science is
how the whole thing was reasoned through, built, tested, and honestly reported.
