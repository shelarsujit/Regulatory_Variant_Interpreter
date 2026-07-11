# Submission — Built with Claude: Life Sciences (Researcher track)

Ready-to-paste answers for the submission form. Field headers match the form.

---

## Project Name
Regulatory Variant Interpreter — trust-first interpretation of non-coding VUS

## Track
Researcher

---

## Project description
*What you built or investigated, what you found, and why it matters.*

**What it is.** A trust-first interpreter for non-coding, regulatory DNA variants — the frontier
where variant-of-uncertain-significance (VUS) calls mostly fail. Coding-variant tools are mature; a
variant in regulatory DNA usually gets *no principled call at all*, because its effect isn't "changes
a protein," it's "changes *how much* a gene is expressed." We fine-tune a DNA language model on the
Deng et al. 2024 cortical MPRA (*Science*, adh0559), score a single-base change's effect via
in-silico saturation mutagenesis, annotate the mechanism via transcription-factor motif gain/loss,
and — the core contribution — **ground every prediction in independent evidence** (held-out MPRA,
GTEx eQTLs, an independent organoid-context model, TSS proximity) to return a **calibrated confidence
with an auditable evidence chain**. When the model and evidence agree, confidence is high; when they
conflict, the tool surfaces it instead of hiding it behind a number.

**What we found.**
1. Optimizing the allelic-difference *directly* — a shared-weight "siamese" model trained on measured
   skew — beats the standard subtract-endpoints proxy by **+47%** on a bidirectional Caduceus
   backbone (variant-effect Δ-Pearson 0.19 → **0.28**, emVar AUC → **0.67**), the project's biggest gain.
2. A stacking meta-learner that fuses the primary model with an *independent* organoid model beats a
   single-feature calibrator (emVar AUC **0.623 vs 0.610**), the independent model being the lever.
3. We reproduce Deng's headline emVar set end-to-end (**163 ≈ 164**), and we report **honest
   negatives** — reverse-complement averaging, lower calibration τ, bigger HyenaDNA, and a frozen
   Enformer feature (tested to full 15k-variant coverage) — each logged with its reason, including one
   case where we caught our own thin-coverage artifact (a promising feature weight that collapsed to
   zero under full coverage).

**Why it matters.** The deliverable isn't a bare score — it's a *calibrated confidence and an
auditable evidence chain* for a class of variants curators currently can't call. A confident-but-wrong
regulatory call is dangerous; a calibrated "uncertain — here's the conflicting evidence" is usable.
The tool works on any variant and outlives the hackathon as a variant-curation aid.

---

## Link to your work
https://github.com/shelarsujit/Regulatory_Variant_Interpreter

> Note for judges: the complete work (README, `docs/`, trained-model code, results, app) is on the
> branch **`docs/hackathon-plan-provenance-prior-art`**. Repo is public for judging.

---

## Demo Video (max 3 min)
`<paste YouTube link>` — a ~2-minute narrated walkthrough: the Variant-coordinates tab showing an
agreement call (95%, all evidence concordant) and a **conflict** call (model vs wet-lab, surfaced not
hidden), then the paste-sequences tab with the meta-learner's stacked confidence.

---

## How did you use Claude? Which products (Claude Science, Claude Code, etc.)? Where did they matter most?

**Claude Code was the co-developer for the entire engineering loop** — it framed the trust-first
thesis, wrote every module (data pipeline, backbone-swappable predictor, the siamese variant-effect
model, the motif / evidence / trust layers, the stacking meta-learner, the demo app), ran the
experiments, and reported results honestly. It didn't just autocomplete: it made and documented
architectural decisions (e.g. that Caduceus's *bidirectionality*, not model size, is what moves
variant-effect), invented the direct-skew siamese objective, and **caught its own mistake** — a
promising frozen-model feature weight that collapsed to zero once run at full coverage, which it
flagged as a small-sample artifact rather than reporting as a win.

**Claude Science connectors grounded the work in real sources:**
- **PubMed** — verified the Deng anchor (PMID 38781390, DOI 10.1126/science.adh0559) and its numbers
  before they shipped in docs, and caught that the paper's term is "DAV," not "emVar."
- **Synapse.org** — confirmed the raw deposit `syn51090452` is DUA-gated, which is *why* the processed
  supplement is our source of truth (not the raw FASTQs).
- **Hugging Face** — verified backbone configs (Caduceus-ph is `rcps: false`, the correct
  non-RC-equivariant variant for an orientation-specific MPRA) and resolved curated brain-CAGE track
  indices from Enformer's targets file.

**Where it mattered most:** the leakage-safe experimental discipline and the honest-negative
reporting. Claude enforced the never-train-on-calibration-variants invariant *in code*, and treated
our own hypotheses with the same skepticism the tool applies to variants — the difference between a
demo and a result. Full narrative: [`docs/05_how_claude_science_built_this.md`](05_how_claude_science_built_this.md).

---

## Thoughts / feedback on building with Claude Science

The standout was that Claude Code operated as a genuine research collaborator, not a code generator —
it held the scientific thesis across a long, multi-day project, proposed experiments, and (critically)
reported negatives and caught its own artifacts instead of only surfacing wins. The connector
grounding (PubMed / Synapse / Hugging Face) meant provenance and model-config claims were *verified
against sources* rather than asserted, which is exactly what you want in science.

Friction points: iterating on GPU-only models (Caduceus, Enformer) meant a Colab round-trip because
the local environment is CPU, and dependency conflicts between GPU libraries (mamba-ssm,
enformer-pytorch) and the rest of the stack needed manual untangling — a tighter "run this on managed
GPU" loop inside Claude Science would remove the biggest source of iteration lag.

Overall: the trust-first *process* — leakage discipline, calibrated confidence, honest negatives — was
something Claude actively reinforced, and that shaped the science, not just the code.
