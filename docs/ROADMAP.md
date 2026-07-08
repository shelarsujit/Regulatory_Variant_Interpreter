# Roadmap & backlog — pick up later

Parked work + guidance. Not the charter (that's `CLAUDE.md`) or the decision log
(`02_decision_log.md`); this is the "what's next / don't forget" list.

---

## A. Research-track submission (the actual judged deliverable)

Track = **[Researcher] Build From the Bench**: *"Submit something discrete — a finding, a
trained model, an analysis others can reproduce — and show us how Claude Science got you there."*

**We already have the discrete deliverable:** a trained model (HyenaDNA on Deng cortical MPRA,
primary r≈0.72), a reproducible analysis (manifests, provenance, `results.json`, decision log),
and an honest finding (variant-effect at the data ceiling; RC/τ/bigger-backbone negative
results; 163≈164 emVar pipeline validation).

**NOT needed for this track (dropped):** Claude-in-product agent, MCP server, demo-video polish.
Those are product-track concerns.

**Gaps to close (highest leverage first):**
1. **`docs/05_how_claude_science_built_this.md`** — the *required* "how Claude Science got you
   there" narrative. Claude's role: framed the trust-first thesis; decoded the Science supplement
   schema; built the hg38 + myvariant.info ingest; enforced the leakage-safe locus split; ran the
   calibration eval; ran the honest negative-result experiments (RC, τ, bigger backbone).
2. **Turnkey reproducibility** — top-level `README.md` with a single clone→download→prep→train→eval
   path that reproduces the numbers. (~90% there; verify a stranger can run it.)
3. **Crisp finding / 100–200w abstract** — headline: *a trust-calibrated regulatory-variant model
   whose deliverable is calibrated confidence, not a bare score — and an honest map of where
   single-base prediction hits the assay's data ceiling.*

Differentiator vs the track's own examples (Corces/ChromBPNet noncoding→chromatin): **the trust /
calibration layer** — nobody else's example has it.

---

## B. Model-quality roadmap (accuracy levers, ranked)

Everything here is optional polish on top of a complete submission. Raw accuracy is the tunable
knob; the trust thesis already holds. See `docs/03_results.md` §6–8 for the tried-and-rejected log.

1. **Caduceus backbone — bidirectional, NOT the RC-equivariant `-ps`** *(WIRED; awaits a GPU run)*.
   HyenaDNA is causal, so a mid-sequence variant suffers left-context bias; a bidirectional model
   sees both flanks. Uses the **`-ph` (post-hoc)** variant — bidirectional without forcing
   `f(seq)=f(rc(seq))` (RC-invariance is *wrong* for an orientation-specific MPRA — proven by the
   RC-averaging negative result, `03_results.md §6`). `--backbone caduceus` now defaults to
   `kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16` (verified via the HF Claude
   Science tool: `rcps: false`, `bidirectional: true`, d_model 256, 7.7M params, Apache-2.0; D16).
   GPU-only (`mamba-ssm` + `causal-conv1d`, CUDA build). Colab recipe below.
2. **Deep ensemble** (`--n-seeds N`, D18) — plumbing done + CPU-tested; needs a real multi-seed run
   to quantify. Gives mean Δ + per-variant σ that shrinks trust confidence.
3. **Lower calibration τ** — TRIED, rejected (`03_results.md §7`): AUC is τ-invariant; τ=0.5 keeps
   P aligned to the emVar rate.
4. **RC test-time averaging** — TRIED, rejected (`03_results.md §6`): hurts (MPRA is
   orientation-specific).

---

## C. Data / evidence follow-ups (optional)

- **HOCOMOCO v11 motif concordance overlay** on the 164 emVars (D17) — validation, not core path.
- **ClinVar + a 2nd external evidence source** in `evidence.py` (injectable seam exists).
- **GTEx**: currently a boolean brain-eQTL flag from Data S2; a signed-effect GTEx table would let
  it carry direction.
- Fold the one-off `gtex_eqtl.parquet` extraction into `prepare_data --deng-dir` for reproducibility.
