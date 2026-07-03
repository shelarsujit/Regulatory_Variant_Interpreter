# CLAUDE.md — Regulatory Variant Interpreter (project charter)

> This file is the operating charter for the project. It is deliberately terse.
> The **in-depth, plain-English theory** — including a primer for non-biologists —
> lives in [`docs/`](docs/). Start there if you are new:
> - [`docs/00_overview_for_non_biologists.md`](docs/00_overview_for_non_biologists.md) — what this whole thing *is*, from first principles.
> - [`docs/01_data_provenance.md`](docs/01_data_provenance.md) — where the data comes from and exactly what it looks like.
> - [`docs/02_decision_log.md`](docs/02_decision_log.md) — every engineering decision and *why* (the "theory of everything we do").

## 1. What this is
A **trust-first interpreter for non-coding, regulatory DNA variants** — the frontier
where variant-of-uncertain-significance (VUS) calls mostly fail today. Coding-variant
tools are mature; a variant in a regulatory region usually gets *no* interpretation,
because its effect isn't "changes a protein," it's "changes *how much* a gene is
expressed." Predicting that requires a model that reads raw DNA.

The engine is a DNA language model fine-tuned on massively parallel reporter assay
(MPRA) data. It runs **in-silico saturation mutagenesis** to score a variant's
reference-vs-alternate effect, then annotates the likely mechanism via
transcription-factor **motif gain/loss**.

**The hard problem is trust, not prediction.** The agent grounds every prediction in
independent evidence (a frozen foundation model, GTEx eQTLs, held-out MPRA
measurements, ClinVar, TSS proximity) and returns a **calibrated confidence** with an
**auditable evidence chain**. When model and evidence agree → high confidence. When
they conflict → the tool says so, honestly, instead of hiding it.

## 2. The named user
A variant-curation scientist or clinical geneticist holding a regulatory VUS with no
principled way to call it. Works on any variant → outlives the hackathon as a tool for
any genomic-medicine / variant-curation group.

## 3. Locked decisions (rationale in `docs/02_decision_log.md`)
- **Backbone:** HyenaDNA — single-nucleotide resolution (essential for single-base
  saturation mutagenesis), sub-quadratic, fine-tunable on one A100 in a week.
  Checkpoint: `LongSafari/hyenadna-tiny-1k-seqlen` (1 kb context ≫ ~200 bp elements).
  **Upgrade path:** Caduceus (reverse-complement equivariant) if we converge early.
- **NOT the plan:** fine-tuning Enformer/AlphaGenome. We use one of them **frozen,
  zero-shot** as an *independent* evidence source in the trust layer.
- **Data source of truth:** the **processed** Deng et al. 2024 *Science* supplement
  (`adh0559`). The **raw** Synapse deposit (`syn51090452`, PsychENCODE, behind a
  Data-Use Agreement) is a fallback only. See `docs/01_data_provenance.md`.
- **Targets:** two **separately fine-tuned single-context** models —
  **primary cortex = the call**; **organoid = an independent second model** feeding the
  trust layer. (Not a shared-trunk multitask head — that would destroy organoid's
  independence, which is the cheapest strong signal we have.)
- **Leakage discipline (non-negotiable):** never train on a sequence that overlaps a
  calibration variant. Enforced by a locus-grouped split in `data/splits.py` and
  asserted at the end of `data/prepare_data.py`.

## 4. Pipeline
```
fine-tune HyenaDNA        # (seq -> primary-cortex activity); 2nd model for organoid
      │
      ▼
saturation mutagenesis    # score ALT vs REF as Δactivity  (predictor.py)
      │
      ▼
motif annotation          # JASPAR PWM gain/loss: "disrupts a predicted CTCF site" (motifs.py)
      │
      ▼
grounding                 # frozen Enformer/AlphaGenome, GTEx, ClinVar, TSS, held-out MPRA (evidence.py)
      │
      ▼
trust / calibration       # agreement -> calibrated confidence + explicit conflicts (trust.py)
      │
      ▼
interpret_variant()       # orchestrator -> Interpretation  (interpret.py)
```

## 5. Interface contract
```python
interpret_variant(chrom: str, pos: int, ref: str, alt: str,
                  build: str = "hg38") -> Interpretation
```
`Interpretation`, `TrustReport`, `EvidenceItem`, `Mechanism` are defined in
[`src/schema.py`](src/schema.py). Every field except the raw model number exists to
make that number *trustable*.

## 6. Repo map
```
data/prepare_data.py   # supplement tables -> train/val + held-out variant calibration set  [IMPLEMENTED]
data/splits.py         # leakage-safe, locus-grouped splitting                              [IMPLEMENTED]
src/schema.py          # the Interpretation contract (dataclasses)                          [IMPLEMENTED]
src/predictor.py       # load checkpoint + saturation-mutagenesis scorer                    [stub]
src/evidence.py        # frozen foundation model, GTEx, ClinVar, TSS, held-out MPRA         [stub]
src/motifs.py          # JASPAR PWM gain/loss (offline-capable + loader)                    [IMPLEMENTED]
src/trust.py           # isotonic calibrator + agreement/conflict aggregation               [IMPLEMENTED]
src/interpret.py       # interpret_variant() orchestrator                                   [wired; needs predictor + genome]
train/finetune_hyenadna.py  # the A100 fine-tune                                            [stub]
app/app.py             # Gradio demo -> HF Space                                            [stub]
tests/test_core.py     # leakage split, motif gain/loss, calibration+conflict, end-to-end   [IMPLEMENTED]
docs/                  # plain-English theory + data provenance + decision log
notebooks/             # calibration validation, scratch
```

## 7. Working agreements
1. **Trust > accuracy.** A well-calibrated "uncertain" beats a confident wrong call.
2. **Always surface conflict.** Never hide a disagreement between the model and the evidence.
3. **Never train on calibration variants.** (See §3.)
4. **Every prediction is auditable.** It carries an evidence chain + provenance
   (checkpoint hash, data versions).
5. **Document as we build.** Each non-trivial decision gets an entry in
   `docs/02_decision_log.md`; each module carries a plain-English "theory" docstring.
