---
title: Regulatory Variant Interpreter
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Trust-first interpretation of non-coding regulatory DNA variants
---

# Regulatory Variant Interpreter

Trust-first interpretation of **non-coding, regulatory DNA variants** — the frontier where
variant-of-uncertain-significance (VUS) calls mostly fail today.

A fine-tuned DNA language model scores a single-base change's effect on regulatory activity, a
transcription-factor motif scan explains the likely mechanism, and — the core of the system — every
prediction is **grounded in independent evidence** (held-out MPRA measurements, GTEx eQTLs, TSS
proximity, an independent organoid-context model) to return a **calibrated confidence** with an
**auditable evidence chain**. When the model and the evidence agree, confidence is high; when they
**conflict, the tool says so, honestly** — instead of hiding it behind a confident number.

## Two ways to use it
- **Variant coordinates** — enter chrom / pos / ref / alt. Curated demo variants (agreement ·
  conflict · eQTL-flag) run offline; arbitrary coordinates need an hg38 FASTA (`RVI_GENOME`).
- **Paste sequences** — paste a reference and alternate 270 bp element (one differing base). No
  genome needed.

## What the confidence means
A stacking **meta-learner** fuses the primary cortex model, an independent organoid-context model,
and the motif signal into one calibrated confidence, and shows which features drove it. A
well-calibrated "uncertain" is a success here, not a failure.

This Space runs on CPU and serves the HyenaDNA activity model. The best variant-effect model (a
direct-skew siamese model on the Caduceus-ph backbone, Δ-Pearson ≈ 0.28) is GPU-only; on CPU the app
falls back to the activity model automatically.

Code, methods, and the full results (including honest negatives): the project repository.
Built with **Claude Science**.
