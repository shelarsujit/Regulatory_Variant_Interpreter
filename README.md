# Regulatory Variant Interpreter

**A trust-first tool for interpreting non-coding, regulatory DNA variants.**

Most variant-interpretation tools work on *coding* variants — changes that alter a
protein. But a huge fraction of disease-associated genetic variation sits in
**regulatory** DNA: sequence that doesn't code for a protein but controls **how much**
a gene is switched on. For those, a curator today usually gets *no principled call at
all* — the variant is filed as "of uncertain significance" (a VUS).

This project reads raw DNA with a fine-tuned **DNA language model**, predicts what a
single-base change does to regulatory activity (**in-silico saturation mutagenesis**),
explains the likely mechanism (**transcription-factor motif** gain/loss), and — the
core of the system — **grounds every prediction in independent evidence** to return a
**calibrated confidence** with an **auditable evidence chain**. When the model and the
independent evidence agree, confidence is high; when they conflict, the tool says so
honestly.

> New here, or not a biologist? Read
> **[`docs/00_overview_for_non_biologists.md`](docs/00_overview_for_non_biologists.md)** —
> it explains the whole subject from first principles (what DNA, genes, regulation,
> variants, MPRA, and "reading DNA with a model" actually mean) before any code.

## Status
Early build (Life Sciences hackathon). **Day 1 = data foundation.**

| Component | State |
|---|---|
| Data prep → training set + held-out variant calibration set | ✅ implemented (`data/prepare_data.py`) |
| Leakage-safe locus split | ✅ implemented (`data/splits.py`) |
| `interpret_variant()` contract | ✅ defined (`src/schema.py`) |
| Motif gain/loss (`src/motifs.py`) + trust calibration (`src/trust.py`) | ✅ implemented + tested |
| `interpret_variant` orchestration | ✅ wired (needs the predictor + genome window) |
| HyenaDNA fine-tune, saturation mutagenesis, evidence sources, demo | 🚧 stubs |

## The data
Deng et al. 2024, *Science* (`adh0559`) — a lentiMPRA in the developing human cortex
that measured regulatory activity for **102,767** sequences and the allelic effect of
**17,069** psychiatric-disorder-associated variants (**164** significant at 10% FDR).
We train on the activity measurements and hold out the variants as ground truth for
**calibrating confidence**. Full provenance, formats, and access notes:
[`docs/01_data_provenance.md`](docs/01_data_provenance.md).

## Quickstart (Day 1)
```bash
pip install -r requirements.txt        # top block is enough for data prep

# Prove the whole pipeline end-to-end today, before the real supplement is downloaded,
# using a small synthetic MPRA-like fixture:
python data/prepare_data.py --synthetic

# Once the real Science supplement tables are in data/raw/ :
python data/prepare_data.py --raw-dir data/raw
```
Outputs (git-ignored) land in `data/processed/`: `train.parquet`, `val.parquet`,
`calibration_variants.parquet`, and a `manifest.json` recording every parameter, the
column mapping used, row counts, and the leakage-check result.

## Design principles
1. **Trust > accuracy** — a calibrated "uncertain" beats a confident wrong call.
2. **Always surface conflict** — never hide model-vs-evidence disagreement.
3. **Never train on calibration variants** — enforced and asserted in code.
4. **Everything is auditable** — evidence chain + provenance on every prediction.

See [`CLAUDE.md`](CLAUDE.md) for the full charter and
[`docs/02_decision_log.md`](docs/02_decision_log.md) for the reasoning behind every
choice.

## License
Code: see [`LICENSE`](LICENSE). The Deng et al. MPRA data is CC-BY-NC and is **not**
redistributed here — `prepare_data.py` consumes files you download yourself.
