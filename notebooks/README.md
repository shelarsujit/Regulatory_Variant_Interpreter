# notebooks/

Scratch + validation notebooks (kept out of the import path on purpose).

Planned:
- `calibration_validation.ipynb` — fit the `Calibrator` on
  `data/processed/calibration_variants.parquet`, plot reliability curves, and check that
  stated confidence matches real accuracy (the core trust claim). Uses only the held-out
  variants — never the training elements (see `docs/02_decision_log.md`, D5/D6).
- `ism_sanity.ipynb` — eyeball saturation-mutagenesis maps for a few known emVars.
