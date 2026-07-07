# data/raw/ — put the downloaded Deng supplement here (NOT committed)

Source of truth = the **processed *Science* adh0559 supplement** (decision D3;
`docs/01_data_provenance.md`). Download it yourself; this folder is git-ignored
(CC-BY-NC, not redistributed).

Download from:
- Science: https://www.science.org/doi/10.1126/science.adh0559 → Supplementary Materials
- Mirror (if paywalled): https://www.biorxiv.org/content/10.1101/2023.02.15.528663v2

Put two tables here (any `.xlsx`/`.csv` name):
- **element-activity** table → training labels (`activity_primary`, `activity_organoid`)
- **variant allelic-skew** table → calibration ground truth (`measured_skew`, `fdr`, `is_emvar`)

Then build the processed dataset:
```
python data/prepare_data.py \
    --element-table data/raw/<element_activity>.xlsx \
    --variant-table data/raw/<variant_skew>.xlsx
```
Check `data/processed/manifest.json` — confirm the fuzzy column resolver picked the right
headers; override any it got wrong with `--col logical_name=RealHeader` (see §4).

⚠️ If the tables give only genomic coordinates and no `sequence` column, sequences must be
reconstructed from an hg38 FASTA (pyfaidx, 200 bp centered) — ask before running.
