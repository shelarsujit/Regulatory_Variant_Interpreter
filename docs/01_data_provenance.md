# Data provenance — Deng et al. cortical MPRA

This is the permanent record of **where our data comes from, what it looks like, and how
much of that we have actually verified**. Read it before touching `data/prepare_data.py`.
Plain-English background for every term here is in
[`00_overview_for_non_biologists.md`](00_overview_for_non_biologists.md).

Legend: ✅ confirmed from a primary/authoritative source · ⚠️ **to verify on first
download** (inferred, not yet seen directly).

---

## 1. The paper

| Field | Value |
|---|---|
| Preprint | bioRxiv **2023.02.15.528663** (v1 Feb 2023; v2). "Massively parallel characterization of psychiatric disorder-associated and cell-type-specific regulatory elements in the developing human cortex." ✅ |
| Journal version | ***Science* 384:eadh0559** (24 May 2024). "Massively parallel characterization of regulatory elements in the developing human cortex." **PMID 38781390**, **doi:10.1126/science.adh0559**. ✅ |
| Senior authors | **Nadav Ahituv** & **Katherine S. Pollard** (with Nowakowski & Pollen labs); PsychENCODE ("NeuREs" project). ✅ |
| License | Preprint CC-BY-NC. ✅ |

## 2. The assay and its scale
- **lentiMPRA** in **primary human mid-gestation cortical cells** and **10-week cerebral
  organoids**. ✅
- **102,767** sequences tested; **46,802** active enhancers in primary cells. ✅
- Variant library: **17,069 brain QTLs** (within 100 kb of differentially-expressed
  cross-disorder neurodevelopmental genes, or in LD with psychiatric-disorder GWAS SNPs). ✅
- **164** variants show significant allelic effects at **10% FDR** (~51% down / 49% up) —
  these are the "emVars." ✅
- Deep-learning models in the paper: **MPRAnn** (a CNN) and **SeiMPRA** (a lasso over
  ~21,907 Sei chromatin features). Useful as sanity baselines; **not** our backbone. ✅

## 3. Where the data lives — two tiers

### Tier A — RAW sequencing (DNA + RNA barcode counts)
| | |
|---|---|
| Repository | **Synapse**, PsychENCODE "NeuREs" project **`syn21392931`** → MPRA subfolder **`MPRA_CapstoneII` = `syn51090452`**. ✅ |
| Contents | ~96 paired DNA/RNA files: 4-rep cerebral-organoid DNA+RNA, 4-rep primary-cortex DNA+RNA, 5-donor bulk-RNA control. ✅ |
| Format | FASTQ (raw reads). ⚠️ exact file naming |
| **Access** | ⚠️→✅ **PsychENCODE Data-Use Agreement + `SYNAPSE_AUTH_TOKEN` required to *download*.** Anonymous access gives metadata / tree-walk only. |

> **Consequence (important).** Building from Tier A means (a) waiting on a DUA approval and
> (b) running a full barcode-mapping + normalization pipeline (MPRAflow) just to obtain a
> training label. Neither is affordable in a one-week build. **We do not start here.**

### Tier B — PROCESSED tables (our source of truth)
| | |
|---|---|
| Location | The ***Science* `adh0559` supplementary materials** ("Data S" / Table S files), open with the article; mirrored in the bioRxiv v2 supplement. ✅ (existence) / ⚠️ (exact table numbers) |
| What we need from it | **(a)** per-element **activity** table → training labels. **(b)** per-**variant** ref/alt **allelic-skew** table → held-out calibration ground truth. ✅ (both are reported in the paper) |
| Format | Excel / CSV tables. ⚠️ exact column names |
| License | CC-BY-NC. Not redistributed in this repo — you download it; `prepare_data.py` consumes it locally from `data/raw/`. |

## 4. Expected schema (⚠️ verify column names on first download)

`prepare_data.py` does **not** hard-code these names — it uses a fuzzy **column
resolver** (case-insensitive substring match) and writes the mapping it chose into
`manifest.json`. Adjust the resolver map once you see the real headers.

**Element-activity table → `train`/`val`** (one row per tested sequence):

| Logical field | Likely source column(s) | Notes |
|---|---|---|
| `sequence` | the ~200 bp tested element sequence | ⚠️ or reconstruct from coordinates + hg38 |
| `chrom`, `start`, `end` | genomic coordinates of the element | needed for the locus split |
| `activity_primary` | normalized **log2(RNA/DNA)** in primary cortex | the **training target** |
| `activity_organoid` | normalized log2(RNA/DNA) in organoid | second model's target (optional) |
| `element_id` | library element identifier | |

**Variant allelic-skew table → `calibration_variants`** (one row per tested SNV):

| Logical field | Likely source column(s) | Notes |
|---|---|---|
| `chrom`, `pos`, `ref`, `alt` | variant coordinates + alleles | the tool's input tuple |
| `seq_ref`, `seq_alt` | the two allelic sequences | ⚠️ or reconstruct from hg38 (alt = ref with center base swapped) |
| `measured_skew` | log2 allelic fold-change (alt vs ref) | **calibration ground truth (regression)** |
| `fdr` / `padj` | significance of the allelic effect | |
| `is_emvar` | significant at 10% FDR? (the 164) | **calibration ground truth (classification)** |

## 5. Sequence length ⚠️
Ahituv lentiMPRA uses a **~200 bp core** (Agarwal et al. describe "200-nucleotide cores");
oligos carry ~15 bp cloning adapters per side (~230–270 nt total). Exact modeled length
for Deng is **to verify** in Methods / Table S1. **Fallback:** treat as **200 bp,
centered**, pad to the model's fixed input length. Either way it's far inside HyenaDNA's
1 kb context — no architectural consequence.

## 6. Two activity contexts (design consequence)
There are **two** readouts per element — **primary cortex** and **organoid** — not one.
We make **primary cortex the prediction target** (the 164 emVars are defined there) and
use **organoid as an independent second model** in the trust layer. See decision **D4**.
⚠️ Verify whether the supplement reports *variant-level* skew for organoid separately; if
it does, we gain a bonus second calibration target. Not required — the organoid model
yields a predicted Δ regardless.

## 7. Calibration set — size and shape
- Usable calibration pairs: **~17,069** variants with a measured skew. ✅
- Positives (emVars): **164** (~1%). ✅
- **Implication:** calibrate primarily on the **continuous** predicted-Δ vs measured-skew
  relationship (isotonic/Platt); use the 164 as a **classification** reliability check.
  Do **not** calibrate on 164 alone. (Decision D6.)

## 8. What we could NOT verify from the build sandbox
Outbound access to `science.org`, `biorxiv.org`, and NCBI is blocked by the environment's
egress policy, so §1–3 were confirmed from PubMed/PMC abstracts, the Gladstone press
writeup, and a third-party benchmark repo that catalogs the exact Synapse tree — **not**
from the supplement files themselves. Everything marked ⚠️ (exact table numbers, column
names, sequence length) must be confirmed the first time the real supplement is opened.
`prepare_data.py` is built to absorb that: fuzzy column resolution + a `manifest.json`
that records what it actually found.

## 9. Reusable external asset
The sibling paper **Agarwal et al. 2024/2025** (same lab, same lentiMPRA + MPRAnn family,
3 cell types) ships a **Zenodo** record with **MPRAnn training/testing + in-silico
mutagenesis** code (zenodo.org/records/10558183). It is a strong **reference
implementation** to adapt for our saturation-mutagenesis scorer — same assay family,
same ISM convention. (It is *not* the Deng dataset.)

---

### Source index
- bioRxiv preprint: https://www.biorxiv.org/content/10.1101/2023.02.15.528663v2
- *Science* article: https://www.science.org/doi/10.1126/science.adh0559
- PubMed: https://pubmed.ncbi.nlm.nih.gov/38781390/
- PMC (preprint full text): https://pmc.ncbi.nlm.nih.gov/articles/PMC9949039/
- Gladstone writeup: https://gladstone.org/news/scientists-leverage-machine-learning-decode-gene-regulation-developing-human-brain
- lentiMPRA protocol (Gordon et al., Nat Protoc 2020): https://www.nature.com/articles/s41596-020-0333-5
- Sibling MPRAnn + ISM reference (Zenodo): https://zenodo.org/records/10558183
- HyenaDNA checkpoints: https://hf.co/LongSafari/hyenadna-tiny-1k-seqlen
