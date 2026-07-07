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
| File count | Supplement ships **Data S1–S3** (three files). We use S1 (elements) + S2 (variants). **S3 content ⚠️ unknown** — inspect on download. |
| Format | Excel / CSV tables. ⚠️ exact column names |
| License | CC-BY-NC. Not redistributed in this repo — you download it; `prepare_data.py` consumes it locally from `data/raw/`. |

## 4. Schema — ✅ CONFIRMED from the downloaded supplement (2026-07; see decision D14)

The real tables differ from the generic guesses below; the confirmed layout drives
`data/load_deng.py` (invoked via `--deng-dir`). Two fields the tool needs are **absent** and
are derived: `sequence` (reconstructed from hg38) and variant `ref`/`alt` (resolved from the
`rsid` via myvariant.info).

**Data S1 → elements** (file `adh0559_data_s1.xlsx`, sheets `Primary` + `Organoids`):
| Logical field | Real column | Notes |
|---|---|---|
| `chrom`,`start`,`end` | `insert_chrom`, `insert_start`, `insert_end` | **270 bp** windows (BED 0-based) |
| `element_id` | `insert_name` (= `chr:start-end`) | join key across the two sheets |
| `activity_primary` | `rna_dna_ratio` (sheet `Primary`) | training target |
| `activity_organoid` | `rna_dna_ratio` (sheet `Organoids`) | 2nd model target (join on `insert_name`) |
| `sequence` | **absent** → reconstruct from hg38 | `genome.fetch(chrom, start, end)` |

**Data S2 → calibration variants** (file `DataS2-Variant-library-ratios.xlsx`, sheet `Primary`):
| Logical field | Real column | Notes |
|---|---|---|
| `chrom`,`pos` | `variant_chrom`, `variant_pos` (1-based) | |
| `rsid` | `rsid` | → myvariant.info → `ref`/`alt` (dbsnp.py) |
| `ref`,`alt` | **absent** → resolve from `rsid` | multi-allelic: first alt tested |
| `seq_ref`,`seq_alt` | **absent** → reconstruct from hg38 | alt = ref window w/ variant base swapped |
| `measured_skew` | `logFC` | continuous calibration target (D6) |
| `fdr` | `adj.P.Val` | ✅ emVar def = **limma FDR<10% AND ≥1 allele active** → 164. Raw `adj.P.Val≤0.10` alone → ~600 (includes inactive variants; over-counts). Apply active-filter FIRST. (D14; confirmed from PMC full text 2026-07-07) |
| `is_active` | RNA/DNA > median of positive controls | variant-lib threshold = **1.068** (DA-lib = 1.047). Gate before FDR. ✅ |
| — | ~576 rows have null `logFC` | untested → dropped |

> The generic `--element-table/--variant-table` path (below) still exists for other datasets;
> it uses a fuzzy **column resolver** and records the mapping in `manifest.json`.

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

## 5. Sequence length ✅ CONFIRMED (PMC full text, 2026-07-07)
**Modeled length = 270 bp, centered** on the variant (variant library) or DA-peak summit
(DA library). Paper: *"We designed 270 base pair (bp) oligos, each centered on the DA peak
summit or variant, flanked by 15-bp adapters on either side"*; the deep-learning model
one-hot encodes *"270bp × 4 nucleotides per sequence."* This matches Data S1's 270 bp
`insert_start/insert_end` windows (§4). Downstream of the oligo (not modeled): 31-bp minimal
promoter + 15-bp random barcode. Far inside HyenaDNA's 1 kb context — no architectural
consequence. (Prior fallback said 200 bp — **superseded, use 270**.)

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
Outbound access to `science.org`, `biorxiv.org`, and NCBI was blocked by the original build
sandbox's egress policy, so §1–3 were first confirmed from PubMed/PMC abstracts, the
Gladstone press writeup, and a third-party benchmark repo that catalogs the exact Synapse
tree — not from the supplement files themselves. **Update 2026-07-07:** the *Science* article
full text is now on PMC (**PMC12085231**) and reachable via the PubMed MCP; the Synapse tree
(`syn21392931` → `syn51090452`) is reachable via the Synapse MCP. Sequence length (§5) and
the emVar definition (§4) are now ✅ confirmed. Remaining ⚠️: exact S1/S2 column names and
DataS3 content — confirmed only when the real supplement files are opened. `prepare_data.py`
absorbs that: fuzzy column resolution + a `manifest.json` that records what it actually found.

## 8a. Facts nailed from PMC full text (2026-07-07) — reference
- **Modeled sequence = 270 bp, variant/summit-centered** (§5).
- **emVar (164) = limma FDR<10% AND ≥1 allele active** (active gate: RNA/DNA > pos-control
  median; variant-lib 1.068 / DA-lib 1.047). Raw FDR≤0.10 alone over-counts (~600). (§4)
- Variant library = **5 replicates**; DA library = **3 replicates**.
- Counts: 17,069 variants → 15,335 both-allele QC-pass → 8,029 active (≥1 allele) → **164 DAVs**.
- Organoid DAVs = **420** at FDR<10%; 74 shared with primary; effect-size correlation r=0.91.
  ⚠️→partially ✅: organoid *does* report variant-level skew → bonus 2nd calibration target (§6).
- **Paper's own model split: chr3 = validation, chr4 = held-out test**, all other chroms train.
  (Ours is locus-grouped — stricter — but keep chr4 handling consistent to compare.)
- Paper baseline model = 1 conv + 2 recurrent layers (MPRAnn family). **chr4 Pearson: DA 0.82,
  variant 0.78** (Spearman 0.81 / 0.70). Numbers to beat/compare for our HyenaDNA fine-tune.
- Paper motif gain/loss = **motifbreakR + HOCOMOCO v11**, threshold 1e-4 → we use **JASPAR**
  in `motifs.py`. Divergence — log a decision (align to HOCOMOCO, or justify JASPAR).
- Paper ISM = 270 pos × 17,069 oligos = **18.4M alleles** — direct precedent for our
  saturation-mutagenesis scorer.

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
- PMC (**Science article** full text): https://pmc.ncbi.nlm.nih.gov/articles/PMC12085231/
- PMC (preprint full text): https://pmc.ncbi.nlm.nih.gov/articles/PMC9949039/
- Gladstone writeup: https://gladstone.org/news/scientists-leverage-machine-learning-decode-gene-regulation-developing-human-brain
- lentiMPRA protocol (Gordon et al., Nat Protoc 2020): https://www.nature.com/articles/s41596-020-0333-5
- Sibling MPRAnn + ISM reference (Zenodo): https://zenodo.org/records/10558183
- HyenaDNA checkpoints: https://hf.co/LongSafari/hyenadna-tiny-1k-seqlen
