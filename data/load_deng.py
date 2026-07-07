"""Ingest the real Deng et al. Science adh0559 supplement into our train/calibration schema.

THEORY (plain English — see docs/01_data_provenance.md, decisions D3/D13):
The supplement gives coordinates and measured ratios, but NOT the DNA letters (no `sequence`
column) and NOT the variant ref/alt bases (only an rsID). This module fills both gaps:
  * elements  : join Data S1 `Primary` + `Organoids` on `insert_name`, reconstruct each tested
                270 bp element sequence from a local hg38 FASTA (data/genome.py).
  * variants  : from Data S2 `Primary`, resolve ref/alt per rsID via myvariant.info (data/dbsnp.py),
                reconstruct the tested element's ref sequence from hg38 and build the alt
                sequence by swapping the variant base. measured_skew = `logFC`, fdr = `adj.P.Val`.

COORDINATE SAFETY. The 0-based(BED)-vs-1-based convention of `insert_*` / `variant_pos` is not
documented, so we DETECT it empirically: on a sample we check which convention makes the hg38
base equal the dbSNP ref allele, and report the concordance. If concordance is low the run
warns loudly rather than silently emitting wrong sequences.

Output DataFrames match what data/splits.leakage_safe_split expects, so prepare_data.py reuses
the same locus split + leakage assertions + manifest machinery.
"""
from __future__ import annotations

import os
import re

import pandas as pd

from genome import Genome, revcomp
import dbsnp

_INSERT_RE = re.compile(r"^(chr[\w]+):(\d+)-(\d+)$")

_S1 = "adh0559_data_s1.xlsx"
_S2 = os.path.join("adh0559_data_s2", "DataS2-Variant-library-ratios.xlsx")


# --------------------------------------------------------------------------- parsing
def _parse_insert(name: str):
    """'chr10:100006370-100006640' -> ('chr10', 100006370, 100006640). None if malformed."""
    m = _INSERT_RE.match(str(name))
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


# --------------------------------------------------------------------------- elements
def build_elements(genome: Genome, s1_path: str, *, base_offset: int) -> pd.DataFrame:
    """Join S1 Primary+Organoids on insert_name; reconstruct each element's sequence.

    `base_offset` (0 for BED/0-based, 1 for 1-based-inclusive insert coords) is chosen by the
    detector in load_deng() and applied to the window start.
    """
    prim = pd.read_excel(s1_path, sheet_name="Primary",
                         usecols=["insert_chrom", "insert_start", "insert_end",
                                  "insert_name", "rna_dna_ratio"])
    org = pd.read_excel(s1_path, sheet_name="Organoids",
                        usecols=["insert_name", "rna_dna_ratio"])
    org = org.rename(columns={"rna_dna_ratio": "activity_organoid"})
    df = prim.rename(columns={"rna_dna_ratio": "activity_primary",
                              "insert_chrom": "chrom"}).merge(org, on="insert_name", how="left")

    seqs, keep = [], []
    for chrom, start, end in zip(df["chrom"], df["insert_start"], df["insert_end"]):
        if chrom in genome:
            seqs.append(genome.fetch(chrom, int(start) - base_offset, int(end) - base_offset))
        else:
            seqs.append(None)
        keep.append(chrom in genome)
    df["sequence"] = seqs
    df["element_id"] = df["insert_name"]
    df["pos"] = (df["insert_start"] + df["insert_end"]) // 2
    df = df.rename(columns={"insert_start": "start", "insert_end": "end"})
    df = df[df["sequence"].notna()].reset_index(drop=True)
    return df[["element_id", "chrom", "start", "end", "pos", "sequence",
               "activity_primary", "activity_organoid"]]


# --------------------------------------------------------------------------- variants
def build_variants(genome: Genome, s2_path: str, allele_map: dict, *, base_offset: int,
                   emvar_fdr: float = 0.10, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Build the calibration variant set: alleles from dbSNP, sequences from hg38.

    Returns (variants_df, diagnostics). Drops rows with no measured skew, an unresolved rsID,
    a ref/hg38 mismatch, or a variant that falls outside its element window.
    """
    v = pd.read_excel(s2_path, sheet_name="Primary",
                      usecols=["rsid", "variant_chrom", "variant_pos", "insert_name",
                               "logFC", "adj.P.Val"])
    v = v.rename(columns={"adj.P.Val": "fdr"})             # clean identifier for itertuples
    v = v[v["logFC"].notna()].reset_index(drop=True)

    rows = []
    diag = {"n_in": int(len(v)), "no_allele": 0, "bad_insert": 0, "out_of_window": 0,
            "ref_mismatch_fwd": 0, "rescued_revcomp": 0, "multiallelic": 0, "not_in_fasta": 0}
    for r in v.itertuples(index=False):
        rec = allele_map.get(str(r.rsid))
        if not rec:
            diag["no_allele"] += 1
            continue
        parsed = _parse_insert(r.insert_name)
        if not parsed:
            diag["bad_insert"] += 1
            continue
        chrom, i_start, i_end = parsed
        if chrom not in genome:
            diag["not_in_fasta"] += 1
            continue
        w_start = i_start - base_offset
        seq_ref = genome.fetch(chrom, w_start, i_end - base_offset)
        idx = (int(r.variant_pos) - 1) - w_start           # 0-based index of variant in window
        if idx < 0 or idx >= len(seq_ref):
            diag["out_of_window"] += 1
            continue

        ref, alts = rec["ref"], rec["alts"]
        genome_base = seq_ref[idx]
        alt = alts[0]
        if len(alts) > 1:
            diag["multiallelic"] += 1
        if genome_base == ref:
            pass
        elif genome_base == revcomp(ref) and len(ref) == 1:   # variant reported on - strand
            ref, alt = revcomp(ref), revcomp(alts[0])
            diag["rescued_revcomp"] += 1
        else:
            diag["ref_mismatch_fwd"] += 1
            continue
        seq_alt = seq_ref[:idx] + alt + seq_ref[idx + 1:]
        fdr = float(r.fdr) if pd.notna(r.fdr) else None
        rows.append({
            "chrom": chrom, "pos": int(r.variant_pos), "ref": ref, "alt": alt,
            "seq_ref": seq_ref, "seq_alt": seq_alt,
            "measured_skew": float(r.logFC),
            "fdr": fdr,
            "is_emvar": bool(fdr is not None and fdr <= emvar_fdr),
            "rsid": str(r.rsid),
        })
    out = pd.DataFrame(rows)
    diag["n_out"] = int(len(out))
    if verbose:
        print(f"[deng] variants: {diag['n_out']}/{diag['n_in']} kept | "
              f"no_allele={diag['no_allele']} ref_mismatch={diag['ref_mismatch_fwd']} "
              f"revcomp_rescued={diag['rescued_revcomp']} out_of_window={diag['out_of_window']} "
              f"multiallelic={diag['multiallelic']}")
    return out, diag


# --------------------------------------------------------------------------- coord detection
def _detect_base_offset(genome: Genome, s2_path: str, allele_map: dict, sample: int = 800) -> dict:
    """Sanity-check ref-allele concordance and pick the insert base_offset.

    NOTE: the concordance is essentially invariant to the 0-vs-1 element-window offset (both
    place the VARIANT base identically; they differ only by ±1 bp at the window EDGE). So the
    real job of this check is a genome/build guard — if the FASTA is the wrong assembly (hg19)
    or has mismatched contig names, concordance collapses and load_deng() warns. We default to
    BED 0-based (the UCSC/`insert_*` convention) when the signal can't separate them.
    """
    v = pd.read_excel(s2_path, sheet_name="Primary",
                      usecols=["rsid", "variant_pos", "insert_name", "logFC"])
    v = v[v["logFC"].notna()].reset_index(drop=True).head(sample)
    scores = {}
    for off in (0, 1):
        ok = tot = 0
        for r in v.itertuples(index=False):
            rec = allele_map.get(str(r.rsid))
            parsed = _parse_insert(r.insert_name)
            if not rec or not parsed or parsed[0] not in genome:
                continue
            chrom, i_start, i_end = parsed
            w_start = i_start - off
            idx = (int(r.variant_pos) - 1) - w_start
            seq = genome.fetch(chrom, w_start, i_end - off)
            if 0 <= idx < len(seq):
                tot += 1
                b = seq[idx]
                if b == rec["ref"] or (len(rec["ref"]) == 1 and b == revcomp(rec["ref"])):
                    ok += 1
        scores[off] = (ok / tot) if tot else 0.0
    best = max(scores, key=scores.get)
    return {"base_offset": best, "concordance_by_offset": scores,
            "chosen_concordance": scores[best]}


# --------------------------------------------------------------------------- orchestrator
def load_deng(deng_dir: str, genome_path: str, *, emvar_fdr: float = 0.10,
              cache_path: str | None = None, verbose: bool = True):
    """Load the Deng supplement -> (elements_df, variants_df, meta) for prepare_data.py."""
    s1_path = os.path.join(deng_dir, _S1)
    s2_path = os.path.join(deng_dir, _S2)
    for p in (s1_path, s2_path):
        if not os.path.isfile(p):
            raise SystemExit(f"Deng table not found: {p}\n"
                             "Point --deng-dir at the folder holding adh0559_data_s1.xlsx etc.")
    genome = Genome(genome_path)

    # 1. resolve alleles for every S2 rsID (cached)
    rsids = pd.read_excel(s2_path, sheet_name="Primary", usecols=["rsid", "logFC"])
    rsids = rsids[rsids["logFC"].notna()]["rsid"].tolist()
    kw = {"cache_path": cache_path} if cache_path else {}
    allele_map = dbsnp.resolve_rsids(rsids, verbose=verbose, **kw)
    n_resolved = sum(1 for a in allele_map.values() if a)
    if verbose:
        print(f"[deng] alleles resolved: {n_resolved}/{len(allele_map)} rsIDs")

    # 2. detect the coordinate convention empirically
    det = _detect_base_offset(genome, s2_path, allele_map)
    if verbose:
        print(f"[deng] coord detect: offset={det['base_offset']} "
              f"concordance={det['concordance_by_offset']}")
    if det["chosen_concordance"] < 0.9:
        print(f"[deng] ⚠ WARNING: ref-allele concordance only {det['chosen_concordance']:.1%} — "
              "check FASTA build (hg38?) and contig naming before trusting sequences.")

    # 3. build both tables under the chosen convention
    off = det["base_offset"]
    elements = build_elements(genome, s1_path, base_offset=off)
    variants, vdiag = build_variants(genome, s2_path, allele_map, base_offset=off,
                                     emvar_fdr=emvar_fdr, verbose=verbose)
    if verbose:
        print(f"[deng] elements: {len(elements)} | variants: {len(variants)} | "
              f"emVars(fdr<={emvar_fdr}): {int(variants['is_emvar'].sum())}")

    meta = {"mode": "deng", "genome": os.path.basename(genome_path),
            "n_rsid": len(allele_map), "n_alleles_resolved": n_resolved,
            "coord_detection": det, "variant_diag": vdiag, "emvar_fdr": emvar_fdr,
            "element_len": int((elements["end"] - elements["start"]).mode().iloc[0]) if len(elements) else None}
    return elements, variants, meta
