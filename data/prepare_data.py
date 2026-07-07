#!/usr/bin/env python
"""Day-1 data preparation: Deng et al. cortical MPRA -> training set + held-out calibration set.

WHAT THIS PRODUCES
  data/processed/train.parquet                (seq, activity_primary[, activity_organoid], locus)
  data/processed/val.parquet                  same schema, locus-disjoint from train
  data/processed/calibration_variants.parquet (chrom,pos,ref,alt, seq_ref, seq_alt, measured_skew,
                                               fdr, is_emvar, locus)  <- held-out ground truth
  data/processed/manifest.json                params, column mapping, row counts, git commit,
                                               and the leakage-check result (provenance)

WHY IT'S SHAPED THIS WAY (see docs/02_decision_log.md, D3/D5/D8)
  * Source of truth = the PROCESSED *Science* supplement tables, not the raw Synapse
    FASTQs (which need a Data-Use Agreement + a full reprocessing pipeline). You download
    the supplement yourself and drop the tables in data/raw/; this script consumes them.
  * The exact column headers of that supplement are not yet verified (egress to the
    journal is blocked in the build sandbox), so we resolve columns *fuzzily* and record
    the mapping we chose in manifest.json. Fix RESOLVER / pass --col if a header is missed.
  * Leakage discipline is enforced by data/splits.py and ASSERTED here — the run aborts if
    any calibration variant shares a genomic locus with a training sequence.

RUN IT TODAY (no download needed) — proves the whole pipeline on a synthetic MPRA-like fixture:
    python data/prepare_data.py --synthetic

RUN IT FOR REAL (once the supplement tables are in data/raw/):
    python data/prepare_data.py \
        --element-table data/raw/elements_activity.xlsx \
        --variant-table data/raw/variant_skew.xlsx
    # add --col activity_primary=<header> etc. if the fuzzy resolver misses a column.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

# make sibling module `splits` importable when run as `python data/prepare_data.py`
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import splits  # noqa: E402

ALPHABET = "ACGT"

# logical field -> ordered candidate header substrings (lowercased, first match wins)
RESOLVER: dict[str, list[str]] = {
    "sequence": ["sequence", "seq", "oligo", "element"],
    "chrom": ["chrom", "chr", "seqnames"],
    "start": ["start", "chromstart"],
    "end": ["end", "chromend"],
    "pos": ["pos", "position", "snp_pos", "variant_pos"],
    "activity_primary": ["primary", "cortex", "activity_primary", "log2_primary", "mean_primary"],
    "activity_organoid": ["organoid", "activity_organoid", "log2_organoid", "mean_organoid"],
    "activity": ["activity", "log2", "rna_dna", "expression"],  # generic fallback for primary
    "element_id": ["element_id", "oligo_id", "name", "id"],
    "ref": ["ref", "reference", "allele1", "a1"],
    "alt": ["alt", "alternate", "allele2", "a2", "effect_allele"],
    "seq_ref": ["seq_ref", "ref_seq", "sequence_ref"],
    "seq_alt": ["seq_alt", "alt_seq", "sequence_alt"],
    "measured_skew": ["skew", "allelic", "log2fc", "logfc", "fold", "effect_size", "beta"],
    "fdr": ["fdr", "padj", "qval", "adj_p"],
    "is_emvar": ["emvar", "significant", "is_sig"],
}


# --------------------------------------------------------------------------- column resolver
def resolve(columns, field, override):
    if override and field in override:
        return override[field]
    lower = {c.lower(): c for c in columns}
    for cand in RESOLVER.get(field, []):
        for lc, original in lower.items():
            if cand in lc:
                return original
    return None


# --------------------------------------------------------------------------- synthetic fixture
def _rand_seq(rng, n):
    return "".join(rng.choice(ALPHABET) for _ in range(n))


def make_synthetic(n_elements, n_variants, seq_len, seed, n_leak_decoys):
    """Generate a small MPRA-like dataset so the pipeline runs end-to-end with no download.

    Activity depends on GC-content + a toy motif so the labels are learnable (not noise),
    and organoid activity is a noisier copy of primary (mirroring the real 'comparable
    across contexts' finding). We deliberately inject `n_leak_decoys` elements placed
    exactly on variant loci to PROVE the leakage filter removes them.
    """
    rng = np.random.default_rng(seed)
    prng = __import__("random").Random(seed)
    chroms = [f"chr{i}" for i in range(1, 23)]
    mid = seq_len // 2

    el = []
    for i in range(n_elements):
        seq = _rand_seq(prng, seq_len)
        gc = (seq.count("G") + seq.count("C")) / seq_len
        motif = 1.0 if "TGACTCA" in seq else 0.0            # toy AP-1-like motif
        base = 2.5 * gc + 0.8 * motif
        start = int(prng.randint(1_000_000, 50_000_000))
        el.append({
            "element_id": f"E{i:06d}",
            "chrom": prng.choice(chroms),
            "start": start,
            "end": start + seq_len,
            "pos": start + mid,
            "sequence": seq,
            "activity_primary": round(base + rng.normal(0, 0.30), 4),
            "activity_organoid": round(base + rng.normal(0, 0.45), 4),
        })
    elements = pd.DataFrame(el)

    va = []
    for _ in range(n_variants):
        seq = _rand_seq(prng, seq_len)
        ref = seq[mid]
        alt = prng.choice([b for b in ALPHABET if b != ref])
        skew = float(rng.normal(0, 0.15))
        is_em = abs(skew) > 0.40                              # ~1% end up "significant"
        if is_em:
            skew *= 2.0
        start = int(prng.randint(60_000_000, 120_000_000))    # different genomic band from elements
        va.append({
            "chrom": prng.choice(chroms),
            "pos": start + mid,
            "ref": ref,
            "alt": alt,
            "seq_ref": seq,
            "seq_alt": seq[:mid] + alt + seq[mid + 1:],
            "measured_skew": round(skew, 4),
            "fdr": round(float(rng.uniform(0, 1)), 4),
            "is_emvar": bool(is_em),
        })
    variants = pd.DataFrame(va)

    # Inject leak decoys: elements sitting exactly on variant loci. A correct split must
    # drop every one of these from training.
    decoys = []
    for k in range(min(n_leak_decoys, len(variants))):
        v = variants.iloc[k]
        decoys.append({
            "element_id": f"DECOY{k:04d}",
            "chrom": v["chrom"],
            "start": int(v["pos"]) - mid,
            "end": int(v["pos"]) + mid,
            "pos": int(v["pos"]),
            "sequence": _rand_seq(prng, seq_len),
            "activity_primary": 0.0,
            "activity_organoid": 0.0,
        })
    if decoys:
        elements = pd.concat([elements, pd.DataFrame(decoys)], ignore_index=True)

    return elements, variants, {"mode": "synthetic", "n_leak_decoys_injected": len(decoys)}


# --------------------------------------------------------------------------- real loader
def _read_any(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if path.lower().endswith((".tsv", ".txt")):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def load_real(element_table, variant_table, override, seq_len):
    if not (element_table and variant_table):
        raise SystemExit(
            "real mode needs --element-table and --variant-table (or use --synthetic).\n"
            "See docs/01_data_provenance.md for what these tables are."
        )
    e_raw, v_raw = _read_any(element_table), _read_any(variant_table)

    emap = {f: resolve(e_raw.columns, f, override)
            for f in ["element_id", "chrom", "start", "end", "pos",
                      "sequence", "activity_primary", "activity_organoid", "activity"]}
    vmap = {f: resolve(v_raw.columns, f, override)
            for f in ["chrom", "pos", "ref", "alt", "seq_ref", "seq_alt",
                      "measured_skew", "fdr", "is_emvar"]}

    elements = pd.DataFrame()
    elements["element_id"] = e_raw[emap["element_id"]] if emap["element_id"] else [f"E{i}" for i in range(len(e_raw))]
    if not emap["chrom"]:
        raise SystemExit("element table: could not find a chromosome column (pass --col chrom=<header>)")
    elements["chrom"] = e_raw[emap["chrom"]].astype(str)
    if emap["pos"]:
        elements["pos"] = e_raw[emap["pos"]].astype(int)
    elif emap["start"] and emap["end"]:
        elements["pos"] = (e_raw[emap["start"]].astype(int) + e_raw[emap["end"]].astype(int)) // 2
    else:
        raise SystemExit("element table needs `pos` OR `start`+`end` for the locus split")
    for opt in ("start", "end", "sequence"):
        if emap[opt]:
            elements[opt] = e_raw[emap[opt]]
    ap = emap["activity_primary"] or emap["activity"]
    if not ap:
        raise SystemExit("element table: could not find a primary-activity column (pass --col activity_primary=<header>)")
    elements["activity_primary"] = e_raw[ap]
    if emap["activity_organoid"]:
        elements["activity_organoid"] = e_raw[emap["activity_organoid"]]

    variants = pd.DataFrame()
    for req in ("chrom", "pos", "ref", "alt"):
        if not vmap[req]:
            raise SystemExit(f"variant table: missing required column `{req}` (pass --col {req}=<header>)")
    variants["chrom"] = v_raw[vmap["chrom"]].astype(str)
    variants["pos"] = v_raw[vmap["pos"]].astype(int)
    variants["ref"] = v_raw[vmap["ref"]].astype(str)
    variants["alt"] = v_raw[vmap["alt"]].astype(str)
    for opt in ("seq_ref", "seq_alt", "measured_skew", "fdr", "is_emvar"):
        if vmap[opt]:
            variants[opt] = v_raw[vmap[opt]]

    return elements, variants, {"mode": "real", "element_map": emap, "variant_map": vmap}


# --------------------------------------------------------------------------- io helpers
def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_HERE,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _write_table(df, path):
    """Parquet if pyarrow is available, else CSV (records the choice by extension)."""
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        alt = path.rsplit(".", 1)[0] + ".csv"
        df.to_csv(alt, index=False)
        return alt


def parse_col_overrides(pairs):
    override = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--col expects field=header, got: {p}")
        field, header = p.split("=", 1)
        override[field.strip()] = header.strip()
    return override


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="generate a fixture instead of reading real tables")
    ap.add_argument("--deng-dir", help="folder with the real Deng supplement (adh0559_data_s1.xlsx etc.); "
                                       "uses data/load_deng.py (hg38 reconstruction + dbSNP alleles)")
    ap.add_argument("--genome", help="path to a local hg38 FASTA (required with --deng-dir)")
    ap.add_argument("--dbsnp-cache", help="path for the rsID->allele cache (default data/raw/dbsnp_cache.json)")
    ap.add_argument("--emvar-fdr", type=float, default=0.10, help="adj.P.Val threshold flagged as an emVar")
    ap.add_argument("--element-table", help="path to the element-activity supplement table (generic loader)")
    ap.add_argument("--variant-table", help="path to the variant allelic-skew supplement table (generic loader)")
    ap.add_argument("--col", action="append", default=[], help="override a column mapping, e.g. --col activity_primary=Primary_log2")
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "processed"))
    ap.add_argument("--seq-len", type=int, default=200, help="modeled element length (bp); verify against Methods")
    ap.add_argument("--locus-bin", type=int, default=splits.LOCUS_BIN_DEFAULT, help="bp per locus bucket for the leakage-safe split")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--guard-neighbors", type=int, default=1, help="also hold out +/- N neighbor buckets around each variant")
    ap.add_argument("--seed", type=int, default=7)
    # synthetic knobs
    ap.add_argument("--n-elements", type=int, default=5000)
    ap.add_argument("--n-variants", type=int, default=2000)
    ap.add_argument("--n-leak-decoys", type=int, default=25)
    args = ap.parse_args(argv)

    override = parse_col_overrides(args.col)

    if args.synthetic:
        elements, variants, load_meta = make_synthetic(
            args.n_elements, args.n_variants, args.seq_len, args.seed, args.n_leak_decoys)
    elif args.deng_dir:
        if not args.genome:
            raise SystemExit("--deng-dir requires --genome <hg38.fa> (sequences are reconstructed "
                             "from the reference; see docs/01_data_provenance.md §4).")
        import load_deng  # lazy: pulls pyfaidx + network (Ensembl)
        elements, variants, load_meta = load_deng.load_deng(
            args.deng_dir, args.genome, emvar_fdr=args.emvar_fdr, cache_path=args.dbsnp_cache)
    else:
        elements, variants, load_meta = load_real(args.element_table, args.variant_table, override, args.seq_len)

    split = splits.leakage_safe_split(
        elements, variants,
        locus_bin=args.locus_bin, val_frac=args.val_frac,
        seed=args.seed, guard_neighbors=args.guard_neighbors,
    )

    # HARD GATES — abort the run if leakage discipline is violated.
    splits.assert_no_leakage(split["train"], split["val"], split["calibration"])
    splits.assert_no_sequence_overlap(split["train"], split["calibration"])

    os.makedirs(args.out_dir, exist_ok=True)
    outputs = {
        "train": _write_table(split["train"], os.path.join(args.out_dir, "train.parquet")),
        "val": _write_table(split["val"], os.path.join(args.out_dir, "val.parquet")),
        "calibration_variants": _write_table(split["calibration"], os.path.join(args.out_dir, "calibration_variants.parquet")),
    }

    manifest = {
        "created_utc": _dt.datetime.utcnow().isoformat() + "Z",
        "git_commit": _git_commit(),
        "load": load_meta,
        "params": {
            "seq_len": args.seq_len, "locus_bin": args.locus_bin, "val_frac": args.val_frac,
            "guard_neighbors": args.guard_neighbors, "seed": args.seed,
        },
        "stats": split["stats"],
        "leakage_check": "PASS",
        "outputs": outputs,
        "notes": "Source of truth = processed Science supplement (docs/01_data_provenance.md). "
                 "Verify columns flagged ⚠ on first real download.",
    }
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    s = split["stats"]
    print("── data prep complete ──────────────────────────────────────────")
    print(f"  mode                 : {load_meta['mode']}")
    print(f"  elements in          : {s['n_elements_in']}")
    print(f"  removed for leakage  : {s['n_removed_for_leakage']}  (variant-locus elements held out of training)")
    print(f"  train / val elements : {s['n_train']} / {s['n_val']}")
    print(f"  calibration variants : {s['n_calibration']}  (emVars: {s['n_emvar']})")
    print(f"  leakage check        : PASS  ✅")
    print(f"  outputs              : {args.out_dir}/  (+ manifest.json)")
    print("────────────────────────────────────────────────────────────────")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
