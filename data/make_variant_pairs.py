#!/usr/bin/env python
"""Carve a leakage-safe SIAMESE training set out of the held-out variant library.

WHY THIS EXISTS (docs/07_enhancement_design.md, Enhancement #1)
  The activity models regress single-sequence activity, then `score_variant` *subtracts*
  alt-ref to get a Δ. That indirect Δ is the suspected cause of the 0.70 -> 0.15 drop.
  Enhancement #1 trains a shared-weight (siamese) head DIRECTLY on measured allelic skew
  (`measured_skew` / logFC). That needs a *training* set of (seq_ref, seq_alt, skew) pairs
  that is locus-disjoint from whatever we grade the model on.

ADDITIVE BY DESIGN — touches no earlier work.
  * READS  data/processed/calibration_variants.parquet   (unchanged, byte-identical)
  * WRITES data/processed/train_variants.parquet          (siamese TRAIN pairs)          [NEW]
           data/processed/val_variants.parquet            (siamese in-training VAL pairs) [NEW]
           data/processed/eval_variants_siamese.parquet   (held-out siamese EVAL subset)  [NEW]
           data/processed/variant_pairs_manifest.json     (split provenance + leakage check)[NEW]
  The original calibration table is never rewritten, so the existing activity-model
  eval/calibration numbers (docs/03) are unaffected. The siamese model is graded on
  `eval_variants_siamese.parquet` (a locus-disjoint held-out slice); for an apples-to-apples
  comparison the activity baseline should be re-scored on that SAME slice at eval time.

LEAKAGE DISCIPLINE (CLAUDE.md §3, non-negotiable)
  Split is locus-grouped: an entire locus bucket goes to train/val OR to eval, never both.
  Whole loci move together so overlapping MPRA tiles never straddle the split. Asserted
  before anything is written; the run aborts on any shared locus.

  Note on the element models: `prepare_data.py` already bans EVERY variant locus from the
  element training pool, so a siamese model initialised from an element-trained backbone is
  clean w.r.t. these eval variants too.

RUN
    python data/make_variant_pairs.py                       # defaults: 70/15/15 by locus
    python data/make_variant_pairs.py --eval-frac 0.2 --val-frac 0.1 --seed 7
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import subprocess
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import splits  # noqa: E402  (assign_locus, for parity with the element split)

_DEFAULT_IN = os.path.join(_HERE, "processed", "calibration_variants.parquet")
_REQUIRED = ("chrom", "pos", "ref", "alt", "seq_ref", "seq_alt", "measured_skew")


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


def _ensure_locus(df, locus_bin):
    """Guarantee a `locus` column; recompute if absent or a different bin is requested."""
    if "locus" in df.columns and locus_bin is None:
        return df
    bin_size = locus_bin or splits.LOCUS_BIN_DEFAULT
    df = df.copy()
    df["locus"] = [splits.assign_locus(c, p, bin_size) for c, p in zip(df["chrom"], df["pos"])]
    return df


def split_variant_pairs(variants: pd.DataFrame, *, eval_frac: float, val_frac: float,
                        seed: int, locus_bin: int | None):
    """Locus-grouped split of the variant library into train / val / held-out eval.

    Returns (train_df, val_df, eval_df, stats). Splitting is over whole loci so no locus
    bucket is shared across the three sets.
    """
    missing = [c for c in _REQUIRED if c not in variants.columns]
    if missing:
        raise SystemExit(f"input table missing required columns: {missing}")

    variants = _ensure_locus(variants, locus_bin)
    # drop rows with no measurable skew label — siamese needs a target
    labeled = variants[variants["measured_skew"].notna()].copy()
    n_dropped = int(len(variants) - len(labeled))

    loci = sorted(labeled["locus"].unique())
    rng = random.Random(seed)
    rng.shuffle(loci)

    n_eval = max(1, round(len(loci) * eval_frac)) if loci else 0
    n_val = max(1, round(len(loci) * val_frac)) if loci else 0
    eval_loci = set(loci[:n_eval])
    val_loci = set(loci[n_eval:n_eval + n_val])
    # everything else is train

    def _bucket(locus):
        if locus in eval_loci:
            return "eval"
        if locus in val_loci:
            return "val"
        return "train"

    labeled["pair_split"] = [_bucket(l) for l in labeled["locus"]]
    train = labeled[labeled["pair_split"] == "train"].reset_index(drop=True)
    val = labeled[labeled["pair_split"] == "val"].reset_index(drop=True)
    ev = labeled[labeled["pair_split"] == "eval"].reset_index(drop=True)

    def _emvars(df):
        return int(df["is_emvar"].sum()) if "is_emvar" in df else None

    stats = {
        "n_input": int(len(variants)),
        "n_dropped_no_skew": n_dropped,
        "n_labeled": int(len(labeled)),
        "n_loci": len(loci),
        "n_train": int(len(train)), "n_val": int(len(val)), "n_eval": int(len(ev)),
        "emvar_train": _emvars(train), "emvar_val": _emvars(val), "emvar_eval": _emvars(ev),
        "eval_frac": eval_frac, "val_frac": val_frac, "seed": seed,
        "locus_bin": locus_bin or splits.LOCUS_BIN_DEFAULT,
    }
    return train, val, ev, stats


def assert_pairs_disjoint(train, val, ev):
    """Hard gate: train / val / eval variant loci must be pairwise disjoint."""
    tr, va, ee = set(train["locus"]), set(val["locus"]), set(ev["locus"])
    if tr & ee:
        raise AssertionError(f"LEAKAGE: {len(tr & ee)} loci shared between siamese train and eval")
    if va & ee:
        raise AssertionError(f"LEAKAGE: {len(va & ee)} loci shared between siamese val and eval")
    if tr & va:
        raise AssertionError(f"LEAKAGE: {len(tr & va)} loci shared between siamese train and val")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", default=_DEFAULT_IN,
                    help="held-out variant library to carve from (default: calibration_variants.parquet)")
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "processed"))
    ap.add_argument("--eval-frac", type=float, default=0.15, help="fraction of LOCI held out for siamese eval")
    ap.add_argument("--val-frac", type=float, default=0.15, help="fraction of LOCI used for in-training val")
    ap.add_argument("--locus-bin", type=int, default=None,
                    help="recompute locus buckets at this bp size (default: reuse existing `locus` column)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    if not os.path.isfile(args.in_path):
        raise SystemExit(f"input not found: {args.in_path}\nRun data/prepare_data.py first.")

    variants = pd.read_parquet(args.in_path)
    train, val, ev, stats = split_variant_pairs(
        variants, eval_frac=args.eval_frac, val_frac=args.val_frac,
        seed=args.seed, locus_bin=args.locus_bin,
    )

    # HARD GATE — abort before writing if any locus is shared.
    assert_pairs_disjoint(train, val, ev)

    os.makedirs(args.out_dir, exist_ok=True)
    outputs = {
        "train_variants": _write_table(train, os.path.join(args.out_dir, "train_variants.parquet")),
        "val_variants": _write_table(val, os.path.join(args.out_dir, "val_variants.parquet")),
        "eval_variants_siamese": _write_table(ev, os.path.join(args.out_dir, "eval_variants_siamese.parquet")),
    }

    manifest = {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "source_table": os.path.abspath(args.in_path),
        "purpose": "siamese variant-effect objective (docs/07_enhancement_design.md #1)",
        "stats": stats,
        "leakage_check": "PASS",
        "outputs": outputs,
        "note": "Original calibration_variants.parquet is NOT modified. Grade the siamese model "
                "on eval_variants_siamese.parquet; re-score the activity baseline on the SAME "
                "slice for an apples-to-apples comparison.",
    }
    with open(os.path.join(args.out_dir, "variant_pairs_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print("── siamese variant-pair split ─────────────────────────────────")
    print(f"  source                : {os.path.basename(args.in_path)}  ({stats['n_input']} rows)")
    print(f"  labeled (has skew)    : {stats['n_labeled']}  (dropped {stats['n_dropped_no_skew']})")
    print(f"  loci                  : {stats['n_loci']}")
    print(f"  train / val / eval    : {stats['n_train']} / {stats['n_val']} / {stats['n_eval']}")
    print(f"  emVars  tr / va / ev  : {stats['emvar_train']} / {stats['emvar_val']} / {stats['emvar_eval']}")
    print(f"  leakage check         : PASS  ✅  (train/val/eval loci disjoint)")
    print(f"  outputs               : {args.out_dir}/  (+ variant_pairs_manifest.json)")
    print("───────────────────────────────────────────────────────────────")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
