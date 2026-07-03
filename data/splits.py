"""Leakage-safe, locus-grouped dataset splitting.

THEORY (plain English — see docs/00_overview_for_non_biologists.md §4.3):
Our model learns "DNA sequence -> regulatory activity." We then test our *trust* by
comparing the model's predicted effect of a single-base change against the lab-measured
effect for ~17,000 real variants. The cardinal sin here is **leakage**: if the model was
trained on a sequence that is (nearly) identical to one we grade it on, our confidence
numbers are self-congratulatory fiction.

Why this is a real trap, not a theoretical one: a variant's *reference* sequence is
literally one of the MPRA library elements (with one letter swapped), and the library
tiles genomic regions with *overlapping* windows. So two different rows can share most of
their bases. A naive random row split would scatter near-duplicates across train and
calibration.

Defense: group every sequence into a genomic **locus** bucket and make the split
*group-disjoint* — an entire locus goes to training OR to calibration, never both — and
guard the immediate neighbor buckets too (to catch tiles that straddle a bucket edge).
The disjointness is asserted by `assert_no_leakage`; callers must abort if it fails.

Inputs expected (pandas DataFrames):
  elements : needs columns  chrom, pos   (+ whatever payload: sequence, activity_*)
  variants : needs columns  chrom, pos   (+ payload: ref, alt, seq_ref, measured_skew, ...)
             `pos` for an element = its genomic midpoint (compute upstream if you only
             have start/end).
"""
from __future__ import annotations

import random
from typing import Iterable

import pandas as pd

LOCUS_BIN_DEFAULT = 10_000   # bases per locus bucket; bigger = stronger independence, less train data


def assign_locus(chrom: str, pos: int, bin_size: int) -> str:
    """Map a genomic coordinate to its locus bucket id, e.g. ('chr2', 162279995) -> 'chr2:16227'."""
    return f"{chrom}:{int(pos) // int(bin_size)}"


def _expand_neighbors(loci: Iterable[str], guard: int) -> set[str]:
    """Return the input loci plus ±`guard` neighboring buckets on each chromosome."""
    out: set[str] = set()
    for locus in loci:
        chrom, idx = locus.rsplit(":", 1)
        idx = int(idx)
        for delta in range(-guard, guard + 1):
            out.add(f"{chrom}:{idx + delta}")
    return out


def held_out_loci(variants: pd.DataFrame, *, locus_bin: int, guard_neighbors: int = 1) -> set[str]:
    """The set of locus buckets that must NEVER appear in training (variant loci + guard)."""
    loci = {assign_locus(c, p, locus_bin) for c, p in zip(variants["chrom"], variants["pos"])}
    return _expand_neighbors(loci, guard_neighbors) if guard_neighbors else loci


def leakage_safe_split(
    elements: pd.DataFrame,
    variants: pd.DataFrame,
    *,
    locus_bin: int = LOCUS_BIN_DEFAULT,
    val_frac: float = 0.1,
    seed: int = 7,
    guard_neighbors: int = 1,
) -> dict:
    """Split into train/val (elements) + calibration (variants) with NO locus overlap.

    Returns dict(train=df, val=df, calibration=df, stats=dict). Every returned frame
    carries a `locus` column so `assert_no_leakage` can verify disjointness.
    """
    elements = elements.copy()
    variants = variants.copy()

    elements["locus"] = [assign_locus(c, p, locus_bin) for c, p in zip(elements["chrom"], elements["pos"])]
    variants["locus"] = [assign_locus(c, p, locus_bin) for c, p in zip(variants["chrom"], variants["pos"])]

    banned = held_out_loci(variants, locus_bin=locus_bin, guard_neighbors=guard_neighbors)
    keep_mask = ~elements["locus"].isin(banned)
    n_removed = int((~keep_mask).sum())
    pool = elements[keep_mask].copy()

    # Group-level train/val split: shuffle whole loci, not rows, so overlapping tiles
    # stay together on the same side.
    loci = sorted(pool["locus"].unique())
    rng = random.Random(seed)
    rng.shuffle(loci)
    n_val = max(1, round(len(loci) * val_frac)) if loci else 0
    val_loci = set(loci[:n_val])

    pool["split"] = ["val" if locus in val_loci else "train" for locus in pool["locus"]]
    train = pool[pool["split"] == "train"].reset_index(drop=True)
    val = pool[pool["split"] == "val"].reset_index(drop=True)
    calibration = variants.reset_index(drop=True)

    stats = {
        "n_elements_in": int(len(elements)),
        "n_removed_for_leakage": n_removed,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_calibration": int(len(calibration)),
        "n_emvar": int(variants["is_emvar"].sum()) if "is_emvar" in variants else None,
        "locus_bin": locus_bin,
        "guard_neighbors": guard_neighbors,
        "val_frac": val_frac,
        "seed": seed,
    }
    return {"train": train, "val": val, "calibration": calibration, "stats": stats}


def assert_no_leakage(train: pd.DataFrame, val: pd.DataFrame, calibration: pd.DataFrame) -> None:
    """Hard gate: train/val/calibration must occupy disjoint loci. Raise otherwise."""
    tr, va, ca = set(train["locus"]), set(val["locus"]), set(calibration["locus"])
    if tr & ca:
        raise AssertionError(f"LEAKAGE: {len(tr & ca)} loci shared between train and calibration")
    if va & ca:
        raise AssertionError(f"LEAKAGE: {len(va & ca)} loci shared between val and calibration")
    if tr & va:
        raise AssertionError(f"LEAKAGE: {len(tr & va)} loci shared between train and val")


def assert_no_sequence_overlap(train: pd.DataFrame, calibration: pd.DataFrame) -> None:
    """Extra belt-and-braces check when sequences are available: no training sequence may
    equal a calibration reference/alternate sequence."""
    if "sequence" not in train:
        return
    train_seqs = set(train["sequence"])
    for col in ("seq_ref", "seq_alt"):
        if col in calibration:
            overlap = train_seqs & set(calibration[col].dropna())
            if overlap:
                raise AssertionError(
                    f"LEAKAGE: {len(overlap)} training sequences identical to calibration '{col}'"
                )
