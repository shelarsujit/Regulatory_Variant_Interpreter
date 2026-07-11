"""Additive tests for the siamese enhancement (docs/07 #1).

Scope kept to the leakage-critical, torch-free logic: the locus-grouped split in
data/make_variant_pairs.py must produce train/val/eval slices with PAIRWISE-DISJOINT loci
(the non-negotiable invariant, CLAUDE.md §3). The train->load->score roundtrip is exercised
by the CPU smoke run in the workflow, not here (it needs torch + a backbone download).

Run:  python tests/test_siamese.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "data"))
import make_variant_pairs as mvp  # noqa: E402


def _toy_variants(n=600, seed=3):
    import random
    rng = random.Random(seed)
    chroms = [f"chr{i}" for i in range(1, 6)]
    rows = []
    for _ in range(n):
        chrom = rng.choice(chroms)
        pos = rng.randint(1_000_000, 5_000_000)
        skew = rng.gauss(0, 0.2)
        rows.append({
            "chrom": chrom, "pos": pos, "ref": "A", "alt": "G",
            "seq_ref": "ACGT" * 20, "seq_alt": "ACGA" * 20,
            "measured_skew": round(skew, 4),
            "is_emvar": abs(skew) > 0.4,
        })
    return pd.DataFrame(rows)


def test_split_loci_are_pairwise_disjoint():
    df = _toy_variants()
    train, val, ev, stats = mvp.split_variant_pairs(
        df, eval_frac=0.2, val_frac=0.2, seed=7, locus_bin=10_000)
    # the hard gate must not raise
    mvp.assert_pairs_disjoint(train, val, ev)
    tr, va, ee = set(train["locus"]), set(val["locus"]), set(ev["locus"])
    assert not (tr & ee), "train/eval loci overlap"
    assert not (va & ee), "val/eval loci overlap"
    assert not (tr & va), "train/val loci overlap"
    # every labeled row is accounted for exactly once
    assert stats["n_train"] + stats["n_val"] + stats["n_eval"] == stats["n_labeled"]
    print(f"  ok: disjoint split — train/val/eval loci = "
          f"{len(tr)}/{len(va)}/{len(ee)}; rows = "
          f"{stats['n_train']}/{stats['n_val']}/{stats['n_eval']}")


def test_split_is_deterministic():
    df = _toy_variants()
    a = mvp.split_variant_pairs(df, eval_frac=0.2, val_frac=0.2, seed=11, locus_bin=10_000)[0]
    b = mvp.split_variant_pairs(df, eval_frac=0.2, val_frac=0.2, seed=11, locus_bin=10_000)[0]
    assert list(a["locus"]) == list(b["locus"]), "same seed must give identical split"
    print("  ok: split deterministic under fixed seed")


def test_unlabeled_rows_dropped():
    df = _toy_variants(n=50)
    df.loc[df.index[:5], "measured_skew"] = float("nan")
    _, _, _, stats = mvp.split_variant_pairs(
        df, eval_frac=0.2, val_frac=0.2, seed=7, locus_bin=10_000)
    assert stats["n_dropped_no_skew"] == 5, "rows with null skew must be dropped"
    assert stats["n_labeled"] == 45
    print("  ok: unlabeled (null-skew) rows dropped before splitting")


def _run_all():
    tests = [test_split_loci_are_pairwise_disjoint, test_split_is_deterministic,
             test_unlabeled_rows_dropped]
    print(f"running {len(tests)} siamese tests")
    print("-" * 60)
    for t in tests:
        print(t.__name__)
        t()
    print("-" * 60)
    print("ALL SIAMESE TESTS PASSED ✅")


if __name__ == "__main__":
    _run_all()
