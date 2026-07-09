#!/usr/bin/env python
"""Precompute the frozen-model (Enformer) zero-shot Δ for every variant — ONCE, then cache.

WHY (docs/08 §4): Enformer is ~1 s/variant on GPU; ~15k variants ≈ hours. Score once, persist to
`data/processed/frozen_delta_cache.parquet`, and every later meta refit is CPU-only (same pattern
as `siamese_eval_predictions.parquet`). GPU-only (the model is large); run on Colab/an A100.

Reads the variant tables (train/val/eval variant slices + the full calibration set), scores each
UNIQUE (chrom,pos,ref,alt) with `src.foundation.make_enformer_fn`, and writes
    data/processed/frozen_delta_cache.parquet   [chrom, pos, ref, alt, frozen_delta]
keyed by variant identity so all slices draw from one file. Coverage + ref-mismatch drops logged.

RUN (GPU):
    python eval/precompute_frozen.py --genome data/raw/genome/hg38.fa
    python eval/precompute_frozen.py --genome ... --tracks 4980,4981 --limit 50   # smoke test
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "data"))


def main(argv=None):
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    proc = os.path.join(_ROOT, "data", "processed")
    ap.add_argument("--genome", default=os.path.join(_ROOT, "data", "raw", "genome", "hg38.fa"),
                    help="hg38 FASTA (Enformer pulls its own 196 kb window from this)")
    ap.add_argument("--tables", nargs="+",
                    default=[os.path.join(proc, f) for f in
                             ("train_variants.parquet", "val_variants.parquet",
                              "eval_variants_siamese.parquet", "calibration_variants.parquet")],
                    help="variant tables to union over (missing ones are skipped)")
    ap.add_argument("--out", default=os.path.join(proc, "frozen_delta_cache.parquet"))
    ap.add_argument("--checkpoint", default="EleutherAI/enformer-official-rough")
    ap.add_argument("--tracks", default=None, help="comma-separated Enformer track indices (else default)")
    ap.add_argument("--center-bins", type=int, default=3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--require-ref-match", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap unique variants (smoke test)")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.genome):
        raise SystemExit(f"genome FASTA not found: {args.genome}")

    # union unique variants across the available tables
    frames = []
    for t in args.tables:
        if os.path.isfile(t):
            df = pd.read_parquet(t, columns=["chrom", "pos", "ref", "alt"])
            frames.append(df)
            print(f"[frozen] {os.path.basename(t)}: {len(df)} rows")
    if not frames:
        raise SystemExit("no variant tables found; run data/make_variant_pairs.py first")
    allv = pd.concat(frames, ignore_index=True).drop_duplicates(["chrom", "pos", "ref", "alt"])
    allv = allv.reset_index(drop=True)
    if args.limit:
        allv = allv.head(args.limit)
    print(f"[frozen] {len(allv)} unique variants to score")

    # resume: skip variants already cached
    done = {}
    if os.path.isfile(args.out):
        prev = pd.read_parquet(args.out)
        done = {(r.chrom, int(r.pos), r.ref, r.alt): r.frozen_delta for r in prev.itertuples()}
        print(f"[frozen] resuming — {len(done)} already cached")

    import torch
    from genome import Genome
    from src.foundation import make_enformer_fn
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device != "cuda":
        print("[frozen] ⚠️ no CUDA — Enformer on CPU is extremely slow; this is meant for GPU.")
    tracks = tuple(int(x) for x in args.tracks.split(",")) if args.tracks else None
    genome = Genome(args.genome)
    fn = make_enformer_fn(genome, device=device, checkpoint=args.checkpoint,
                          require_ref_match=args.require_ref_match,
                          **({"tracks": tracks} if tracks else {}),
                          center_bins=args.center_bins)

    rows, dropped, t0 = [], 0, time.time()
    for i, r in enumerate(allv.itertuples(), 1):
        key = (r.chrom, int(r.pos), r.ref, r.alt)
        d = done.get(key)
        if d is None:
            try:
                d = fn(r.chrom, int(r.pos), r.ref, r.alt)
            except Exception as e:
                print(f"[frozen] {key} failed: {type(e).__name__}: {e}")
                d = None
        if d is None:
            dropped += 1
            continue
        rows.append({"chrom": r.chrom, "pos": int(r.pos), "ref": r.ref, "alt": r.alt,
                     "frozen_delta": float(d)})
        if i % 100 == 0:
            rate = i / (time.time() - t0)
            print(f"[frozen] {i}/{len(allv)}  ({rate:.1f}/s, ~{(len(allv)-i)/max(rate,1e-9)/60:.0f} min left)")
            pd.DataFrame(rows).to_parquet(args.out, index=False)   # checkpoint periodically

    out = pd.DataFrame(rows)
    out.to_parquet(args.out, index=False)
    print(f"\n[frozen] DONE — cached {len(out)} Δ ({dropped} dropped) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
