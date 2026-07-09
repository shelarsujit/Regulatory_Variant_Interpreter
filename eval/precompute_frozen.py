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
                             # eval FIRST so a partial --limit covers the grade slice (which is what
                             # a meaningful meta grade needs); then val + train for the fit set.
                             ("eval_variants_siamese.parquet", "val_variants.parquet",
                              "train_variants.parquet", "calibration_variants.parquet")],
                    help="variant tables to union over (missing ones are skipped). Order matters "
                         "with --limit: earlier tables are scored first.")
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

    # cache is ACCUMULATIVE: seed with everything already computed, add newly scored, write the
    # whole thing. So partial/limited/reordered runs never lose prior work — coverage only grows.
    cache = {k: float(v) for k, v in done.items()}

    def _flush():
        pd.DataFrame([{"chrom": c, "pos": p, "ref": rf, "alt": al, "frozen_delta": dv}
                      for (c, p, rf, al), dv in cache.items()]).to_parquet(args.out, index=False)

    scored, dropped, t0 = 0, 0, time.time()
    for i, r in enumerate(allv.itertuples(), 1):
        key = (r.chrom, int(r.pos), r.ref, r.alt)
        if key in cache:
            continue
        try:
            d = fn(r.chrom, int(r.pos), r.ref, r.alt)
        except Exception as e:
            print(f"[frozen] {key} failed: {type(e).__name__}: {e}")
            d = None
        if d is None:
            dropped += 1
            continue
        cache[key] = float(d)
        scored += 1
        if scored % 100 == 0:
            rate = scored / (time.time() - t0)
            print(f"[frozen] {i}/{len(allv)}  (+{scored} new, {rate:.1f}/s)")
            _flush()                                                # checkpoint

    _flush()
    print(f"\n[frozen] DONE — cache now {len(cache)} Δ (+{scored} new, {dropped} dropped) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
