#!/usr/bin/env python
"""Fit the siamese isotonic calibrator OFFLINE (CPU, no torch) from a predictions dump.

The siamese model is Caduceus (GPU-only), so scoring it needs a GPU. But once
`eval/eval_siamese.py --dump-predictions` has saved the per-variant siamese Δ + labels, the
calibrator is just an isotonic fit on those numbers — pure pandas/numpy, runnable anywhere.
So GPU is needed exactly ONCE (to produce the dump); every later refit (different τ, etc.) is CPU.

RUN
    python eval/fit_calibrator_from_dump.py --dump weights/siamese_eval_predictions.parquet
    #   -> weights/calibrator_siamese.json   (drop-in for app.py; auto-preferred when serving siamese)
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main(argv=None):
    import pandas as pd
    from src.trust import Calibrator

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True,
                    help="parquet/csv from eval_siamese.py --dump-predictions "
                         "(cols: siamese_delta, measured_skew, is_emvar)")
    ap.add_argument("--out", default=os.path.join(_ROOT, "weights", "calibrator_siamese.json"))
    ap.add_argument("--tau", type=float, default=0.5, help="|skew| counted as a real effect (D6)")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.dump):
        raise SystemExit(f"dump not found: {args.dump}\n"
                         "Produce it on a GPU with: eval/eval_siamese.py --dump-predictions <path>")
    df = pd.read_parquet(args.dump) if args.dump.endswith(".parquet") else pd.read_csv(args.dump)
    for col in ("siamese_delta", "measured_skew"):
        if col not in df.columns:
            raise SystemExit(f"dump missing column {col!r}; has {list(df.columns)}")
    is_emvar = df["is_emvar"].astype(int).to_numpy() if "is_emvar" in df.columns else None

    cal = Calibrator().fit(df["siamese_delta"].to_numpy(), df["measured_skew"].to_numpy(),
                           is_emvar=is_emvar, tau=args.tau)
    cal.save(args.out)
    print(f"[calib] fit on {len(df)} variants -> {args.out}")
    print(f"[calib] diagnostics: {cal.diagnostics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
