#!/usr/bin/env python
"""Grade the siamese variant-effect model — and the activity baseline — on the SAME held-out slice.

WHY (docs/07_enhancement_design.md #1):
The siamese model is trained on `train_variants.parquet` and MUST be graded only on the
locus-disjoint `eval_variants_siamese.parquet` (produced by data/make_variant_pairs.py). For an
honest comparison this harness ALSO re-scores the existing activity model (ActivityPredictor's
subtract-the-endpoints Δ) on that exact same slice — so any win is attributable to the objective,
not to a different test set. The success gate is: siamese emVar AUC (and Δ-Pearson) beats the
activity baseline ON THIS SLICE.

ADDITIVE: NEW file. Reads existing artifacts; writes only its own results json. Does not touch
calibration_variants.parquet, the fitted calibrators, or docs/03 numbers.

RUN
    # baseline only (no siamese trained yet) — establishes the activity Δ number on the eval slice:
    python eval/eval_siamese.py --activity-weights weights/primary
    # full comparison once a siamese model exists:
    python eval/eval_siamese.py --siamese-weights weights/siamese_primary --activity-weights weights/primary
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _pearson(a, b):
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else float("nan")


def _spearman(a, b):
    import numpy as np
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return _pearson(ra, rb)


def _auc(scores, labels):
    import numpy as np
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    pos, neg = int(y.sum()), int((~y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _strict_emvar(df, s2_path, fdr_thresh):
    """The paper's active-gated emVar (FDR<thresh AND >=1 allele active), joined by rsID.
    Mirrors eval/calibrate.py so strict-AUC numbers are comparable across harnesses."""
    import pandas as pd
    s2 = pd.read_excel(s2_path, sheet_name="Primary",
                       usecols=["rsid", "alt_is_active", "ref_is_active", "adj.P.Val"])
    s2 = s2.rename(columns={"adj.P.Val": "fdr2"})
    s2["rsid"] = s2["rsid"].astype(str)
    active = (s2["alt_is_active"].fillna(False).astype(bool) |
              s2["ref_is_active"].fillna(False).astype(bool))
    s2["strict_emvar"] = (s2["fdr2"] <= fdr_thresh) & active
    m = dict(zip(s2["rsid"], s2["strict_emvar"]))
    return df["rsid"].astype(str).map(m).fillna(False).astype(int)


def _metrics(name, delta, skew, emvar_loose, emvar_strict):
    import numpy as np
    ad = np.abs(delta)
    row = {
        "model": name,
        "pearson": round(_pearson(delta, skew), 4),
        "spearman": round(_spearman(delta, skew), 4),
        "emvar_auc_loose": round(_auc(ad, emvar_loose), 4),
    }
    if emvar_strict is not None:
        a = _auc(ad, emvar_strict)
        row["emvar_auc_strict"] = None if a != a else round(a, 4)
    return row


def main(argv=None):
    import numpy as np
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-table", default=os.path.join(_ROOT, "data", "processed",
                                                         "eval_variants_siamese.parquet"),
                    help="held-out siamese eval slice (from data/make_variant_pairs.py)")
    ap.add_argument("--siamese-weights", default=None, help="siamese checkpoint dir")
    ap.add_argument("--activity-weights", default=None,
                    help="activity checkpoint dir (ActivityPredictor) to score on the SAME slice")
    ap.add_argument("--activity-context", default="primary", choices=("primary", "organoid"))
    ap.add_argument("--s2", default=None, help="Data S2 xlsx -> enables the strict active-gated emVar")
    ap.add_argument("--fdr", type=float, default=0.10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default=None)
    ap.add_argument("--results-out", default=os.path.join(_ROOT, "weights", "results_siamese.json"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    if not (args.siamese_weights or args.activity_weights):
        raise SystemExit("give --siamese-weights and/or --activity-weights (at least one)")
    if not os.path.isfile(args.eval_table):
        raise SystemExit(f"eval slice not found: {args.eval_table}\n"
                         "Run data/make_variant_pairs.py first.")

    df = pd.read_parquet(args.eval_table)
    df = df.dropna(subset=["seq_ref", "seq_alt", "measured_skew"]).reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    refs, alts = df["seq_ref"].tolist(), df["seq_alt"].tolist()
    skew = df["measured_skew"].to_numpy(float)
    emvar_loose = df["is_emvar"].astype(int).to_numpy() if "is_emvar" in df else np.zeros(len(df), int)
    emvar_strict = None
    if args.s2 and "rsid" in df.columns:
        emvar_strict = _strict_emvar(df, args.s2, args.fdr).to_numpy()
        print(f"[eval] strict emVar positives on slice: {int(emvar_strict.sum())}")
    print(f"[eval] {len(df)} held-out variants | loose emVars: {int(emvar_loose.sum())}")

    rows = []

    if args.siamese_weights:
        from src.siamese_predictor import SiameseVariantScorer
        s = SiameseVariantScorer(args.siamese_weights, device=args.device,
                                 batch_size=args.batch_size).load()
        d = np.array(s.score_variants_batch(refs, alts))
        rows.append(_metrics("siamese", d, skew, emvar_loose, emvar_strict))
        print(f"[eval] siamese scored (backbone={s.backbone_type})")

    if args.activity_weights:
        from src.predictor import ActivityPredictor
        p = ActivityPredictor(context=args.activity_context, weights=args.activity_weights,
                              device=args.device, batch_size=args.batch_size).load()
        if not p.is_finetuned:
            print("[eval] ⚠️ activity predictor is NOT fine-tuned — baseline will be noise.")
        d = np.array(p.score_variants_batch(refs, alts))
        rows.append(_metrics(f"activity[{args.activity_context}]", d, skew, emvar_loose, emvar_strict))
        print("[eval] activity baseline scored (subtract-endpoints Δ)")

    # report
    cols = ["model", "pearson", "spearman", "emvar_auc_loose"] + (
        ["emvar_auc_strict"] if emvar_strict is not None else [])
    width = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("\n── siamese vs activity, SAME held-out slice ────────────────────")
    print("  " + " | ".join(c.ljust(width[c]) for c in cols))
    print("  " + "-+-".join("-" * width[c] for c in cols))
    for r in rows:
        print("  " + " | ".join(str(r.get(c, "")).ljust(width[c]) for c in cols))
    if len(rows) == 2:
        key = "emvar_auc_strict" if emvar_strict is not None else "emvar_auc_loose"
        sia = next((r[key] for r in rows if r["model"] == "siamese"), None)
        act = next((r[key] for r in rows if r["model"].startswith("activity")), None)
        if sia is not None and act is not None:
            verdict = "SIAMESE WINS" if sia > act else ("TIE" if sia == act else "baseline wins")
            print(f"\n  gate ({key}): siamese {sia} vs activity {act} -> {verdict}")
    print("────────────────────────────────────────────────────────────────")

    out = {"eval_table": os.path.abspath(args.eval_table), "n": int(len(df)),
           "loose_emvars": int(emvar_loose.sum()),
           "strict_emvars": int(emvar_strict.sum()) if emvar_strict is not None else None,
           "results": rows}
    os.makedirs(os.path.dirname(args.results_out), exist_ok=True)
    with open(args.results_out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval] wrote {args.results_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
