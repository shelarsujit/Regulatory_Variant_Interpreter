"""Fit + evaluate the trust calibrator on the held-out MPRA variants.

THEORY (plain English — see docs/00_overview_for_non_biologists.md §4.2, decisions D1/D6/D11):
This is where a raw prediction becomes a TRUSTABLE one. We take the fine-tuned primary model,
score every held-out calibration variant (predicted Δactivity = alt − ref), and compare it to
the wet-lab measured allelic skew (`logFC`). Because the locus split (D5) guarantees the model
never trained on these variants, the comparison is honest. We then fit the isotonic calibrator
(trust.Calibrator) that maps |predicted Δ| → P(real regulatory effect), and report:

  * Pearson / Spearman of predicted Δ vs measured skew        (does the model track reality?)
  * emVar classification AUC                                   (can it rank the real movers?)
  * a reliability table                                        (are the probabilities honest?)

emVar definition (docs/01_data_provenance §4, confirmed): the paper's emVar = limma FDR<10%
AND at least one allele active. Our calibration parquet's `is_emvar` used the loose FDR≤0.10
alone (~596). If Data S2 is passed via --s2, we recompute the STRICT active-gated flag (~164)
by joining `alt_is_active`/`ref_is_active` on rsID — the honest classification target.

RUN (after the primary fine-tune finishes):
    python eval/calibrate.py --weights weights/primary \
        --s2 data/raw/science.adh0559_data_s1_to_s3/adh0559_data_s2/DataS2-Variant-library-ratios.xlsx
Writes the fitted calibrator to weights/calibrator_primary.json (the app loads it via
RVI_CALIBRATION / trust.Calibrator.load).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

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


def _strict_emvar(df, s2_path, fdr_thresh):
    """Recompute the paper's active-gated emVar (FDR<thresh AND >=1 allele active) via rsID join."""
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


def main(argv=None):
    import numpy as np
    import pandas as pd
    from src.predictor import ActivityPredictor
    from src.trust import Calibrator, _auc

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="fine-tuned checkpoint dir (e.g. weights/primary)")
    ap.add_argument("--context", default="primary", choices=("primary", "organoid"))
    ap.add_argument("--calibration", default=os.path.join(_ROOT, "data", "processed", "calibration_variants.parquet"))
    ap.add_argument("--s2", default=None, help="Data S2 xlsx -> enables the strict active-gated emVar")
    ap.add_argument("--out", default=None, help="calibrator json out (default weights/calibrator_<context>.json)")
    ap.add_argument("--tau", type=float, default=0.5, help="|skew| counted as a real effect (D6)")
    ap.add_argument("--fdr", type=float, default=0.10, help="emVar FDR threshold")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--rc", action="store_true", help="test-time reverse-complement averaging of Δ")
    ap.add_argument("--ensemble", action="store_true",
                    help="treat --weights as an ensemble root (<weights>/seed*/); scores the mean Δ")
    ap.add_argument("--results-out", default=None,
                    help="machine-readable results json (default weights/results_<context>.json)")
    ap.add_argument("--limit", type=int, default=0, help="cap variants (smoke test)")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.calibration)
    df = df.dropna(subset=["seq_ref", "seq_alt", "measured_skew"]).reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"[eval] {len(df)} held-out variants | context={args.context} | weights={args.weights}")

    if args.ensemble:
        from src.predictor import EnsemblePredictor
        pred = EnsemblePredictor.from_dir(args.weights, context=args.context, device=args.device,
                                          batch_size=args.batch_size, rc_average=args.rc).load()
        print(f"[eval] ensemble of {len(pred.members)} member(s)")
    else:
        pred = ActivityPredictor(context=args.context, weights=args.weights, device=args.device,
                                 batch_size=args.batch_size, rc_average=args.rc).load()
    if not pred.is_finetuned:
        print("[eval] ⚠️ WARNING: predictor is NOT fine-tuned (random head) — metrics will be noise.")
    if args.rc:
        print("[eval] reverse-complement averaging: ON")

    # batched Δ = activity(alt) − activity(ref)  (RC-averaged if --rc)
    delta = np.array(pred.score_variants_batch(df["seq_ref"].tolist(), df["seq_alt"].tolist()))
    skew = df["measured_skew"].to_numpy(float)

    # emVar targets: loose (from parquet) and strict (active-gated, if S2 given)
    emvar_loose = df["is_emvar"].astype(int).to_numpy()
    emvar_strict = _strict_emvar(df, args.s2, args.fdr).to_numpy() if args.s2 else None

    pear, spear = _pearson(delta, skew), _spearman(delta, skew)
    auc_loose = _auc(np.abs(delta), emvar_loose)
    auc_strict = _auc(np.abs(delta), emvar_strict) if emvar_strict is not None else None

    def _fmt(x):
        return f"{x:.4f}" if x is not None else "n/a (single-class; need more variants)"

    print("\n── variant-effect metrics (predicted Δ vs measured skew) ──")
    print(f"  Pearson r         : {pear:.4f}")
    print(f"  Spearman r        : {spear:.4f}")
    print(f"  emVar AUC (loose) : {_fmt(auc_loose)}  (n_pos={int(emvar_loose.sum())}, FDR≤{args.fdr})")
    if emvar_strict is not None:
        print(f"  emVar AUC (strict): {_fmt(auc_strict)}  (n_pos={int(emvar_strict.sum())}, "
              f"active-gated — the paper's ~164)")

    # fit the calibrator on the strict target if available, else loose
    target_emvar = emvar_strict if emvar_strict is not None else emvar_loose
    cal = Calibrator().fit(delta, skew, is_emvar=target_emvar, tau=args.tau)
    print("\n── calibrator ──")
    print(f"  diagnostics: {cal.diagnostics}")

    # reliability: bin calibrated prob, show observed emVar rate per bin
    probs = np.array([cal.transform(d) for d in delta])
    print("\n── reliability (calibrated P vs observed emVar rate) ──")
    edges = np.linspace(0, 1, 6)
    reliability = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        if m.sum():
            row = {"bin": [round(float(lo), 2), round(float(hi), 2)], "n": int(m.sum()),
                   "mean_P": round(float(probs[m].mean()), 4),
                   "observed_emVar": round(float(target_emvar[m].mean()), 4)}
            reliability.append(row)
            print(f"  P∈[{lo:.1f},{hi:.1f})  n={row['n']:5d}  "
                  f"mean_P={row['mean_P']:.3f}  observed_emVar={row['observed_emVar']:.3f}")

    out = args.out or os.path.join(_ROOT, "weights", f"calibrator_{args.context}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cal.save(out)
    print(f"\n[eval] calibrator saved → {out}")
    print("       app: set RVI_CALIBRATION or load via trust.Calibrator.load()")

    # machine-readable results log (auto-written every run — no hand-copying)
    results = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {"weights": args.weights, "context": args.context,
                   "calibration": os.path.abspath(args.calibration), "rc_average": args.rc,
                   "tau": args.tau, "fdr": args.fdr, "limit": args.limit or None},
        "predictor": pred.provenance(),
        "n_variants": int(len(df)),
        "metrics": {"pearson": round(pear, 4), "spearman": round(spear, 4),
                    "emvar_auc_loose": (round(auc_loose, 4) if auc_loose is not None else None),
                    "emvar_auc_strict": (round(auc_strict, 4) if auc_strict is not None else None),
                    "n_emvar_loose": int(emvar_loose.sum()),
                    "n_emvar_strict": (int(emvar_strict.sum()) if emvar_strict is not None else None)},
        "calibrator": {"path": os.path.abspath(out), "diagnostics": cal.diagnostics},
        "reliability": reliability,
    }
    rout = args.results_out or os.path.join(_ROOT, "weights", f"results_{args.context}.json")
    with open(rout, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] results logged → {rout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
