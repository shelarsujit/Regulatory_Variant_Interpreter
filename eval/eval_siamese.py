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


# metric names computed per bootstrap resample; keep in sync with _metric_vec below.
_BOOT_METRICS = ("pearson", "spearman", "emvar_auc_loose", "emvar_auc_strict")


def _metric_vec(delta, skew, emvar_loose, emvar_strict, idx):
    """Raw (unrounded) metrics for a delta array on the resampled indices `idx`.
    Returns a dict metric-name -> float (nan where undefined, e.g. AUC with no positives)."""
    import numpy as np
    d = np.asarray(delta, float)[idx]
    ad = np.abs(d)
    sk = np.asarray(skew, float)[idx]
    lo = np.asarray(emvar_loose)[idx]
    out = {
        "pearson": _pearson(d, sk),
        "spearman": _spearman(d, sk),
        "emvar_auc_loose": _auc(ad, lo),
    }
    if emvar_strict is not None:
        out["emvar_auc_strict"] = _auc(ad, np.asarray(emvar_strict)[idx])
    return out


def _ci(samples, lo=2.5, hi=97.5):
    """Percentile CI over a 1-D list of bootstrap replicates, ignoring nan (undefined resamples)."""
    import numpy as np
    a = np.asarray(samples, float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return None
    return [round(float(np.percentile(a, lo)), 4), round(float(np.percentile(a, hi)), 4)]


def _bootstrap(deltas_by_model, skew, emvar_loose, emvar_strict, n_boot, seed):
    """Paired percentile bootstrap over the eval slice.

    Resamples variant indices WITH REPLACEMENT and applies the SAME indices to every model, so the
    per-metric CIs and — crucially — the siamese−activity paired-difference CI share resamples and the
    difference test is valid (the two models see identical resampled variants each replicate).

    Returns {"per_model": {name: {metric: {ci, ...}}}, "paired": {metric: {diff_ci, p_one_sided, ...}}}.
    The one-sided bootstrap p-value = fraction of replicates where (siamese−activity) <= 0, i.e. the
    probability the observed siamese>activity gain is not reproduced under resampling. Paired block is
    emitted only when both a 'siamese' and an 'activity[...]' model are present.
    """
    import numpy as np
    n = len(skew)
    rng = np.random.default_rng(seed)
    names = list(deltas_by_model)
    metrics = [m for m in _BOOT_METRICS if m != "emvar_auc_strict" or emvar_strict is not None]

    reps = {name: {m: [] for m in metrics} for name in names}
    sia_name = next((k for k in names if k == "siamese"), None)
    act_name = next((k for k in names if k.startswith("activity")), None)
    paired = sia_name is not None and act_name is not None
    diff_reps = {m: [] for m in metrics} if paired else None

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)  # one resample shared across all models (paired)
        per = {name: _metric_vec(deltas_by_model[name], skew, emvar_loose, emvar_strict, idx)
               for name in names}
        for name in names:
            for m in metrics:
                reps[name][m].append(per[name][m])
        if paired:
            for m in metrics:
                diff_reps[m].append(per[sia_name][m] - per[act_name][m])

    per_model = {name: {m: {"ci95": _ci(reps[name][m])} for m in metrics} for name in names}
    result = {"n_boot": int(n_boot), "seed": int(seed), "per_model": per_model}
    if paired:
        pd_block = {}
        for m in metrics:
            arr = np.asarray(diff_reps[m], float)
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                pd_block[m] = {"diff_ci95": None, "p_one_sided": None, "n_valid": 0}
                continue
            p = float((arr <= 0).mean())  # H0: siamese <= activity
            pd_block[m] = {"diff_median": round(float(np.median(arr)), 4),
                           "diff_ci95": _ci(diff_reps[m]),
                           "p_one_sided": round(p, 4),
                           "n_valid": int(arr.size)}
        result["paired"] = {"siamese_minus": act_name, "by_metric": pd_block}
    return result


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
    ap.add_argument("--fit-calibrator", action="store_true",
                    help="fit + save an isotonic Calibrator on the siamese Δ (held-out slice) so the "
                         "trust layer calibrates the siamese scorer (default out: weights/calibrator_siamese.json)")
    ap.add_argument("--calibrator-out", default=os.path.join(_ROOT, "weights", "calibrator_siamese.json"))
    ap.add_argument("--tau", type=float, default=0.5, help="|skew| counted as a real effect (D6)")
    ap.add_argument("--dump-predictions", default=None,
                    help="save per-variant siamese Δ (+ skew/emVar labels) to this parquet/csv so the "
                         "calibrator can be REFIT offline on CPU later — GPU is then needed only once")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="N paired bootstrap resamples of the eval slice -> 95%% CIs per metric and a "
                         "CI + one-sided p-value on the siamese−activity difference (0 = off; 1000 for the paper)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for --bootstrap (reproducible CIs)")
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
    siamese_delta = None
    deltas_by_model = {}  # model-name -> per-variant Δ, shared across metrics + the paired bootstrap

    if args.siamese_weights:
        from src.siamese_predictor import SiameseVariantScorer
        s = SiameseVariantScorer(args.siamese_weights, device=args.device,
                                 batch_size=args.batch_size).load()
        siamese_delta = np.array(s.score_variants_batch(refs, alts))
        deltas_by_model["siamese"] = siamese_delta
        rows.append(_metrics("siamese", siamese_delta, skew, emvar_loose, emvar_strict))
        print(f"[eval] siamese scored (backbone={s.backbone_type})")

    if args.activity_weights:
        from src.predictor import ActivityPredictor
        p = ActivityPredictor(context=args.activity_context, weights=args.activity_weights,
                              device=args.device, batch_size=args.batch_size).load()
        if not p.is_finetuned:
            print("[eval] ⚠️ activity predictor is NOT fine-tuned — baseline will be noise.")
        d = np.array(p.score_variants_batch(refs, alts))
        deltas_by_model[f"activity[{args.activity_context}]"] = d
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

    # paired percentile bootstrap -> 95% CIs + a significance test on the siamese−activity gain.
    # This is the rigor the preprint headline (Δ-Pearson 0.19->0.28) needs: a point estimate alone
    # is a reviewer kill (docs/10_preprint_draft.md §Limitations #2).
    boot = None
    if args.bootstrap > 0:
        boot = _bootstrap(deltas_by_model, skew, emvar_loose, emvar_strict, args.bootstrap, args.seed)
        print(f"\n── paired bootstrap ({args.bootstrap} resamples, seed {args.seed}) ─────────────")
        for name, mm in boot["per_model"].items():
            cis = ", ".join(f"{m}={mm[m]['ci95']}" for m in mm)
            print(f"  {name}: {cis}")
        if "paired" in boot:
            print(f"  Δ (siamese − {boot['paired']['siamese_minus']}), 95% CI [p one-sided]:")
            for m, b in boot["paired"]["by_metric"].items():
                if b["diff_ci95"] is None:
                    print(f"    {m}: undefined (no valid resamples)")
                    continue
                sig = "*" if (b["p_one_sided"] is not None and b["p_one_sided"] < 0.05) else " "
                print(f"    {m}: {b['diff_ci95']}  p={b['p_one_sided']} {sig}")
        print("────────────────────────────────────────────────────────────────")

    # dump per-variant siamese Δ so the calibrator can be refit OFFLINE (CPU) later — GPU once.
    if args.dump_predictions and siamese_delta is not None:
        dump = df[["chrom", "pos", "ref", "alt", "measured_skew", "is_emvar"]].copy()
        dump["siamese_delta"] = siamese_delta
        if "rsid" in df.columns:
            dump["rsid"] = df["rsid"].values
        try:
            dump.to_parquet(args.dump_predictions, index=False)
        except Exception:
            dump.to_csv(args.dump_predictions.rsplit(".", 1)[0] + ".csv", index=False)
        print(f"[eval] dumped {len(dump)} per-variant siamese Δ -> {args.dump_predictions}")

    # fit + save an isotonic calibrator on the siamese Δ so the trust layer can calibrate it.
    # Fit on this held-out slice (the only siamese-clean labels); isotonic is monotonic so it does
    # not change ranking/AUC — only maps |Δ| -> P(effect). Reliability is descriptive.
    calibrator_out = None
    if args.fit_calibrator:
        if siamese_delta is None:
            print("[eval] --fit-calibrator needs --siamese-weights; skipping")
        else:
            from src.trust import Calibrator
            cal = Calibrator().fit(siamese_delta, skew, is_emvar=emvar_loose, tau=args.tau)
            cal.save(args.calibrator_out)
            calibrator_out = args.calibrator_out
            print(f"[eval] fit siamese calibrator -> {args.calibrator_out}  "
                  f"(diagnostics: {cal.diagnostics})")

    out = {"eval_table": os.path.abspath(args.eval_table), "n": int(len(df)),
           "loose_emvars": int(emvar_loose.sum()),
           "strict_emvars": int(emvar_strict.sum()) if emvar_strict is not None else None,
           "calibrator_out": calibrator_out,
           "results": rows,
           "bootstrap": boot}
    os.makedirs(os.path.dirname(args.results_out), exist_ok=True)
    with open(args.results_out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval] wrote {args.results_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
