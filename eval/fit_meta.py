#!/usr/bin/env python
"""Fit + grade the stacking meta-learner on real held-out variants (docs/07 #2).

Assembles the per-variant evidence signals that are actually computable, fits src/meta.MetaCombiner
on the variant TRAIN+VAL loci, and grades on the locus-disjoint eval slice — the SAME partition the
siamese eval uses (data/make_variant_pairs.py), so numbers are directly comparable and leakage-safe.

FEATURES (missing ones impute; coverage is logged honestly):
  dna_lm_delta   ActivityPredictor primary Δ (or a siamese scorer via --siamese-weights)   [dense]
  motif_dscore   top JASPAR/illustrative motif gain-loss Δscore (src/motifs)               [sparse]
  frozen_delta   frozen big-model zero-shot Δ — only if a foundation_fn is wired           [off by default]
  gtex_signed    signed brain-eQTL effect — only if a signed GTEx table is supplied        [off by default]
  dna_lm_sigma   ensemble std — only with --ensemble                                       [off by default]
  tss_dist_kb    nearest-TSS distance — only if a TSS table is supplied                    [off by default]

BASELINE = the single-feature story (|dna_lm_delta| ranking, i.e. what trust.Calibrator uses).
The meta-learner is a WIN only if it beats that baseline AUC on the held-out slice. Offline, with
only dna+sparse-motif live, expect a near-tie: the stacking payoff needs the independent frozen
signal (wire a foundation_fn / GPU Enformer) — logged as such, honestly.

ADDITIVE: NEW file. Writes only weights/meta_<ctx>.json + weights/results_meta.json.

RUN
    python eval/fit_meta.py --activity-weights weights/primary
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _auc(scores, labels):
    import numpy as np
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    pos, neg = int(y.sum()), int((~y).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _signals_for(df, delta, motif_lib, organoid_delta=None, frozen_lookup=None):
    """Build a raw-signal dict per row. delta = DNA-LM Δ array aligned to df; organoid_delta =
    optional independent organoid-model Δ array; motif_lib = a real JASPAR library or None;
    frozen_lookup = optional {(chrom,pos,ref,alt): frozen_delta} from the precomputed cache."""
    from src import motifs
    out = []
    for i, (_, row) in enumerate(df.iterrows()):
        sig = {"dna_lm_delta": float(delta[i])}
        if organoid_delta is not None:
            sig["organoid_delta"] = float(organoid_delta[i])
        if frozen_lookup is not None:
            fd = frozen_lookup.get((row["chrom"], int(row["pos"]), row["ref"], row["alt"]))
            if fd is not None:
                sig["frozen_delta"] = float(fd)
        # motif Δscore: top |gain/loss| overlapping the variant
        try:
            ev = motifs.annotate_motifs(str(row["seq_ref"]), str(row["seq_alt"]), library=motif_lib)
            if ev:
                sig["motif_dscore"] = float(ev[0].delta_score)
        except Exception:
            pass
        out.append(sig)
    return out


def _coverage(signals, names):
    cov = {}
    for n in names:
        c = sum(1 for s in signals if s.get(n) is not None)
        cov[n] = f"{c}/{len(signals)}"
    return cov


def main(argv=None):
    import numpy as np
    import pandas as pd
    from src.meta import MetaCombiner, build_matrix, RAW_SIGNALS

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    proc = os.path.join(_ROOT, "data", "processed")
    ap.add_argument("--fit-tables", nargs="+",
                    default=[os.path.join(proc, "train_variants.parquet"),
                             os.path.join(proc, "val_variants.parquet")],
                    help="variant tables to FIT on (loci disjoint from the eval slice)")
    ap.add_argument("--eval-table", default=os.path.join(proc, "eval_variants_siamese.parquet"))
    ap.add_argument("--activity-weights", default=os.path.join(_ROOT, "weights", "primary"))
    ap.add_argument("--activity-context", default="primary", choices=("primary", "organoid"))
    ap.add_argument("--siamese-weights", default=None,
                    help="use a siamese scorer for dna_lm_delta instead of the activity model")
    ap.add_argument("--organoid-weights", default=os.path.join(_ROOT, "weights", "organoid"),
                    help="independent organoid-context model -> organoid_delta feature ('' to skip)")
    ap.add_argument("--jaspar", default=None,
                    help="path to a JASPAR .pfm/.jaspar file -> real motif library (else illustrative)")
    ap.add_argument("--frozen-cache", default=None,
                    help="precomputed frozen-model Δ parquet (eval/precompute_frozen.py) -> frozen_delta feature")
    ap.add_argument("--context", default="primary")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="meta json out (default weights/meta_<ctx>.json)")
    ap.add_argument("--results-out", default=os.path.join(_ROOT, "weights", "results_meta.json"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    for p in list(args.fit_tables) + [args.eval_table]:
        if not os.path.isfile(p):
            raise SystemExit(f"missing {p}\nRun data/make_variant_pairs.py first.")

    fit_df = pd.concat([pd.read_parquet(p) for p in args.fit_tables], ignore_index=True)
    ev_df = pd.read_parquet(args.eval_table)
    for d in (fit_df, ev_df):
        d.dropna(subset=["seq_ref", "seq_alt"], inplace=True)
    fit_df = fit_df.reset_index(drop=True)
    ev_df = ev_df.reset_index(drop=True)
    if args.limit:
        fit_df, ev_df = fit_df.head(args.limit), ev_df.head(args.limit)

    # leakage guard: fit and eval loci must be disjoint
    if "locus" in fit_df and "locus" in ev_df:
        shared = set(fit_df["locus"]) & set(ev_df["locus"])
        if shared:
            raise SystemExit(f"LEAKAGE: {len(shared)} loci shared between fit and eval slices")

    # DNA-LM Δ for both slices
    if args.siamese_weights:
        from src.siamese_predictor import SiameseVariantScorer
        scorer = SiameseVariantScorer(args.siamese_weights, device=args.device,
                                      batch_size=args.batch_size).load()
        dna_source = f"siamese:{args.siamese_weights}"
    else:
        from src.predictor import ActivityPredictor
        scorer = ActivityPredictor(context=args.activity_context, weights=args.activity_weights,
                                   device=args.device, batch_size=args.batch_size).load()
        if not scorer.is_finetuned:
            print("[meta] ⚠️ DNA-LM is NOT fine-tuned — features will be noise.")
        dna_source = f"activity:{args.activity_weights}"

    print(f"[meta] scoring DNA-LM Δ ({dna_source}) on {len(fit_df)} fit + {len(ev_df)} eval variants…")
    fit_delta = np.array(scorer.score_variants_batch(fit_df["seq_ref"].tolist(), fit_df["seq_alt"].tolist()))
    ev_delta = np.array(scorer.score_variants_batch(ev_df["seq_ref"].tolist(), ev_df["seq_alt"].tolist()))

    # independent organoid-context model Δ (charter's 2nd model) — an offline independent feature
    fit_org = ev_org = None
    if args.organoid_weights and os.path.isdir(args.organoid_weights):
        from src.predictor import ActivityPredictor
        org = ActivityPredictor(context="organoid", weights=args.organoid_weights,
                                device=args.device, batch_size=args.batch_size).load()
        if org.is_finetuned:
            print(f"[meta] scoring organoid Δ ({args.organoid_weights})…")
            fit_org = np.array(org.score_variants_batch(fit_df["seq_ref"].tolist(), fit_df["seq_alt"].tolist()))
            ev_org = np.array(org.score_variants_batch(ev_df["seq_ref"].tolist(), ev_df["seq_alt"].tolist()))
        else:
            print("[meta] organoid model not fine-tuned — skipping organoid feature.")

    # real JASPAR motif library (else the illustrative default)
    motif_lib = None
    if args.jaspar and os.path.isfile(args.jaspar):
        from src import motifs
        motif_lib = motifs.load_jaspar_pfms(args.jaspar)
        print(f"[meta] loaded {len(motif_lib)} JASPAR motifs from {args.jaspar}")

    # optional precomputed frozen-model Δ (Enformer) -> the independent stacking feature (docs/08)
    frozen_lookup = None
    if args.frozen_cache and os.path.isfile(args.frozen_cache):
        fc = pd.read_parquet(args.frozen_cache)
        frozen_lookup = {(r.chrom, int(r.pos), r.ref, r.alt): float(r.frozen_delta)
                         for r in fc.itertuples()}
        print(f"[meta] loaded {len(frozen_lookup)} frozen Δ from {args.frozen_cache}")

    fit_sig = _signals_for(fit_df, fit_delta, motif_lib, fit_org, frozen_lookup)
    ev_sig = _signals_for(ev_df, ev_delta, motif_lib, ev_org, frozen_lookup)
    print(f"[meta] fit feature coverage:  {_coverage(fit_sig, RAW_SIGNALS)}")
    print(f"[meta] eval feature coverage: {_coverage(ev_sig, RAW_SIGNALS)}")

    y_fit = fit_df["is_emvar"].astype(int).to_numpy()
    y_ev = ev_df["is_emvar"].astype(int).to_numpy()

    X_fit = build_matrix(fit_sig)
    X_ev = build_matrix(ev_sig)
    meta = MetaCombiner().fit(X_fit, y_fit)
    p_ev = meta.predict_proba(X_ev)
    meta_auc = _auc(p_ev, y_ev)

    # baseline: |dna_lm_delta| ranking (the single-feature calibrator story) on the SAME slice
    base_auc = _auc(np.abs(ev_delta), y_ev)

    print("\n── meta-learner vs single-feature baseline, held-out slice ─────")
    print(f"  eval variants        : {len(ev_df)}  (emVars: {int(y_ev.sum())})")
    print(f"  baseline |Δ| AUC     : {None if base_auc is None else round(base_auc,4)}")
    print(f"  META AUC             : {None if meta_auc is None else round(meta_auc,4)}")
    if meta_auc is not None and base_auc is not None:
        verdict = ("META WINS" if meta_auc > base_auc else
                   "tie" if abs(meta_auc - base_auc) < 1e-4 else "baseline wins")
        print(f"  gate                 : {verdict}  (Δ={meta_auc-base_auc:+.4f})")
    print(f"  feature weights      : {meta.diagnostics['feature_weights']}")
    print("────────────────────────────────────────────────────────────────")

    out = args.out or os.path.join(_ROOT, "weights", f"meta_{args.context}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)   # weights/ may not exist on a fresh clone
    meta.save(out)
    results = {
        "dna_source": dna_source, "n_fit": int(len(fit_df)), "n_eval": int(len(ev_df)),
        "eval_emvars": int(y_ev.sum()),
        "baseline_abs_delta_auc": None if base_auc is None else round(base_auc, 4),
        "meta_auc": None if meta_auc is None else round(meta_auc, 4),
        "feature_coverage_eval": _coverage(ev_sig, RAW_SIGNALS),
        "feature_weights": meta.diagnostics["feature_weights"],
        "note": "Offline, only dna_lm_delta is dense and motif is sparse; the stacking payoff needs "
                "an independent frozen-model Δ (wire a foundation_fn / GPU Enformer). "
                "A near-tie here is the EXPECTED honest result, not a failure of the machinery.",
    }
    os.makedirs(os.path.dirname(args.results_out), exist_ok=True)
    with open(args.results_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[meta] wrote {out} and {args.results_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
