"""Additive tests for the stacking meta-learner (docs/07 #2, src/meta.py).

Pure-numpy, no torch / no network. Covers the properties that matter:
  * it recovers a MAGNITUDE-driven effect label (the reason features are |Δ|, not signed Δ),
  * missing evidence sources (NaN) are imputed and never crash,
  * save/load is exact,
  * explain() ranks the true drivers first.

Run:  python tests/test_meta.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from src.meta import MetaCombiner, build_matrix, _auc  # noqa: E402


def _toy(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    dna = rng.normal(0, 1, n)
    frozen = dna * 0.6 + rng.normal(0, 0.5, n)          # correlated independent signal
    motif = rng.normal(0, 1, n)
    gtex = rng.normal(0, 1, n)                           # noise
    tss = rng.uniform(0, 50, n)                          # noise
    sigma = np.abs(rng.normal(0, 0.3, n))
    score = np.abs(dna) + np.abs(frozen) + 0.3 * np.abs(motif)   # MAGNITUDE-driven label
    y = (score > np.quantile(score, 0.95)).astype(int)
    sigs = [{"dna_lm_delta": dna[i], "dna_lm_sigma": sigma[i], "frozen_delta": frozen[i],
             "motif_dscore": motif[i], "gtex_signed": gtex[i], "tss_dist_kb": tss[i]}
            for i in range(n)]
    return sigs, y


def test_recovers_magnitude_effect():
    sigs, y = _toy()
    X = build_matrix(sigs)
    tr, te = slice(0, 1500), slice(1500, None)
    m = MetaCombiner().fit(X[tr], y[tr])
    auc = _auc(m.predict_proba(X[te]), y[te])
    assert auc is not None and auc > 0.9, f"held-out AUC too low: {auc}"
    w = m.diagnostics["feature_weights"]
    assert w["abs_dna_lm_delta"] > w["abs_gtex_signed"], "true driver must outweigh noise"
    print(f"  ok: held-out AUC={auc:.4f}; abs_dna_lm_delta weight={w['abs_dna_lm_delta']:.3f}")


def test_missing_sources_are_imputed():
    sigs, y = _toy()
    for i in range(0, len(sigs), 5):          # drop frozen for 20% of rows
        sigs[i]["frozen_delta"] = None
    X = build_matrix(sigs)
    m = MetaCombiner().fit(X[:1500], y[:1500])
    # a signals dict missing several sources must still score without error
    p = m.transform({"dna_lm_delta": 2.0})
    assert 0.0 <= p <= 1.0, f"transform out of range: {p}"
    print(f"  ok: NaN-imputed features score fine (P={p:.3f} for dna-only)")


def test_save_load_roundtrip():
    sigs, y = _toy()
    X = build_matrix(sigs)
    m = MetaCombiner().fit(X[:1500], y[:1500])
    scratch = os.environ.get("TMP") or os.environ.get("TMPDIR") or _ROOT
    path = os.path.join(scratch, "_meta_roundtrip_test.json")
    m.save(path)
    m2 = MetaCombiner.load(path)
    assert np.allclose(m.predict_proba(X[1500:]), m2.predict_proba(X[1500:])), "roundtrip mismatch"
    os.remove(path)
    print("  ok: save/load roundtrip exact")


def test_explain_ranks_drivers_first():
    sigs, y = _toy()
    X = build_matrix(sigs)
    m = MetaCombiner().fit(X[:1500], y[:1500])
    contribs = m.explain({"dna_lm_delta": 3.0, "frozen_delta": 2.5, "motif_dscore": -4.0,
                          "gtex_signed": 0.1, "tss_dist_kb": 2.0})
    top = contribs[0][0]
    assert top in ("abs_dna_lm_delta", "abs_frozen_delta", "abs_motif_dscore"), \
        f"top contributor should be a real driver, got {top}"
    print(f"  ok: explain ranks '{top}' first")


def _run_all():
    tests = [test_recovers_magnitude_effect, test_missing_sources_are_imputed,
             test_save_load_roundtrip, test_explain_ranks_drivers_first]
    print(f"running {len(tests)} meta-learner tests")
    print("-" * 60)
    for t in tests:
        print(t.__name__)
        t()
    print("-" * 60)
    print("ALL META TESTS PASSED ✅")


if __name__ == "__main__":
    _run_all()
