"""Core tests: leakage-safe split, motif gain/loss, calibration + honest conflict.

Runs with plain python (`python tests/test_core.py`) or pytest. No GPU, no downloads.
These lock the invariants that the trust thesis depends on.
"""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "data"))

import splits                              # data/splits.py
import prepare_data                        # data/prepare_data.py
from src import motifs, trust
from src.schema import Call, Direction, EvidenceItem


# --------------------------------------------------------------------------- 1. leakage
def test_leakage_split_holds_out_variant_loci():
    elements, variants, meta = prepare_data.make_synthetic(
        n_elements=3000, n_variants=1200, seq_len=200, seed=11, n_leak_decoys=40)
    out = splits.leakage_safe_split(elements, variants, locus_bin=10_000, seed=11)
    # the split must not raise, and loci must be disjoint
    splits.assert_no_leakage(out["train"], out["val"], out["calibration"])
    splits.assert_no_sequence_overlap(out["train"], out["calibration"])
    # every injected decoy (an element sitting on a variant locus) must be removed
    assert out["stats"]["n_removed_for_leakage"] >= meta["n_leak_decoys_injected"] == 40
    tr = set(out["train"]["locus"]); ca = set(out["calibration"]["locus"])
    assert tr.isdisjoint(ca)
    print("  ok: leakage split — decoys removed, train/calibration loci disjoint")


# --------------------------------------------------------------------------- 2. motifs
def _flank(n, rng):
    # neutral-ish filler unlikely to contain the tested motifs
    return "".join(rng.choice("AT") for _ in range(n))


def test_motif_loss_detected():
    import random
    rng = random.Random(3)
    motif = "TGACTCA"                       # AP-1 site
    left, right = _flank(40, rng), _flank(40, rng)
    seq_ref = left + motif + right          # intact AP-1
    center = len(left) + 3                  # the 'C' in TGA C TCA
    seq_alt = seq_ref[:center] + "A" + seq_ref[center + 1:]   # TGA A TCA -> broken
    events = motifs.annotate_motifs(seq_ref, seq_alt)
    ap1 = [m for m in events if m.tf == "AP-1"]
    assert ap1, f"expected an AP-1 event, got {[m.tf for m in events]}"
    assert ap1[0].event == "loss" and ap1[0].delta_score < 0
    print(f"  ok: motif loss — {ap1[0].describe()}")


def test_motif_gain_detected():
    import random
    rng = random.Random(4)
    left, right = _flank(40, rng), _flank(40, rng)
    broken = "TGAATCA"                       # one base off AP-1
    seq_ref = left + broken + right
    center = len(left) + 3
    seq_alt = seq_ref[:center] + "C" + seq_ref[center + 1:]   # -> TGACTCA (creates AP-1)
    events = motifs.annotate_motifs(seq_ref, seq_alt)
    ap1 = [m for m in events if m.tf == "AP-1"]
    assert ap1 and ap1[0].event == "gain" and ap1[0].delta_score > 0
    print(f"  ok: motif gain — {ap1[0].describe()}")


# --------------------------------------------------------------------------- 3. calibration + trust
def _sim_calibration(n=4000, seed=0):
    """Simulate an imperfect predictor: predicted Δ correlates with measured skew."""
    rng = np.random.default_rng(seed)
    skew = rng.normal(0, 0.4, n)
    predicted = 0.7 * skew + rng.normal(0, 0.25, n)     # decent-but-noisy model
    is_emvar = (np.abs(skew) > 0.8).astype(int)
    return predicted, skew, is_emvar


def test_calibrator_is_monotone_and_informative():
    predicted, skew, is_emvar = _sim_calibration()
    cal = trust.Calibrator().fit(predicted, skew, is_emvar=is_emvar, tau=0.5)
    # calibrated probability must be non-decreasing in |Δ|
    grid = np.linspace(0, 2.0, 25)
    probs = [cal.transform(x) for x in grid]
    assert all(b >= a - 1e-9 for a, b in zip(probs, probs[1:])), "calibration not monotone"
    assert probs[-1] > probs[0] + 0.1, "calibration carries no signal"
    # the model should rank emVars above non-emVars
    assert cal.diagnostics["auc_emvar"] is not None and cal.diagnostics["auc_emvar"] > 0.7
    print(f"  ok: calibrator monotone; AUC(emVar)={cal.diagnostics['auc_emvar']}, "
          f"Pearson={cal.diagnostics['pearson_pred_vs_skew']}")


def test_trust_agreement_beats_conflict_and_surfaces_it():
    predicted, skew, is_emvar = _sim_calibration()
    cal = trust.Calibrator().fit(predicted, skew, is_emvar=is_emvar)
    delta = -0.9                              # a strong predicted down-effect

    agree = [EvidenceItem("held_out_MPRA", "measured skew agrees", direction=Direction.DOWN, concordant=True),
             EvidenceItem("GTEx_eQTL", "eQTL agrees", direction=Direction.DOWN, concordant=True)]
    conflict = [EvidenceItem("frozen_foundation_model", "foundation model disagrees",
                             direction=Direction.UP, concordant=False)]

    r_agree = trust.build_trust_report(delta, agree, calibrator=cal)
    r_conf = trust.build_trust_report(delta, conflict, calibrator=cal)
    r_mixed = trust.build_trust_report(delta, agree + conflict, calibrator=cal)

    assert r_agree.confidence > r_conf.confidence, "agreement should beat conflict"
    assert r_conf.has_conflict and r_conf.conflicts[0].source == "frozen_foundation_model"
    assert r_mixed.has_conflict and len(r_mixed.agreements) == 2   # conflict surfaced, not hidden
    assert r_agree.call == Call.DISRUPTIVE
    print(f"  ok: trust — agree={r_agree.confidence:.2f} > conflict={r_conf.confidence:.2f}; "
          f"mixed surfaces {len(r_mixed.conflicts)} conflict")


def test_interpret_end_to_end_runs():
    """The orchestrator runs live given sequences + an injected model Δ (no GPU/genome)."""
    import random
    from src import interpret_variant
    rng = random.Random(7)
    left, right = _flank(60, rng), _flank(60, rng)
    seq_ref = left + "TGACTCA" + right                 # intact AP-1
    center = len(left) + 3
    seq_alt = seq_ref[:center] + "A" + seq_ref[center + 1:]   # break it

    predicted, skew, is_emvar = _sim_calibration()
    cal = trust.Calibrator().fit(predicted, skew, is_emvar=is_emvar)

    result = interpret_variant(
        "chr2", 162279995, "C", "A",
        seq_ref=seq_ref, seq_alt=seq_alt,
        model_delta=-0.9, model_delta_organoid=-0.7,      # injected primary + organoid Δ
        evidence=[EvidenceItem("GTEx_eQTL", "eQTL agrees", direction=Direction.DOWN, concordant=True),
                  EvidenceItem("frozen_foundation_model", "foundation model disagrees",
                               direction=Direction.UP, concordant=False)],
        calibrator=cal,
    )
    assert result.predicted_direction == Direction.DOWN
    assert any(m.tf == "AP-1" and m.event == "loss" for m in result.mechanisms)
    assert any(e.source == "organoid_model" and e.concordant is True for e in result.evidence)
    assert result.trust.has_conflict and result.trust.n_independent_sources >= 2
    assert len(result.build_evidence_chain()) >= 5
    assert len(result.to_json()) > 100                    # serializes cleanly
    print("  ok: interpret_variant end-to-end —")
    print("      " + result.summary_line())
    for line in result.build_evidence_chain():
        print("        · " + line)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"running {len(tests)} tests\n" + "-" * 60)
    for t in tests:
        print(t.__name__)
        t()
    print("-" * 60 + f"\nALL {len(tests)} TESTS PASSED ✅")


if __name__ == "__main__":
    _run_all()
