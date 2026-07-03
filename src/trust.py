"""Turn a prediction + evidence into a CALIBRATED confidence and an honest verdict.

THEORY (plain English — see docs/00_overview_for_non_biologists.md §4.2-4.4):
This module is the heart of the project. Two jobs:

1. CALIBRATION. A raw model score is not a probability. We learn a monotone mapping from
   |predicted Δactivity| -> P(the variant has a real regulatory effect), fit on the
   held-out MPRA variants whose true effects were measured — and which the model was NEVER
   trained on (guaranteed by the locus split in data/splits.py). Because only ~1% of tested
   variants are significant, we derive the calibration label from the CONTINUOUS measured
   skew (|skew| >= tau) rather than the sparse significance flag (decision D6), and keep the
   significance flag only as a secondary classification check (AUC).

   The calibrator is a self-contained isotonic regression (pool-adjacent-violators) — numpy
   only, so the trust layer has no heavy dependency and stays easy to test and to ship in
   the demo.

2. AGGREGATION WITH HONEST CONFLICT. We combine the calibrated model confidence with the
   independent evidence in log-odds space: concordant sources push confidence up, conflicts
   push it down. Conflicts are ALWAYS reported (never averaged away). A well-calibrated
   "uncertain" is a success, not a failure.
"""
from __future__ import annotations

import math

import numpy as np

from .schema import Call, EvidenceItem, TrustReport


# --------------------------------------------------------------------------- isotonic (PAVA)
def _pava(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: nearest non-decreasing fit to y (unit weights)."""
    blocks: list[list[float]] = []  # [sum, count] per block
    for val in y:
        blocks.append([float(val), 1.0])
        while len(blocks) >= 2 and (blocks[-2][0] / blocks[-2][1]) >= (blocks[-1][0] / blocks[-1][1]):
            s2, c2 = blocks.pop()
            s1, c1 = blocks.pop()
            blocks.append([s1 + s2, c1 + c2])
    out = []
    for s, c in blocks:
        out.extend([s / c] * int(c))
    return np.array(out)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Rank-based ROC-AUC (Mann-Whitney). None if labels are single-class."""
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = scores.argsort()
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


class Calibrator:
    """Maps a raw model Δactivity to a calibrated probability of a real regulatory effect."""

    def __init__(self):
        self._xp: np.ndarray | None = None   # sorted |delta| knots
        self._fp: np.ndarray | None = None   # calibrated probability at each knot
        self.diagnostics: dict = {}

    @property
    def is_fitted(self) -> bool:
        return self._xp is not None

    def fit(self, predicted_delta, measured_skew, is_emvar=None, tau: float = 0.5) -> "Calibrator":
        """Fit on held-out variants. `tau` = |skew| magnitude counted as a 'real effect'."""
        pred = np.asarray(predicted_delta, dtype=float)
        skew = np.asarray(measured_skew, dtype=float)
        target = (np.abs(skew) >= tau).astype(float)      # continuous-derived label (D6)

        scores = np.abs(pred)
        order = scores.argsort()
        xs, ys = scores[order], target[order]
        fitted = _pava(ys)

        # collapse duplicate x to a strictly increasing lookup for np.interp
        xp, fp = [], []
        for x, f in zip(xs, fitted):
            if xp and x == xp[-1]:
                fp[-1] = f                                 # nondecreasing -> keep the later (>=)
            else:
                xp.append(float(x))
                fp.append(float(f))
        self._xp, self._fp = np.array(xp), np.clip(np.array(fp), 1e-3, 1 - 1e-3)

        pear = float(np.corrcoef(pred, skew)[0, 1]) if len(pred) > 1 else float("nan")
        sp_p = np.argsort(np.argsort(pred)).astype(float)
        sp_s = np.argsort(np.argsort(skew)).astype(float)
        spear = float(np.corrcoef(sp_p, sp_s)[0, 1]) if len(pred) > 1 else float("nan")
        self.diagnostics = {
            "n": int(len(pred)), "tau": tau,
            "pearson_pred_vs_skew": round(pear, 4),
            "spearman_pred_vs_skew": round(spear, 4),
            "auc_emvar": (round(_auc(scores, np.asarray(is_emvar, dtype=int)), 4)
                          if is_emvar is not None and _auc(scores, np.asarray(is_emvar, dtype=int)) is not None
                          else None),
            "frac_effect_label": round(float(target.mean()), 4),
        }
        return self

    def transform(self, predicted_delta: float) -> float:
        """Raw Δ -> calibrated probability in [0, 1]."""
        if not self.is_fitted:
            raise RuntimeError("Calibrator is not fitted")
        return float(np.interp(abs(predicted_delta), self._xp, self._fp))


# --------------------------------------------------------------------------- aggregation
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p = min(max(p, 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


def _fallback_confidence(model_delta: float, scale: float = 0.4) -> float:
    """Confidence when no calibrator is available: a squashed |Δ| (documented heuristic)."""
    return _sigmoid((abs(model_delta) - scale) / (scale / 2))


def build_trust_report(model_delta: float, evidence: list[EvidenceItem], *,
                       calibrator: Calibrator | None = None,
                       evidence_weight: float = 0.6,
                       hi: float = 0.66, lo: float = 0.34) -> TrustReport:
    """Combine calibrated model confidence with independent evidence into a verdict.

    Confidence is aggregated in log-odds space: each concordant source adds
    `evidence_weight * item.weight`, each conflict subtracts it. Conflicts are surfaced
    explicitly. Call bands: >= `hi` -> altering, <= `lo` -> benign, else uncertain.
    """
    base = calibrator.transform(model_delta) if (calibrator and calibrator.is_fitted) \
        else _fallback_confidence(model_delta)

    agreements = [e for e in evidence if e.concordant is True]
    conflicts = [e for e in evidence if e.concordant is False]

    z = _logit(base)
    z += evidence_weight * sum(e.weight for e in agreements)
    z -= evidence_weight * sum(e.weight for e in conflicts)
    confidence = _sigmoid(z)

    if confidence >= hi:
        call = Call.DISRUPTIVE
    elif confidence <= lo:
        call = Call.BENIGN
    else:
        call = Call.UNCERTAIN

    rationale = (
        f"model confidence {base:.0%} "
        f"({'calibrated' if (calibrator and calibrator.is_fitted) else 'uncalibrated heuristic'}); "
        f"{len(agreements)} concordant source(s), {len(conflicts)} conflict(s) "
        f"-> {confidence:.0%}."
    )
    if conflicts:
        rationale += " Conflict(s) surfaced, not averaged away."

    return TrustReport(confidence=round(confidence, 4), call=call,
                       agreements=agreements, conflicts=conflicts, rationale=rationale)
