"""Turn a prediction + evidence into a CALIBRATED confidence and an honest verdict.

THEORY (plain English — see docs/00_overview_for_non_biologists.md §4.2–4.4):
This module is the heart of the project. Two jobs:

1. CALIBRATION. A raw model score is not a probability. We learn a mapping (isotonic /
   Platt regression) from raw scores to honest probabilities using the held-out variants
   whose true effects the MPRA measured — WITHOUT ever having trained the model on them
   (the leakage discipline in data/splits.py guarantees this). After calibration, "70%
   confidence" should mean "right about 70% of the time." Because only ~1% of tested
   variants are significant, we calibrate primarily on the CONTINUOUS predicted-Δ vs
   measured-skew relationship and use the significant set as a classification check
   (decision D6).

2. AGGREGATION WITH HONEST CONFLICT. We combine the model's prediction with the
   independent evidence. Agreement raises confidence; conflict LOWERS it and is reported
   explicitly (never averaged away). A well-calibrated "uncertain" is a success, not a
   failure.

STATUS: stub (Phase 3). The Calibrator is fit once on data/processed/calibration_variants.
"""
from __future__ import annotations

from .schema import Call, Direction, EvidenceItem, TrustReport


class Calibrator:
    """Maps a raw model Δactivity to a calibrated probability of a real regulatory effect."""

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self._fitted = None

    def fit(self, predicted_delta, measured_skew, is_emvar=None) -> "Calibrator":
        """Fit on held-out variants (predicted Δ vs measured skew). See leakage note above."""
        raise NotImplementedError("Phase 3: isotonic/Platt fit on calibration_variants")

    def transform(self, predicted_delta: float) -> float:
        """Raw Δ -> calibrated probability in [0, 1]."""
        raise NotImplementedError("Phase 3")


def build_trust_report(model_delta: float, evidence: list[EvidenceItem], *,
                       calibrator: Calibrator | None = None) -> TrustReport:
    """Combine calibrated model confidence with independent evidence into a verdict.

    Partitions evidence into agreements vs conflicts, derives a calibrated confidence,
    picks a Call, and writes a one-paragraph rationale. Conflicts are always surfaced.
    """
    raise NotImplementedError("Phase 3: aggregate calibrated confidence + conflict accounting")
