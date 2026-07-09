"""Stacking meta-learner — fuse DNA-LM Δ + frozen big-model Δ + motif + eQTL into P(effect).

THEORY (plain English — docs/07_enhancement_design.md #2):
The single-nucleotide DNA-LM and a big frozen expression model (Enformer/Borzoi/AlphaGenome)
make DIFFERENT errors. A small supervised combiner over their outputs — plus the motif and eQTL
signals we already compute — beats any one alone at flagging real regulatory variants. It is also
the honest, charter-clean use of the frozen model: a FEATURE, never fine-tuned. And its per-feature
weights ARE the "why we trust this call" explanation the tool already promises.

WHY HAND-ROLLED (no sklearn): the repo deliberately keeps its trust math dependency-light and
inspectable (see trust.Calibrator's hand-rolled PAVA isotonic). This is a plain logistic
regression fit by gradient descent with L2 + class-imbalance weighting, standardized inputs, and
NaN-safe features (a missing evidence source imputes to the training mean, i.e. contributes
nothing beyond the intercept). Same JSON save/load contract as Calibrator.

ADDITIVE: NEW module. Nothing in trust.py / evidence.py / interpret.py is required to change for
this to exist; wiring is opt-in (interpret loads weights/meta_<ctx>.json IF present, else falls
back to today's single-feature isotonic calibrator).

LEAKAGE: fit ONLY on labeled held-out variants the DNA-LM never trained on, and hold out a slice
of THOSE for the reported AUC so the meta-learner is graded on variants it also never saw.
"""
from __future__ import annotations

import json
import warnings

import numpy as np

# Raw signed signals the caller supplies (what interpret_variant naturally has). These are NOT
# the model's features — they are transformed into effect-detection features below.
RAW_SIGNALS = ["dna_lm_delta", "dna_lm_sigma", "organoid_delta", "frozen_delta", "motif_dscore",
               "gtex_signed", "tss_dist_kb"]

# MODEL feature order (persisted; shown in the evidence chain). Effect presence is driven by
# MAGNITUDE, not sign (same reason trust.Calibrator uses |Δ|): a logistic model is linear, so it
# cannot learn |x| from a signed x. We therefore feed magnitudes, plus signed sign-CONCORDANCE
# terms (do two independent models agree on direction?) which carry the useful sign info AND are
# the trust thesis in a feature: agreement between independent predictors -> higher confidence.
DEFAULT_FEATURES = [
    "abs_dna_lm_delta",         # |Δ| from the fine-tuned model — primary effect-strength signal
    "dna_lm_sigma",             # ensemble std of Δ (higher -> less trustworthy; expect negative weight)
    "abs_organoid_delta",       # |Δ| from the independent organoid-context model (charter's 2nd model)
    "concordance_dna_organoid", # signed agreement primary<->organoid: two of OUR models concurring
    "abs_frozen_delta",         # |Δ| from the frozen big model (independent model class)
    "concordance_dna_frozen",   # signed agreement primary<->frozen big model
    "abs_motif_dscore",         # |TF-motif gain/loss Δscore| (mechanism strength)
    "abs_gtex_signed",          # |brain-eQTL effect| (0 if only a boolean flag exists)
    "tss_dist_kb",              # distance to nearest TSS in kb (proximity prior)
]


def _concord(a, b):
    """Signed direction-agreement of two effects: +min(|a|,|b|) if same sign, - if opposite;
    NaN if either is missing. Encodes 'independent predictors agree' as one number."""
    if np.isfinite(a) and np.isfinite(b):
        return np.sign(a) * np.sign(b) * min(abs(a), abs(b))
    return np.nan


def _sig(x):
    return float(x) if (x is not None and np.isfinite(x)) else np.nan


def assemble_features(signals: dict, feature_names=DEFAULT_FEATURES) -> np.ndarray:
    """Transform raw SIGNED signals into the effect-detection MODEL features.

    Accepts a dict of RAW_SIGNALS (signed Δ's etc.). Absent/None signals -> NaN so the fitted
    combiner imputes them to the training mean (contributing nothing beyond the intercept).
    """
    dna = _sig(signals.get("dna_lm_delta"))
    organoid = _sig(signals.get("organoid_delta"))
    frozen = _sig(signals.get("frozen_delta"))
    motif = _sig(signals.get("motif_dscore"))
    gtex = _sig(signals.get("gtex_signed"))

    def _absnan(x):
        return abs(x) if np.isfinite(x) else np.nan

    built = {
        "abs_dna_lm_delta": _absnan(dna),
        "dna_lm_sigma": _sig(signals.get("dna_lm_sigma")),
        "abs_organoid_delta": _absnan(organoid),
        "concordance_dna_organoid": _concord(dna, organoid),
        "abs_frozen_delta": _absnan(frozen),
        "concordance_dna_frozen": _concord(dna, frozen),
        "abs_motif_dscore": _absnan(motif),
        "abs_gtex_signed": _absnan(gtex),
        "tss_dist_kb": _sig(signals.get("tss_dist_kb")),
    }
    return np.array([built.get(name, np.nan) for name in feature_names], dtype=float)


def build_matrix(signal_dicts, feature_names=DEFAULT_FEATURES) -> np.ndarray:
    """Stack many raw-signal dicts into a model feature matrix (n, d)."""
    return np.vstack([assemble_features(s, feature_names) for s in signal_dicts]) \
        if signal_dicts else np.empty((0, len(feature_names)))


def _auc(scores: np.ndarray, labels: np.ndarray):
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    pos, neg = int(y.sum()), int((~y).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


class MetaCombiner:
    """Logistic stacking model over evidence features -> calibrated P(real regulatory effect)."""

    def __init__(self, feature_names=DEFAULT_FEATURES):
        self.feature_names = list(feature_names)
        self.w: np.ndarray | None = None      # standardized-space weights
        self.b: float = 0.0
        self.mean: np.ndarray | None = None    # per-feature train mean (for standardize + NaN impute)
        self.std: np.ndarray | None = None
        self.diagnostics: dict = {}

    @property
    def is_fitted(self) -> bool:
        return self.w is not None

    # ------------------------------------------------------------------ internals
    def _standardize(self, X: np.ndarray) -> np.ndarray:
        """Impute NaN -> train mean, then z-score. After imputation a missing feature sits at
        z=0 and contributes only the intercept."""
        Xf = np.where(np.isnan(X), self.mean, X)
        return (Xf - self.mean) / self.std

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, *, l2: float = 1.0, lr: float = 0.1, epochs: int = 2000,
            balance: bool = True) -> "MetaCombiner":
        """Fit on rows X (n, d) with binary labels y (n,). Class-imbalance weighted (emVars ~1%)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError(f"X must be (n, {len(self.feature_names)}); got {X.shape}")

        # standardization stats from columns that have any finite value. An all-NaN column
        # (an evidence source absent for every variant, e.g. an unwired frozen model) yields a
        # NaN mean/std from nanmean/nanstd — expected, so we silence the warning and fall back to
        # mean 0 / std 1, which makes that feature a no-op (imputes to 0, contributes intercept only).
        Xn = np.where(np.isfinite(X), X, np.nan)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mean = np.nanmean(Xn, axis=0)
            std = np.nanstd(Xn, axis=0)
        self.mean = np.where(np.isfinite(self.mean), self.mean, 0.0)
        self.std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)

        Z = self._standardize(X)
        n, d = Z.shape
        self.w = np.zeros(d)
        self.b = 0.0

        # per-sample weights: upweight the rare positive class to its inverse frequency
        if balance and y.sum() > 0 and y.sum() < n:
            w_pos = n / (2.0 * y.sum())
            w_neg = n / (2.0 * (n - y.sum()))
            sw = np.where(y > 0.5, w_pos, w_neg)
        else:
            sw = np.ones(n)
        sw = sw / sw.mean()

        for _ in range(epochs):
            p = 1.0 / (1.0 + np.exp(-(Z @ self.w + self.b)))
            g = sw * (p - y)
            grad_w = Z.T @ g / n + l2 * self.w / n
            grad_b = float(g.sum() / n)
            self.w -= lr * grad_w
            self.b -= lr * grad_b

        p = 1.0 / (1.0 + np.exp(-(Z @ self.w + self.b)))
        self.diagnostics = {
            "n": int(n), "n_positive": int(y.sum()), "l2": l2, "epochs": epochs,
            "auc_train": (round(_auc(p, y), 4) if _auc(p, y) is not None else None),
            # weights in standardized space are directly comparable across features -> importance
            "feature_weights": {name: round(float(wi), 4)
                                for name, wi in zip(self.feature_names, self.w)},
            "intercept": round(float(self.b), 4),
        }
        return self

    # ------------------------------------------------------------------ predict
    def predict_proba(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("MetaCombiner is not fitted")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Z = self._standardize(X)
        return 1.0 / (1.0 + np.exp(-(Z @ self.w + self.b)))

    def transform(self, signals: dict) -> float:
        """Convenience: dict of signals -> single P(effect)."""
        x = assemble_features(signals, self.feature_names)
        return float(self.predict_proba(x)[0])

    def explain(self, signals: dict) -> list[tuple[str, float]]:
        """Per-feature signed contribution to the log-odds (for the evidence chain).
        Sorted by absolute contribution — the biggest reasons first."""
        x = assemble_features(signals, self.feature_names)
        z = self._standardize(x.reshape(1, -1))[0]
        contribs = [(name, float(zi * wi)) for name, zi, wi in zip(self.feature_names, z, self.w)]
        contribs.sort(key=lambda t: abs(t[1]), reverse=True)
        return contribs

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        if not self.is_fitted:
            raise RuntimeError("MetaCombiner is not fitted")
        return {"feature_names": self.feature_names, "w": self.w.tolist(), "b": float(self.b),
                "mean": self.mean.tolist(), "std": self.std.tolist(),
                "diagnostics": self.diagnostics}

    def save(self, path: str) -> str:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "MetaCombiner":
        with open(path) as f:
            d = json.load(f)
        m = cls(d["feature_names"])
        m.w = np.asarray(d["w"], dtype=float)
        m.b = float(d["b"])
        m.mean = np.asarray(d["mean"], dtype=float)
        m.std = np.asarray(d["std"], dtype=float)
        m.diagnostics = d.get("diagnostics", {})
        return m
