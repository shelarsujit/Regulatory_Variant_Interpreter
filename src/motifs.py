"""Explain the mechanism: which transcription-factor motif does the variant gain or lose?

THEORY (plain English — see docs/00_overview_for_non_biologists.md §3.6):
Transcription factors (TFs) are proteins that dock onto specific short DNA "words"
(motifs) to tune a gene's volume. A motif is scored with a Position Weight Matrix (PWM):
a per-position log-odds table saying how much each base at each offset favors a real
binding site over random DNA. If a single-base change destroys a strong motif match, a TF
can no longer dock (a "loss"); if it creates one, a new TF can (a "gain"). Turning the
model's bare number into "ALT disrupts a predicted CTCF motif" gives the curator a
testable, recognizable hypothesis.

DESIGN (decision D10): this module is dependency-light (numpy only) and OFFLINE-CAPABLE.
  * It ships a small library of *illustrative consensus* motifs (AP-1, CRE, GC-box, TATA,
    E-box, CCAAT) so it runs and is testable with zero downloads.
  * For production, load the real JASPAR database via `load_jaspar_pfms(path)` (parses
    JASPAR .pfm/.jaspar files) or `library_from_pyjaspar()` if pyjaspar + its DB are
    installed. Swap the library; the scoring math is identical.
Only motif windows that OVERLAP the variant are compared between alleles, so a strong
site elsewhere in the window cannot mask (or fake) the variant's local effect.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .schema import Mechanism

BASES = "ACGT"
_IDX = {b: i for i, b in enumerate(BASES)}
_BACKGROUND = np.array([0.25, 0.25, 0.25, 0.25])

# IUPAC degeneracy -> allowed bases (used to compactly specify illustrative motifs)
_IUPAC = {
    "A": "A", "C": "C", "G": "G", "T": "T",
    "R": "AG", "Y": "CT", "S": "GC", "W": "AT", "K": "GT", "M": "AC",
    "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG", "N": "ACGT",
}

# Illustrative consensus motifs (textbook cores; NOT full JASPAR matrices). Clearly labelled
# so nobody mistakes them for the real database — production should load JASPAR.
_ILLUSTRATIVE_CONSENSUS = {
    "AP-1": "TGASTCA",     # AP-1 / Jun-Fos
    "CRE": "TGACGTCA",     # CREB
    "GC-box(SP1)": "GGGGCGGGG",
    "TATA": "TATAAA",
    "E-box": "CACGTG",     # bHLH (MYC, etc.)
    "CCAAT": "CCAAT",      # NF-Y / CEBP core
}


@dataclass
class PWM:
    """A motif as a per-position log2-odds matrix (length L x 4 bases)."""
    name: str
    log_odds: np.ndarray          # shape (L, 4), log2(P(base|pos) / background)

    @property
    def length(self) -> int:
        return self.log_odds.shape[0]

    @property
    def max_score(self) -> float:
        return float(self.log_odds.max(axis=1).sum())

    @property
    def min_score(self) -> float:
        return float(self.log_odds.min(axis=1).sum())

    def relative(self, score: float) -> float:
        """Map a raw log-odds score to [0, 1] between the motif's worst and best possible."""
        lo, hi = self.min_score, self.max_score
        return 0.0 if hi <= lo else (score - lo) / (hi - lo)

    @classmethod
    def from_pfm(cls, name: str, pfm: np.ndarray, pseudocount: float = 0.8) -> "PWM":
        """Build from a position frequency (count) matrix of shape (L, 4)."""
        pfm = np.asarray(pfm, dtype=float)
        ppm = (pfm + pseudocount) / (pfm.sum(axis=1, keepdims=True) + 4 * pseudocount)
        return cls(name=name, log_odds=np.log2(ppm / _BACKGROUND))

    @classmethod
    def from_consensus(cls, name: str, consensus: str, n: int = 100, noise: float = 2.0) -> "PWM":
        """Build an illustrative PWM from an (optionally IUPAC) consensus string."""
        L = len(consensus)
        pfm = np.full((L, 4), noise, dtype=float)
        for i, letter in enumerate(consensus.upper()):
            allowed = _IUPAC[letter]
            share = n / len(allowed)
            for b in allowed:
                pfm[i, _IDX[b]] += share
        return cls.from_pfm(name, pfm)


# --------------------------------------------------------------------------- sequence utils
def _encode(seq: str) -> np.ndarray:
    """Sequence -> int8 indices; non-ACGT becomes -1 (treated as an invalid window)."""
    return np.array([_IDX.get(b, -1) for b in seq.upper()], dtype=np.int64)


def _rc_window(idx_window: np.ndarray) -> np.ndarray:
    """Reverse-complement a window of base indices (A0 C1 G2 T3 -> complement = 3 - i)."""
    return (3 - idx_window)[::-1]


def _score_window(log_odds: np.ndarray, idx_window: np.ndarray) -> float:
    if (idx_window < 0).any():
        return -math.inf
    return float(log_odds[np.arange(len(idx_window)), idx_window].sum())


def _best_overlapping(pwm: PWM, seq_idx: np.ndarray, var_offset: int) -> tuple[float, int]:
    """Best PWM score (both strands) among windows that OVERLAP the variant position.

    Returns (best_raw_score, window_start). Restricting to overlapping windows isolates
    the variant's effect: non-overlapping sites are identical on ref and alt and cancel.
    """
    L = pwm.length
    n = len(seq_idx)
    best, best_start = -math.inf, -1
    first = max(0, var_offset - L + 1)
    last = min(n - L, var_offset)
    for start in range(first, last + 1):
        window = seq_idx[start:start + L]
        s = max(_score_window(pwm.log_odds, window), _score_window(pwm.log_odds, _rc_window(window)))
        if s > best:
            best, best_start = s, start
    return best, best_start


# --------------------------------------------------------------------------- library
def default_library() -> list[PWM]:
    """The built-in illustrative motif set (offline)."""
    return [PWM.from_consensus(name, cons) for name, cons in _ILLUSTRATIVE_CONSENSUS.items()]


def load_jaspar_pfms(path: str) -> list[PWM]:
    """Parse a JASPAR .pfm/.jaspar file (one or more matrices) into PWMs (production path)."""
    pwms, name, rows = [], None, []

    def flush():
        if name and len(rows) == 4:
            pwms.append(PWM.from_pfm(name, np.array(rows, dtype=float).T))

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                name, rows = line[1:].split()[0], []
            else:
                nums = [float(x) for x in line.replace("[", " ").replace("]", " ").split()
                        if x.lstrip("-").replace(".", "", 1).isdigit()]
                if nums:
                    rows.append(nums)
    flush()
    return pwms


# --------------------------------------------------------------------------- public API
def annotate_motifs(seq_ref: str, seq_alt: str, *,
                    library: list[PWM] | None = None,
                    present_rel: float = 0.75,
                    min_rel_delta: float = 0.10,
                    top_k: int = 5) -> list[Mechanism]:
    """Scan both alleles for TF motifs overlapping the variant; return top gain/loss events.

    A motif is reported when it is plausibly present on at least one allele
    (relative score >= `present_rel`) AND the two alleles differ by at least
    `min_rel_delta` in relative score. `event`="loss" if ALT weakens the motif, else "gain".
    Events are ranked by |Δscore| (log-odds bits) and truncated to `top_k`.
    """
    if len(seq_ref) != len(seq_alt):
        return []  # indels handled in a later phase
    diffs = [i for i, (a, b) in enumerate(zip(seq_ref.upper(), seq_alt.upper())) if a != b]
    if len(diffs) != 1:
        return []
    var_offset = diffs[0]

    library = library or default_library()
    ref_idx, alt_idx = _encode(seq_ref), _encode(seq_alt)

    events: list[Mechanism] = []
    for pwm in library:
        ref_raw, ref_start = _best_overlapping(pwm, ref_idx, var_offset)
        alt_raw, alt_start = _best_overlapping(pwm, alt_idx, var_offset)
        if not (math.isfinite(ref_raw) and math.isfinite(alt_raw)):
            continue
        rel_ref, rel_alt = pwm.relative(ref_raw), pwm.relative(alt_raw)
        if max(rel_ref, rel_alt) < present_rel:
            continue
        if abs(rel_ref - rel_alt) < min_rel_delta:
            continue
        event = "loss" if alt_raw < ref_raw else "gain"
        position = ref_start if event == "loss" else alt_start
        events.append(Mechanism(tf=pwm.name, event=event, position=int(position),
                                delta_score=round(alt_raw - ref_raw, 3)))

    events.sort(key=lambda m: abs(m.delta_score), reverse=True)
    return events[:top_k]
