"""Explain the mechanism: which transcription-factor motif does the variant gain or lose?

THEORY (plain English — see docs/00_overview_for_non_biologists.md §3.6):
A number ("activity drops 0.8") is not an explanation a curator can act on. Transcription
factors (TFs) are proteins that dock onto specific short DNA "words" (motifs) to tune a
gene's volume. If a single-base change destroys a motif, a TF can no longer dock there; if
it creates one, a new TF can. We scan the reference and alternate sequences against a
library of known motifs (JASPAR, expressed as position-weight matrices / PWMs) and report
the biggest gains/losses as candidate mechanisms:
    "ALT disrupts a predicted CTCF motif at offset 97 (Δscore -6.1)"

This turns the model's number into a testable, recognizable hypothesis. STATUS: stub (Phase 2/3).
"""
from __future__ import annotations

from .schema import Mechanism

DEFAULT_JASPAR_RELEASE = "JASPAR2024"


def annotate_motifs(seq_ref: str, seq_alt: str, *,
                    jaspar_release: str = DEFAULT_JASPAR_RELEASE,
                    score_threshold: float = 0.8,
                    top_k: int = 5) -> list[Mechanism]:
    """Scan both alleles against JASPAR PWMs; return the top motif gain/loss events.

    A "loss" = a motif that scores above threshold on REF but not ALT; a "gain" = the
    reverse. Events are ranked by |Δscore| and truncated to `top_k`.
    """
    raise NotImplementedError("Phase 2/3: PWM scan of ref vs alt window via pyjaspar")
