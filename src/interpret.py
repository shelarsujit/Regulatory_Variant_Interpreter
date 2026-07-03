"""The orchestrator: interpret_variant() ties every layer into one auditable result.

THEORY (plain English — see docs/00_overview_for_non_biologists.md §5):
This is the single public entry point. It runs the pipeline end to end and returns an
`Interpretation` (src/schema.py) — the call, the calibrated confidence, the mechanism, the
independent evidence, and the auditable evidence chain + provenance. The steps below are
the whole system in miniage; each delegates to a dedicated, independently-testable module.

STATUS: scaffold. The orchestration flow is written out so the contract is unambiguous;
the component calls raise NotImplementedError until their phases land.
"""
from __future__ import annotations

from .schema import Direction, Interpretation


def _direction(delta: float, dead_zone: float = 0.05) -> Direction:
    if delta > dead_zone:
        return Direction.UP
    if delta < -dead_zone:
        return Direction.DOWN
    return Direction.NONE


def interpret_variant(chrom: str, pos: int, ref: str, alt: str, build: str = "hg38",
                      *, predictor=None, organoid_predictor=None,
                      calibrator=None, genome=None) -> Interpretation:
    """Interpret a single non-coding variant. Returns an auditable `Interpretation`.

    Pipeline:
      1. Build the ~200 bp reference window around `pos`; make the alt copy (one base swapped).
      2. Predict Δactivity with the primary-cortex model (and the organoid model).
      3. Annotate mechanism: JASPAR motif gain/loss (motifs.annotate_motifs).
      4. Gather independent evidence (evidence.gather_evidence).
      5. Build the calibrated trust verdict (trust.build_trust_report).
      6. Assemble the Interpretation, its evidence chain, and provenance.
    """
    # --- 1. sequence windows ------------------------------------------------------------
    #   seq_ref, seq_alt = build_windows(chrom, pos, ref, alt, genome, build)
    # --- 2. model prediction ------------------------------------------------------------
    #   delta_primary  = (predictor or ActivityPredictor("...","primary")).score_variant(seq_ref, seq_alt)
    #   delta_organoid = organoid_predictor.score_variant(...) if organoid_predictor else None
    # --- 3. mechanism -------------------------------------------------------------------
    #   mechanisms = motifs.annotate_motifs(seq_ref, seq_alt)
    # --- 4. grounding -------------------------------------------------------------------
    #   evidence = evidence.gather_evidence(chrom, pos, ref, alt, _direction(delta_primary), build=build)
    # --- 5. trust -----------------------------------------------------------------------
    #   trust = trust.build_trust_report(delta_primary, evidence, calibrator=calibrator)
    # --- 6. assemble --------------------------------------------------------------------
    #   result = Interpretation(chrom, pos, ref, alt, build, delta_primary, delta_organoid,
    #                           _direction(delta_primary), mechanisms, evidence, trust,
    #                           provenance={...checkpoint hash, data versions...})
    #   result.build_evidence_chain()
    #   return result
    raise NotImplementedError("Phases 2–3: wire predictor + motifs + evidence + trust per the flow above")
