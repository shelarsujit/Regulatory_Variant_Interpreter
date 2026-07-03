"""The orchestrator: interpret_variant() ties every layer into one auditable result.

THEORY (plain English — see docs/00_overview_for_non_biologists.md §5):
This is the single public entry point. It runs the pipeline end to end and returns an
`Interpretation` (src/schema.py) — the call, the calibrated confidence, the mechanism, the
independent evidence, and the auditable evidence chain + provenance.

STATUS: the motif layer, the organoid-agreement signal, and the calibrated trust verdict
are LIVE. The two remaining injection points are (a) genome-window extraction (pass
`seq_ref`/`seq_alt` for now, or wire pyfaidx + hg38 later) and (b) the fine-tuned predictor
(pass a `predictor`, or a precomputed `model_delta`). Everything else runs today.
"""
from __future__ import annotations

from . import evidence as evidence_mod
from . import motifs, trust
from .schema import Direction, EvidenceItem, Interpretation


def _direction(delta: float, dead_zone: float = 0.05) -> Direction:
    if delta > dead_zone:
        return Direction.UP
    if delta < -dead_zone:
        return Direction.DOWN
    return Direction.NONE


def interpret_variant(chrom: str, pos: int, ref: str, alt: str, build: str = "hg38",
                      *, seq_ref: str | None = None, seq_alt: str | None = None,
                      predictor=None, organoid_predictor=None,
                      model_delta: float | None = None, model_delta_organoid: float | None = None,
                      evidence: list[EvidenceItem] | None = None,
                      calibrator=None, genome=None, motif_library=None,
                      provenance: dict | None = None) -> Interpretation:
    """Interpret a single non-coding variant. Returns an auditable `Interpretation`.

    Pipeline: (1) sequence windows -> (2) model Δactivity -> (3) motif gain/loss ->
    (4) independent evidence -> (5) calibrated trust verdict -> (6) assemble + evidence chain.
    """
    # --- 1. sequence windows ------------------------------------------------------------
    if seq_ref is None or seq_alt is None:
        raise NotImplementedError(
            "pass seq_ref/seq_alt, or wire genome-window extraction (pyfaidx + hg38) — Phase 2/3"
        )

    # --- 2. model prediction ------------------------------------------------------------
    if model_delta is None:
        if predictor is None:
            raise ValueError("need either model_delta or a predictor")
        model_delta = predictor.score_variant(seq_ref, seq_alt)
    if model_delta_organoid is None and organoid_predictor is not None:
        model_delta_organoid = organoid_predictor.score_variant(seq_ref, seq_alt)
    direction = _direction(model_delta)

    # --- 3. mechanism -------------------------------------------------------------------
    mechanisms = motifs.annotate_motifs(seq_ref, seq_alt, library=motif_library)

    # --- 4. grounding evidence ----------------------------------------------------------
    if evidence is None:
        try:
            evidence = evidence_mod.gather_evidence(chrom, pos, ref, alt, direction, build=build)
        except NotImplementedError:
            evidence = []
    evidence = list(evidence)
    # the organoid model is our cheapest independent second opinion (decision D4)
    if model_delta_organoid is not None:
        org_dir = _direction(model_delta_organoid)
        concordant = (org_dir == direction) if direction is not Direction.NONE else None
        evidence.append(EvidenceItem(
            source="organoid_model",
            summary=f"independent organoid-context model Δ={model_delta_organoid:+.3f}",
            value=round(model_delta_organoid, 3), direction=org_dir, concordant=concordant))

    # --- 5. trust verdict ---------------------------------------------------------------
    report = trust.build_trust_report(model_delta, evidence, calibrator=calibrator)

    # --- 6. assemble --------------------------------------------------------------------
    result = Interpretation(
        chrom=chrom, pos=pos, ref=ref, alt=alt, build=build,
        model_delta_primary=round(float(model_delta), 4),
        model_delta_organoid=(round(float(model_delta_organoid), 4)
                              if model_delta_organoid is not None else None),
        predicted_direction=direction,
        mechanisms=mechanisms, evidence=evidence, trust=report,
        provenance=provenance or {"note": "predictor not yet wired; sequences supplied by caller"},
    )
    result.build_evidence_chain()
    return result
