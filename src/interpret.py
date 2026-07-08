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


def _resolve_genome(genome):
    """Accept a Genome-like object (has .window) or an hg38 FASTA path -> Genome. None passes through."""
    if genome is None or hasattr(genome, "window"):
        return genome
    import os
    import sys
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    if data_dir not in sys.path:
        sys.path.insert(0, data_dir)
    from genome import Genome  # data/genome.py
    return Genome(genome)


def _windows_from_genome(genome, chrom, pos, ref, alt, length):
    """Build (seq_ref, seq_alt, ref_matches) by centering a `length`-bp window on `pos`.

    Handles SNVs and simple indels. `ref_matches` reports whether the reference genome base(s)
    at `pos` actually equal `ref` (a sanity flag surfaced in provenance, not a hard failure —
    the caller asserts the alleles).
    """
    seq_ref, off = genome.window(chrom, pos, length)
    ref, alt = ref.upper(), alt.upper()
    ref_matches = bool(0 <= off < len(seq_ref) and seq_ref[off:off + len(ref)] == ref)
    seq_alt = seq_ref[:off] + alt + seq_ref[off + len(ref):]
    return seq_ref, seq_alt, ref_matches


def interpret_variant(chrom: str, pos: int, ref: str, alt: str, build: str = "hg38",
                      *, seq_ref: str | None = None, seq_alt: str | None = None,
                      predictor=None, organoid_predictor=None,
                      model_delta: float | None = None, model_delta_organoid: float | None = None,
                      evidence: list[EvidenceItem] | None = None,
                      evidence_resources: dict | None = None,
                      calibrator=None, genome=None, motif_library=None,
                      window_len: int | None = None,
                      provenance: dict | None = None) -> Interpretation:
    """Interpret a single non-coding variant. Returns an auditable `Interpretation`.

    Pipeline: (1) sequence windows -> (2) model Δactivity -> (3) motif gain/loss ->
    (4) independent evidence -> (5) calibrated trust verdict -> (6) assemble + evidence chain.

    Sequence windows come from one of: explicit `seq_ref`/`seq_alt`, or a `genome` (a Genome
    instance or an hg38 FASTA path) from which a window centered on `pos` is extracted. Window
    length = `window_len` or the predictor's `seq_len` or 270 (the Deng element length).
    """
    # --- 1. sequence windows ------------------------------------------------------------
    ref_matches = None
    if seq_ref is None or seq_alt is None:
        genome = _resolve_genome(genome)
        if genome is None:
            raise ValueError(
                "need sequence windows: pass seq_ref/seq_alt, or a genome (Genome or hg38 FASTA path)"
            )
        length = window_len or getattr(predictor, "seq_len", None) or 270
        seq_ref, seq_alt, ref_matches = _windows_from_genome(genome, chrom, pos, ref, alt, length)

    # --- 2. model prediction ------------------------------------------------------------
    model_std = None
    if model_delta is None:
        if predictor is None:
            raise ValueError("need either model_delta or a predictor")
        # an ensemble predictor also returns its member-disagreement (std); a single model doesn't
        if hasattr(predictor, "score_variant_with_uncertainty"):
            model_delta, model_std = predictor.score_variant_with_uncertainty(seq_ref, seq_alt)
        else:
            model_delta = predictor.score_variant(seq_ref, seq_alt)
    if model_delta_organoid is None and organoid_predictor is not None:
        model_delta_organoid = organoid_predictor.score_variant(seq_ref, seq_alt)
    direction = _direction(model_delta)

    # --- 3. mechanism -------------------------------------------------------------------
    mechanisms = motifs.annotate_motifs(seq_ref, seq_alt, library=motif_library)

    # --- 4. grounding evidence ----------------------------------------------------------
    if evidence is None:
        try:
            evidence = evidence_mod.gather_evidence(chrom, pos, ref, alt, direction, build=build,
                                                    **(evidence_resources or {}))
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
    report = trust.build_trust_report(model_delta, evidence, calibrator=calibrator,
                                      model_uncertainty=model_std)

    # --- 6. assemble --------------------------------------------------------------------
    prov = dict(provenance) if provenance else {}
    if predictor is not None and hasattr(predictor, "provenance"):
        prov.setdefault("predictor", predictor.provenance())
    if ref_matches is not None:
        prov["genome_ref_match"] = ref_matches
        if not ref_matches:
            prov["genome_ref_warning"] = (
                f"reference base at {chrom}:{pos} does not equal ref '{ref}' in {build}; "
                "check allele orientation / build"
            )
    result = Interpretation(
        chrom=chrom, pos=pos, ref=ref, alt=alt, build=build,
        model_delta_primary=round(float(model_delta), 4),
        model_delta_organoid=(round(float(model_delta_organoid), 4)
                              if model_delta_organoid is not None else None),
        model_delta_primary_std=(round(float(model_std), 4) if model_std is not None else None),
        predicted_direction=direction,
        mechanisms=mechanisms, evidence=evidence, trust=report,
        provenance=prov or {"note": "sequences supplied by caller"},
    )
    result.build_evidence_chain()
    return result
