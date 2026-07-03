"""Gather INDEPENDENT evidence about a variant — the grounding layer.

THEORY (plain English — see docs/00_overview_for_non_biologists.md §4.1):
Our fine-tuned model gives one opinion. To make that opinion trustworthy we collect
signals produced *independently* of it and check whether they agree. The more genuinely
independent sources concur, the higher our confidence; when they conflict, the trust
layer surfaces it instead of hiding it.

Independence ladder (weakest -> strongest independence from our model, decision D4/D7):
  * organoid_model          — our 2nd model, different biological context (near-independent)
  * frozen_foundation_model — Enformer / AlphaGenome, zero-shot: a different model class
  * held_out_MPRA           — the actual wet-lab measured skew for THIS variant, if tested
  * GTEx_eQTL               — population genetics: is the variant linked to expression IRL?
  * ClinVar                 — has anyone clinically classified it?
  * TSS                     — proximity to a gene start (context, not a direction)

Each source returns an EvidenceItem carrying a direction and whether it is `concordant`
with the primary model's predicted direction. STATUS: stub (Phase 3).
"""
from __future__ import annotations

from .schema import Direction, EvidenceItem


def gather_evidence(chrom: str, pos: int, ref: str, alt: str,
                    model_direction: Direction, *, build: str = "hg38") -> list[EvidenceItem]:
    """Collect all available independent evidence for a variant.

    Missing sources are simply omitted (absence is not conflict). Each item's
    `concordant` flag is set relative to `model_direction`.
    """
    raise NotImplementedError("Phase 3: query the sources below and assemble EvidenceItems")


# --- per-source stubs (each returns an EvidenceItem or None) -----------------------------
def from_held_out_mpra(chrom, pos, ref, alt, model_direction, calibration_table=None): ...
def from_gtex_eqtl(chrom, pos, ref, alt, model_direction, build="hg38"): ...
def from_clinvar(chrom, pos, ref, alt, build="hg38"): ...
def from_tss_proximity(chrom, pos, build="hg38"): ...
def from_frozen_foundation_model(chrom, pos, ref, alt, model_direction, build="hg38"): ...
def from_organoid_model(seq_ref, seq_alt, model_direction): ...
