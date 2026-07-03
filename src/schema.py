"""The interface contract for the whole system.

THEORY (plain English — see docs/00_overview_for_non_biologists.md):
The product of this tool is NOT a bare number ("activity drops 0.8"). It is a
structured, auditable judgement a variant curator can act on. This module fixes the
*shape* of that judgement so every other module (predictor, evidence, motifs, trust,
app) agrees on one return type.

`interpret_variant(chrom, pos, ref, alt) -> Interpretation`

An `Interpretation` bundles four things, in increasing order of what makes it
trustworthy:
  1. the model's raw prediction         (model_delta_primary / _organoid)
  2. the mechanism                       (mechanisms: which TF motif is gained/lost)
  3. the independent grounding evidence  (evidence: eQTL, ClinVar, frozen model, ...)
  4. the trust verdict                   (trust: calibrated confidence + explicit conflicts)
plus an auditable evidence_chain and provenance (which checkpoint, which data versions).

Only #1 comes from our fine-tuned model. Everything else exists to make #1 trustable.
This file is deliberately dependency-free (pure dataclasses) so it can be imported
anywhere, including into notebooks and the demo.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class Direction(int, Enum):
    """Direction of a regulatory effect, sign-aligned to the ALT allele."""
    DOWN = -1   # ALT lowers activity (turns the volume down)
    NONE = 0    # no meaningful change
    UP = 1      # ALT raises activity (turns the volume up)


class Call(str, Enum):
    """The headline classification returned to the curator."""
    DISRUPTIVE = "likely regulatory-altering"
    BENIGN = "likely benign"
    UNCERTAIN = "uncertain"


@dataclass
class EvidenceItem:
    """One independent piece of evidence about the variant.

    `concordant` answers: does this source agree with the primary model's predicted
    direction? True/False for a real agreement/conflict, None when not applicable
    (e.g. TSS proximity has no direction). The trust layer reads these to decide
    confidence and to build the explicit conflict list.
    """
    source: str                     # "held_out_MPRA" | "GTEx_eQTL" | "ClinVar" | "TSS" |
                                    # "frozen_foundation_model" | "organoid_model" | "motif"
    summary: str                    # human-readable one-liner for the evidence chain
    value: Optional[Any] = None     # numeric effect / categorical class (JSON-serializable)
    direction: Optional[Direction] = None
    concordant: Optional[bool] = None
    weight: float = 1.0             # relative influence on the confidence aggregation
    detail: dict = field(default_factory=dict)


@dataclass
class Mechanism:
    """A candidate mechanistic explanation: a TF motif created or destroyed by the variant."""
    tf: str                         # e.g. "CTCF"
    event: str                      # "loss" | "gain"
    position: int                   # 0-based offset within the modeled element
    delta_score: float              # PWM log-odds change (ref -> alt)

    def describe(self) -> str:
        verb = "disrupts a predicted" if self.event == "loss" else "creates a predicted"
        return f"ALT {verb} {self.tf} motif at offset {self.position} (Δscore {self.delta_score:+.2f})"


@dataclass
class TrustReport:
    """The verdict: a calibrated confidence plus an HONEST accounting of agreement/conflict."""
    confidence: float               # calibrated probability in [0, 1] (NOT a raw softmax)
    call: Call
    agreements: list[EvidenceItem] = field(default_factory=list)
    conflicts: list[EvidenceItem] = field(default_factory=list)
    rationale: str = ""             # one paragraph: how confidence was reached

    @property
    def n_independent_sources(self) -> int:
        return len({e.source for e in (self.agreements + self.conflicts)})

    @property
    def has_conflict(self) -> bool:
        return len(self.conflicts) > 0


@dataclass
class Interpretation:
    """The full, auditable result of interpreting one variant."""
    # --- input variant ---
    chrom: str
    pos: int
    ref: str
    alt: str
    build: str = "hg38"

    # --- model prediction (the only part from our fine-tuned model) ---
    model_delta_primary: float = 0.0            # predicted ALT - REF activity, primary cortex
    model_delta_organoid: Optional[float] = None  # independent second-context model
    predicted_direction: Direction = Direction.NONE

    # --- mechanism ---
    mechanisms: list[Mechanism] = field(default_factory=list)

    # --- grounding evidence + trust verdict ---
    evidence: list[EvidenceItem] = field(default_factory=list)
    trust: Optional[TrustReport] = None

    # --- audit ---
    evidence_chain: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)  # checkpoint hash, data versions, model tags

    # ------------------------------------------------------------------ helpers
    @property
    def variant_id(self) -> str:
        return f"{self.chrom}:{self.pos}:{self.ref}>{self.alt}"

    def summary_line(self) -> str:
        conf = f"{self.trust.confidence:.0%}" if self.trust else "n/a"
        call = self.trust.call.value if self.trust else "unscored"
        flag = "  ⚠ CONFLICT" if (self.trust and self.trust.has_conflict) else ""
        return f"{self.variant_id}  →  {call} (confidence {conf}){flag}"

    def build_evidence_chain(self) -> list[str]:
        """Compose the human-readable audit trail from the structured fields."""
        chain: list[str] = [f"Variant {self.variant_id} ({self.build})"]
        chain.append(
            f"Primary-cortex model Δactivity = {self.model_delta_primary:+.3f} "
            f"(direction: {self.predicted_direction.name})"
        )
        if self.model_delta_organoid is not None:
            chain.append(f"Organoid model Δactivity = {self.model_delta_organoid:+.3f} (independent context)")
        for m in self.mechanisms:
            chain.append("Mechanism: " + m.describe())
        for e in self.evidence:
            tag = {True: "agrees", False: "CONFLICTS", None: "context"}[e.concordant]
            chain.append(f"[{e.source} · {tag}] {e.summary}")
        if self.trust:
            chain.append(f"Verdict: {self.trust.call.value} @ {self.trust.confidence:.0%} — {self.trust.rationale}")
        self.evidence_chain = chain
        return chain

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        def default(o):
            if isinstance(o, Enum):
                return o.value
            return str(o)
        return json.dumps(self.to_dict(), indent=indent, default=default)
