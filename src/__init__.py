"""Regulatory Variant Interpreter — trust-first interpretation of non-coding variants.

Public surface:
    from src import interpret_variant, Interpretation
`interpret_variant` is imported lazily so that merely importing the package (e.g. in a
notebook or the data layer) does not pull in torch / transformers.
"""
from .schema import Call, Direction, EvidenceItem, Interpretation, Mechanism, TrustReport

__all__ = [
    "interpret_variant",
    "Interpretation", "TrustReport", "EvidenceItem", "Mechanism", "Direction", "Call",
]


def interpret_variant(*args, **kwargs):
    from .interpret import interpret_variant as _impl
    return _impl(*args, **kwargs)
