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
with the primary model's predicted direction.

DESIGN (decision D13): offline-first. The held-out MPRA source is wired for real against
`data/processed/calibration_variants.parquet` (a genuine measurement, independent of our
model). Every external source (GTEx / ClinVar / TSS / frozen foundation model) takes an
injectable local table or callable and returns None when its resource is absent — a MISSING
source is omitted, never treated as a conflict. pandas is imported lazily so importing this
module stays cheap.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Callable, Optional

from .schema import Direction, EvidenceItem

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CALIBRATION = os.path.join(_HERE, os.pardir, "data", "processed", "calibration_variants.parquet")

# per-source influence on the confidence aggregation, encoding the independence ladder (D13).
_WEIGHTS = {
    "held_out_MPRA": 2.0,           # a direct wet-lab measurement of THIS variant — strongest
    "frozen_foundation_model": 1.5, # a different model class, zero-shot
    "GTEx_eQTL": 1.2,               # population-genetics link to real expression
    "ClinVar": 1.0,                 # human clinical classification (often direction-less)
    "TSS": 0.3,                     # context only, no direction
}


# --------------------------------------------------------------------------- helpers
def _direction_of(value: float, dead_zone: float = 0.05) -> Direction:
    if value > dead_zone:
        return Direction.UP
    if value < -dead_zone:
        return Direction.DOWN
    return Direction.NONE


def _concordant(src_dir: Optional[Direction], model_dir: Direction) -> Optional[bool]:
    """True/False only when both sides carry a real direction; None otherwise (context)."""
    if src_dir is None or src_dir is Direction.NONE or model_dir is Direction.NONE:
        return None
    return src_dir == model_dir


@lru_cache(maxsize=8)
def _load_table(path: str):
    """Cache-load a parquet/csv table by path; None if missing/unreadable."""
    if not path or not os.path.isfile(path):
        return None
    import pandas as pd
    try:
        return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    except Exception:
        return None


def _as_frame(table):
    """Accept a DataFrame, a path, or None -> DataFrame or None."""
    if table is None:
        return None
    if isinstance(table, str):
        return _load_table(table)
    return table  # assume DataFrame-like


def _match_variant(df, chrom, pos, ref, alt):
    """Return the single matching row (Series) for a variant, or None."""
    if df is None or len(df) == 0:
        return None
    m = (df["chrom"].astype(str) == str(chrom)) & (df["pos"].astype("int64") == int(pos))
    for col, want in (("ref", ref), ("alt", alt)):
        if col in df.columns:
            m &= (df[col].astype(str).str.upper() == str(want).upper())
    hit = df[m]
    return hit.iloc[0] if len(hit) else None


# --- per-source functions (each returns an EvidenceItem or None) -------------------------
def from_held_out_mpra(chrom, pos, ref, alt, model_direction, calibration_table=None):
    """The measured allelic skew for THIS variant, if it was in the held-out MPRA set.

    This is a real wet-lab measurement independent of our model (and, by the D5 locus split,
    the model was never trained on it) — the strongest grounding we have when present.
    """
    df = _as_frame(calibration_table if calibration_table is not None else _DEFAULT_CALIBRATION)
    row = _match_variant(df, chrom, pos, ref, alt)
    if row is None or "measured_skew" not in row:
        return None
    skew = float(row["measured_skew"])
    src_dir = _direction_of(skew)
    fdr = float(row["fdr"]) if "fdr" in row and row["fdr"] == row["fdr"] else None
    is_emvar = bool(row["is_emvar"]) if "is_emvar" in row else None
    sig = (f", FDR={fdr:.3g}" if fdr is not None else "")
    return EvidenceItem(
        source="held_out_MPRA",
        summary=f"held-out MPRA measured skew {skew:+.3f}{sig}"
                + (" (significant emVar)" if is_emvar else ""),
        value=round(skew, 4), direction=src_dir,
        concordant=_concordant(src_dir, model_direction),
        weight=_WEIGHTS["held_out_MPRA"] * (1.5 if is_emvar else 1.0),
        detail={"fdr": fdr, "is_emvar": is_emvar},
    )


def from_gtex_eqtl(chrom, pos, ref, alt, model_direction, build="hg38", gtex_table=None):
    """GTEx eQTL: does the variant associate with a gene's expression in the population?

    Expects a local table with columns chrom,pos,ref,alt and a signed effect ('slope' or
    'nes'). Returns None when no table is supplied (absence != conflict).
    """
    df = _as_frame(gtex_table)
    row = _match_variant(df, chrom, pos, ref, alt)
    if row is None:
        return None
    eff_col = next((c for c in ("slope", "nes", "effect", "beta") if c in row), None)
    if eff_col is None:
        return None
    slope = float(row[eff_col])
    src_dir = _direction_of(slope)
    gene = row.get("gene") if hasattr(row, "get") else None
    return EvidenceItem(
        source="GTEx_eQTL",
        summary=f"GTEx eQTL {eff_col}={slope:+.3f}" + (f" on {gene}" if gene else ""),
        value=round(slope, 4), direction=src_dir,
        concordant=_concordant(src_dir, model_direction),
        weight=_WEIGHTS["GTEx_eQTL"], detail={"gene": gene, "effect_col": eff_col},
    )


def from_clinvar(chrom, pos, ref, alt, build="hg38", clinvar_table=None):
    """ClinVar clinical classification. Usually direction-less -> attaches as context."""
    df = _as_frame(clinvar_table)
    row = _match_variant(df, chrom, pos, ref, alt)
    if row is None:
        return None
    sig_col = next((c for c in ("clinical_significance", "clnsig", "significance") if c in row), None)
    if sig_col is None:
        return None
    significance = str(row[sig_col])
    return EvidenceItem(
        source="ClinVar", summary=f"ClinVar: {significance}",
        value=significance, direction=None, concordant=None,   # no regulatory direction
        weight=_WEIGHTS["ClinVar"], detail={"significance": significance},
    )


def from_tss_proximity(chrom, pos, build="hg38", tss_table=None, near_bp=2000):
    """Distance to the nearest transcription start site — context, not a direction.

    Expects a table with columns chrom, tss (position). Returns None if none supplied.
    """
    df = _as_frame(tss_table)
    if df is None or "tss" not in getattr(df, "columns", []):
        return None
    sub = df[df["chrom"].astype(str) == str(chrom)]
    if len(sub) == 0:
        return None
    dist = int((sub["tss"].astype("int64") - int(pos)).abs().min())
    return EvidenceItem(
        source="TSS", summary=f"{dist:,} bp from nearest TSS"
                + (" (proximal)" if dist <= near_bp else " (distal)"),
        value=dist, direction=None, concordant=None,
        weight=_WEIGHTS["TSS"] * (1.0 if dist <= near_bp else 0.5),
        detail={"distance_bp": dist, "proximal": dist <= near_bp},
    )


def from_frozen_foundation_model(chrom, pos, ref, alt, model_direction, build="hg38",
                                 foundation_fn: Callable[..., float] | None = None):
    """A frozen Enformer/AlphaGenome zero-shot Δ (decision D7) — the top-of-ladder independent
    model. `foundation_fn(chrom,pos,ref,alt,build) -> signed Δ`. None if not provided.
    """
    if foundation_fn is None:
        return None
    delta = foundation_fn(chrom, pos, ref, alt, build)
    if delta is None:
        return None
    delta = float(delta)
    src_dir = _direction_of(delta)
    return EvidenceItem(
        source="frozen_foundation_model",
        summary=f"frozen foundation model Δ={delta:+.3f} (zero-shot)",
        value=round(delta, 4), direction=src_dir,
        concordant=_concordant(src_dir, model_direction),
        weight=_WEIGHTS["frozen_foundation_model"], detail={},
    )


def from_organoid_model(seq_ref, seq_alt, model_direction, organoid_predictor=None,
                        organoid_delta=None):
    """Our independent organoid-context model (decision D4). Note: `interpret_variant`
    already attaches this when given `model_delta_organoid`; this helper exists so
    `gather_evidence` can also produce it directly from an organoid predictor + sequences.
    """
    if organoid_delta is None:
        if organoid_predictor is None or seq_ref is None or seq_alt is None:
            return None
        organoid_delta = organoid_predictor.score_variant(seq_ref, seq_alt)
    organoid_delta = float(organoid_delta)
    src_dir = _direction_of(organoid_delta)
    return EvidenceItem(
        source="organoid_model",
        summary=f"independent organoid-context model Δ={organoid_delta:+.3f}",
        value=round(organoid_delta, 3), direction=src_dir,
        concordant=_concordant(src_dir, model_direction), weight=1.0, detail={},
    )


# --------------------------------------------------------------------------- orchestration
def gather_evidence(chrom: str, pos: int, ref: str, alt: str,
                    model_direction: Direction, *, build: str = "hg38",
                    calibration_table: Any = None, gtex_table: Any = None,
                    clinvar_table: Any = None, tss_table: Any = None,
                    foundation_fn: Callable[..., float] | None = None,
                    seq_ref: str | None = None, seq_alt: str | None = None,
                    organoid_predictor=None, organoid_delta: float | None = None,
                    ) -> list[EvidenceItem]:
    """Collect all AVAILABLE independent evidence for a variant.

    Missing sources are simply omitted (absence is not conflict, decision D13). Each item's
    `concordant` flag is set relative to `model_direction`. Resources are injectable so the
    same code runs offline (held-out MPRA only) or fully grounded (all tables + foundation
    model wired). `interpret_variant` attaches the organoid signal itself, so the organoid
    args here default off to avoid double-counting.
    """
    items: list[Optional[EvidenceItem]] = [
        from_held_out_mpra(chrom, pos, ref, alt, model_direction, calibration_table),
        from_frozen_foundation_model(chrom, pos, ref, alt, model_direction, build, foundation_fn),
        from_gtex_eqtl(chrom, pos, ref, alt, model_direction, build, gtex_table),
        from_clinvar(chrom, pos, ref, alt, build, clinvar_table),
        from_tss_proximity(chrom, pos, build, tss_table),
    ]
    if organoid_predictor is not None or organoid_delta is not None:
        items.append(from_organoid_model(seq_ref, seq_alt, model_direction,
                                         organoid_predictor, organoid_delta))
    return [it for it in items if it is not None]
