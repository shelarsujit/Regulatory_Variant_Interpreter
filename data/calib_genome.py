"""CalibrationGenome — a genome shim backed by the held-out variant table (no hg38 FASTA).

WHY (docs/01 §4): the real Variant-coordinates path needs an hg38 FASTA (~3 GB) to turn a
(chrom, pos) into the 270 bp element window. But the held-out `calibration_variants.parquet`
ALREADY stores the exact reference + alternate element sequences for every variant it contains.
So for those variants we can serve the coordinates tab with zero download — enabling the rich
curated demos (agreement / conflict / eQTL-catch) offline.

It duck-types the one method `interpret_variant` needs — `window(chrom, pos, length) -> (seq, off)`
— so it drops straight into `interpret_variant(genome=CalibrationGenome(...))`. Unknown variants
raise KeyError (the caller surfaces it), exactly as an out-of-range FASTA lookup would.

This is a demo/offline aid, NOT the source of truth: real arbitrary-coordinate interpretation
still uses the FASTA-backed `data/genome.py::Genome`. Both satisfy the same `.window` contract.
"""
from __future__ import annotations


class CalibrationGenome:
    """Serve stored element sequences for variants present in a calibration/variant table."""

    def __init__(self, table):
        """`table`: a DataFrame (or path) with chrom, pos, seq_ref, seq_alt columns."""
        import pandas as pd
        if isinstance(table, str):
            table = pd.read_parquet(table) if table.endswith(".parquet") else pd.read_csv(table)
        need = {"chrom", "pos", "seq_ref", "seq_alt"}
        missing = need - set(table.columns)
        if missing:
            raise ValueError(f"CalibrationGenome table missing columns: {missing}")
        # index by (chrom, pos) -> (seq_ref, seq_alt); keep the first if duplicated
        self._by_pos = {}
        for c, p, sr, sa in zip(table["chrom"].astype(str), table["pos"].astype("int64"),
                                table["seq_ref"].astype(str), table["seq_alt"].astype(str)):
            self._by_pos.setdefault((c, int(p)), (sr, sa))

    def __contains__(self, chrom_pos) -> bool:
        return tuple(chrom_pos) in self._by_pos

    def _variant_offset(self, seq_ref: str, seq_alt: str) -> int:
        """The single differing position between the stored ref/alt (the variant's window offset)."""
        diffs = [i for i, (a, b) in enumerate(zip(seq_ref, seq_alt)) if a != b]
        # fall back to the window centre if lengths differ / no single diff (indels)
        return diffs[0] if len(diffs) == 1 else len(seq_ref) // 2

    def window(self, chrom: str, center_1based: int, length: int):
        """Return (seq_ref, offset) for a KNOWN variant. `length` is advisory — the stored element
        length wins (real Deng elements are 270 bp). Raises KeyError for unknown variants."""
        key = (str(chrom), int(center_1based))
        if key not in self._by_pos:
            raise KeyError(f"{chrom}:{center_1based} not in the calibration table "
                           f"(CalibrationGenome only serves held-out variants; set RVI_GENOME "
                           f"to an hg38 FASTA for arbitrary coordinates)")
        seq_ref, seq_alt = self._by_pos[key]
        return seq_ref, self._variant_offset(seq_ref, seq_alt)
