"""Pull reference sequence windows from a local hg38 FASTA (pyfaidx).

THEORY (plain English):
The Deng supplement tables give genomic COORDINATES, not DNA letters (docs/01_data_provenance
§4). To turn a variant or an MPRA element into the string our model reads, we look the region
up in a reference genome FASTA. This module is that lookup — used by data/load_deng.py to
reconstruct element/allele sequences, and reusable by src/interpret.py for the genome-window
extraction step it still stubs.

COORDINATES. We standardize on the UCSC/BED convention used by the Deng `insert_*` fields:
0-based, half-open [start, end). pyfaidx slicing `fa[chrom][start:end]` is already 0-based
half-open, so element windows map directly. Variant positions from dbSNP/Ensembl are 1-based,
so `base_at(chrom, pos_1based)` converts for you. Everything is upper-cased; N is preserved.
"""
from __future__ import annotations

import os

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


class Genome:
    """Thin pyfaidx wrapper. Lazy: the FASTA index (.fai) is built on first access."""

    def __init__(self, fasta_path: str):
        if not os.path.isfile(fasta_path):
            raise FileNotFoundError(
                f"hg38 FASTA not found: {fasta_path}\n"
                "Download it, e.g.:\n"
                "  curl -L -o data/raw/genome/hg38.fa.gz "
                "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz\n"
                "  gunzip data/raw/genome/hg38.fa.gz"
            )
        import pyfaidx
        # rebuild_index=False: reuse an existing .fai; as_raw for plain str slicing
        self._fa = pyfaidx.Fasta(fasta_path, as_raw=True, sequence_always_upper=True,
                                 rebuild=False)
        self.path = fasta_path

    def __contains__(self, chrom: str) -> bool:
        return chrom in self._fa

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """Sequence for the 0-based, half-open interval [start, end). Upper-case."""
        if chrom not in self._fa:
            raise KeyError(f"contig {chrom!r} not in FASTA {self.path}")
        return str(self._fa[chrom][start:end])

    def base_at(self, chrom: str, pos_1based: int) -> str:
        """The single reference base at a 1-based genomic position."""
        return self.fetch(chrom, pos_1based - 1, pos_1based)

    def window(self, chrom: str, center_1based: int, length: int) -> tuple[str, int]:
        """A `length`-bp window centered on a 1-based position.

        Returns (sequence, offset) where `offset` is the 0-based index of the centered base
        within the returned sequence (so callers can place the alt allele). Windows near a
        contig start are clamped, which shifts `offset` accordingly.
        """
        half = length // 2
        c0 = center_1based - 1                      # 0-based center
        start = max(0, c0 - half)
        end = start + length
        seq = self.fetch(chrom, start, end)
        return seq, c0 - start
