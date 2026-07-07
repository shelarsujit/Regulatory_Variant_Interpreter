"""Gradio demo for the Regulatory Variant Interpreter (deploy target: a HF Space).

THEORY (plain English — see docs/00_overview_for_non_biologists.md §5):
The UI exists to make the *trust story* legible. A curator types a variant and gets back not
just a number but a calibrated confidence, the likely mechanism (a gained/lost TF motif), and
the full independent-evidence chain — with conflicts shown explicitly, never hidden. A
well-calibrated "uncertain" is a success here, not a failure (working agreement #1).

CONFIGURATION (all optional; the app degrades gracefully):
    RVI_GENOME            path to a local hg38 FASTA -> enables the (chrom,pos,ref,alt) input
    RVI_WEIGHTS_PRIMARY   fine-tuned primary-cortex checkpoint dir (weights/primary)
    RVI_WEIGHTS_ORGANOID  fine-tuned organoid checkpoint dir (weights/organoid)
    RVI_CALIBRATION       calibration_variants.parquet (enables the held-out-MPRA evidence)
Without weights the model still loads (untrained head) and a banner says the Δ is not yet
meaningful — the pipeline, mechanism, evidence, and trust plumbing all still run live.

Two input modes:
    1. Variant coordinates  chrom/pos/ref/alt   (requires RVI_GENOME)
    2. Paste sequences      seq_ref / seq_alt    (no genome needed — great for a quick demo)
"""
from __future__ import annotations

import functools
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "data")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --------------------------------------------------------------------------- lazy singletons
@functools.lru_cache(maxsize=1)
def _genome():
    path = os.environ.get("RVI_GENOME")
    if not path or not os.path.isfile(path):
        return None
    from genome import Genome
    return Genome(path)


@functools.lru_cache(maxsize=2)
def _predictor(context: str):
    """Load a (possibly untrained) ActivityPredictor for a context, cached."""
    from src.predictor import ActivityPredictor
    weights = os.environ.get(f"RVI_WEIGHTS_{context.upper()}")
    weights = weights if (weights and os.path.isdir(weights)) else None
    return ActivityPredictor(context=context, weights=weights).load()


@functools.lru_cache(maxsize=1)
def _calibrator():
    """Fit a Calibrator from the held-out MPRA set by scoring it with the primary model.

    Returns None if the calibration table or model isn't available (trust then falls back to
    the documented uncalibrated heuristic).
    """
    path = os.environ.get("RVI_CALIBRATION")
    if not path or not os.path.isfile(path):
        return None
    try:
        import numpy as np
        import pandas as pd
        from src.trust import Calibrator
        df = pd.read_parquet(path)
        pred = _predictor("primary")
        # score a capped sample for speed on CPU
        df = df.dropna(subset=["seq_ref", "seq_alt", "measured_skew"]).head(1500)
        deltas = [pred.score_variant(r.seq_ref, r.seq_alt) for r in df.itertuples()]
        emv = df["is_emvar"].astype(int).to_numpy() if "is_emvar" in df else None
        return Calibrator().fit(np.array(deltas), df["measured_skew"].to_numpy(), is_emvar=emv)
    except Exception:
        return None


# --------------------------------------------------------------------------- rendering
def _render(result, banner: str = "") -> str:
    t = result.trust
    conf = f"{t.confidence:.0%}" if t else "n/a"
    call = t.call.value if t else "unscored"
    flag = " ⚠️ **CONFLICT**" if (t and t.has_conflict) else ""
    lines = []
    if banner:
        lines.append(banner + "\n")
    lines.append(f"## {result.variant_id} → **{call}**  ·  confidence **{conf}**{flag}\n")
    lines.append(f"- **Predicted direction:** {result.predicted_direction.name} "
                 f"(primary-cortex Δactivity = {result.model_delta_primary:+.3f})")
    if result.model_delta_organoid is not None:
        lines.append(f"- **Organoid model (independent):** Δ = {result.model_delta_organoid:+.3f}")

    if result.mechanisms:
        lines.append("\n### Mechanism")
        for m in result.mechanisms:
            lines.append(f"- {m.describe()}")

    if t:
        if t.agreements:
            lines.append("\n### ✅ Concordant evidence")
            for e in t.agreements:
                lines.append(f"- **{e.source}** — {e.summary}")
        if t.conflicts:
            lines.append("\n### ⚠️ Conflicting evidence (surfaced, not averaged away)")
            for e in t.conflicts:
                lines.append(f"- **{e.source}** — {e.summary}")
        context_ev = [e for e in result.evidence if e.concordant is None]
        if context_ev:
            lines.append("\n### Context")
            for e in context_ev:
                lines.append(f"- **{e.source}** — {e.summary}")
        lines.append(f"\n> {t.rationale}")

    if result.provenance.get("genome_ref_warning"):
        lines.append(f"\n⚠️ {result.provenance['genome_ref_warning']}")
    return "\n".join(lines)


def _untrained_banner(*contexts) -> str:
    untrained = [c for c in contexts if not _predictor(c).is_finetuned]
    if untrained:
        return ("> ⚠️ **Model head is untrained** (" + ", ".join(untrained) + "). The Δactivity "
                "numbers are placeholders until the Phase-2 fine-tune — the mechanism, evidence, "
                "and trust plumbing below are live and real.")
    return ""


# --------------------------------------------------------------------------- callbacks
def interpret_coords(chrom, pos, ref, alt):
    from src import interpret_variant
    g = _genome()
    if g is None:
        return ("**No genome configured.** Set `RVI_GENOME` to a local hg38 FASTA, or use the "
                "*Paste sequences* tab.")
    try:
        result = interpret_variant(
            str(chrom).strip(), int(pos), ref.strip().upper(), alt.strip().upper(),
            genome=g, predictor=_predictor("primary"), organoid_predictor=_predictor("organoid"),
            calibrator=_calibrator())
        return _render(result, _untrained_banner("primary", "organoid"))
    except Exception as e:
        return f"**Error:** {type(e).__name__}: {e}"


def interpret_seqs(seq_ref, seq_alt):
    from src import interpret_variant
    seq_ref, seq_alt = (seq_ref or "").strip().upper(), (seq_alt or "").strip().upper()
    if not seq_ref or not seq_alt or len(seq_ref) != len(seq_alt):
        return "**Provide seq_ref and seq_alt of equal length** (a single-base substitution)."
    diffs = [i for i, (a, b) in enumerate(zip(seq_ref, seq_alt)) if a != b]
    if len(diffs) != 1:
        return f"Expected exactly one differing base; found {len(diffs)}."
    i = diffs[0]
    try:
        result = interpret_variant(
            "chrNA", i, seq_ref[i], seq_alt[i], seq_ref=seq_ref, seq_alt=seq_alt,
            predictor=_predictor("primary"), organoid_predictor=_predictor("organoid"),
            calibrator=_calibrator())
        return _render(result, _untrained_banner("primary", "organoid"))
    except Exception as e:
        return f"**Error:** {type(e).__name__}: {e}"


# --------------------------------------------------------------------------- UI
def build_demo():
    import gradio as gr
    genome_ready = _genome() is not None
    with gr.Blocks(title="Regulatory Variant Interpreter") as demo:
        gr.Markdown("# Regulatory Variant Interpreter\n"
                    "Trust-first interpretation of **non-coding regulatory variants**. "
                    "Every prediction carries a calibrated confidence and an auditable evidence "
                    "chain — and any disagreement between the model and the evidence is shown, "
                    "not hidden.")
        with gr.Tab("Variant coordinates"):
            if not genome_ready:
                gr.Markdown("> ⚠️ `RVI_GENOME` not set — this tab needs a local hg38 FASTA. "
                            "Use **Paste sequences** meanwhile.")
            with gr.Row():
                chrom = gr.Textbox(label="chrom", value="chr2")
                pos = gr.Number(label="pos (1-based)", value=162279995, precision=0)
                ref = gr.Textbox(label="ref", value="A")
                alt = gr.Textbox(label="alt", value="G")
            go1 = gr.Button("Interpret", variant="primary")
            out1 = gr.Markdown()
            go1.click(interpret_coords, [chrom, pos, ref, alt], out1)
        with gr.Tab("Paste sequences"):
            gr.Markdown("Paste the reference and alternate element sequences (equal length, "
                        "one differing base). No genome needed.")
            seq_ref = gr.Textbox(label="seq_ref", lines=3)
            seq_alt = gr.Textbox(label="seq_alt", lines=3)
            go2 = gr.Button("Interpret", variant="primary")
            out2 = gr.Markdown()
            go2.click(interpret_seqs, [seq_ref, seq_alt], out2)
        gr.Markdown("<sub>HyenaDNA saturation-mutagenesis Δactivity · JASPAR motif gain/loss · "
                    "held-out MPRA / GTEx / ClinVar / frozen-model grounding · isotonic-calibrated "
                    "confidence. See docs/ for the full method.</sub>")
    return demo


if __name__ == "__main__":
    build_demo().launch()
