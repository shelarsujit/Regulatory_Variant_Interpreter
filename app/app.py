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


def _resolve_weights(context: str):
    """Weights dir for a context: RVI_WEIGHTS_<CTX> env, else weights/<ctx>_s32, else weights/<ctx>."""
    env = os.environ.get(f"RVI_WEIGHTS_{context.upper()}")
    for cand in (env, os.path.join(_ROOT, "weights", f"{context}_s32"),
                 os.path.join(_ROOT, "weights", context)):
        if cand and os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "model.pt")):
            return cand
    return None


@functools.lru_cache(maxsize=2)
def _predictor(context: str):
    """Load a (possibly untrained) ActivityPredictor for a context, cached.

    The backbone is auto-detected from the weights (small-32k etc.), so no checkpoint flag needed.
    """
    from src.predictor import ActivityPredictor
    return ActivityPredictor(context=context, weights=_resolve_weights(context)).load()


@functools.lru_cache(maxsize=1)
def _calibrator():
    """Load the saved isotonic calibrator (fit offline by eval/calibrate.py). Fast — no refit.

    Path: RVI_CALIBRATION env, else weights/calibrator_primary.json. None if absent (trust then
    falls back to the documented uncalibrated heuristic).
    """
    path = os.environ.get("RVI_CALIBRATION") or os.path.join(_ROOT, "weights", "calibrator_primary.json")
    if not os.path.isfile(path):
        return None
    try:
        from src.trust import Calibrator
        return Calibrator.load(path)
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _gtex():
    """GTEx brain-eQTL lookup (RVI_GTEX env, else data/processed/gtex_eqtl.parquet). None if absent."""
    path = os.environ.get("RVI_GTEX") or os.path.join(_ROOT, "data", "processed", "gtex_eqtl.parquet")
    if not os.path.isfile(path):
        return None
    try:
        import pandas as pd
        return pd.read_parquet(path)
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _seq_lookup():
    """Map (chrom,pos,ref,alt) -> (seq_ref,seq_alt) from the calibration parquet.

    A genome fallback so coordinate lookups work for the 15k tested variants WITHOUT a 3 GB hg38
    FASTA — which is exactly what a hosted HF Space needs. None if the table is absent.
    """
    path = os.path.join(_ROOT, "data", "processed", "calibration_variants.parquet")
    if not os.path.isfile(path):
        return None
    try:
        import pandas as pd
        df = pd.read_parquet(path).dropna(subset=["seq_ref", "seq_alt"])
        return {(str(r.chrom), int(r.pos), str(r.ref).upper(), str(r.alt).upper()):
                (r.seq_ref, r.seq_alt) for r in df.itertuples()}
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
    chrom, ref, alt = str(chrom).strip(), ref.strip().upper(), alt.strip().upper()
    pos = int(pos)
    g = _genome()
    seq_kwargs = {}
    if g is None:
        # no FASTA (e.g. a hosted Space) -> fall back to the bundled per-variant sequence lookup
        lk = _seq_lookup()
        hit = lk.get((chrom, pos, ref, alt)) if lk else None
        if hit is None:
            return ("**No genome configured** and this variant isn't in the bundled set. "
                    "Set `RVI_GENOME` to a local hg38 FASTA, try a listed demo variant, or use "
                    "the *Paste sequences* tab.")
        seq_kwargs = {"seq_ref": hit[0], "seq_alt": hit[1]}
    try:
        result = interpret_variant(
            chrom, pos, ref, alt, genome=g,
            predictor=_predictor("primary"), organoid_predictor=_predictor("organoid"),
            calibrator=_calibrator(), evidence_resources={"gtex_table": _gtex()}, **seq_kwargs)
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
                chrom = gr.Textbox(label="chrom", value="chr11")
                pos = gr.Number(label="pos (1-based)", value=64443180, precision=0)
                ref = gr.Textbox(label="ref", value="G")
                alt = gr.Textbox(label="alt", value="C")
            go1 = gr.Button("Interpret", variant="primary")
            # curated demo variants (real, from the calibration set) — each shows a trust scenario
            gr.Examples(
                examples=[
                    ["chr11", 64443180, "G", "C"],   # concordant: model + MPRA + GTEx + organoid agree
                    ["chr1", 41041729, "G", "A"],    # conflict: model UP vs lab strong-DOWN emVar + organoid
                    ["chr1", 920661, "G", "A"],      # GTEx catches a miss: model none, but known brain eQTL
                ],
                inputs=[chrom, pos, ref, alt], label="Demo variants (agreement · conflict · eQTL-flag)")
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
    # RVI_SHARE=1 -> temporary public gradio.live link (handy for a live demo without deploying).
    # Only bind 0.0.0.0 on a hosted Space (container needs it); locally use gradio's default
    # 127.0.0.1 so the printed URL is browsable (0.0.0.0 is not a navigable address).
    in_space = bool(os.environ.get("SPACE_ID") or os.environ.get("SYSTEM") == "spaces")
    build_demo().launch(share=os.environ.get("RVI_SHARE") == "1",
                        server_name="0.0.0.0" if in_space else None)
