"""Gradio demo for the Regulatory Variant Interpreter (deploy target: a HF Space).

STATUS: stub (Phase 4). Once the pipeline is wired, this exposes interpret_variant() as a
simple form — the user types a variant (chrom, pos, ref, alt) and gets back the call, the
calibrated confidence, the mechanism, and the full evidence chain, with conflicts shown in
red rather than hidden. The point of the UI is to make the *trust story* legible: a curator
should see not just the number but why to believe it (or not).
"""
from __future__ import annotations


def interpret(chrom: str, pos: int, ref: str, alt: str):
    from src import interpret_variant
    result = interpret_variant(chrom, int(pos), ref, alt)
    return result.summary_line(), "\n".join(result.build_evidence_chain())


def build_demo():
    import gradio as gr
    with gr.Blocks(title="Regulatory Variant Interpreter") as demo:
        gr.Markdown("# Regulatory Variant Interpreter\nTrust-first interpretation of non-coding variants.")
        with gr.Row():
            chrom = gr.Textbox(label="chrom", value="chr2")
            pos = gr.Number(label="pos", value=162279995, precision=0)
            ref = gr.Textbox(label="ref", value="A")
            alt = gr.Textbox(label="alt", value="G")
        go = gr.Button("Interpret", variant="primary")
        verdict = gr.Textbox(label="Verdict")
        chain = gr.Textbox(label="Evidence chain", lines=12)
        go.click(interpret, [chrom, pos, ref, alt], [verdict, chain])
    return demo


if __name__ == "__main__":
    build_demo().launch()
