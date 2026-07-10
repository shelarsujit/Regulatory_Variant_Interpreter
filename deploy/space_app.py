"""Hugging Face Space entry point (CPU) for the Regulatory Variant Interpreter.

A CPU Space serves the HyenaDNA activity model + the stacking meta-learner + the full trust layer.
The Caduceus siamese model is GPU-only, so it is NOT bundled here — `app.py` falls back to the
activity model automatically. The coordinates tab works WITHOUT a 3 GB hg38 FASTA via the
calibration-backed genome shim (curated demo variants); the paste-sequences tab works for any
270 bp ref/alt pair. Set these to change behavior: RVI_META (default on), RVI_SIAMESE (default off).
"""
import os
import sys

os.environ.setdefault("RVI_META", "1")       # serve the stacking meta-learner
os.environ.setdefault("RVI_SIAMESE", "0")    # CPU Space: HyenaDNA activity model (no mamba-ssm)

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "data")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.app import build_demo  # noqa: E402

demo = build_demo()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
