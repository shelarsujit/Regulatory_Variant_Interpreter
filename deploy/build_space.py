#!/usr/bin/env python
"""Stage + deploy the CPU Hugging Face Space for the Regulatory Variant Interpreter.

Assembles a self-contained Space tree (code + the small HyenaDNA weights + calibration data + the
Space entry/README/requirements) and, with --push, creates/updates the Space and uploads it.

The Space serves the activity model + meta-learner + trust layer on CPU (~8 MB of weights); the
Caduceus siamese model is GPU-only and intentionally NOT bundled (app.py falls back automatically).

RUN
    huggingface-cli login                       # once, so the token is cached
    python deploy/build_space.py                # dry run: just stage into deploy/_stage/
    python deploy/build_space.py --push         # create/update the Space and upload
    python deploy/build_space.py --push --repo-id you/your-space-name
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_STAGE = os.path.join(_HERE, "_stage")

# (source relative to repo root, dest relative to Space root). Directories copied whole.
FILES = [
    ("deploy/space_app.py", "app.py"),
    ("deploy/space_README.md", "README.md"),
    ("deploy/space_requirements.txt", "requirements.txt"),
    # code
    ("src", "src"),
    ("app/app.py", "app/app.py"),
    ("data/genome.py", "data/genome.py"),
    ("data/calib_genome.py", "data/calib_genome.py"),
    # trained artifacts (small HyenaDNA activity models + calibrator + meta)
    ("weights/primary", "weights/primary"),
    ("weights/organoid", "weights/organoid"),
    ("weights/calibrator_primary.json", "weights/calibrator_primary.json"),
    ("weights/meta_primary.json", "weights/meta_primary.json"),
    # calibration data (held-out MPRA evidence + the offline coords shim) + GTEx
    ("data/processed/calibration_variants.parquet", "data/processed/calibration_variants.parquet"),
    ("data/processed/gtex_eqtl.parquet", "data/processed/gtex_eqtl.parquet"),
]


def stage():
    if os.path.isdir(_STAGE):
        shutil.rmtree(_STAGE)
    os.makedirs(_STAGE)
    missing = []
    for src_rel, dst_rel in FILES:
        src = os.path.join(_ROOT, src_rel)
        dst = os.path.join(_STAGE, dst_rel)
        if not os.path.exists(src):
            missing.append(src_rel)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, dst)
    # app/ must be an importable package for `from app.app import build_demo`
    open(os.path.join(_STAGE, "app", "__init__.py"), "a").close()
    if missing:
        raise SystemExit("missing required files (train/eval first):\n  " + "\n  ".join(missing))
    total = sum(os.path.getsize(f) for f in glob.glob(os.path.join(_STAGE, "**", "*"), recursive=True)
                if os.path.isfile(f))
    print(f"[space] staged -> {_STAGE}  ({total/1e6:.1f} MB)")
    return _STAGE


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--push", action="store_true", help="create/update the Space and upload the staged tree")
    ap.add_argument("--repo-id", default="SujitShelar/regulatory-variant-interpreter")
    ap.add_argument("--private", action="store_true", help="create the Space private")
    args = ap.parse_args(argv)

    path = stage()
    if not args.push:
        print("[space] dry run — inspect deploy/_stage/, then re-run with --push")
        return 0

    from huggingface_hub import HfApi, create_repo
    create_repo(args.repo_id, repo_type="space", space_sdk="gradio",
                private=args.private, exist_ok=True)
    HfApi().upload_folder(folder_path=path, repo_id=args.repo_id, repo_type="space",
                          commit_message="Deploy Regulatory Variant Interpreter (CPU)")
    print(f"[space] pushed -> https://huggingface.co/spaces/{args.repo_id}")
    print("[space] first build takes a few minutes (installs torch/transformers); watch the Space logs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
