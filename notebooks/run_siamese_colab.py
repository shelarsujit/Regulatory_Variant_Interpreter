# Siamese variant-effect fine-tune on Colab GPU (Enhancement #1, docs/07).
# Warm-started from the fine-tuned CADUCEUS activity backbone (best backbone: variant r=0.19).
# = bidirectional encoder (Caduceus -ph) + direct allelic-skew objective (siamese head).
# Paste into Colab (Runtime -> T4 or A100). Cells marked; run top to bottom.
# Data (calibration_variants.parquet + the fine-tuned Caduceus weights) is gitignored -> upload once.

# ===================== CELL 1 — GPU + clone the branch =====================
# (bash)
"""
!nvidia-smi -L
!git clone --branch docs/hackathon-plan-provenance-prior-art \
    https://github.com/shelarsujit/Regulatory_Variant_Interpreter.git
%cd Regulatory_Variant_Interpreter
!ls train/finetune_siamese.py src/siamese_predictor.py data/make_variant_pairs.py  # sanity: branch has #1
"""

# ===================== CELL 2 — deps (incl. Mamba CUDA kernels for Caduceus) =====================
# (bash) SAME pins as the Caduceus run — the siamese backbone IS Caduceus, so mamba-ssm is required
# and transformers is PINNED (caduceus remote code predates the >=5.x tie-weights refactor).
# After this cell: Runtime -> Restart session, then continue at Cell 3.
"""
!pip install -q "transformers==4.44.2" einops pandas pyarrow scipy openpyxl
!pip install -q causal-conv1d mamba-ssm --no-build-isolation
"""

# ===================== CELL 3 — upload data + the fine-tuned Caduceus weights =====================
# (python) upload:
#   - calibration_variants.parquet         (the held-out variant library -> pairs are carved from it)
#   - DataS2-Variant-library-ratios.xlsx   (enables the strict active-gated emVar in eval)
#   - caduceus_out.zip                      (from the Caduceus run: weights/primary_cad -> warm start)
import os
from google.colab import files
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)
up = files.upload()
for f in up:
    if f.endswith(".parquet"):
        os.replace(f, f"data/processed/{f}")
    elif f.endswith(".xlsx"):
        os.replace(f, "data/raw/DataS2.xlsx")
    elif f.endswith(".zip"):
        os.system(f"unzip -o {f}")   # restores weights/primary_cad/ (+ organoid_cad, results, calibrator)
print("processed:", os.listdir("data/processed"))
print("weights:", [d for d in os.listdir("weights")] if os.path.isdir("weights") else "MISSING")

# ===================== CELL 4 — carve the leakage-safe siamese pairs =====================
# (bash) locus-grouped train/val/eval split OUT of the calibration library. Original file untouched.
"""
!python data/make_variant_pairs.py
!echo '--- pair manifest ---'
!python -c "import json;s=json.load(open('data/processed/variant_pairs_manifest.json'))['stats'];print(s)"
"""

# ===================== CELL 5 — de-risk: load the warm-start backbone =====================
# (python) confirm the Caduceus activity checkpoint is present and loads (d_model=256).
import os
assert os.path.isfile("weights/primary_cad/model.pt"), "upload caduceus_out.zip in Cell 3"
import torch
blob = torch.load("weights/primary_cad/model.pt", map_location="cpu")
print("warm-start ckpt OK | objective:", blob.get("meta", {}).get("backbone"),
      "| keys:", list(blob.keys()))

# ===================== CELL 6 — train the siamese model (Caduceus backbone, warm-started) =====================
# (bash) --backbone caduceus auto-resolves the -ph checkpoint; --init-from overlays the fine-tuned
# activity backbone (only the backbone transfers; the difference head is new). batch 64, AMP.
"""
!python train/finetune_siamese.py \
    --backbone caduceus \
    --init-from weights/primary_cad \
    --data data/processed \
    --out weights/siamese_cad \
    --epochs 10 --batch-size 64 --lr 2e-4 --amp
"""

# ===================== CELL 7 — eval: siamese vs activity baseline, SAME held-out slice =====================
# (bash) scores BOTH models on eval_variants_siamese.parquet. Gate: siamese emVar AUC beats
# the activity baseline (and beats the full-set Caduceus 0.6303). Writes weights/results_siamese.json.
"""
!python eval/eval_siamese.py \
    --siamese-weights weights/siamese_cad \
    --activity-weights weights/primary_cad --activity-context primary \
    --s2 data/raw/DataS2.xlsx \
    --results-out weights/results_siamese.json
!echo '--- headline ---'
!python -c "import json;r=json.load(open('weights/results_siamese.json'));print('n=',r['n'],'strict_emvars=',r['strict_emvars']);[print(x) for x in r['results']]"
"""

# ===================== CELL 8 — download weights + results =====================
# (python)
from google.colab import files
os.system("zip -r siamese_out.zip weights/siamese_cad weights/results_siamese.json "
          "data/processed/variant_pairs_manifest.json")
files.download("siamese_out.zip")
