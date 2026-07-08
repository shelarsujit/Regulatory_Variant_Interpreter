# Caduceus (-ph, bidirectional, non-RC-equivariant) fine-tune on Colab GPU.
# Paste into a Colab notebook (Runtime -> T4 or A100). Cells are marked; run top to bottom.
# Data (train/val/calibration parquet + Data S2 xlsx) is gitignored, so you upload it once.

# ===================== CELL 1 — GPU + clone the branch =====================
# (bash)
"""
!nvidia-smi -L
!git clone --branch docs/hackathon-plan-provenance-prior-art \
    https://github.com/shelarsujit/Regulatory_Variant_Interpreter.git
%cd Regulatory_Variant_Interpreter
!grep -n "caduceus-ph" src/predictor.py     # sanity: branch has the -ph default
"""

# ===================== CELL 2 — deps (incl. Mamba CUDA kernels) =====================
# (bash) torch is preinstalled on Colab; mamba-ssm/causal-conv1d build against its CUDA.
# transformers is PINNED: caduceus remote code predates the >=5.x tie-weights refactor
# (`all_tied_weights_keys`). An unpinned install pulls 5.x and load fails. Do NOT unpin.
# After this cell: Runtime -> Restart session (mamba/transformers load into RAM), then Cell 4.
"""
!pip install -q "transformers==4.44.2" einops pandas pyarrow scipy openpyxl
!pip install -q causal-conv1d mamba-ssm --no-build-isolation
"""

# ===================== CELL 3 — upload data =====================
# (python) pick: train.parquet, val.parquet, calibration_variants.parquet, and the
# DataS2-Variant-library-ratios.xlsx. They route to the right places automatically.
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
print("processed:", os.listdir("data/processed"), "| raw:", os.listdir("data/raw"))

# ===================== CELL 4 — smoke-test the Caduceus load (de-risk) =====================
# (python) confirms mamba-ssm + remote code work and the base model returns d_model=256.
import torch
from transformers import AutoModel
CK = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
m = AutoModel.from_pretrained(CK, trust_remote_code=True).cuda().eval()
h = m(torch.randint(7, 12, (2, 270)).cuda()).last_hidden_state
print("caduceus load OK:", tuple(h.shape))   # expect (2, 270, 256); if so, proceed
del m; torch.cuda.empty_cache()

# ===================== CELL 5 — fine-tune both contexts (Caduceus -ph) =====================
# (bash) --backbone caduceus auto-resolves the -ph checkpoint; batch 64 (bigger than HyenaDNA).
"""
!python train/finetune_hyenadna.py --context primary  --backbone caduceus \
    --data data/processed --out weights/primary_cad  --epochs 8 --batch-size 64 --amp
!python train/finetune_hyenadna.py --context organoid --backbone caduceus \
    --data data/processed --out weights/organoid_cad --epochs 8 --batch-size 64 --amp
"""

# ===================== CELL 6 — eval + calibrator (backbone auto-detected) =====================
# (bash) writes weights/results_cad.json; compare emVar AUC to HyenaDNA 0.615 / small-32k 0.628.
"""
!python eval/calibrate.py --weights weights/primary_cad \
    --s2 data/raw/DataS2.xlsx --results-out weights/results_cad.json
!echo '--- headline metrics ---'
!python -c "import json;print(json.load(open('weights/results_cad.json'))['metrics'])"
"""

# ===================== CELL 7 — download weights + results =====================
# (python)
from google.colab import files
os.system("zip -r caduceus_out.zip weights/primary_cad weights/organoid_cad "
          "weights/calibrator_primary.json weights/results_cad.json")
files.download("caduceus_out.zip")
