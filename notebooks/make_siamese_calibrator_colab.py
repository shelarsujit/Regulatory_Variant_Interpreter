# Produce calibrator_siamese.json — NO retraining. ~10 min on a Colab T4.
# The siamese model (siamese_cad) is already trained; this only SCORES it on the held-out slice,
# fits the isotonic calibrator, and dumps per-variant Δ so any future refit is pure-CPU (no GPU again).
# Paste into Colab (Runtime -> T4). Cells marked; run top to bottom.

# ===================== CELL 1 — GPU + clone =====================
# (bash)
"""
!nvidia-smi -L
!git clone --branch docs/hackathon-plan-provenance-prior-art \
    https://github.com/shelarsujit/Regulatory_Variant_Interpreter.git
%cd Regulatory_Variant_Interpreter
"""

# ===================== CELL 2 — deps (Mamba CUDA kernels for Caduceus) =====================
# (bash) transformers PINNED (caduceus remote code predates the 5.x tie-weights refactor).
# After this cell: Runtime -> Restart session, then continue at Cell 3.
"""
!pip install -q "transformers==4.44.2" einops pandas pyarrow scipy openpyxl
!pip install -q causal-conv1d mamba-ssm --no-build-isolation
"""

# ===================== CELL 3 — upload =====================
# (python) upload exactly these:
#   - calibration_variants.parquet          (the held-out variant library)
#   - siamese_out.zip  (or caduceus siamese weights) -> must contain weights/siamese_cad/
#   - DataS2-Variant-library-ratios.xlsx    (OPTIONAL — only enables the strict emVar count)
import os, shutil
from google.colab import files
os.makedirs("data/processed", exist_ok=True); os.makedirs("data/raw", exist_ok=True); os.makedirs("weights", exist_ok=True)
up = files.upload()
for f in up:
    if f.endswith(".parquet"):
        os.replace(f, f"data/processed/{f}")
    elif f.endswith(".xlsx"):
        os.replace(f, "data/raw/DataS2.xlsx")
    elif f.endswith(".zip"):
        os.system(f"unzip -o {f}")
# normalize: the zip may root at siamese_cad/ OR weights/siamese_cad/ — put it at weights/siamese_cad/
if os.path.isfile("siamese_cad/model.pt") and not os.path.isfile("weights/siamese_cad/model.pt"):
    shutil.move("siamese_cad", "weights/siamese_cad")
assert os.path.isfile("weights/siamese_cad/model.pt"), "siamese_cad/model.pt missing — upload the weights zip"
print("OK: siamese_cad present; calibration:", os.listdir("data/processed"))

# ===================== CELL 4 — carve the SAME leakage-safe eval slice =====================
# (bash) deterministic (seed=7) -> identical slice the model was graded on. Original file untouched.
"""
!python data/make_variant_pairs.py
"""

# ===================== CELL 5 — score + fit calibrator + dump (the whole point) =====================
# (python) INLINE — uses only classes already on the cloned branch (SiameseVariantScorer, Calibrator),
# so it works even if the branch's eval_siamese.py predates the --fit-calibrator flag. No --s2 needed:
# the calibrator uses the `is_emvar` already in the eval table.
import numpy as np, pandas as pd
from src.siamese_predictor import SiameseVariantScorer
from src.trust import Calibrator

df = pd.read_parquet("data/processed/eval_variants_siamese.parquet")
df = df.dropna(subset=["seq_ref", "seq_alt", "measured_skew"]).reset_index(drop=True)
print(f"eval slice: {len(df)} variants, {int(df['is_emvar'].sum())} emVars")

scorer = SiameseVariantScorer("weights/siamese_cad").load()
delta = np.array(scorer.score_variants_batch(df["seq_ref"].tolist(), df["seq_alt"].tolist()))
emv = df["is_emvar"].astype(int).to_numpy()

cal = Calibrator().fit(delta, df["measured_skew"].to_numpy(), is_emvar=emv, tau=0.5)
cal.save("weights/calibrator_siamese.json")
print("calibrator diagnostics:", cal.diagnostics)

# dump per-variant Δ so any future refit (different τ, etc.) is pure-CPU — no GPU again
dump = df[["chrom", "pos", "ref", "alt", "measured_skew", "is_emvar"]].copy()
dump["siamese_delta"] = delta
if "rsid" in df.columns:
    dump["rsid"] = df["rsid"].values
dump.to_parquet("weights/siamese_eval_predictions.parquet", index=False)
print("wrote weights/calibrator_siamese.json + weights/siamese_eval_predictions.parquet")

# ===================== CELL 6 — download =====================
# (python) calibrator_siamese.json is the file you need; the .parquet dump lets you REFIT on CPU later.
from google.colab import files
os.system("zip -r calibrator_out.zip weights/calibrator_siamese.json "
          "weights/siamese_eval_predictions.parquet weights/results_siamese.json")
files.download("calibrator_out.zip")
