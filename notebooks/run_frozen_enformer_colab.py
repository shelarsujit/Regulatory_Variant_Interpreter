# Precompute the frozen-Enformer Δ feature for the meta-learner (docs/08), then refit the meta.
# GPU-only. Paste into Colab (Runtime -> A100 or T4). Cells marked; run top to bottom.
#
# COST: Enformer is ~1 s/variant on GPU; ~15k unique variants ≈ a few hours. precompute_frozen.py
# checkpoints every 100 variants and RESUMES, so a disconnect is not fatal — re-run Cell 5 to continue.
# For a quick signal first, use --limit 500 in Cell 5 (a partial cache still lets the meta fit).

# ===================== CELL 1 — GPU + clone =====================
# (bash)
"""
!nvidia-smi -L
!git clone --branch docs/hackathon-plan-provenance-prior-art \
    https://github.com/shelarsujit/Regulatory_Variant_Interpreter.git
%cd Regulatory_Variant_Interpreter
!ls src/foundation.py eval/precompute_frozen.py    # sanity: branch has the frozen wiring
"""

# ===================== CELL 2 — deps =====================
# (bash) enformer-pytorch pulls a compatible transformers/hub on Colab; pyfaidx for the FASTA.
"""
!pip install -q enformer-pytorch pyfaidx pandas pyarrow scipy
"""

# ===================== CELL 3 — hg38 FASTA (Enformer needs 196 kb genomic context) =====================
# (bash) ~1 GB gz -> ~3 GB unzipped; pyfaidx builds the .fai on first use (a minute).
# ROBUST download: aria2 (multi-connection + resume) is far more reliable on Colab than a plain
# wget, which silently truncates on a network hiccup ("unexpected end of file"). `gzip -t` verifies
# the archive BEFORE gunzip so a partial download is caught, not fed to gunzip. Re-run this cell to
# resume if it drops — aria2 continues the partial file.
"""
!apt-get -qq install -y aria2 >/dev/null
!mkdir -p data/raw/genome
!aria2c -c -x8 -s8 --retry-wait=5 --max-tries=10 --summary-interval=0 \
    --dir=data/raw/genome -o hg38.fa.gz \
    https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
# integrity gate: only gunzip if the archive is complete/valid
!gzip -t data/raw/genome/hg38.fa.gz && gunzip -f data/raw/genome/hg38.fa.gz \
    && ls -la data/raw/genome/hg38.fa \
    || echo "download incomplete — re-run this cell (aria2 will resume)"
"""

# Fallback if aria2 is unavailable — wget with resume + retries + the same integrity gate:
"""
!mkdir -p data/raw/genome
!wget --continue --tries=10 --read-timeout=30 --timeout=30 \
    -O data/raw/genome/hg38.fa.gz \
    https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
!gzip -t data/raw/genome/hg38.fa.gz && gunzip -f data/raw/genome/hg38.fa.gz \
    && ls -la data/raw/genome/hg38.fa || echo "incomplete — re-run to resume"
"""

# ===================== CELL 4 — upload the variant tables =====================
# (python) upload: train_variants.parquet, val_variants.parquet, eval_variants_siamese.parquet,
# calibration_variants.parquet  (produced locally by data/make_variant_pairs.py).
import os
from google.colab import files
os.makedirs("data/processed", exist_ok=True)
up = files.upload()
for f in up:
    if f.endswith(".parquet"):
        os.replace(f, f"data/processed/{f}")
print("processed:", os.listdir("data/processed"))

# ===================== CELL 5 — precompute the frozen Δ cache (the expensive, one-time step) =====================
# (bash) drop --limit for the full run. Resumable: re-run to continue after a disconnect.
# Track selection: the default is a brain CAGE track; refine via targets_human.txt if the fitted
# weight is weak (see docs/08 §3). Pass --tracks 4980,4981,... to average a curated brain set.
"""
!python eval/precompute_frozen.py \
    --genome data/raw/genome/hg38.fa \
    --out data/processed/frozen_delta_cache.parquet \
    --limit 500
!python -c "import pandas as pd;d=pd.read_parquet('data/processed/frozen_delta_cache.parquet');print('cached',len(d),'| Δ range',round(d.frozen_delta.min(),3),round(d.frozen_delta.max(),3))"
"""

# ===================== CELL 6 — download the frozen cache (fit LOCALLY, not here) =====================
# (python) DO NOT fit the meta on Colab: the trained weights (weights/primary, organoid,
# siamese_cad) live on your machine, not this fresh clone — fitting here loads an UNTRAINED model
# and produces noise. Colab's only job is the expensive Enformer precompute. Download the cache and
# fit locally where the weights are.
from google.colab import files
files.download("data/processed/frozen_delta_cache.parquet")

# ===================== THEN, LOCALLY (CPU — where your weights are) =====================
# 1. Put the downloaded file at:  data/processed/frozen_delta_cache.parquet
# 2. Fit + grade the meta with your best DNA-LM signal:
#      python eval/fit_meta.py --activity-weights weights/primary \
#          --organoid-weights weights/organoid \
#          --frozen-cache data/processed/frozen_delta_cache.parquet
#    (or --siamese-weights weights/siamese_cad to pair with the 0.28 siamese Δ)
# 3. Read weights/results_meta.json: nonzero abs_frozen_delta / concordance_dna_frozen = the win.
#
# NOTE on --limit: a partial cache must cover the EVAL slice to grade. precompute_frozen.py now
# scores eval_variants first, so `--limit 2273` already covers the whole grade slice; a larger
# limit adds fit-set coverage. For the full, robust number, run the precompute with no --limit.
