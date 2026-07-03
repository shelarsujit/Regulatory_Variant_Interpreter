"""Fine-tune HyenaDNA on the cortical MPRA (runs on one A100). STATUS: stub (Phase 2).

THEORY (plain English — see docs/00_overview_for_non_biologists.md §3.4):
"Fine-tuning" = taking a general pre-trained DNA reader and training it a little more on
our labelled examples so it specializes in predicting regulatory activity. We train TWO
separate single-context models (decision D4):
    context="primary"  -> target = activity_primary   (THE call)
    context="organoid" -> target = activity_organoid  (independent 2nd opinion for trust)

Data contract (produced by data/prepare_data.py):
    train.parquet / val.parquet : columns  sequence, activity_primary, activity_organoid, locus
    The val split is locus-disjoint from train (data/splits.py), so early-stopping is honest.

Sketch:
    backbone  = AutoModel.from_pretrained("LongSafari/hyenadna-tiny-1k-seqlen",
                                          trust_remote_code=True)
    head      = Linear(hidden, 1)               # regression: sequence -> activity
    loss      = MSE(pred, activity_<context>)
    optimizer = AdamW; cosine schedule; mixed precision
    select    = best val Pearson r; save checkpoint + a provenance sidecar (hash, data manifest)

CLI (planned):
    python train/finetune_hyenadna.py --context primary  --data data/processed --out weights/primary
    python train/finetune_hyenadna.py --context organoid --data data/processed --out weights/organoid
"""
from __future__ import annotations


def main(argv=None):
    raise NotImplementedError("Phase 2: implement the HyenaDNA fine-tune per the sketch above")


if __name__ == "__main__":
    raise SystemExit(main())
