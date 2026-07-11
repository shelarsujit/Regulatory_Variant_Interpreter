"""Fine-tune a SIAMESE variant-effect model — direct allelic-skew regression (Enhancement #1).

THEORY (docs/07_enhancement_design.md #1):
The activity models regress single-sequence activity; `score_variant` then subtracts alt-ref.
That indirect Δ is the suspected cause of the 0.70 -> 0.15 element-vs-variant accuracy gap. Here
we optimise the difference DIRECTLY: a shared-weight backbone encodes ref and alt, and a small
head reads (h_alt - h_ref) to predict the measured skew (logFC). The loss is on the target we
actually score, so gradients optimise variant effect, not activity.

DATA CONTRACT (produced by data/make_variant_pairs.py — locus-disjoint, leakage-safe):
    train_variants.parquet / val_variants.parquet : seq_ref, seq_alt, measured_skew[, is_emvar]
    eval_variants_siamese.parquet                  : held-out grading slice (used by eval, not here)

ADDITIVE: NEW script + NEW head (src/siamese_predictor.build_siamese_head). Touches neither
finetune_hyenadna.py nor predictor.py; existing activity checkpoints and numbers are unaffected.

WARM START (recommended — mitigates the smaller ~10k-pair train set):
    --init-from weights/primary   # load a fine-tuned activity backbone as the starting point
The head is new (different shape), so only the backbone state is transferred.

OUTPUT — a dir SiameseVariantScorer(weights=OUT).load() consumes:
    OUT/model.pt        {"backbone": sd, "head": sd, "meta": {objective:"siamese", ...}}
    OUT/provenance.json

RUN (CPU smoke test):
    python train/finetune_siamese.py --out weights/siamese_primary --epochs 1 --limit 256
RUN (GPU):
    python train/finetune_siamese.py --out weights/siamese_primary --init-from weights/primary --amp
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.predictor import BACKBONES, build_tokenizer, default_checkpoint_for  # noqa: E402
from src.siamese_predictor import SiameseSkewRegressor  # noqa: E402


# --------------------------------------------------------------------------- data
def _load_pairs(data_dir, name):
    import pandas as pd
    df = pd.read_parquet(os.path.join(data_dir, f"{name}.parquet"))
    for col in ("seq_ref", "seq_alt", "measured_skew"):
        if col not in df.columns:
            raise KeyError(f"{name}.parquet missing {col!r}; has {list(df.columns)}")
    df = df[df["seq_ref"].notna() & df["seq_alt"].notna() & df["measured_skew"].notna()]
    refs = df["seq_ref"].astype(str).tolist()
    alts = df["seq_alt"].astype(str).tolist()
    skew = df["measured_skew"].astype("float32").tolist()
    emv = df["is_emvar"].astype(bool).tolist() if "is_emvar" in df.columns else [False] * len(df)
    return refs, alts, skew, emv


def _iter_pairs(refs, alts, skew, batch_size, tokenizer, device, shuffle, seed=0):
    import numpy as np
    import torch
    n = len(refs)
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for i in range(0, n, batch_size):
        b = idx[i:i + batch_size]
        r_ids, r_mask = tokenizer.encode([refs[j] for j in b])
        a_ids, a_mask = tokenizer.encode([alts[j] for j in b])
        y = torch.tensor([skew[j] for j in b], dtype=torch.float32)
        yield (r_ids.to(device), r_mask.to(device),
               a_ids.to(device), a_mask.to(device), y.to(device))


def _pearson(pred, true):
    import numpy as np
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    if len(pred) < 2 or pred.std() == 0 or true.std() == 0:
        return float("nan")
    return float(np.corrcoef(pred, true)[0, 1])


def _auc(scores, labels):
    import numpy as np
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    pos, neg = int(y.sum()), int((~y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


# --------------------------------------------------------------------------- eval
def _evaluate(module, refs, alts, skew, emv, batch_size, tokenizer, device):
    import numpy as np
    import torch
    module.eval()
    preds = []
    with torch.no_grad():
        for r_ids, r_mask, a_ids, a_mask, _ in _iter_pairs(refs, alts, skew, batch_size,
                                                           tokenizer, device, shuffle=False):
            preds.extend(module(r_ids, r_mask, a_ids, a_mask).float().cpu().reshape(-1).tolist())
    mse = float(np.mean([(p - t) ** 2 for p, t in zip(preds, skew)])) if skew else float("nan")
    # emVar AUC on |Δ| (unsigned magnitude, matching the activity-model eval convention)
    auc = _auc(np.abs(preds), emv)
    return {"val_pearson": round(_pearson(preds, skew), 4),
            "val_mse": round(mse, 5),
            "val_emvar_auc": None if auc != auc else round(auc, 4)}


# --------------------------------------------------------------------------- provenance
def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _save(out_dir, module, args, hidden, backbone_type, best, epoch):
    import torch
    os.makedirs(out_dir, exist_ok=True)
    meta = {"objective": "siamese", "context": args.context, "backbone": backbone_type,
            "hidden": hidden, "pool": args.pool, "seq_len": args.max_len,
            "checkpoint": args.checkpoint, "init_from": args.init_from,
            "best_epoch": epoch, **best}
    torch.save({"backbone": module.backbone.state_dict(),
                "head": module.head.state_dict(), "meta": meta},
               os.path.join(out_dir, "model.pt"))
    prov = {
        "objective": "siamese", "checkpoint": args.checkpoint, "backbone": backbone_type,
        "context": args.context, "target_col": "measured_skew", "init_from": args.init_from,
        "git_commit": _git_commit(),
        "hyperparams": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                        "weight_decay": args.weight_decay, "warmup_frac": args.warmup_frac,
                        "pool": args.pool, "seed": args.seed, "max_len": args.max_len,
                        "amp": args.amp},
        "best_epoch": epoch, "best_metrics": best,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(out_dir, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)


def _maybe_init_backbone(module, init_from):
    """Warm-start: overlay a fine-tuned activity checkpoint's backbone into the siamese backbone.
    Head is new (different shape) and left at init. Silent no-op if the path is missing."""
    if not init_from:
        return "scratch"
    import torch
    mp = os.path.join(init_from, "model.pt")
    if not os.path.isfile(mp):
        print(f"[siamese] --init-from {init_from!r} has no model.pt; starting from base backbone")
        return "base"
    blob = torch.load(mp, map_location="cpu")
    if isinstance(blob, dict) and "backbone" in blob:
        missing, unexpected = module.backbone.load_state_dict(blob["backbone"], strict=False)
        print(f"[siamese] warm-started backbone from {init_from} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
        return init_from
    print(f"[siamese] {mp} has no backbone state; starting from base backbone")
    return "base"


# --------------------------------------------------------------------------- train
def train(args):
    import torch

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    tr = _load_pairs(args.data, args.train_name)
    va = _load_pairs(args.data, args.val_name)
    tr_refs, tr_alts, tr_skew, tr_emv = tr
    va_refs, va_alts, va_skew, va_emv = va
    if args.limit:
        tr_refs, tr_alts, tr_skew, tr_emv = (x[:args.limit] for x in tr)
        va_refs, va_alts, va_skew, va_emv = (x[:args.limit] for x in va)

    builder = SiameseSkewRegressor(args.checkpoint, args.pool)
    module = builder.build().to(device)
    hidden = builder.hidden
    backbone_type = getattr(getattr(module.backbone, "config", None), "model_type", "hyenadna")
    _maybe_init_backbone(module, args.init_from)
    tok = build_tokenizer(backbone_type)
    print(f"[siamese/{args.context}] backbone={backbone_type} ({args.checkpoint}) "
          f"train={len(tr_refs)} val={len(va_refs)} target=measured_skew device={device}")

    opt = torch.optim.AdamW(module.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(max(len(tr_refs), 1) / args.batch_size)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup = int(args.warmup_frac * total_steps)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = args.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = torch.nn.MSELoss()

    best = {"val_pearson": -2.0}
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        module.train()
        running, seen = 0.0, 0
        t0 = time.time()
        for r_ids, r_mask, a_ids, a_mask, y in _iter_pairs(
                tr_refs, tr_alts, tr_skew, args.batch_size, tok, device,
                shuffle=True, seed=args.seed + epoch):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                pred = module(r_ids, r_mask, a_ids, a_mask)
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(module.parameters(), args.max_grad_norm)
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                sched.step()
            running += loss.item() * len(y)
            seen += len(y)
        metrics = _evaluate(module, va_refs, va_alts, va_skew, va_emv,
                            args.batch_size, tok, device)
        print(f"  epoch {epoch:02d} | train_mse {running/max(seen,1):.5f} | "
              f"{metrics} | lr {sched.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s")

        vp = metrics["val_pearson"]
        if vp == vp and vp > best["val_pearson"]:
            best, best_epoch = metrics, epoch
            _save(args.out, module, args, hidden, backbone_type, best, epoch)
            print(f"    ↳ new best (val_pearson={vp}) — saved to {args.out}")

    if best_epoch < 0:
        _save(args.out, module, args, hidden, backbone_type, best, args.epochs)
        print(f"  no val improvement; saved final model to {args.out}")
    print(f"[siamese/{args.context}] done. best_epoch={best_epoch} best={best}")
    return best


# --------------------------------------------------------------------------- CLI
def build_argparser():
    ap = argparse.ArgumentParser(description="Fine-tune a siamese DNA-LM for direct variant-effect "
                                             "(allelic-skew) regression.")
    ap.add_argument("--context", default="primary",
                    help="label only (siamese trains on variant skew, not a context activity column)")
    ap.add_argument("--data", default=os.path.join(_ROOT, "data", "processed"),
                    help="dir holding train_variants.parquet / val_variants.parquet")
    ap.add_argument("--train-name", default="train_variants")
    ap.add_argument("--val-name", default="val_variants")
    ap.add_argument("--out", required=True, help="output checkpoint dir (SiameseVariantScorer loads it)")
    ap.add_argument("--backbone", choices=tuple(BACKBONES), default="hyenadna",
                    help="DNA-LM backbone family; caduceus needs a CUDA GPU (mamba-ssm)")
    ap.add_argument("--checkpoint", default=None, help="specific backbone checkpoint")
    ap.add_argument("--init-from", default=None,
                    help="fine-tuned activity checkpoint dir to warm-start the backbone (recommended)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--pool", choices=("mean", "last"), default="mean")
    ap.add_argument("--max-len", type=int, default=270)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0, help="cap rows per split for a fast smoke test")
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    if args.checkpoint is None:
        args.checkpoint = default_checkpoint_for(args.backbone)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
