"""Fine-tune HyenaDNA on the cortical MPRA (runs on one A100 / a Colab GPU).

THEORY (plain English — see docs/00_overview_for_non_biologists.md §3.4):
"Fine-tuning" = taking a general pre-trained DNA reader and training it a little more on
our labelled examples so it specializes in predicting regulatory activity. We train TWO
separate single-context models (decision D4):
    context="primary"  -> target = activity_primary   (THE call)
    context="organoid" -> target = activity_organoid  (independent 2nd opinion for trust)

Data contract (produced by data/prepare_data.py):
    train.parquet / val.parquet : columns  sequence, activity_primary, activity_organoid, locus
    The val split is locus-disjoint from train (data/splits.py), so early-stopping is honest.

ARCHITECTURE — identical to src/predictor.ActivityPredictor (decision D12), which is what
consumes the checkpoint at inference time. We deliberately import the SAME tokenizer and the
SAME pooling function so the sequence->activity map is byte-identical between training and
scoring; any drift there would silently invalidate every calibrated confidence downstream.
    backbone = AutoModel.from_pretrained(DEFAULT_CHECKPOINT, trust_remote_code=True)
    head     = Linear(hidden, 1)
    activity = head( pool( backbone(ids).last_hidden_state, attention_mask ) )
    loss     = MSE(activity, target); AdamW; cosine schedule w/ warmup; AMP; select best val Pearson r

OUTPUT — a directory ActivityPredictor(weights=OUT) loads directly:
    OUT/model.pt         # {"backbone": state_dict, "head": state_dict, "meta": {...}}
    OUT/provenance.json  # checkpoint id, context, git commit, hyperparams, best val metrics

COLAB (A100) — minimal:
    !pip install -q torch transformers einops pandas pyarrow scipy
    !python train/finetune_hyenadna.py --context primary  --data data/processed --out weights/primary
    !python train/finetune_hyenadna.py --context organoid --data data/processed --out weights/organoid
Then ActivityPredictor(context="primary", weights="weights/primary").load() is fine-tuned.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

# make `src` importable whether run from repo root or the train/ dir
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.predictor import (BACKBONES, DEFAULT_CHECKPOINT, build_tokenizer,  # noqa: E402
                           default_checkpoint_for, pool_hidden)

_TARGET_COL = {"primary": "activity_primary", "organoid": "activity_organoid"}


# --------------------------------------------------------------------------- model
def _build_regressor(checkpoint, pool):
    """Backbone + Linear(hidden, 1) regression head, as one trainable module."""
    import torch.nn as nn
    from transformers import AutoModel

    class HyenaDNARegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
            self.pool = pool
            hidden = self._hidden_size()
            self.head = nn.Linear(hidden, 1)

        def _hidden_size(self):
            cfg = self.backbone.config
            for a in ("hidden_size", "d_model", "n_embd"):
                v = getattr(cfg, a, None)
                if isinstance(v, int):
                    return v
            raise RuntimeError("could not infer backbone hidden size")

        def forward(self, input_ids, attention_mask):
            out = self.backbone(input_ids)
            last = getattr(out, "last_hidden_state", None)
            if last is None:
                last = out[0]
            return self.head(pool_hidden(last, attention_mask, self.pool)).squeeze(-1)

    return HyenaDNARegressor()


# --------------------------------------------------------------------------- data
def _load_split(data_dir, name, target_col):
    import pandas as pd
    df = pd.read_parquet(os.path.join(data_dir, f"{name}.parquet"))
    if target_col not in df.columns:
        raise KeyError(f"{name}.parquet missing target column {target_col!r}; "
                       f"has {list(df.columns)}")
    # Drop rows with a null target (e.g. organoid activity is absent for elements not tested in
    # the organoid sheet — ~1.9k). A NaN target silently poisons MSE, so we remove them here.
    before = len(df)
    df = df[df[target_col].notna() & df["sequence"].notna()]
    dropped = before - len(df)
    if dropped:
        print(f"[data] {name}: dropped {dropped}/{before} rows with null {target_col}")
    seqs = df["sequence"].astype(str).tolist()
    targets = df[target_col].astype("float32").tolist()
    return seqs, targets


def _iter_batches(seqs, targets, batch_size, tokenizer, device, shuffle, seed=0):
    """Yield (input_ids, attention_mask, target) tensors. Tokenizes per batch (cheap)."""
    import numpy as np
    import torch
    n = len(seqs)
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for i in range(0, n, batch_size):
        b = idx[i:i + batch_size]
        input_ids, attn = tokenizer.encode([seqs[j] for j in b])
        y = torch.tensor([targets[j] for j in b], dtype=torch.float32)
        yield input_ids.to(device), attn.to(device), y.to(device)


def _pearson(pred, true):
    import numpy as np
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    if len(pred) < 2 or pred.std() == 0 or true.std() == 0:
        return float("nan")
    return float(np.corrcoef(pred, true)[0, 1])


# --------------------------------------------------------------------------- eval
def _evaluate(model, seqs, targets, batch_size, tokenizer, device):
    import torch
    model.eval()
    preds = []
    with torch.no_grad():
        for input_ids, attn, _ in _iter_batches(seqs, targets, batch_size, tokenizer,
                                                 device, shuffle=False):
            preds.extend(model(input_ids, attn).float().cpu().tolist())
    mse = float(sum((p - t) ** 2 for p, t in zip(preds, targets)) / max(len(targets), 1))
    return {"val_pearson": round(_pearson(preds, targets), 4), "val_mse": round(mse, 5)}


# --------------------------------------------------------------------------- provenance
def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _save_checkpoint(out_dir, model, args, hidden, best_metrics, epoch):
    import torch
    os.makedirs(out_dir, exist_ok=True)
    backbone_type = getattr(getattr(model.backbone, "config", None), "model_type", "hyenadna")
    tok = build_tokenizer(backbone_type)
    meta = {"context": args.context, "backbone": backbone_type, "hidden": hidden,
            "pool": args.pool, "seq_len": args.max_len, "best_epoch": epoch,
            "checkpoint": args.checkpoint, **best_metrics}
    # One torch checkpoint holding both fine-tuned state_dicts. We do NOT use
    # backbone.save_pretrained: HyenaDNA reuses one `freq` buffer across filter layers
    # (shared tensors), which transformers' save_pretrained rejects (decision D12).
    # torch.save preserves shared storage; ActivityPredictor rebuilds the base backbone
    # from `checkpoint` then load_state_dict's these.
    torch.save({"backbone": model.backbone.state_dict(),
                "head": model.head.state_dict(), "meta": meta},
               os.path.join(out_dir, "model.pt"))
    prov = {
        "checkpoint": args.checkpoint, "backbone": backbone_type, "context": args.context,
        "target_col": _TARGET_COL[args.context], "git_commit": _git_commit(),
        "tokenizer_vocab": tok.vocab,
        "hyperparams": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                        "weight_decay": args.weight_decay, "warmup_frac": args.warmup_frac,
                        "pool": args.pool, "seed": args.seed, "max_len": args.max_len,
                        "amp": args.amp},
        "best_epoch": epoch, "best_metrics": best_metrics,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(out_dir, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)


# --------------------------------------------------------------------------- train loop
def train(args):
    import torch

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    target_col = _TARGET_COL[args.context]

    tr_seqs, tr_y = _load_split(args.data, "train", target_col)
    va_seqs, va_y = _load_split(args.data, "val", target_col)
    if args.limit:                                    # fast smoke-test path
        tr_seqs, tr_y = tr_seqs[:args.limit], tr_y[:args.limit]
        va_seqs, va_y = va_seqs[:args.limit], va_y[:args.limit]

    model = _build_regressor(args.checkpoint, args.pool).to(device)
    hidden = model.head.in_features
    # tokenizer follows the backbone (HyenaDNA -> identical to before; Caduceus -> shared vocab)
    backbone_type = getattr(getattr(model.backbone, "config", None), "model_type", "hyenadna")
    tok = build_tokenizer(backbone_type)
    print(f"[{args.context}] backbone={backbone_type} ({args.checkpoint}) "
          f"train={len(tr_seqs)} val={len(va_seqs)} target={target_col} device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(tr_seqs) / args.batch_size)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup = int(args.warmup_frac * total_steps)

    def lr_lambda(step):                              # linear warmup -> cosine decay
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
        model.train()
        running, seen = 0.0, 0
        t0 = time.time()
        for input_ids, attn, y in _iter_batches(tr_seqs, tr_y, args.batch_size, tok, device,
                                                shuffle=True, seed=args.seed + epoch):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                pred = model(input_ids, attn)
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item() * len(y)
            seen += len(y)
        metrics = _evaluate(model, va_seqs, va_y, args.batch_size, tok, device)
        print(f"  epoch {epoch:02d} | train_mse {running/max(seen,1):.5f} | "
              f"{metrics} | lr {sched.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s")

        # select on best val Pearson r (NaN guarded)
        vp = metrics["val_pearson"]
        if vp == vp and vp > best["val_pearson"]:
            best, best_epoch = metrics, epoch
            _save_checkpoint(args.out, model, args, hidden, best, epoch)
            print(f"    ↳ new best (val_pearson={vp}) — saved to {args.out}")

    if best_epoch < 0:                                # never improved (e.g. constant target)
        _save_checkpoint(args.out, model, args, hidden, best, args.epochs)
        print(f"  no val improvement; saved final model to {args.out}")
    print(f"[{args.context}] done. best_epoch={best_epoch} best={best}")
    return best


# --------------------------------------------------------------------------- CLI
def build_argparser():
    ap = argparse.ArgumentParser(description="Fine-tune a DNA LM (HyenaDNA | Caduceus) for MPRA "
                                             "regulatory activity.")
    ap.add_argument("--context", choices=("primary", "organoid"), required=True,
                    help="primary = the call; organoid = independent 2nd model (decision D4)")
    ap.add_argument("--data", default=os.path.join(_ROOT, "data", "processed"),
                    help="dir holding train.parquet / val.parquet")
    ap.add_argument("--out", required=True, help="output checkpoint dir (loaded by ActivityPredictor)")
    ap.add_argument("--backbone", choices=tuple(BACKBONES), default="hyenadna",
                    help="DNA-LM backbone family; picks a default --checkpoint if none given. "
                         "caduceus needs a CUDA GPU (mamba-ssm) — see decision D16")
    ap.add_argument("--checkpoint", default=None,
                    help="specific backbone checkpoint (default: the --backbone family default)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--pool", choices=("mean", "last"), default="mean",
                    help="MUST match ActivityPredictor(pool=...) at inference")
    ap.add_argument("--max-len", type=int, default=270, help="element length (bp); metadata/provenance "
                                                             "(real Deng elements are 270 bp)")
    ap.add_argument("--amp", action="store_true", help="mixed precision (recommended on A100)")
    ap.add_argument("--device", default=None, help="cuda|cpu (auto if unset)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-seeds", type=int, default=1,
                    help="train N seeded ensemble members into <out>/seed{0..N-1} (decision D17)")
    ap.add_argument("--limit", type=int, default=0, help="cap rows per split for a fast smoke test")
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    if args.checkpoint is None:                        # resolve from the backbone family default
        args.checkpoint = default_checkpoint_for(args.backbone)
    if args.n_seeds > 1:                               # ensemble: one member per seed (D17)
        root, base_seed = args.out, args.seed
        for k in range(args.n_seeds):
            args.out = os.path.join(root, f"seed{k}")
            args.seed = base_seed + k
            print(f"=== ensemble member {k + 1}/{args.n_seeds} (seed={args.seed}) -> {args.out} ===")
            train(args)
    else:
        train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
