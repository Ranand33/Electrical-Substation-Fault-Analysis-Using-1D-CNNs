# Helpers shared by the sensitivity-study scripts (05a, 05b).

import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lib.config import (BATCH_SIZE, WEIGHT_DECAY, GRAD_CLIP, USE_AMP,
                        PATIENCE, FILTER_HIGH_VIOLATION_PCMS,
                        VIOLATION_RATE_THRESHOLD,
                        THRESHOLD_METRIC, MIN_PRECISION)
from lib.models import build_model, set_seed
from lib.evaluation import evaluate
from lib.dataset import PCMWindowDataset, make_sampler, apply_viol_filter

MAX_EPOCHS = 100


def load_arrays(data_dir, split):
    return (
        np.load(os.path.join(data_dir, f"X_{split}.npy")),
        np.load(os.path.join(data_dir, f"y_{split}.npy")),
        np.load(os.path.join(data_dir, f"pcm_{split}.npy")),
    )


def train_one(arch, lr, base_ch, dropout, seed,
              X_tr, y_tr, pc_tr, X_v, y_v, pc_v,
              in_features, device, max_epochs=MAX_EPOCHS):
    set_seed(seed)

    train_ds = PCMWindowDataset(X_tr, y_tr, pc_tr, augment=True)
    val_ds = PCMWindowDataset(X_v, y_v, pc_v, augment=False)
    for ds in (train_ds, val_ds):
        torch.nan_to_num_(ds.X, nan=0.0, posinf=0.0, neginf=0.0)
    if FILTER_HIGH_VIOLATION_PCMS:
        train_ds = apply_viol_filter(train_ds, VIOLATION_RATE_THRESHOLD,
                                     verbose=False)

    tr = DataLoader(train_ds, batch_size=BATCH_SIZE,
                    sampler=make_sampler(train_ds), pin_memory=True)
    vl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                    pin_memory=True)

    model = build_model(arch, in_features=in_features,
                        base_ch=base_ch, dropout=dropout).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, steps_per_epoch=len(tr),
        epochs=max_epochs, pct_start=0.1, anneal_strategy="cos",
    )
    amp = USE_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_auc, best_thr, bad = 0.0, 0.5, 0
    for _ in range(max_epochs):
        model.train()
        for batch in tr:
            x = batch[0].to(device)
            y_b = batch[1].float().to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                loss = crit(model(x), y_b)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            if not torch.isfinite(gn):
                opt.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(opt)
            scaler.update()
            sched.step()

        m, _, _ = evaluate(model, vl, device,
                           threshold_metric=THRESHOLD_METRIC,
                           min_precision=MIN_PRECISION)
        if m["auc_pr"] > best_auc:
            best_auc, best_thr, bad = m["auc_pr"], m["opt_threshold"], 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    return model, best_thr


def eval_on_test(model, X_te, y_te, pc_te, threshold, device):
    ds = PCMWindowDataset(X_te, y_te, pc_te, augment=False)
    torch.nan_to_num_(ds.X, nan=0.0, posinf=0.0, neginf=0.0)
    ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    m, _, _ = evaluate(model, ld, device, threshold=threshold,
                       bootstrap_ci=True,
                       threshold_metric=THRESHOLD_METRIC,
                       min_precision=MIN_PRECISION)
    return m


def load_best_hparams(arch, hparam_file, default_lr, default_base_ch, default_dropout):
    if not hparam_file or not os.path.exists(hparam_file):
        return default_lr, default_base_ch, default_dropout
    with open(hparam_file) as f:
        hpo = json.load(f)
    if arch not in hpo:
        return default_lr, default_base_ch, default_dropout
    best = max(hpo[arch], key=lambda r: r["val_auc_pr"])
    return best["lr"], best["base_ch"], best["dropout"]
