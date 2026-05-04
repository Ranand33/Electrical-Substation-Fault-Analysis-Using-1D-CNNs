"""
Test architectural choices specific to the PCM pre-failure prediction problem,
as opposed to generic training choices (LR, dropout) covered by
04_hyperparameter_search.py.

Usage (run from project root):
    PYTHONPATH=. python steps/06_ablation.py
    PYTHONPATH=. python steps/06_ablation.py --ablation kernel_size
    PYTHONPATH=. python steps/06_ablation.py --ablation dilation
    PYTHONPATH=. python steps/06_ablation.py --ablation depth
    PYTHONPATH=. python steps/06_ablation.py --seed 33 --epochs 60
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lib.config import (DATA_DIR, BATCH_SIZE, WEIGHT_DECAY, GRAD_CLIP, USE_AMP,
                        PATIENCE, FILTER_HIGH_VIOLATION_PCMS, VIOLATION_RATE_THRESHOLD)
from lib.models import set_seed, ResBlock1D, count_parameters, _BN_EPS
from lib.evaluation import evaluate
from lib.dataset import PCMWindowDataset, make_sampler, apply_viol_filter

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


# Ablation-specific architectures
class TemporalBlock(nn.Module):
    # Dilated causal conv block used by TCN_VarDilation (ablation B only)
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation))
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation))
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.pad = pad
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def _chomp(self, x):
        return x[:, :, :-self.pad] if self.pad > 0 else x

    def forward(self, x):
        out = self.drop(self.relu(self._chomp(self.conv1(x))))
        out = self.drop(self.relu(self._chomp(self.conv2(out))))
        return self.relu(out + self.downsample(x))


class SimpleCNN_VarKernel(nn.Module):
    # SimpleCNN with a configurable first-layer kernel size.
    def __init__(self, in_features: int = 9, base_ch: int = 32,
                 dropout: float = 0.3, k1: int = 7):
        super().__init__()
        self.k1 = k1
        self.stage1 = nn.Sequential(
            nn.Conv1d(in_features, base_ch, kernel_size=k1, padding=k1 // 2, bias=False),
            nn.BatchNorm1d(base_ch, eps=_BN_EPS), nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv1d(base_ch, base_ch * 2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(base_ch * 2, eps=_BN_EPS), nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv1d(base_ch * 2, base_ch * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(base_ch * 4, eps=_BN_EPS), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(base_ch * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.stage3(self.stage2(self.stage1(x)))
        x = self.gap(x).squeeze(-1)
        return self.head(self.drop(x)).squeeze(-1)

    def rf(self) -> int:
        return 1 + (self.k1 - 1) + (2 - 1) + (5 - 1) * 2 + (2 - 1) * 2 + (3 - 1) * 4


class ResNet1D_VarDepth(nn.Module):
    # ResNet1D with a configurable number of residual blocks.
    def __init__(self, in_features: int = 9, base_ch: int = 32,
                 dropout: float = 0.3, n_blocks: int = 3):
        super().__init__()
        self.n_blocks = n_blocks
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, base_ch, 7, padding=3, bias=False),
            nn.BatchNorm1d(base_ch, eps=_BN_EPS), nn.ReLU(),
        )
        ch = base_ch
        blocks = []
        for i in range(n_blocks):
            out_ch = ch * 2 if i < n_blocks - 1 else ch
            blocks.append(ResBlock1D(ch, out_ch, stride=2, dropout=dropout))
            ch = out_ch
        self.layers = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.gap(self.layers(self.stem(x))).squeeze(-1)
        return self.head(x).squeeze(-1)

    def rf(self) -> int:
        return 7 + 6 * (2 ** self.n_blocks - 1)


class TCN_VarDilation(nn.Module):
    # TCN with a configurable dilation schedule.
    def __init__(self, in_features=9, base_ch=32, dropout=0.2, dilations=None):
        super().__init__()
        if dilations is None:
            dilations = [2 ** i for i in range(8)]
        self.dilations = dilations
        channels = [base_ch] * len(dilations)
        layers = []
        for i, d in enumerate(dilations):
            in_ch = in_features if i == 0 else channels[i - 1]
            layers.append(TemporalBlock(in_ch, channels[i], 3, d, dropout))
        self.network = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        return self.head(self.drop(self.gap(self.network(x)).squeeze(-1))).squeeze(-1)

    def rf(self) -> int:
        return 1 + 4 * sum(self.dilations)


# Training helpers

def load_data(data_dir):
    splits = {}
    for split in ("train", "val", "test"):
        X = np.load(os.path.join(data_dir, f"X_{split}.npy"))
        y = np.load(os.path.join(data_dir, f"y_{split}.npy"))
        p = np.load(os.path.join(data_dir, f"pcm_{split}.npy"))
        splits[split] = (X, y, p)
    return splits


def make_loader(X, y, pcm_ids, augment, batch_size):
    ds = PCMWindowDataset(X, y, pcm_ids, augment=augment)
    torch.nan_to_num_(ds.X, nan=0.0, posinf=0.0, neginf=0.0)
    if augment:
        if FILTER_HIGH_VIOLATION_PCMS:
            ds = apply_viol_filter(ds, VIOLATION_RATE_THRESHOLD, verbose=False)
        return DataLoader(ds, batch_size=batch_size, sampler=make_sampler(ds), num_workers=0)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def train_and_eval(model, splits, device, lr, epochs, patience):
    X_tr, y_tr, p_tr = splits["train"]
    X_vl, y_vl, p_vl = splits["val"]
    X_te, y_te, p_te = splits["test"]

    tr_loader = make_loader(X_tr, y_tr, p_tr, augment=True, batch_size=BATCH_SIZE)
    vl_loader = make_loader(X_vl, y_vl, p_vl, augment=False, batch_size=BATCH_SIZE)
    te_loader = make_loader(X_te, y_te, p_te, augment=False, batch_size=BATCH_SIZE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, steps_per_epoch=len(tr_loader),
        epochs=epochs, pct_start=0.1, anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")

    best_val, best_state, wait = -1.0, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y, _ in tr_loader:
            x, y = x.to(device), y.float().to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type,
                                    enabled=USE_AMP and device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            scheduler.step()

        val_m, _, _ = evaluate(model, vl_loader, device)
        vl_auc = val_m["auc_pr"]
        if vl_auc > best_val:
            best_val = vl_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_m, _, _ = evaluate(model, te_loader, device)
    return best_val, test_m


# Ablation runners

def ablation_kernel_size(splits, device, seed, epochs, patience, results_dir):
    kernel_sizes = [7, 13, 25, 49, 97]
    results = []

    print("ABLATION A: First-layer kernel size (SimpleCNN)")
    for k in kernel_sizes:
        set_seed(seed)
        model = SimpleCNN_VarKernel(
            in_features=splits["train"][0].shape[2],
            base_ch=32, dropout=0.3, k1=k,
        ).to(device)
        rf_steps = model.rf()

        t0 = time.time()
        val_auc, test_m = train_and_eval(model, splits, device, lr=1e-3,
                                         epochs=epochs, patience=patience)
        elapsed = time.time() - t0

        test_auc = test_m["auc_pr"]
        print(f"  k={k:>3d}  RF={rf_steps:>4d}  {rf_steps / 24:>6.1f}d  "
              f"val={val_auc:.4f}  test={test_auc:.4f}  ({elapsed / 60:.1f}min)")
        results.append({"ablation": "kernel_size", "k": k,
                        "rf_steps": rf_steps,
                        "val_auc_pr": round(val_auc, 4),
                        "test_auc_pr": round(test_auc, 4),
                        "test_metrics": test_m})

    _save_ablation_results(results, "ablation_A_kernel_size", results_dir)
    return results


def ablation_dilation(splits, device, seed, epochs, patience, results_dir):
    schedules = {
        "geometric [1..128]": [2 ** i for i in range(8)],
        "coarse [1,4,16,64]": [1, 4, 16, 64],
        "dense [1,1,2,2,4,4,8,8]": [1, 1, 2, 2, 4, 4, 8, 8],
        "uniform [1..8]": list(range(1, 9)),
    }
    results = []

    print("ABLATION B: TCN dilation schedule")
    for label, dils in schedules.items():
        set_seed(seed)
        model = TCN_VarDilation(
            in_features=splits["train"][0].shape[2],
            base_ch=32, dropout=0.2, dilations=dils,
        ).to(device)
        rf = model.rf()

        t0 = time.time()
        val_auc, test_m = train_and_eval(model, splits, device, lr=1e-3,
                                         epochs=epochs, patience=patience)
        elapsed = time.time() - t0

        test_auc = test_m["auc_pr"]
        cover = "yes" if rf >= 336 else f"no({rf})"
        print(f"  {label:<30}  RF={rf:>4d}  {cover:>8}  "
              f"val={val_auc:.4f}  test={test_auc:.4f}  ({elapsed / 60:.1f}min)")
        results.append({"ablation": "dilation_schedule", "label": label,
                        "dilations": dils, "rf": rf, "covers_window": rf >= 336,
                        "val_auc_pr": round(val_auc, 4),
                        "test_auc_pr": round(test_auc, 4),
                        "test_metrics": test_m})

    _save_ablation_results(results, "ablation_B_dilation", results_dir)
    return results


def ablation_depth(splits, device, seed, epochs, patience, results_dir):
    depths = [2, 3, 4, 5]
    results = []

    print("ABLATION C: ResNet1D depth (number of residual blocks)")
    for d in depths:
        set_seed(seed)
        model = ResNet1D_VarDepth(
            in_features=splits["train"][0].shape[2],
            base_ch=32, dropout=0.3, n_blocks=d,
        ).to(device)
        rf_steps = model.rf()
        n_params = count_parameters(model)

        t0 = time.time()
        val_auc, test_m = train_and_eval(model, splits, device, lr=5e-4,
                                         epochs=epochs, patience=patience)
        elapsed = time.time() - t0

        test_auc = test_m["auc_pr"]
        print(f"  depth={d:>2d}  RF={rf_steps:>4d}  {rf_steps / 24:>6.1f}d  "
              f"{n_params:>8,d}  val={val_auc:.4f}  test={test_auc:.4f}  ({elapsed / 60:.1f}min)")
        results.append({"ablation": "depth", "depth": d,
                        "rf_steps": rf_steps, "n_params": n_params,
                        "val_auc_pr": round(val_auc, 4),
                        "test_auc_pr": round(test_auc, 4),
                        "test_metrics": test_m})

    _save_ablation_results(results, "ablation_C_depth", results_dir)
    return results


def _save_ablation_results(results: list, name: str, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {path}")


def main():
    p = argparse.ArgumentParser(
        description="Domain-motivated architecture ablations for PCM pre-failure detection"
    )
    p.add_argument("--ablation", default="all",
                   choices=["all", "kernel_size", "dilation", "depth"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=60,
                   help="Max epochs per run (60 is sufficient with early stopping).")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--results_dir", default="arch_ablation_results")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  seed={args.seed}  |  max_epochs={args.epochs}")

    print(f"\nLoading data from '{DATA_DIR}'")
    splits = load_data(DATA_DIR)
    for split, (X, y, _) in splits.items():
        n_pos = int(y.sum())
        print(f"  {split}: {X.shape}  pos={n_pos} ({n_pos / len(y):.1%})")

    t_start = time.time()

    if args.ablation in ("all", "kernel_size"):
        ablation_kernel_size(splits, device, args.seed,
                             args.epochs, args.patience, args.results_dir)
    if args.ablation in ("all", "dilation"):
        ablation_dilation(splits, device, args.seed,
                          args.epochs, args.patience, args.results_dir)
    if args.ablation in ("all", "depth"):
        ablation_depth(splits, device, args.seed,
                       args.epochs, args.patience, args.results_dir)

    total = (time.time() - t_start) / 60
    print(f"\nAll ablations complete in {total:.1f} min.  Results: {args.results_dir}/")


if __name__ == "__main__":
    main()
