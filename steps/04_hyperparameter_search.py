# Optuna TPE search over (lr, base_ch, dropout, batch_size, optimizer, ...) per arch.

import argparse
import json
import os
import time

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lib.config import (DATA_DIR, WEIGHT_DECAY, PATIENCE,
                        FILTER_HIGH_VIOLATION_PCMS, VIOLATION_RATE_THRESHOLD,
                        THRESHOLD_METRIC, MIN_PRECISION)
from lib.models import build_model, set_seed
from lib.evaluation import evaluate
from lib.dataset import load_split, make_sampler, apply_viol_filter

SEED = 3
MAX_EPOCHS = 100
N_TRIALS = 50


def train_one_config(arch, params, train_ds, val_ds, in_features, device, max_epochs):
    set_seed(SEED)

    model_kwargs = {}
    if arch == "resnet":
        model_kwargs["n_blocks"] = params["n_blocks"]
    elif arch == "tcn":
        model_kwargs["depth"] = params["depth"]
        model_kwargs["kernel_size"] = params["kernel_size"]

    model = build_model(arch, in_features=in_features,
                        base_ch=params["base_ch"], dropout=params["dropout"],
                        **model_kwargs).to(device)
    criterion = nn.BCEWithLogitsLoss()

    bs = params["batch_size"]
    tr = DataLoader(train_ds, batch_size=bs, sampler=make_sampler(train_ds),
                    pin_memory=True)
    vl = DataLoader(val_ds, batch_size=bs, shuffle=False, pin_memory=True)

    opt_name = params["optimizer"]
    if opt_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=params["lr"],
                              momentum=params["momentum"], weight_decay=WEIGHT_DECAY)
    elif opt_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=params["lr"])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=params["lr"],
                                weight_decay=WEIGHT_DECAY)

    best = 0.0
    bad = 0
    for _ in range(max_epochs):
        model.train()
        for batch in tr:
            x = batch[0].to(device)
            y = batch[1].float().to(device)
            opt.zero_grad()
            criterion(model(x), y).backward()
            opt.step()
        val_m, _, _ = evaluate(model, vl, device,
                               threshold_metric=THRESHOLD_METRIC,
                               min_precision=MIN_PRECISION)
        if val_m["auc_pr"] > best:
            best = val_m["auc_pr"]
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", nargs="+", default=["simple", "resnet", "tcn"],
                    choices=["simple", "resnet", "tcn"])
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--n_trials", type=int, default=N_TRIALS)
    ap.add_argument("--results_dir", default="hparam_runs")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = load_split(DATA_DIR, "train", augment=True)
    val_ds = load_split(DATA_DIR, "val")
    for ds in (train_ds, val_ds):
        torch.nan_to_num_(ds.X, nan=0.0, posinf=0.0, neginf=0.0)
    if FILTER_HIGH_VIOLATION_PCMS:
        train_ds = apply_viol_filter(train_ds, VIOLATION_RATE_THRESHOLD, verbose=False)
    in_features = train_ds.X.shape[-1]
    print(f"device={device}  train={len(train_ds):,}  val={len(val_ds):,}  "
          f"F={in_features}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    all_records = {}

    for arch in args.arch:
        print(f"\n{arch}: {args.n_trials} trials")

        def objective(trial):
            params = {
                "lr":         trial.suggest_float("lr", 1e-4, 0.1, log=True),
                "base_ch":    trial.suggest_categorical("base_ch", [16, 32, 64]),
                "dropout":    trial.suggest_float("dropout", 0.1, 0.5),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
                "optimizer":  trial.suggest_categorical("optimizer", ["adamw", "adam", "sgd"]),
            }
            params["momentum"] = (
                trial.suggest_float("momentum", 0.85, 0.99)
                if params["optimizer"] == "sgd" else 0.9
            )
            if arch == "resnet":
                params["n_blocks"] = trial.suggest_int("n_blocks", 2, 4)
            elif arch == "tcn":
                params["depth"] = trial.suggest_categorical("depth", [6, 8, 10])
                params["kernel_size"] = trial.suggest_categorical("kernel_size", [3, 5, 7])

            t0 = time.time()
            score = train_one_config(arch, params, train_ds, val_ds,
                                     in_features, device, args.epochs)
            print(f"  trial {trial.number:3d}  val={score:.4f}  "
                  f"({time.time() - t0:.0f}s)  {params}")
            return score

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
        )
        study.optimize(objective, n_trials=args.n_trials)
        print(f"  best: {study.best_params}  -> {study.best_value:.4f}")

        all_records[arch] = [
            {"arch": arch, **t.params, "val_auc_pr": round(t.value, 4)}
            for t in study.trials if t.value is not None
        ]

    out_path = os.path.join(args.results_dir, "hparam_results.json")
    with open(out_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
