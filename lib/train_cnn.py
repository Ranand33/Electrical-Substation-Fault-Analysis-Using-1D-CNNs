"""
Usage:
    python train_cnn.py --model simple
    python train_cnn.py --model resnet --lr 5e-4 --seed 0
All experiment combinations are orchestrated by 03_train_cnns.py.
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (DATA_DIR, BATCH_SIZE, WEIGHT_DECAY, GRAD_CLIP, USE_AMP,
                    PATIENCE, MODEL_CONFIGS, FILTER_HIGH_VIOLATION_PCMS,
                    VIOLATION_RATE_THRESHOLD, THRESHOLD_METRIC, MIN_PRECISION)
from models import build_model, FocalLoss, count_parameters, set_seed
from evaluation import (evaluate, fit_platt_calibrator,
                        apply_platt_calibration, metrics_from_probs)
from plots import plot_training_history, plot_test_evaluation, plot_per_pcm
from dataset import PCMWindowDataset, load_split, make_sampler, apply_viol_filter

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# CONFIG
MODEL_NAME = "simple"  # 'simple' | 'resnet' | 'tcn'
SEED = 42
NUM_EPOCHS = 100

FOCAL_GAMMA = 2.0

_cfg = MODEL_CONFIGS.get(MODEL_NAME, MODEL_CONFIGS["simple"])
LR = _cfg["lr"]
BASE_CH = _cfg["base_ch"]
DROPOUT = _cfg["dropout"]

# Platt scaling
CALIBRATE_PROBABILITIES = True

ABLATION_NO_SAMPLER = False
ABLATION_NO_AUGMENT = False
ABLATION_POS_WEIGHT = None
ABLATION_FOCAL_LOSS = False
WINDOW_SIZE = None

CKPT_PATH = f"best_{MODEL_NAME}.pt"
RESULTS_PATH = f"results_{MODEL_NAME}.json"


def train():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")

    print("Loading datasets")
    train_ds = load_split(DATA_DIR, "train",
                          window_size=WINDOW_SIZE,
                          augment=(not ABLATION_NO_AUGMENT))
    val_ds = load_split(DATA_DIR, "val", window_size=WINDOW_SIZE)
    test_ds = load_split(DATA_DIR, "test", window_size=WINDOW_SIZE)
    print(f"Loaded: {len(train_ds):,} train / {len(val_ds):,} val / {len(test_ds):,} test")

    # Replace NaN and inf from RobustScaler on zero-IQR channels.
    for ds in [train_ds, val_ds, test_ds]:
        torch.nan_to_num_(ds.X, nan=0.0, posinf=0.0, neginf=0.0)
    nan_counts = {s: (~torch.isfinite(ds.X)).sum().item()
                  for s, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]}
    if any(v > 0 for v in nan_counts.values()):
        print(f"WARNING - NaN/inf found and zeroed: {nan_counts}")
    else:
        print("Data sanity check passed (no NaN/inf).")

    # Dead channel audit
    _ch_std = train_ds.X.std(dim=(0, 1)).numpy()
    _dead = np.where(_ch_std < 1e-4)[0].tolist()
    if _dead:
        print(f"WARNING - {len(_dead)} near-constant channel(s) (std < 1e-4, indices {_dead}).")
    else:
        print(f"Channel variance check: all {train_ds.X.shape[-1]} channels healthy.")

    # High violation rate PCM filter
    if FILTER_HIGH_VIOLATION_PCMS:
        train_ds = apply_viol_filter(train_ds, VIOLATION_RATE_THRESHOLD)

    # Sampler / DataLoaders
    _disable_sampler = (ABLATION_NO_SAMPLER or
                        ABLATION_POS_WEIGHT is not None or
                        ABLATION_FOCAL_LOSS)
    sampler = None if _disable_sampler else make_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, shuffle=(sampler is None),
                              num_workers=0, pin_memory=True)
    if ABLATION_NO_SAMPLER:
        print("Ablation: WeightedRandomSampler disabled, uniform shuffle active.")

    # Train AUC-PR is computed on an augment-free copy
    train_eval_ds = PCMWindowDataset(train_ds.X, train_ds.y, train_ds.pcm_ids, augment=False)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)

    in_features = train_ds.X.shape[-1]

    if ABLATION_POS_WEIGHT is not None:
        _pw = ABLATION_POS_WEIGHT
        if _pw < 0:
            _n_pos = max(int(train_ds.y.sum()), 1)
            _n_neg = int((train_ds.y == 0).sum())
            _pw = _n_neg / _n_pos
        pos_weight = torch.tensor([float(_pw)], device=device)
        print(f"Ablation: pos_weight={_pw:.2f} (BCE class weighting replaces sampler)")
    else:
        pos_weight = torch.tensor([1.0], device=device)

    model = build_model(MODEL_NAME, in_features=in_features,
                        base_ch=BASE_CH, dropout=DROPOUT).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    if os.name != "nt":
        try:
            model = torch.compile(model)
            print("torch.compile: enabled")
        except RuntimeError:
            pass
    else:
        print("torch.compile: skipped on Windows")

    # Loss, optimiser, scheduler
    if ABLATION_FOCAL_LOSS:
        criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=0.25)
        print(f"Loss: FocalLoss(gamma={FOCAL_GAMMA}, alpha=0.25)")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Loss: BCEWithLogitsLoss(pos_weight={pos_weight.item():.2f})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(train_loader),
        epochs=NUM_EPOCHS, pct_start=0.1, anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and device.type == "cuda"))

    # Training loop
    best_auc_pr = 0.0
    best_threshold = 0.5
    patience_count = 0
    best_state = None
    history = []

    print(f"\n{'Epoch':>5} {'Train Loss':>11} {'Tr AUC-PR':>10} "
          f"{'Val AUC-PR':>11} {'Val AUC-ROC':>12} {'Val F1':>8} {'LR':>10}")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        batch_iter = (
            tqdm(train_loader, desc=f"Ep {epoch:3d}/{NUM_EPOCHS}", leave=False, ncols=90)
            if tqdm else train_loader
        )
        for batch in batch_iter:
            x = batch[0].to(device)
            y = batch[1].float().to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(USE_AMP and device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            if not torch.isfinite(loss):
                print(f"\nWARNING - non-finite loss at epoch {epoch} ({loss.item()}). "
                      f"Skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            if not torch.isfinite(grad_norm):
                print(f"\nWARNING - non-finite gradient norm at epoch {epoch}. "
                      f"Skipping optimizer step.")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        train_metrics, _, _ = evaluate(model, train_eval_loader, device,
                                       threshold_metric=THRESHOLD_METRIC,
                                       min_precision=MIN_PRECISION)
        val_metrics, _, _ = evaluate(model, val_loader, device,
                                     threshold_metric=THRESHOLD_METRIC,
                                     min_precision=MIN_PRECISION)
        current_lr = scheduler.get_last_lr()[0]

        row = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 5),
            "train_auc_pr": train_metrics["auc_pr"],
            "auc_pr": val_metrics["auc_pr"],
            "auc_roc": val_metrics["auc_roc"],
            "f1": val_metrics["f1"],
            "threshold": val_metrics["threshold"],
            "opt_threshold": val_metrics["opt_threshold"],
            "lr": round(current_lr, 7),
        }
        history.append(row)

        gap = train_metrics["auc_pr"] - val_metrics["auc_pr"]
        gap_flag = f"  [gap={gap:+.3f}]" if abs(gap) > 0.15 else ""
        print(f"{epoch:>5}  {avg_loss:>11.5f}  {train_metrics['auc_pr']:>10.4f}  "
              f"{val_metrics['auc_pr']:>11.4f}  {val_metrics['auc_roc']:>12.4f}  "
              f"{val_metrics['f1']:>8.4f}  {current_lr:>10.2e}{gap_flag}")

        # Early stopping on val AUC-PR
        if val_metrics["auc_pr"] > best_auc_pr:
            best_auc_pr = val_metrics["auc_pr"]
            best_threshold = val_metrics["opt_threshold"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
            torch.save(best_state, CKPT_PATH)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\nStopping early after {PATIENCE} epochs with no improvement.")
                break

    # Platt scaling calibration
    platt_cal = None
    if CALIBRATE_PROBABILITIES:
        _, raw_val_probs, val_labels_np = evaluate(
            model, val_loader, device, threshold=best_threshold,
            threshold_metric=THRESHOLD_METRIC, min_precision=MIN_PRECISION,
        )
        platt_cal = fit_platt_calibrator(raw_val_probs, val_labels_np)
        a_cal = platt_cal.coef_[0][0]
        b_cal = platt_cal.intercept_[0]
        print(f"\nPlatt calibrator: logit_cal = {a_cal:.4f} * p_raw + {b_cal:.4f}")
        cal_val_probs = apply_platt_calibration(platt_cal, raw_val_probs)
        print(f"  Val prob mean: raw={raw_val_probs.mean():.4f} cal={cal_val_probs.mean():.4f}")

    # Test evaluation
    print(f"\nBest val AUC-PR: {best_auc_pr:.4f} (threshold={best_threshold:.4f})")
    model.load_state_dict(best_state)
    test_metrics, test_probs, test_labels = evaluate(
        model, test_loader, device,
        threshold=best_threshold, bootstrap_ci=True,
        threshold_metric=THRESHOLD_METRIC, min_precision=MIN_PRECISION,
    )

    if platt_cal is not None:
        cal_test_probs = apply_platt_calibration(platt_cal, test_probs)
        cal_metrics = metrics_from_probs(cal_test_probs, test_labels,
                                         best_threshold, bootstrap_ci=True)
        test_metrics["calibrated"] = cal_metrics

    print(f"TEST RESULTS ({MODEL_NAME})")
    ci_lo = test_metrics.get("auc_pr_ci_lo")
    ci_hi = test_metrics.get("auc_pr_ci_hi")
    ci_str = f"  [95% CI: {ci_lo:.4f}-{ci_hi:.4f}]" if ci_lo is not None else ""
    print(f"  AUC-PR: {test_metrics['auc_pr']}{ci_str}")
    print(f"  AUC-ROC: {test_metrics['auc_roc']}")
    print(f"  F1: {test_metrics['f1']}")
    print(f"  Recall: {test_metrics['recall']}")
    print(f"  Precision: {test_metrics['precision']}")
    print(f"  Threshold: {test_metrics['threshold']} ({test_metrics['threshold_method']})")
    print(f"  Positives: {test_metrics['n_positive']} / {test_metrics['n_samples']}")
    if test_metrics.get("calibrated"):
        cm = test_metrics["calibrated"]
        cm_lo = cm.get("auc_pr_ci_lo")
        cm_hi = cm.get("auc_pr_ci_hi")
        cm_ci_str = f"  [95% CI: {cm_lo:.4f}-{cm_hi:.4f}]" if cm_lo is not None else ""
        print(f"\n  Platt-calibrated probabilities (same threshold):")
        print(f"  Cal AUC-PR: {cm['auc_pr']}{cm_ci_str}")
        print(f"  Cal Recall: {cm['recall']}")
        print(f"  Cal Prec: {cm['precision']}")
    if test_metrics.get("per_pcm_auc_pr"):
        print(f"\n  Per-PCM AUC-PR ({len(test_metrics['per_pcm_auc_pr'])} scoreable PCMs):")
        print(f"  {'PCM':>6}  {'AUC-PR':>7}  {'95% CI':>15}  {'n_pos':>6}  {'n_win':>6}")
        for pcm_id, entry in sorted(test_metrics["per_pcm_auc_pr"].items()):
            lo, hi = entry.get("ci_lo"), entry.get("ci_hi")
            entry_ci_str = f"[{lo:.3f}, {hi:.3f}]" if lo is not None else "n/a"
            print(f"  {pcm_id:>6}  {entry['auc_pr']:>7.4f}  {entry_ci_str:>15}  "
                  f"{entry['n_pos']:>6}  {entry['n_windows']:>6}")
    if test_metrics.get("per_pcm_no_positives"):
        n = len(test_metrics["per_pcm_no_positives"])
        print(f"\n  PCMs with no positives in test split ({n}): "
              f"{test_metrics['per_pcm_no_positives']}")

    # Save results
    results = {
        "model": MODEL_NAME,
        "seed": SEED,
        "best_val_auc_pr": best_auc_pr,
        "best_threshold": best_threshold,
        "threshold_method": test_metrics.get("threshold_method", "unknown"),
        "test": test_metrics,
        "history": history,
        "ablations": {
            "no_sampler": ABLATION_NO_SAMPLER,
            "no_augment": ABLATION_NO_AUGMENT,
            "pos_weight": float(ABLATION_POS_WEIGHT) if ABLATION_POS_WEIGHT is not None else None,
            "focal_loss": ABLATION_FOCAL_LOSS,
            "window_size": WINDOW_SIZE,
        },
        "hyperparams": {
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "base_ch": BASE_CH,
            "patience": PATIENCE,
            "grad_clip": GRAD_CLIP,
            "use_amp": USE_AMP,
            "focal_gamma": FOCAL_GAMMA if ABLATION_FOCAL_LOSS else None,
            "threshold_metric": THRESHOLD_METRIC,
            "min_precision": MIN_PRECISION,
            "calibrate_probs": CALIBRATE_PROBABILITIES,
        },
        "predictions": {
            "test_probs": test_probs.tolist(),
            "test_labels": test_labels.astype(int).tolist(),
            "test_preds": (test_probs >= best_threshold).astype(int).tolist(),
        },
    }
    if platt_cal is not None:
        results["predictions"]["test_cal_probs"] = cal_test_probs.tolist()

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCheckpoint: {CKPT_PATH}")
    print(f"Results: {RESULTS_PATH}")

    # Plots
    print("\nGenerating plots...")
    plot_training_history(history, f"history_{MODEL_NAME}.png")
    plot_test_evaluation(test_probs, test_labels, best_threshold,
                         MODEL_NAME, f"eval_{MODEL_NAME}.png")
    if test_metrics.get("per_pcm_auc_pr"):
        plot_per_pcm(test_metrics["per_pcm_auc_pr"], MODEL_NAME,
                     f"per_pcm_{MODEL_NAME}.png",
                     n_no_pos=len(test_metrics.get("per_pcm_no_positives", [])))

    return model, results


def load_and_predict(checkpoint_path: str, x_numpy: np.ndarray,
                     threshold: float = 0.5, model_name: str = "simple",
                     base_ch: int = 32, dropout: float = 0.3):
    if x_numpy.ndim != 3:
        raise ValueError(f"x_numpy must be 3-D (B, T, F), got shape {x_numpy.shape}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, in_features=x_numpy.shape[2],
                        base_ch=base_ch, dropout=dropout).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    x = torch.from_numpy(x_numpy).float().to(device)
    with torch.no_grad():
        logits = model(x)
    probs = torch.sigmoid(logits).cpu().numpy()
    return probs, (probs >= threshold).astype(int)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Train a 1D CNN for PCM pre-failure detection."
    )
    p.add_argument("--model", default=None, choices=["simple", "resnet", "tcn"])
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--base_ch", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no_sampler", action="store_true")
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--pos_weight", type=float, default=None,
                   help="BCE pos_weight instead of sampler. Pass -1 to auto-compute.")
    p.add_argument("--focal_loss", action="store_true")
    p.add_argument("--window_size", type=int, default=None,
                   help="Keep only last N timesteps (ablation).")
    p.add_argument("--results_dir", default=".")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    if args.model is not None:
        MODEL_NAME = args.model

    _resolved = MODEL_CONFIGS.get(MODEL_NAME, MODEL_CONFIGS["simple"])
    LR = args.lr if args.lr is not None else _resolved["lr"]
    BASE_CH = args.base_ch if args.base_ch is not None else _resolved["base_ch"]
    DROPOUT = args.dropout if args.dropout is not None else _resolved["dropout"]

    if args.seed is not None: SEED = args.seed
    if args.no_sampler: ABLATION_NO_SAMPLER = True
    if args.no_augment: ABLATION_NO_AUGMENT = True
    if args.pos_weight is not None: ABLATION_POS_WEIGHT = args.pos_weight
    if args.focal_loss: ABLATION_FOCAL_LOSS = True
    if args.window_size is not None: WINDOW_SIZE = args.window_size

    _tag = args.tag or (
        f"{MODEL_NAME}"
        f"_lr{LR:.0e}_ch{BASE_CH}_bs{BATCH_SIZE}_wd{WEIGHT_DECAY:.0e}"
        f"{'_nosampler' if ABLATION_NO_SAMPLER else ''}"
        f"{'_noaugment' if ABLATION_NO_AUGMENT else ''}"
        f"{'_posw' if ABLATION_POS_WEIGHT is not None else ''}"
        f"{'_focal' if ABLATION_FOCAL_LOSS else ''}"
        f"{'_win' + str(WINDOW_SIZE) if WINDOW_SIZE else ''}"
        f"_seed{SEED}"
    )
    os.makedirs(args.results_dir, exist_ok=True)
    CKPT_PATH = os.path.join(args.results_dir, f"best_{_tag}.pt")
    RESULTS_PATH = os.path.join(args.results_dir, f"results_{_tag}.json")

    train()
