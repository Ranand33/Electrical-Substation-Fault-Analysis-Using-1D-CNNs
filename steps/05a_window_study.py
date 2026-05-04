# Window-size sensitivity study. Retrain at several W and report metrics.

import argparse
import json
import os
import subprocess
import sys
import time

import torch

from lib.config import DATA_DIR, MODEL_CONFIGS
from lib.models import set_seed
from lib.study_utils import (load_arrays, train_one, eval_on_test,
                             load_best_hparams)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# (window_h, label, tag)
WINDOW_SIZE_SPECS = [
    (24,   "1 day",    "1d"),
    (168,  "7 days",   "7d"),
    (336,  "14 days",  "14d"),
    (720,  "30 days",  "30d"),
    (2160, "90 days",  "90d"),
    (4380, "180 days", "180d"),
    (8760, "365 days", "365d"),
]


def truncate(X, w):
    return X if w >= X.shape[1] else X[:, -w:, :]


def preprocess_w(w, out_dir, intermediate_data):
    w_dir = os.path.join(out_dir, f"data_w{w}")
    if os.path.exists(os.path.join(w_dir, "X_train.npy")):
        return w_dir, True
    cmd = [sys.executable, "01_preprocess.py",
           "--window_size", str(w), "--lead_time", "24", "--out_dir", w_dir]
    if intermediate_data and os.path.exists(intermediate_data):
        cmd += ["--from_intermediate", intermediate_data]
    print(f"  preprocess W={w}h -> {w_dir}")
    rc = subprocess.run(cmd).returncode
    return w_dir, rc == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="simple", choices=["simple", "resnet", "tcn"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="window_size_study")
    p.add_argument("--data_dir", default=DATA_DIR)
    p.add_argument("--window_sizes", nargs="+", type=int,
                   default=[s[0] for s in WINDOW_SIZE_SPECS])
    p.add_argument("--preprocess", action="store_true")
    p.add_argument("--intermediate_data", default=None)
    p.add_argument("--hparam_file", default=None)
    args = p.parse_args()

    cfg = MODEL_CONFIGS[args.arch]
    lr, base_ch, dropout = load_best_hparams(
        args.arch, args.hparam_file, cfg["lr"], cfg["base_ch"], cfg["dropout"]
    )
    print(f"arch={args.arch}  lr={lr:.0e}  base_ch={base_ch}  "
          f"dropout={dropout}  seed={args.seed}")

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr, y_tr, pc_tr = load_arrays(args.data_dir, "train")
    X_v,  y_v,  pc_v  = load_arrays(args.data_dir, "val")
    X_te, y_te, pc_te = load_arrays(args.data_dir, "test")
    T_max = X_tr.shape[1]

    records = []
    for w, label, tag in WINDOW_SIZE_SPECS:
        if w not in args.window_sizes:
            continue

        if w > T_max:
            if not args.preprocess:
                print(f"W={w}h ({label}): skip (max available is {T_max}h)")
                continue
            w_dir, ok = preprocess_w(w, args.out_dir, args.intermediate_data)
            if not ok:
                continue
            X_tr_w, y_tr_w, pc_tr_w = load_arrays(w_dir, "train")
            X_v_w,  y_v_w,  pc_v_w  = load_arrays(w_dir, "val")
            X_te_w, y_te_w, pc_te_w = load_arrays(w_dir, "test")
        else:
            X_tr_w, y_tr_w, pc_tr_w = truncate(X_tr, w), y_tr, pc_tr
            X_v_w,  y_v_w,  pc_v_w  = truncate(X_v,  w), y_v,  pc_v
            X_te_w, y_te_w, pc_te_w = truncate(X_te, w), y_te, pc_te

        print(f"W={w}h ({label})")
        t0 = time.time()
        model, thr = train_one(args.arch, lr, base_ch, dropout, args.seed,
                               X_tr_w, y_tr_w, pc_tr_w,
                               X_v_w,  y_v_w,  pc_v_w,
                               X_tr_w.shape[-1], device)
        m = eval_on_test(model, X_te_w, y_te_w, pc_te_w, thr, device)
        records.append({
            "window_h": w, "label": label, "tag": tag,
            "auc_pr": m["auc_pr"], "auc_roc": m["auc_roc"],
            "recall": m["recall"], "precision": m["precision"], "f1": m["f1"],
            "threshold": m["threshold"],
            "n_positive": m["n_positive"], "n_samples": m["n_samples"],
            "train_time_s": round(time.time() - t0, 1),
        })
        print(f"  AUC-PR={m['auc_pr']:.4f}  AUC-ROC={m['auc_roc']:.4f}  "
              f"R={m['recall']:.4f}  P={m['precision']:.4f}")

    out = {"arch": args.arch, "lr": lr, "base_ch": base_ch, "dropout": dropout,
           "seed": args.seed, "results": records}
    with open(os.path.join(args.out_dir, "ws_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    with open(os.path.join(args.out_dir, "ws_summary_table.txt"), "w") as f:
        f.write(f"window_size_study  arch={args.arch}  seed={args.seed}\n")
        for r in records:
            f.write(f"  {r['label']:<10s} AUC-PR={r['auc_pr']:.4f} "
                    f"AUC-ROC={r['auc_roc']:.4f} R={r['recall']:.4f} "
                    f"P={r['precision']:.4f} F1={r['f1']:.4f} "
                    f"n_pos={r['n_positive']}\n")

    if HAS_MPL and records:
        labels = [r["label"] for r in records]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(labels, [r["auc_pr"] for r in records], "o-")
        ax.set_xlabel("Window Size"); ax.set_ylabel("AUC-PR")
        ax.set_title(f"Window Size Study, {args.arch}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "ws_auc_pr.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(labels, [r["recall"] for r in records], "o-", label="Recall")
        ax.plot(labels, [r["precision"] for r in records], "s--", label="Precision")
        ax.set_xlabel("Window Size"); ax.set_ylabel("Score")
        ax.set_title(f"Window Size Study, {args.arch}")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "ws_recall.png"), dpi=150)
        plt.close(fig)

    print(f"saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
