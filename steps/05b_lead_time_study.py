# Lead-time sensitivity study. Re-label train/val/test for several target lead
# times, retrain, and report metrics on both matched and fixed-24h horizons.

import argparse
import json
import os
import time

import numpy as np
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

STRIDE_H = 168          # 50% overlap of 336-step window
CURRENT_LEAD_H = 24

# (lead_h, label, tag)
LEAD_TIME_SPECS = [
    (24,   "1 day",    "1d"),
    (168,  "7 days",   "7d"),
    (720,  "30 days",  "30d"),
    (2160, "90 days",  "90d"),
    (4380, "180 days", "180d"),
    (8760, "365 days", "365d"),
]


def relabel(y, pcm_ids, target_h, current_h=CURRENT_LEAD_H, stride_h=STRIDE_H):
    # extend each positive run backwards by ceil((target - current)/stride) windows
    if target_h <= current_h:
        return y.copy()
    n_extra = int(np.ceil((target_h - current_h) / stride_h))
    out = y.copy()
    for pcm in np.unique(pcm_ids):
        idx = np.where(pcm_ids == pcm)[0]
        y_p = y[idx]
        starts = []
        in_run = False
        for i, lbl in enumerate(y_p):
            if lbl == 1 and not in_run:
                starts.append(i)
                in_run = True
            elif lbl == 0 and in_run:
                in_run = False
        for s in starts:
            out[idx[max(0, s - n_extra):s]] = 1
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="simple", choices=["simple", "resnet", "tcn"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="lead_time_study")
    p.add_argument("--data_dir", default=DATA_DIR)
    p.add_argument("--lead_times", nargs="+", type=int,
                   default=[s[0] for s in LEAD_TIME_SPECS])
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
    in_features = X_tr.shape[-1]

    records = []
    for lt, label, tag in LEAD_TIME_SPECS:
        if lt not in args.lead_times:
            continue
        print(f"lead_time={lt}h ({label})")

        y_tr_r = relabel(y_tr, pc_tr, lt)
        y_v_r  = relabel(y_v,  pc_v,  lt)
        y_te_r = relabel(y_te, pc_te, lt)
        print(f"  pos rate {y_tr.mean():.2%} -> {y_tr_r.mean():.2%}")

        t0 = time.time()
        model, thr = train_one(args.arch, lr, base_ch, dropout, args.seed,
                               X_tr, y_tr_r, pc_tr,
                               X_v,  y_v_r,  pc_v,
                               in_features, device)
        m_match = eval_on_test(model, X_te, y_te_r, pc_te, thr, device)
        m_fixed = eval_on_test(model, X_te, y_te,    pc_te, thr, device)

        records.append({
            "lead_h": lt, "label": label, "tag": tag,
            "train_pos_rate": round(float(y_tr_r.mean()), 4),
            "matched":   {"auc_pr":    m_match["auc_pr"],
                          "recall":    m_match["recall"],
                          "precision": m_match["precision"]},
            "fixed_24h": {"auc_pr":    m_fixed["auc_pr"],
                          "recall":    m_fixed["recall"],
                          "precision": m_fixed["precision"]},
            "threshold": thr,
            "train_time_s": round(time.time() - t0, 1),
        })
        print(f"  matched   AUC-PR={m_match['auc_pr']:.4f} "
              f"R={m_match['recall']:.4f} P={m_match['precision']:.4f}")
        print(f"  fixed_24h AUC-PR={m_fixed['auc_pr']:.4f} "
              f"R={m_fixed['recall']:.4f} P={m_fixed['precision']:.4f}")

    out = {"arch": args.arch, "lr": lr, "base_ch": base_ch, "dropout": dropout,
           "seed": args.seed, "results": records}
    with open(os.path.join(args.out_dir, "lt_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    with open(os.path.join(args.out_dir, "lt_summary_table.txt"), "w") as f:
        f.write(f"lead_time_study  arch={args.arch}  seed={args.seed}\n")
        for r in records:
            f.write(f"  {r['label']:<10s} pos%={r['train_pos_rate']:.2%}  "
                    f"matched AUC-PR={r['matched']['auc_pr']:.4f} "
                    f"R={r['matched']['recall']:.4f}  "
                    f"fixed24h AUC-PR={r['fixed_24h']['auc_pr']:.4f} "
                    f"R={r['fixed_24h']['recall']:.4f}\n")

    if HAS_MPL and records:
        labels = [r["label"] for r in records]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(labels, [r["matched"]["auc_pr"] for r in records],
                "o-", label="Matched horizon")
        ax.plot(labels, [r["fixed_24h"]["auc_pr"] for r in records],
                "s--", label="Fixed 24h horizon")
        ax.set_xlabel("Lead Time"); ax.set_ylabel("AUC-PR")
        ax.set_title(f"Lead Time Study, {args.arch}")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "lt_auc_pr.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(labels, [r["matched"]["recall"] for r in records],
                "o-", label="Matched horizon")
        ax.plot(labels, [r["fixed_24h"]["recall"] for r in records],
                "s--", label="Fixed 24h horizon")
        ax.set_xlabel("Lead Time"); ax.set_ylabel("Recall")
        ax.set_title(f"Lead Time Study, {args.arch}")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "lt_recall.png"), dpi=150)
        plt.close(fig)

    print(f"saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
