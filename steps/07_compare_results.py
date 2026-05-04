# Aggregate experiment result JSONs and compute mean +/- std across seeds.

import argparse
import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_records(dirs):
    out = []
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "results_*.json"))):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"  skip (bad json): {path}")
                continue
            tag = os.path.basename(path).replace("results_", "").replace(".json", "")
            out.append({
                "tag": tag,
                "path": path,
                "model": data.get("model", tag.split("_")[0]),
                "ablations": data.get("ablations", {}),
                "hyperparams": data.get("hyperparams", {}),
                "test": data.get("test", {}),
            })
    return out


def condition_of(r):
    abl = r["ablations"]
    hp = r["hyperparams"]
    if abl.get("focal_loss"):
        return "focal_loss"
    if abl.get("pos_weight") is not None:
        return "pos_weight"
    if abl.get("window_size") and abl["window_size"] not in (None, 336):
        return f"window_{abl['window_size']}"
    if hp.get("base_ch") and hp["base_ch"] < 32:
        return f"reduced_capacity_ch{hp['base_ch']}"
    if abl.get("no_sampler") and abl.get("no_augment"):
        return "no_sampler+no_augment"
    if abl.get("no_sampler"):
        return "no_sampler"
    if abl.get("no_augment"):
        return "no_augment"
    return "full"


def aggregate(records):
    def col(k):
        vs = [float(r["test"][k]) for r in records if r["test"].get(k) is not None]
        return np.array(vs) if vs else np.array([float("nan")])
    return {
        "n_seeds":      len(records),
        "auc_pr_mean":  float(np.nanmean(col("auc_pr"))),
        "auc_pr_std":   float(np.nanstd(col("auc_pr"))),
        "auc_pr_ci_lo": float(np.nanmean(col("auc_pr_ci_lo"))),
        "auc_pr_ci_hi": float(np.nanmean(col("auc_pr_ci_hi"))),
        "auc_roc_mean": float(np.nanmean(col("auc_roc"))),
        "auc_roc_std":  float(np.nanstd(col("auc_roc"))),
        "recall_mean":  float(np.nanmean(col("recall"))),
        "prec_mean":    float(np.nanmean(col("precision"))),
        "f1_mean":      float(np.nanmean(col("f1"))),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_dirs", nargs="*", default=["."])
    p.add_argument("--out_dir", default=".")
    args = p.parse_args()

    dirs = list(args.results_dirs)
    if os.path.isdir("experiments") and "experiments" not in dirs:
        dirs.append("experiments")

    records = load_records(dirs)
    if not records:
        print("no results_*.json files found")
        sys.exit(1)
    print(f"found {len(records)} result file(s)")

    groups = {}
    for r in records:
        groups.setdefault((r["model"], condition_of(r)), []).append(r)
    summary = {k: aggregate(v) for k, v in groups.items()}
    rows = sorted(summary.items(), key=lambda kv: -kv[1]["auc_pr_mean"])

    # console table
    print(f"\n{'model':<22} {'cond':<22} {'AUC-PR':>8} {'std':>5} "
          f"{'AUC-ROC':>8} {'Recall':>7} {'Prec':>7} {'F1':>6} {'N':>3}")
    for (model, cond), s in rows:
        print(f"{model:<22} {cond:<22} {s['auc_pr_mean']:>8.4f} "
              f"{s['auc_pr_std']:>5.3f} {s['auc_roc_mean']:>8.4f} "
              f"{s['recall_mean']:>7.4f} {s['prec_mean']:>7.4f} "
              f"{s['f1_mean']:>6.4f} {s['n_seeds']:>3}")

    os.makedirs(args.out_dir, exist_ok=True)

    csv_path = os.path.join(args.out_dir, "comparison_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "condition", "auc_pr_mean", "auc_pr_std",
                    "auc_pr_ci_lo", "auc_pr_ci_hi",
                    "auc_roc_mean", "auc_roc_std",
                    "recall_mean", "prec_mean", "f1_mean", "n_seeds"])
        for (model, cond), s in rows:
            w.writerow([model, cond,
                        round(s["auc_pr_mean"], 4), round(s["auc_pr_std"], 4),
                        round(s["auc_pr_ci_lo"], 4), round(s["auc_pr_ci_hi"], 4),
                        round(s["auc_roc_mean"], 4), round(s["auc_roc_std"], 4),
                        round(s["recall_mean"], 4), round(s["prec_mean"], 4),
                        round(s["f1_mean"], 4), s["n_seeds"]])
    print(f"saved {csv_path}")

    labels = [f"{m}\n({c})" for (m, c), _ in rows]
    means = [s["auc_pr_mean"] for _, s in rows]
    stds = [s["auc_pr_std"] for _, s in rows]
    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 1.2), 5))
    ax.bar(range(len(rows)), means, yerr=stds, capsize=3)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("AUC-PR (mean +/- std)")
    ax.set_title("Model comparison")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(args.out_dir, "comparison_plot.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"saved {plot_path}")


if __name__ == "__main__":
    main()
