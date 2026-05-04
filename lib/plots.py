import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_auc_score, roc_curve, confusion_matrix,
    precision_score, recall_score,
)
from sklearn.calibration import calibration_curve


def plot_training_history(history: list, save_path: str) -> None:
    if not history:
        return
    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_auc_pr = [r["auc_pr"] for r in history]
    val_auc_roc = [r["auc_roc"] for r in history]
    lrs = [r["lr"] for r in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(epochs, train_loss, color="tab:blue", lw=1.5)
    ax.set(title="Train Loss (BCE)", xlabel="Epoch", ylabel="Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, val_auc_pr, color="tab:green", lw=1.5, label="Val AUC-PR")
    train_auc_pr_hist = [r.get("train_auc_pr", float("nan")) for r in history]
    if any(not np.isnan(v) for v in train_auc_pr_hist):
        ax.plot(epochs, train_auc_pr_hist, color="tab:blue", lw=1.5, ls="--",
                alpha=0.7, label="Train AUC-PR")
        ax.fill_between(epochs, val_auc_pr, train_auc_pr_hist,
                        alpha=0.12, color="tab:red", label="Train-Val gap")
    best = max(history, key=lambda r: r["auc_pr"])
    ax.axvline(best["epoch"], color="tab:red", ls="--", lw=1, alpha=0.7)
    ax.scatter([best["epoch"]], [best["auc_pr"]], color="tab:red", s=70, zorder=5,
               label=f"Best val {best['auc_pr']:.4f} @ ep {best['epoch']}")
    ax.set(title="AUC-PR (Train vs Val)",
           xlabel="Epoch", ylabel="AUC-PR", ylim=(0, 1))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, val_auc_roc, color="tab:orange", lw=1.5)
    ax.set(title="Val AUC-ROC", xlabel="Epoch", ylabel="AUC-ROC", ylim=(0, 1))
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, lrs, color="tab:purple", lw=1.5)
    ax.set(title="Learning-Rate Schedule (OneCycleLR)", xlabel="Epoch", ylabel="LR")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training history: {save_path}")


def plot_test_evaluation(probs: np.ndarray, labels: np.ndarray,
                         threshold: float, model_name: str, save_path: str) -> None:
    preds = (probs >= threshold).astype(int)
    labels_int = labels.astype(int)
    cm = confusion_matrix(labels_int, preds)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Test Evaluation: {model_name}", fontsize=14, fontweight="bold")

    # Precision-Recall
    ax = axes[0, 0]
    precisions, recalls, _ = precision_recall_curve(labels, probs)
    auc_pr = average_precision_score(labels, probs)
    baseline = float(labels.mean())
    ax.plot(recalls, precisions, color="tab:green", lw=2, label=f"Model AUC = {auc_pr:.4f}")
    ax.fill_between(recalls, precisions, alpha=0.08, color="tab:green")
    ax.axhline(baseline, color="gray", ls="--", lw=1, label=f"Random baseline ({baseline:.3f})")
    op_prec = precision_score(labels, preds, zero_division=0)
    op_rec = recall_score(labels, preds, zero_division=0)
    ax.scatter([op_rec], [op_prec], color="tab:red", s=90, zorder=6,
               label=f"Operating point (thr={threshold:.3f})")
    ax.set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ROC
    ax = axes[0, 1]
    fpr_arr, tpr_arr, _ = roc_curve(labels, probs)
    auc_roc = roc_auc_score(labels, probs)
    ax.plot(fpr_arr, tpr_arr, color="tab:blue", lw=2, label=f"Model AUC = {auc_roc:.4f}")
    ax.fill_between(fpr_arr, tpr_arr, alpha=0.08, color="tab:blue")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="Random")
    cm_vals = cm.ravel()
    if len(cm_vals) == 4:
        tn, fp, fn, tp = cm_vals
        op_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        op_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ax.scatter([op_fpr], [op_tpr], color="tab:red", s=90, zorder=6, label="Operating point")
    ax.set(title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Confusion matrix
    ax = axes[1, 0]
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Pre-failure"])
    ax.set_yticklabels(["Normal", "Pre-failure"])
    ax.set(xlabel="Predicted", ylabel="True",
           title=f"Confusion Matrix (threshold={threshold:.3f})")
    mid = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=18, fontweight="bold",
                    color="white" if cm[i, j] > mid else "black")

    # Calibration (reliability diagram)
    ax = axes[1, 1]
    n_pos = int(labels.sum())
    n_bins = min(10, max(2, n_pos))
    try:
        frac_pos, mean_pred = calibration_curve(
            labels, probs, n_bins=n_bins, strategy="quantile"
        )
        ax.plot(mean_pred, frac_pos, "s-", color="tab:orange", lw=2, ms=7, label="Model")
        ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="Perfect calibration")
        ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
        ax.legend(fontsize=8)
    except ValueError as exc:
        ax.text(0.5, 0.5, f"Calibration unavailable:\n{exc}",
                ha="center", va="center", transform=ax.transAxes, fontsize=9, color="gray")
    ax.set(title="Calibration Curve (Reliability Diagram)",
           xlabel="Mean Predicted Probability", ylabel="Fraction of Positives",
           xlim=(0, 1), ylim=(0, 1))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Test evaluation: {save_path}")


def plot_per_pcm(per_pcm_dict: dict, model_name: str,
                 save_path: str, n_no_pos: int = 0) -> None:
    if not per_pcm_dict:
        return
    pcm_ids = sorted(per_pcm_dict.keys())
    entries = [per_pcm_dict[p] for p in pcm_ids]
    aucs = [e["auc_pr"] for e in entries]
    n_pos = [e["n_pos"] for e in entries]
    n_wins = [e["n_windows"] for e in entries]
    ci_lo = [e.get("ci_lo") for e in entries]
    ci_hi = [e.get("ci_hi") for e in entries]

    colors = ["tab:green" if a >= 0.5 else "tab:red" for a in aucs]
    x_pos = list(range(len(pcm_ids)))
    rotate = len(pcm_ids) > 20

    fig, ax = plt.subplots(figsize=(max(10, len(pcm_ids) * 0.5), 5))
    bars = ax.bar(x_pos, aucs, color=colors, edgecolor="white", lw=0.5, zorder=2)

    err_x, err_y, err_lo, err_hi = [], [], [], []
    for i, (lo, hi, a) in enumerate(zip(ci_lo, ci_hi, aucs)):
        if lo is not None and hi is not None:
            err_x.append(i)
            err_y.append(a)
            err_lo.append(a - lo)
            err_hi.append(hi - a)
    if err_x:
        ax.errorbar(err_x, err_y, yerr=[err_lo, err_hi],
                    fmt="none", color="black", capsize=3, lw=1.2, zorder=3)

    for bar, pos, wins in zip(bars, n_pos, n_wins):
        ax.text(bar.get_x() + bar.get_width() / 2, 0.01,
                f"{pos}/{wins}", ha="center", va="bottom",
                fontsize=6, color="white", fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(int(p)) for p in pcm_ids],
                       rotation=90 if rotate else 0, fontsize=8)
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="AUC-PR = 0.5")
    ax.set(ylim=(0, 1.15), xlabel="PCM ID", ylabel="AUC-PR")
    no_pos_note = (f"; {n_no_pos} PCM(s) no test positives (not shown)" if n_no_pos else "")
    ax.set_title(
        f"Per-PCM AUC-PR ({model_name}) [{len(pcm_ids)} scoreable]{no_pos_note}\n"
        f"Error bars: 95% bootstrap CI. Bar labels: n_pos/n_windows."
    )
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Per-PCM AUC-PR: {save_path}")
