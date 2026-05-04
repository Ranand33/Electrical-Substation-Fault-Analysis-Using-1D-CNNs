import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    f1_score, precision_recall_curve,
    precision_score, recall_score,
)
from sklearn.linear_model import LogisticRegression

_MIN_PCM_WINDOWS = 5


def _bootstrap_auc_pr(labels: np.ndarray, probs: np.ndarray,
                      n_boot: int = 300, ci: float = 0.95):
    scores = []
    rng = np.random.default_rng(0)
    n = len(labels)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        lb, pb = labels[idx], probs[idx]
        if lb.sum() == 0 or lb.sum() == n:
            continue
        scores.append(average_precision_score(lb, pb))
    if len(scores) < 20:
        return None, None
    alpha = (1.0 - ci) / 2.0
    return (float(np.percentile(scores, alpha * 100)),
            float(np.percentile(scores, (1 - alpha) * 100)))


def _ece_score(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(probs, bin_edges) - 1, 0, n_bins - 1)
    n = len(labels)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(labels[mask].mean() - probs[mask].mean())
    return float(ece)


def _select_threshold(n_pos, n_neg, labels, probs,
                      threshold_metric="recall_at_precision", min_precision=0.40):
    opt_thresh = 0.5
    threshold_method = "default_0.5"
    if n_pos > 0 and n_neg > 0:
        precisions, recalls, thresholds = precision_recall_curve(labels, probs)
        prec = precisions[:-1]
        rec = recalls[:-1]
        f1s = 2 * prec * rec / (prec + rec + 1e-8)
        if threshold_metric == "recall_at_precision":
            valid = prec >= min_precision
            if valid.any():
                best_idx = int(np.argmax(np.where(valid, rec, 0.0)))
                threshold_method = f"recall_at_prec>={min_precision}"
            else:
                best_idx = int(np.argmax(f1s))
                threshold_method = f"f1_fallback(no_prec>={min_precision})"
        else:
            best_idx = int(np.argmax(f1s))
            threshold_method = "f1"
        opt_thresh = float(thresholds[best_idx])
    return opt_thresh, threshold_method


def _base_metrics(probs, labels, n_pos, n_neg, threshold, bootstrap_ci):
    if n_pos == 0 or n_neg == 0:
        auc_pr, auc_roc = float("nan"), float("nan")
    else:
        auc_pr = float(average_precision_score(labels, probs))
        auc_roc = float(roc_auc_score(labels, probs))
    preds = (probs >= threshold).astype(int)
    out = {
        "auc_pr": round(float(auc_pr), 4),
        "auc_roc": round(float(auc_roc), 4),
        "f1": round(float(f1_score(labels, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, preds, zero_division=0)), 4),
        "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "threshold": round(float(threshold), 4),
        "n_samples": int(len(labels)),
        "n_positive": n_pos,
    }
    if bootstrap_ci and n_pos > 0 and n_neg > 0:
        lo, hi = _bootstrap_auc_pr(labels, probs)
        out["auc_pr_ci_lo"] = round(lo, 4) if lo is not None else None
        out["auc_pr_ci_hi"] = round(hi, 4) if hi is not None else None
    return out


def evaluate(model, loader, device, threshold=None, bootstrap_ci=False,
             threshold_metric="recall_at_precision", min_precision=0.40):
    model.eval()
    all_logits, all_labels, all_pcms = [], [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, y, pcm = batch
            else:
                x, y = batch
                pcm = None
            logits = model(x.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())
            if pcm is not None:
                all_pcms.append(pcm.cpu())

    labels = torch.cat(all_labels).float().numpy()
    probs = torch.sigmoid(torch.cat(all_logits).float()).numpy()

    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())

    opt_thresh, threshold_method = _select_threshold(
        n_pos, n_neg, labels, probs, threshold_metric, min_precision
    )
    if threshold is None:
        threshold = opt_thresh

    metrics = _base_metrics(probs, labels, n_pos, n_neg, threshold, bootstrap_ci)
    metrics["brier_score"] = round(float(((labels - probs) ** 2).mean()), 4)
    metrics["ece"] = round(_ece_score(labels, probs), 4) if (n_pos > 0 and n_neg > 0) else None
    metrics["opt_threshold"] = round(float(opt_thresh), 4)
    metrics["threshold_method"] = threshold_method

    if all_pcms:
        pcms = torch.cat(all_pcms).numpy()
        per_pcm = {}
        no_pos_pcms = []
        for pcm_id in np.unique(pcms):
            mask = pcms == pcm_id
            n_mask = int(mask.sum())
            n_mpos = int(labels[mask].sum())
            if n_mask < _MIN_PCM_WINDOWS:
                continue
            if n_mpos == 0:
                no_pos_pcms.append(int(pcm_id))
                continue
            if n_mpos == n_mask:
                continue
            auc = average_precision_score(labels[mask], probs[mask])
            entry = {
                "auc_pr": round(float(auc), 4),
                "n_pos": n_mpos,
                "n_windows": n_mask,
                "ci_lo": None,
                "ci_hi": None,
            }
            if bootstrap_ci:
                lo, hi = _bootstrap_auc_pr(labels[mask], probs[mask])
                entry["ci_lo"] = round(lo, 4) if lo is not None else None
                entry["ci_hi"] = round(hi, 4) if hi is not None else None
            per_pcm[int(pcm_id)] = entry
        metrics["per_pcm_auc_pr"] = per_pcm
        metrics["per_pcm_no_positives"] = sorted(no_pos_pcms)

    return metrics, probs, labels


def fit_platt_calibrator(val_probs: np.ndarray, val_labels: np.ndarray):
    cal = LogisticRegression(C=1e4, solver="lbfgs", max_iter=1000)
    cal.fit(val_probs.reshape(-1, 1), val_labels.astype(int))
    return cal


def apply_platt_calibration(cal, probs: np.ndarray) -> np.ndarray:
    return cal.predict_proba(probs.reshape(-1, 1))[:, 1].astype(np.float32)


def metrics_from_probs(probs: np.ndarray, labels: np.ndarray,
                       threshold: float, bootstrap_ci: bool = False) -> dict:
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())
    return _base_metrics(probs, labels, n_pos, n_neg, threshold, bootstrap_ci)
