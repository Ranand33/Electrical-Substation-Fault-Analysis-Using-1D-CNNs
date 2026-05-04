# Baseline models (LR / RF / XGB)

import json
import os

import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    f1_score, recall_score, precision_score,
)
from xgboost import XGBClassifier

from lib.evaluation import _select_threshold

DATA_DIR = os.environ.get("PCM_DATA_DIR", "preprocessed_pcm_data")
RESULTS_PATH = "baseline_results.json"
N_TRIALS = 30
MIN_PRECISION = 0.40


def window_to_features(X):
    # X: (N, T, F) -> (N, F*6). Six summary stats per channel.
    T = X.shape[1]
    q = max(1, T // 4)

    mean = np.nanmean(X, axis=1)
    std = np.nanstd(X, axis=1)
    mn = np.nanmin(X, axis=1)
    mx = np.nanmax(X, axis=1)
    last = X[:, -1, :]

    # cheap slope proxy: late-quarter mean minus early-quarter mean
    early = np.nanmean(X[:, :q, :], axis=1)
    late = np.nanmean(X[:, -q:, :], axis=1)
    slope = (late - early) / max(1, T - q)

    out = np.concatenate([mean, std, mn, mx, last, slope], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def evaluate_model(name, probs, labels):
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())

    if n_pos == 0 or n_neg == 0:
        auc_pr = auc_roc = float("nan")
    else:
        auc_pr = float(average_precision_score(labels, probs))
        auc_roc = float(roc_auc_score(labels, probs))

    thr, thr_method = _select_threshold(
        n_pos, n_neg, labels, probs,
        threshold_metric="recall_at_precision",
        min_precision=MIN_PRECISION,
    )
    preds = (probs >= thr).astype(int)
    return {
        "model": name,
        "auc_pr": round(auc_pr, 4),
        "auc_roc": round(auc_roc, 4),
        "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, preds, zero_division=0)), 4),
        "threshold": round(thr, 4),
        "threshold_method": thr_method,
        "n_positive": n_pos,
        "n_samples": int(len(labels)),
    }


def main():
    np.random.seed(42)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("loading windows")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    y_fit = np.concatenate([y_train, y_val])
    print(f"train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}  "
          f"test pos={int(y_test.sum())}/{len(y_test)}")

    Xf_train = window_to_features(X_train)
    Xf_val = window_to_features(X_val)
    Xf_test = window_to_features(X_test)
    Xf_fit = np.concatenate([Xf_train, Xf_val], axis=0)
    print(f"feature matrix: {Xf_fit.shape}")

    n_pos_fit = int(y_fit.sum())
    n_neg_fit = int((1 - y_fit).sum())
    spw = n_neg_fit / max(n_pos_fit, 1)

    n_channels = X_train.shape[2]
    stat_names = ["mean", "std", "min", "max", "last", "slope"]
    feature_names = [f"{s}_ch{c}" for s in stat_names for c in range(n_channels)]

    # Logistic Regression
    print(f"\ntuning LR ({N_TRIALS} trials)")
    def lr_obj(trial):
        C = trial.suggest_float("C", 1e-3, 1e2, log=True)
        m = LogisticRegression(C=C, solver="liblinear", class_weight="balanced",
                               random_state=42, max_iter=1000)
        m.fit(Xf_train, y_train)
        return average_precision_score(y_val, m.predict_proba(Xf_val)[:, 1])
    study_lr = optuna.create_study(direction="maximize",
                                   sampler=optuna.samplers.TPESampler(seed=42))
    study_lr.optimize(lr_obj, n_trials=N_TRIALS)
    print(f"  best C={study_lr.best_params['C']:.4f}  val AUC-PR={study_lr.best_value:.4f}")
    lr = LogisticRegression(C=study_lr.best_params["C"], solver="liblinear",
                            class_weight="balanced", random_state=42, max_iter=1000)
    lr.fit(Xf_fit, y_fit)
    lr_metrics = evaluate_model("LogisticRegression",
                                lr.predict_proba(Xf_test)[:, 1], y_test)

    # Random Forest
    print(f"\ntuning RF ({N_TRIALS} trials)")
    def rf_obj(trial):
        m = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 5, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            class_weight="balanced_subsample", n_jobs=-1, random_state=42,
        )
        m.fit(Xf_train, y_train)
        return average_precision_score(y_val, m.predict_proba(Xf_val)[:, 1])
    study_rf = optuna.create_study(direction="maximize",
                                   sampler=optuna.samplers.TPESampler(seed=42))
    study_rf.optimize(rf_obj, n_trials=N_TRIALS)
    print(f"  best {study_rf.best_params}  val AUC-PR={study_rf.best_value:.4f}")
    rf = RandomForestClassifier(**study_rf.best_params,
                                class_weight="balanced_subsample",
                                n_jobs=-1, random_state=42)
    rf.fit(Xf_fit, y_fit)
    rf_metrics = evaluate_model("RandomForest",
                                rf.predict_proba(Xf_test)[:, 1], y_test)

    # XGBoost
    print(f"\ntuning XGB ({N_TRIALS} trials)")
    def xgb_obj(trial):
        m = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            scale_pos_weight=spw, eval_metric="aucpr",
            n_jobs=-1, random_state=42, verbosity=0,
        )
        m.fit(Xf_train, y_train)
        return average_precision_score(y_val, m.predict_proba(Xf_val)[:, 1])
    study_xgb = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
    study_xgb.optimize(xgb_obj, n_trials=N_TRIALS)
    print(f"  best {study_xgb.best_params}  val AUC-PR={study_xgb.best_value:.4f}")
    xgb = XGBClassifier(**study_xgb.best_params, scale_pos_weight=spw,
                        eval_metric="aucpr", n_jobs=-1,
                        random_state=42, verbosity=0)
    xgb.fit(Xf_fit, y_fit)
    xgb_metrics = evaluate_model("XGBoost",
                                 xgb.predict_proba(Xf_test)[:, 1], y_test)

    # top features for the tree models (sanity check)
    for name, imp in [("RF", rf.feature_importances_),
                      ("XGB", xgb.feature_importances_)]:
        top = np.argsort(imp)[::-1][:10]
        print(f"\n{name} top-10 features:")
        for r, i in enumerate(top, 1):
            print(f"  {r:2d}. {feature_names[i]:<20s} {imp[i]:.4f}")

    all_results = [lr_metrics, rf_metrics, xgb_metrics]
    print()
    for m in all_results:
        print(f"{m['model']:<20s} AUC-PR={m['auc_pr']:.4f} "
              f"AUC-ROC={m['auc_roc']:.4f} "
              f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")

    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "n_trials": N_TRIALS,
            "best_params": {
                "LogisticRegression": study_lr.best_params,
                "RandomForest": study_rf.best_params,
                "XGBoost": study_xgb.best_params,
            },
            "results": all_results,
        }, f, indent=2)
    print(f"saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
