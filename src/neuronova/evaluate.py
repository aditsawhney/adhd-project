"""
Phase 4: evaluation for both models on the held-out test set.

For each model (A and B):
  - standard metrics (F1, ROC-AUC, avg precision, accuracy, ADHD-class P/R)
  - confusion matrix, ROC/PR curves, calibration plot
  - permutation importance
  - fairness audit by sex and age band

Plus a top-level A-vs-B comparison table -- the actual point of running two
feature sets on the identical test rows.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from features import (
    MODEL_A_BINARY,
    MODEL_A_CATEGORICAL,
    MODEL_A_NUMERIC,
    MODEL_B_CATEGORICAL,
    MODEL_B_NUMERIC,
)

AGE_BANDS = [(-np.inf, 10, "<10"), (10, 14, "10-13"), (14, np.inf, "14+")]
MIN_SUBGROUP_N = 5  # don't report a fairness slice too small to mean anything


def age_band(age):
    for lo, hi, label in AGE_BANDS:
        if lo <= age < hi:
            return label
    return "unknown"


def _fairness_lines(y_test, y_pred, y_proba, groups, group_order, label_width=8):
    lines = []
    for g in group_order:
        mask = (groups == g).values
        n = int(mask.sum())
        if n < MIN_SUBGROUP_N:
            continue
        yt, yp, ypr = y_test[mask], y_pred[mask], y_proba[mask]
        adhd_rate = yt.mean()
        if len(set(yt)) > 1:
            f1 = f1_score(yt, yp)
            auc = roc_auc_score(yt, ypr)
            lines.append(f"  {g:{label_width}s} (N={n:3d}, {adhd_rate*100:5.1f}% ADHD): F1={f1:.4f}, ROC-AUC={auc:.4f}")
        else:
            lines.append(f"  {g:{label_width}s} (N={n:3d}, {adhd_rate*100:5.1f}% ADHD): single-class slice, skipped")
    return lines


def evaluate_model(label, model, feature_cols, test_df, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_test = test_df[feature_cols]
    y_test = test_df["target"].astype(int).reset_index(drop=True)

    y_pred = pd.Series(model.predict(X_test)).reset_index(drop=True)
    y_proba = pd.Series(model.predict_proba(X_test)[:, 1]).reset_index(drop=True)

    metrics = {
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "avg_precision": average_precision_score(y_test, y_proba),
        "accuracy": accuracy_score(y_test, y_pred),
        "adhd_precision": precision_score(y_test, y_pred),
        "adhd_recall": recall_score(y_test, y_pred),
    }

    report_lines = [f"=== Model {label} test-set metrics ===", ""]
    report_lines += [f"{k}: {v:.4f}" for k, v in metrics.items()]
    report_lines.append("")
    report_lines.append(classification_report(y_test, y_pred, target_names=["control", "ADHD"]))

    # --- confusion matrix ---
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred), display_labels=["control", "ADHD"]
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Model {label} — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # --- ROC + PR ---
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[0])
    axes[0].set_title(f"Model {label} — ROC")
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=axes[1])
    axes[1].set_title(f"Model {label} — Precision-Recall")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_pr.png", dpi=150)
    plt.close(fig)

    # --- calibration ---
    frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(mean_pred, frac_pos, "o-", label=f"Model {label}")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Model {label} — Calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "calibration.png", dpi=150)
    plt.close(fig)

    # --- permutation importance ---
    perm = permutation_importance(
        model, X_test, y_test, scoring="f1", n_repeats=10, random_state=42, n_jobs=-1
    )
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(out_dir / "permutation_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, max(3, 0.35 * len(feature_cols))))
    ax.barh(importance_df["feature"][::-1], importance_df["importance_mean"][::-1],
            xerr=importance_df["importance_std"][::-1])
    ax.set_xlabel("Permutation importance (F1 drop)")
    ax.set_title(f"Model {label} — Permutation Importance")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close(fig)

    # --- fairness audit ---
    sex = test_df["Gender"].reset_index(drop=True)
    bands = test_df["Age"].apply(age_band).reset_index(drop=True)

    fairness_lines = [f"=== Model {label} fairness audit ===", "", "By sex:"]
    fairness_lines += _fairness_lines(
        y_test, y_pred, y_proba, sex, sorted(sex.dropna().unique()), label_width=8
    )
    fairness_lines.append("")
    fairness_lines.append("By age band:")
    fairness_lines += _fairness_lines(
        y_test, y_pred, y_proba, bands, ["<10", "10-13", "14+"], label_width=6
    )

    full_report = "\n".join(report_lines) + "\n\n" + "\n".join(fairness_lines) + "\n"
    (out_dir / "audit_report.txt").write_text(full_report)
    print(full_report)

    return metrics


if __name__ == "__main__":
    test = pd.read_csv("data/processed/test.csv")

    model_a = joblib.load("models/model_a.joblib")
    model_b = joblib.load("models/model_b.joblib")

    feature_cols_a = MODEL_A_NUMERIC + MODEL_A_CATEGORICAL + MODEL_A_BINARY
    feature_cols_b = MODEL_B_NUMERIC + MODEL_B_CATEGORICAL

    print("=" * 60)
    metrics_a = evaluate_model("A", model_a, feature_cols_a, test, "reports/model_a")
    print("=" * 60)
    metrics_b = evaluate_model("B", model_b, feature_cols_b, test, "reports/model_b")

    comparison = pd.DataFrame([
        {"model": "A (extended)", "n_features": len(feature_cols_a), **metrics_a},
        {"model": "B (pre-assessment)", "n_features": len(feature_cols_b), **metrics_b},
    ])
    comparison.to_csv("reports/comparison_report.csv", index=False)

    print("=" * 60)
    print("=== Model A vs Model B comparison (same test rows) ===")
    print(comparison.to_string(index=False))