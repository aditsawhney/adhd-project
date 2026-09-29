"""
Phase 6 (secondary explainability check): SHAP on a standalone tree model,
for both feature sets.

Neither calibrated VotingClassifier ensemble (the actual reported Model A
or Model B) is compatible with shap.TreeExplainer -- permutation
importance (evaluate.py) is the primary, ensemble-compatible method for
both.

As a secondary, illustrative check, this fits each feature set's single
best-performing model standalone (same tuned hyperparameters, same data)
and runs TreeExplainer on it directly. Which model that is is determined
by the actual grid-search ranking, not hardcoded -- for Model A this is
XGBoost, for Model B it is ExtraTrees, matching train.py's own top-1
selection for each.
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
import shap

from features import (
    MODEL_A_BINARY,
    MODEL_A_CATEGORICAL,
    MODEL_A_NUMERIC,
    MODEL_B_CATEGORICAL,
    MODEL_B_NUMERIC,
    build_preprocessor_a,
    build_preprocessor_b,
)
from train import run_model_selection

FEATURE_COLS_A = MODEL_A_NUMERIC + MODEL_A_CATEGORICAL + MODEL_A_BINARY
FEATURE_COLS_B = MODEL_B_NUMERIC + MODEL_B_CATEGORICAL


def fit_standalone_top1(train_df, val_df, cols, preprocessor_fn, include_xgboost):
    """Re-run grid search (cheap, a few seconds) and take whichever model
    actually ranked first -- not assumed in advance -- as a standalone,
    SHAP-compatible pipeline with the same tuned hyperparameters used
    inside that feature set's calibrated ensemble."""
    results_df, fitted = run_model_selection(train_df, val_df, cols, preprocessor_fn, include_xgboost)
    top1 = results_df.iloc[0]["model"]
    print(f"Standalone top model: {top1}  (val_f1={results_df.iloc[0]['val_f1']:.4f})")
    return top1, fitted[top1]


def get_feature_names(preprocessor):
    """Pull post-one-hot feature names out of the fitted ColumnTransformer,
    in the same concatenation order it outputs transformed columns."""
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            names.extend(cols)
        elif name == "cat":
            encoder = trans.named_steps["encode"]
            names.extend(encoder.get_feature_names_out(cols))
        elif name == "bin":
            names.extend(cols)
    return names


def positive_class_shap(shap_values):
    """Normalize across shap/model version differences: some return a
    per-class list, some a single 2D array (binary XGBoost), some a 3D
    array (n_samples, n_features, n_classes) for sklearn tree ensembles."""
    if isinstance(shap_values, list):
        return shap_values[1]
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


def run_shap_for_variant(label, cols, preprocessor_fn, include_xgboost, train, val, test, out_dir):
    print(f"\n=== Model {label} ({len(cols)} raw inputs) ===")
    model_name, pipe = fit_standalone_top1(train, val, cols, preprocessor_fn, include_xgboost)
    joblib.dump(pipe, f"models/model_{label.lower()}_standalone_{model_name.lower()}.joblib")

    preprocessor = pipe.named_steps["preprocessor"]
    clf = pipe.named_steps["clf"]

    X_test_transformed = preprocessor.transform(test[cols])
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()
    feature_names = get_feature_names(preprocessor)

    explainer = shap.TreeExplainer(clf)
    shap_values = positive_class_shap(explainer.shap_values(X_test_transformed))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, show=False)
    plt.title(f"Model {label} (standalone {model_name}) — SHAP Summary")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranking = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)
    ranking.to_csv(out_dir / "shap_importance.csv", index=False)

    print(f"Top features by mean |SHAP value| (Model {label}, {model_name}):")
    print(ranking.head(10).to_string(index=False))
    print(f"Saved shap_summary.png and shap_importance.csv to {out_dir}/")
    return model_name, ranking


if __name__ == "__main__":
    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/val.csv")
    test = pd.read_csv("data/processed/test.csv")

    model_b, ranking_b = run_shap_for_variant(
        "B", FEATURE_COLS_B, build_preprocessor_b, include_xgboost=False,
        train=train, val=val, test=test, out_dir="reports/model_b",
    )
    model_a, ranking_a = run_shap_for_variant(
        "A", FEATURE_COLS_A, build_preprocessor_a, include_xgboost=True,
        train=train, val=val, test=test, out_dir="reports/model_a",
    )

    print(f"\nModel B standalone model: {model_b}")
    print(f"Model A standalone model: {model_a}")