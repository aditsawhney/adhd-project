"""
Phase 6 (secondary explainability check): SHAP on a standalone tree model.

The calibrated VotingClassifier ensemble (the actual deployed/reported
Model A) is not compatible with shap.TreeExplainer -- permutation
importance (evaluate.py) is the primary, ensemble-compatible method.

As a secondary, illustrative check, this fits Model A's single
best-performing model (XGBoost, same tuned hyperparameters, same data)
standalone and runs TreeExplainer on it directly. This is also a clean
contrast point for the paper: same technique (SHAP + TreeExplainer) the
original team notebook used, applied to a leakage-free feature set
instead of one containing DX_3/ScanDir ID.
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
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from features import (
    MODEL_A_BINARY,
    MODEL_A_CATEGORICAL,
    MODEL_A_NUMERIC,
    build_preprocessor_a,
)
from train import get_model_grid, make_predefined_split

FEATURE_COLS_A = MODEL_A_NUMERIC + MODEL_A_CATEGORICAL + MODEL_A_BINARY


def fit_standalone_xgboost(train_df, val_df):
    """Re-run grid search for XGBoost alone (cheap, ~1-3s) to get the same
    tuned hyperparameters used inside the ensemble, but as a standalone
    model SHAP's TreeExplainer can actually read."""
    combined, ps = make_predefined_split(train_df, val_df)
    X, y = combined[FEATURE_COLS_A], combined["target"]

    estimator, param_grid = get_model_grid(include_xgboost=True)["XGBoost"]
    pipe = Pipeline([("preprocessor", build_preprocessor_a()), ("clf", estimator)])
    gs = GridSearchCV(pipe, param_grid, scoring="f1", cv=ps, n_jobs=-1, refit=True)
    gs.fit(X, y)
    print(f"Standalone XGBoost: val_f1={gs.best_score_:.4f}, params={gs.best_params_}")
    return gs.best_estimator_


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


if __name__ == "__main__":
    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/val.csv")
    test = pd.read_csv("data/processed/test.csv")

    pipe = fit_standalone_xgboost(train, val)
    joblib.dump(pipe, "models/model_a_standalone_xgb.joblib")

    preprocessor = pipe.named_steps["preprocessor"]
    clf = pipe.named_steps["clf"]

    X_test_transformed = preprocessor.transform(test[FEATURE_COLS_A])
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()
    feature_names = get_feature_names(preprocessor)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test_transformed)
    # some shap/xgboost version combos return a per-class list, others a
    # single array for binary classifiers -- handle both
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    out_dir = Path("reports/model_a")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, show=False)
    plt.title("Model A (standalone XGBoost) — SHAP Summary")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_ranking = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)
    shap_ranking.to_csv(out_dir / "shap_importance.csv", index=False)

    print("\nTop features by mean |SHAP value|:")
    print(shap_ranking.head(10).to_string(index=False))
    print(f"\nSaved shap_summary.png and shap_importance.csv to {out_dir}/")