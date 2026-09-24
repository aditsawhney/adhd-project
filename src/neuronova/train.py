"""
Phase 2: model training for both feature sets.

For each feature set (Model A / Model B):
  1. GridSearchCV over a shared model roster, scored on F1, using a
     PredefinedSplit so every model is tuned against the exact same
     validation fold (no k-fold reshuffling, no fold-to-fold model
     rotation, no scoring inconsistency between models).
  2. Take the top-3 models by validation F1.
  3. Wrap each in CalibratedClassifierCV (sigmoid, cv=5) individually,
     then soft-vote them. Never wrap the VotingClassifier itself in
     calibration — that strips the preprocessor out of the sub-estimators
     and breaks inference on string features (Handedness, etc.) at serve
     time.
"""
import sys
import time
from pathlib import Path

# make 'neuronova' importable regardless of whether PYTHONPATH/.env
# actually made it into the process env
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from features import (
    MODEL_A_BINARY,
    MODEL_A_CATEGORICAL,
    MODEL_A_NUMERIC,
    MODEL_B_CATEGORICAL,
    MODEL_B_NUMERIC,
    build_preprocessor_a,
    build_preprocessor_b,
)

RANDOM_STATE = 42


def get_model_grid(include_xgboost: bool = True):
    """Shared roster + hyperparameter grids. One entry per model."""
    grid = {
        "RandomForest": (
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {
                "clf__n_estimators": [100, 200, 300],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_split": [2, 5, 10],
            },
        ),
        "ExtraTrees": (
            ExtraTreesClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {
                "clf__n_estimators": [100, 200, 300],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_split": [2, 5, 10],
            },
        ),
        "SVM": (
            SVC(class_weight="balanced", random_state=RANDOM_STATE),
            {
                "clf__kernel": ["rbf"],
                "clf__C": [0.1, 1, 10],
                "clf__gamma": ["scale", "auto", 0.1, 1],
            },
        ),
        "LogisticRegression": (
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "QDA": (
            QuadraticDiscriminantAnalysis(),
            {"clf__reg_param": [0.01, 0.1, 0.3, 0.5]},
        ),
        "NaiveBayes": (
            GaussianNB(),
            {"clf__var_smoothing": [1e-9, 1e-8, 1e-7]},
        ),
        "LDA": (
            LinearDiscriminantAnalysis(),
            {"clf__solver": ["svd", "lsqr"]},
        ),
    }
    if include_xgboost:
        grid["XGBoost"] = (
            XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
            {
                "clf__n_estimators": [100, 200, 300],
                "clf__max_depth": [3, 4, 5],
                "clf__learning_rate": [0.01, 0.1, 0.2],
                "clf__subsample": [0.8, 1.0],
            },
        )
    return grid


def make_predefined_split(train_df, val_df):
    """-1 for train rows (always training), 0 for val rows (the one held-out
    fold GridSearchCV scores against) -> identical val fold for every model."""
    combined = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    fold = np.array([-1] * len(train_df) + [0] * len(val_df))
    return combined, PredefinedSplit(test_fold=fold)


def run_model_selection(train_df, val_df, feature_cols, preprocessor_fn, include_xgboost=True):
    combined, ps = make_predefined_split(train_df, val_df)
    X, y = combined[feature_cols], combined["target"]

    results, fitted = [], {}
    for name, (estimator, param_grid) in get_model_grid(include_xgboost).items():
        pipe = Pipeline([("preprocessor", preprocessor_fn()), ("clf", estimator)])
        t0 = time.time()
        gs = GridSearchCV(pipe, param_grid, scoring="f1", cv=ps, n_jobs=-1, refit=True)
        gs.fit(X, y)
        elapsed = time.time() - t0

        results.append({
            "model": name,
            "val_f1": gs.best_score_,
            "best_params": gs.best_params_,
            "fit_seconds": round(elapsed, 2),
        })
        fitted[name] = gs.best_estimator_
        print(f"  {name:20s} val_f1={gs.best_score_:.4f}  ({elapsed:.1f}s)")

    results_df = pd.DataFrame(results).sort_values("val_f1", ascending=False).reset_index(drop=True)
    return results_df, fitted


def build_calibrated_ensemble(top3_names, fitted, train_df, val_df, feature_cols):
    """Calibrate each of the top-3 individually (sigmoid, cv=5), soft-vote.
    fitted[name] carries the tuned hyperparameters from grid search;
    CalibratedClassifierCV clones it and refits fresh across its 5 internal
    folds — no leakage, hyperparameters preserved."""
    combined = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    X, y = combined[feature_cols], combined["target"]

    estimators = [
        (name, CalibratedClassifierCV(fitted[name], method="sigmoid", cv=5))
        for name in top3_names
    ]
    ensemble = VotingClassifier(estimators=estimators, voting="soft")
    ensemble.fit(X, y)
    return ensemble


def train_track(label, train_df, val_df, feature_cols, preprocessor_fn, include_xgboost):
    print(f"\n=== Model {label}: hyperparameter search ({len(feature_cols)} raw feature columns) ===")
    results_df, fitted = run_model_selection(train_df, val_df, feature_cols, preprocessor_fn, include_xgboost)

    top3 = results_df.head(3)["model"].tolist()
    print(f"\nTop 3 for Model {label}: {top3}")

    print(f"=== Model {label}: calibration + soft voting ===")
    ensemble = build_calibrated_ensemble(top3, fitted, train_df, val_df, feature_cols)

    return results_df, ensemble, top3


if __name__ == "__main__":
    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/val.csv")

    feature_cols_b = MODEL_B_NUMERIC + MODEL_B_CATEGORICAL
    feature_cols_a = MODEL_A_NUMERIC + MODEL_A_CATEGORICAL + MODEL_A_BINARY

    results_b, ensemble_b, top3_b = train_track(
        "B", train, val, feature_cols_b, build_preprocessor_b, include_xgboost=False
    )
    results_a, ensemble_a, top3_a = train_track(
        "A", train, val, feature_cols_a, build_preprocessor_a, include_xgboost=True
    )

    results_b.to_csv("reports/model_b/model_selection.csv", index=False)
    results_a.to_csv("reports/model_a/model_selection.csv", index=False)

    joblib.dump(ensemble_b, "models/model_b.joblib")
    joblib.dump(ensemble_a, "models/model_a.joblib")
    joblib.dump({"top3": top3_b, "feature_cols": feature_cols_b}, "models/meta_b.joblib")
    joblib.dump({"top3": top3_a, "feature_cols": feature_cols_a}, "models/meta_a.joblib")

    print("\nSaved model_a.joblib, model_b.joblib, meta_a.joblib, meta_b.joblib to models/")