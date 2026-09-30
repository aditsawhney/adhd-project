"""
Does the top-3 soft-voting ensemble actually generalize better than just
deploying the single best model?

The XGBoost check (xgboost_check.py) showed that ranking by validation F1
does not reliably predict leave-one-site-out performance. The ensemble
itself is built on the same validation-F1 ranking ("take the top three,
soft-vote them") and that choice has never been tested against the
alternative of simply deploying the single best-ranked model, calibrated
alone. This script checks that, for both feature sets, across every
evaluation scheme already used elsewhere in this project.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from features import (
    MODEL_A_BINARY,
    MODEL_A_CATEGORICAL,
    MODEL_A_NUMERIC,
    MODEL_B_CATEGORICAL,
    MODEL_B_NUMERIC,
    build_preprocessor_a,
    build_preprocessor_b,
)
from robustness import fit_ensemble, metric_dict, oof_predictions
from train import run_model_selection

OUT = Path("reports")
SEED = 42

FEATURE_SETS = {
    "B": dict(cols=MODEL_B_NUMERIC + MODEL_B_CATEGORICAL,
              preproc=build_preprocessor_b, xgb=False, out="reports/model_b"),
    "A": dict(cols=MODEL_A_NUMERIC + MODEL_A_CATEGORICAL + MODEL_A_BINARY,
              preproc=build_preprocessor_a, xgb=True, out="reports/model_a"),
}


def solo_predictions(name, fitted, train_val_df, cols, test_df):
    """Single best model, individually calibrated, no voting."""
    cal = CalibratedClassifierCV(clone(fitted[name]), method="sigmoid", cv=5)
    cal.fit(train_val_df[cols], train_val_df["target"].astype(int))
    return cal.predict_proba(test_df[cols])[:, 1]


def solo_oof(name, fitted, df, cols, splits):
    p = np.full(len(df), np.nan)
    y = df["target"].astype(int)
    for tr, te in splits:
        cal = CalibratedClassifierCV(clone(fitted[name]), method="sigmoid", cv=5)
        cal.fit(df.iloc[tr][cols], y.iloc[tr])
        p[te] = cal.predict_proba(df.iloc[te][cols])[:, 1]
    return p


def main():
    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/val.csv")
    test = pd.read_csv("data/processed/test.csv")
    full = pd.concat([train, val, test], ignore_index=True)
    trv = pd.concat([train, val], ignore_index=True)

    cv_splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(full, full["target"]))
    sites = full["Site"].values
    loso_splits = [(np.where(sites != s)[0], np.where(sites == s)[0]) for s in sorted(np.unique(sites))]

    all_rows = []
    for label, cfg in FEATURE_SETS.items():
        print(f"\n=== Feature set {label} ===")
        cols = cfg["cols"]
        results, fitted = run_model_selection(train, val, cols, cfg["preproc"], cfg["xgb"])
        top3 = results.head(3)["model"].tolist()
        best1 = results.iloc[0]["model"]
        print(f"top3={top3}  solo best={best1}")

        y_test = test["target"].astype(int).values
        y_full = full["target"].astype(int).values

        ens = fit_ensemble(top3, fitted, trv[cols], trv["target"].astype(int))
        p_ens_test = ens.predict_proba(test[cols])[:, 1]
        p_ens_cv = oof_predictions(full, cols, top3, fitted, cv_splits)
        p_ens_loso = oof_predictions(full, cols, top3, fitted, loso_splits)

        p_solo_test = solo_predictions(best1, fitted, trv, cols, test)
        p_solo_cv = solo_oof(best1, fitted, full, cols, cv_splits)
        p_solo_loso = solo_oof(best1, fitted, full, cols, loso_splits)

        for approach, p_test, p_cv, p_loso in (
            (f"Ensemble (top3={top3})", p_ens_test, p_ens_cv, p_ens_loso),
            (f"Solo best ({best1})", p_solo_test, p_solo_cv, p_solo_loso),
        ):
            for scheme, y, p in (("test (n=190)", y_test, p_test),
                                 ("5-fold CV (n=947)", y_full, p_cv),
                                 ("leave-one-site-out (n=947)", y_full, p_loso)):
                all_rows.append(dict(feature_set=label, approach=approach, scheme=scheme, **metric_dict(y, p)))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "ensemble_vs_solo.csv", index=False)
    report = ["=== Ensemble vs. single best model, both feature sets ===", "",
              df.round(4).to_string(index=False), ""]
    print("\n".join(report))
    (OUT / "ensemble_vs_solo.txt").write_text("\n".join(report) + "\n")
    print(f"Saved ensemble_vs_solo.csv/.txt to {OUT}/")


if __name__ == "__main__":
    main()