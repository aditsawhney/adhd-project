"""
Phase 2b: does including XGBoost in Model B's candidate pool actually help?

train.py excludes XGBoost from Model B on the strength of an earlier,
untested claim ("overfits on 700 rows / 5 features"). This script checks
that claim directly: build Model B two ways -- current roster
(ExtraTrees/RandomForest/SVM) and with XGBoost added to the candidate
pool -- and compare across every evaluation already used elsewhere in
this project (validation fold, held-out test set, 5-fold CV, and
leave-one-site-out). If XGBoost only wins on the fold it was selected
against and not elsewhere, that is the standard signature of overfitting
to that fold, not a reason to change the deployed roster.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from features import MODEL_B_CATEGORICAL, MODEL_B_NUMERIC, build_preprocessor_b
from robustness import fit_ensemble, metric_dict, oof_predictions
from train import run_model_selection

OUT = Path("reports/model_b")
SEED = 42
FEATURE_COLS_B = MODEL_B_NUMERIC + MODEL_B_CATEGORICAL


def evaluate_variant(label, include_xgboost, train, val, test, full, cv_splits, loso_splits):
    print(f"\n=== {label} ===")
    results, fitted = run_model_selection(
        train, val, FEATURE_COLS_B, build_preprocessor_b, include_xgboost
    )
    top3 = results.head(3)["model"].tolist()
    print(f"top3={top3}")

    trv = pd.concat([train, val], ignore_index=True)
    ens = fit_ensemble(top3, fitted, trv[FEATURE_COLS_B], trv["target"].astype(int))

    y_test = test["target"].astype(int).values
    p_test = ens.predict_proba(test[FEATURE_COLS_B])[:, 1]
    m_test = metric_dict(y_test, p_test)

    p_cv = oof_predictions(full, FEATURE_COLS_B, top3, fitted, cv_splits)
    p_loso = oof_predictions(full, FEATURE_COLS_B, top3, fitted, loso_splits)
    y_full = full["target"].astype(int).values
    m_cv, m_loso = metric_dict(y_full, p_cv), metric_dict(y_full, p_loso)

    val_f1 = results.iloc[0]["val_f1"] if not include_xgboost else \
        results.set_index("model").loc["XGBoost", "val_f1"]

    rows = [dict(variant=label, top3=str(top3), scheme="validation (best candidate)", f1=val_f1,
                 roc_auc=np.nan, avg_precision=np.nan, accuracy=np.nan)]
    for scheme, m in (("test (n=190)", m_test), ("5-fold CV (n=947)", m_cv), ("leave-one-site-out (n=947)", m_loso)):
        rows.append(dict(variant=label, top3=str(top3), scheme=scheme, **m))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/val.csv")
    test = pd.read_csv("data/processed/test.csv")
    full = pd.concat([train, val, test], ignore_index=True)

    cv_splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(full, full["target"]))
    sites = full["Site"].values
    loso_splits = [(np.where(sites != s)[0], np.where(sites == s)[0]) for s in sorted(np.unique(sites))]

    current = evaluate_variant("B (current: no XGBoost)", False, train, val, test, full, cv_splits, loso_splits)
    with_xgb = evaluate_variant("B + XGBoost candidate", True, train, val, test, full, cv_splits, loso_splits)

    combined = pd.concat([current, with_xgb], ignore_index=True)
    combined.to_csv(OUT / "xgboost_check.csv", index=False)

    report = ["=== Does including XGBoost help Model B? ===", "", combined.round(4).to_string(index=False), ""]
    print("\n".join(report))
    (OUT / "xgboost_check.txt").write_text("\n".join(report) + "\n")
    print(f"\nSaved xgboost_check.csv and xgboost_check.txt to {OUT}/")