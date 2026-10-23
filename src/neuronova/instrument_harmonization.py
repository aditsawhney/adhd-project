"""
Does correcting for the rating-instrument scale mismatch fix the site-1
leave-one-site-out failure?

Site 1 is the only site using instrument 1 (ADHD-RS, scored 9-36); every
other site uses instrument 2 or 3 (Conners-family, scored 40-90ish), and
both ranges are stored in the same Inattentive / Hyper/Impulsive columns.
This script z-scores those two columns within their recorded instrument,
using per-instrument statistics computed from the training fold only, and
re-runs the same evaluation schemes used elsewhere in this project to see
whether it helps.

Note before running: instrument and site are perfectly confounded for
site 1 (and also site 6). Under leave-one-site-out, holding out site 1
removes every example of instrument 1 from training, so there is no data
left to learn instrument-1 statistics from. Harmonization learned this
way cannot fix site 1's leave-one-site-out failure by construction; this
script exists to confirm that empirically and to check whether it still
helps everywhere the confound doesn't apply (in-distribution evaluation,
and any site sharing an instrument with others).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold

from features import (
    MODEL_A_BINARY,
    MODEL_A_CATEGORICAL,
    MODEL_A_NUMERIC,
    MODEL_B_CATEGORICAL,
    MODEL_B_NUMERIC,
    _categorical_pipeline,
    _numeric_pipeline,
    build_preprocessor_a,
    build_preprocessor_b,
)
from robustness import fit_ensemble, metric_dict, oof_predictions, per_site_table
from train import run_model_selection

OUT = Path("reports/robustness")
SEED = 42
SCORE_COLS = ["Inattentive", "Hyper/Impulsive"]
INSTRUMENT_COL = "ADHD Measure"


class InstrumentHarmonizer(BaseEstimator, TransformerMixin):
    """Z-scores SCORE_COLS within their recorded instrument, using
    per-instrument mean/std fit on the training fold only. Instruments
    with fewer than min_group_n training rows (or fewer than two real
    scores), and missing or unseen instrument codes, fall back to the
    overall training mean/std."""

    def __init__(self, min_group_n=15):
        self.min_group_n = min_group_n

    def _frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=SCORE_COLS + [INSTRUMENT_COL])

    @staticmethod
    def _safe_mean_std(series):
        m, s = series.mean(), series.std()
        if pd.isna(m) or series.notna().sum() < 2:
            return None  # not enough real data to trust this group/column
        if pd.isna(s) or s == 0:
            s = 1.0
        return (m, s)

    def fit(self, X, y=None):
        X = self._frame(X)
        self.global_stats_ = {}
        for c in SCORE_COLS:
            stats = self._safe_mean_std(X[c].astype(float))
            self.global_stats_[c] = stats if stats is not None else (0.0, 1.0)

        instr = X[INSTRUMENT_COL].fillna("missing").astype(str)
        self.group_stats_ = {}
        for g in instr.unique():
            sub = X.loc[(instr == g).values]
            if len(sub) < self.min_group_n:
                continue
            stats = {}
            for c in SCORE_COLS:
                s = self._safe_mean_std(sub[c].astype(float))
                if s is None:
                    break
                stats[c] = s
            if len(stats) == len(SCORE_COLS):
                self.group_stats_[g] = stats
        return self

    def transform(self, X):
        X = self._frame(X)
        instr = X[INSTRUMENT_COL].fillna("missing").astype(str)
        out = np.zeros((len(X), len(SCORE_COLS)))
        for i, c in enumerate(SCORE_COLS):
            gmean, gstd = self.global_stats_[c]
            means = np.full(len(X), gmean)
            stds = np.full(len(X), gstd)
            for g, stats in self.group_stats_.items():
                mask = (instr == g).values
                if mask.any():
                    m, s = stats[c]
                    means[mask], stds[mask] = m, s
            vals = X[c].values.astype(float)
            vals = np.where(np.isnan(vals), means, vals)  # missing score -> group mean -> z of 0
            out[:, i] = (vals - means) / stds
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_harmonized" for c in SCORE_COLS])


def build_preprocessor_b_harmonized():
    return ColumnTransformer([
        ("harm", InstrumentHarmonizer(), SCORE_COLS + [INSTRUMENT_COL]),
        ("num", _numeric_pipeline(), ["Age"]),
        ("cat", _categorical_pipeline(), MODEL_B_CATEGORICAL),
    ])


def build_preprocessor_a_harmonized():
    numeric_rest = [c for c in MODEL_A_NUMERIC if c not in SCORE_COLS]
    return ColumnTransformer([
        ("harm", InstrumentHarmonizer(), SCORE_COLS + [INSTRUMENT_COL]),
        ("num", _numeric_pipeline(), numeric_rest),
        ("cat", _categorical_pipeline(), MODEL_A_CATEGORICAL),
        ("bin", "passthrough", MODEL_A_BINARY),
    ])


# The harmonizer needs ADHD Measure as an input column even for the B variant,
# where it is not a model feature. It is only used to decide how to rescale the
# two subscores; for A it is also one-hot encoded as usual.
FEATURE_COLS_B_RAW = MODEL_B_NUMERIC + MODEL_B_CATEGORICAL
FEATURE_COLS_B_H = FEATURE_COLS_B_RAW + [INSTRUMENT_COL]
FEATURE_COLS_A_RAW = MODEL_A_NUMERIC + MODEL_A_CATEGORICAL + MODEL_A_BINARY
FEATURE_COLS_A_H = list(dict.fromkeys(FEATURE_COLS_A_RAW + [INSTRUMENT_COL]))


def run_variant(label, cols, preproc_fn, xgb, train, val, test, full, cv_splits, loso_splits):
    print(f"\n=== {label} ===")
    results, fitted = run_model_selection(train, val, cols, preproc_fn, xgb)
    top3 = results.head(3)["model"].tolist()
    print(f"top3={top3}")
    trv = pd.concat([train, val], ignore_index=True)
    ens = fit_ensemble(top3, fitted, trv[cols], trv["target"].astype(int))

    y_test = test["target"].astype(int).values
    m_test = metric_dict(y_test, ens.predict_proba(test[cols])[:, 1])
    p_cv = oof_predictions(full, cols, top3, fitted, cv_splits)
    p_loso = oof_predictions(full, cols, top3, fitted, loso_splits)
    y_full = full["target"].astype(int).values
    m_cv, m_loso = metric_dict(y_full, p_cv), metric_dict(y_full, p_loso)
    site_df = per_site_table(full, p_loso, label)

    rows = []
    for scheme, m in (("test (n=190)", m_test),
                      ("5-fold CV (n=947)", m_cv),
                      ("leave-one-site-out (n=947)", m_loso)):
        rows.append(dict(variant=label, scheme=scheme, **m))
    return pd.DataFrame(rows), site_df


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/val.csv")
    test = pd.read_csv("data/processed/test.csv")
    full = pd.concat([train, val, test], ignore_index=True)

    cv_splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(full, full["target"]))
    sites = full["Site"].values
    loso_splits = [(np.where(sites != s)[0], np.where(sites == s)[0]) for s in sorted(np.unique(sites))]

    summaries, site_tables = [], []
    for label, cols, preproc, xgb in [
        ("B (raw scores)", FEATURE_COLS_B_RAW, build_preprocessor_b, False),
        ("B (harmonized scores)", FEATURE_COLS_B_H, build_preprocessor_b_harmonized, False),
        ("A (raw scores)", FEATURE_COLS_A_RAW, build_preprocessor_a, True),
        ("A (harmonized scores)", FEATURE_COLS_A_H, build_preprocessor_a_harmonized, True),
    ]:
        summary, site_df = run_variant(label, cols, preproc, xgb, train, val, test, full, cv_splits, loso_splits)
        summaries.append(summary)
        site_tables.append(site_df)

    summary_df = pd.concat(summaries, ignore_index=True)
    site_df_all = pd.concat(site_tables, ignore_index=True)
    summary_df.to_csv(OUT / "harmonization_check.csv", index=False)
    site_df_all.to_csv(OUT / "harmonization_by_site.csv", index=False)

    report = ["=== Raw vs. instrument-harmonized scores ===", "", summary_df.round(4).to_string(index=False), "",
              "=== Per-site leave-one-site-out, raw vs. harmonized ===", "",
              site_df_all.round(3).to_string(index=False), ""]
    print("\n".join(report))
    (OUT / "harmonization_report.txt").write_text("\n".join(report) + "\n")
    print(f"\nSaved to {OUT}/")