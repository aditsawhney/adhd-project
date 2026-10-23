"""
Confidence intervals for the fairness audit.

evaluate.py reports F1 and ROC-AUC per sex and age band, but only as point
estimates on small subgroups (72 girls, 16 of them with ADHD; 49 participants
aged 14 and over, 14 with ADHD). This script puts bootstrap intervals on
those numbers and, more importantly, on the gaps between groups, so claims
like "Model A's sex gap is wider than Model B's" can be checked against what
a test set this size can actually resolve.

Each subgroup is resampled separately (keeping its own size), and a gap is
the difference between two independently resampled groups.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from features import (
    MODEL_A_BINARY,
    MODEL_A_CATEGORICAL,
    MODEL_A_NUMERIC,
    MODEL_B_CATEGORICAL,
    MODEL_B_NUMERIC,
)

OUT = Path("reports")
N_BOOT = 2000
SEED = 42

FEATURES = {
    "B": MODEL_B_NUMERIC + MODEL_B_CATEGORICAL,
    "A": MODEL_A_NUMERIC + MODEL_A_CATEGORICAL + MODEL_A_BINARY,
}


def age_band(age):
    if age < 10:
        return "under 10"
    if age < 14:
        return "10-13"
    return "14 and over"


def f1_auc(y, p):
    if y.min() == y.max():
        return np.nan, np.nan
    return f1_score(y, (p >= 0.5).astype(int)), roc_auc_score(y, p)


def boot_group(y, p, rng, n_boot):
    f1s, aucs = [], []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        f, a = f1_auc(y[idx], p[idx])
        f1s.append(f)
        aucs.append(a)
    return np.array(f1s), np.array(aucs)


def interval(x):
    x = x[~np.isnan(x)]
    return np.percentile(x, [2.5, 97.5])


def main():
    test = pd.read_csv("data/processed/test.csv")
    y_all = test["target"].astype(int).values
    sex = test["Gender"].values
    bands = test["Age"].apply(age_band).values

    groups = {
        "Female": sex == "Female",
        "Male": sex == "Male",
        "under 10": bands == "under 10",
        "10-13": bands == "10-13",
        "14 and over": bands == "14 and over",
        "under 14 (pooled)": bands != "14 and over",
    }
    gap_defs = [
        ("sex gap (Female minus Male)", "Female", "Male"),
        ("age gap (14 and over minus under 14)", "14 and over", "under 14 (pooled)"),
    ]

    rows, gap_rows = [], []
    for label, cols in FEATURES.items():
        model = joblib.load(f"models/model_{label.lower()}.joblib")
        p_all = model.predict_proba(test[cols])[:, 1]
        rng = np.random.default_rng(SEED)
        draws = {}
        for g, mask in groups.items():
            y, p = y_all[mask], p_all[mask]
            f1_pt, auc_pt = f1_auc(y, p)
            f1_d, auc_d = boot_group(y, p, rng, N_BOOT)
            draws[g] = (f1_d, auc_d, f1_pt, auc_pt)
            f_lo, f_hi = interval(f1_d)
            a_lo, a_hi = interval(auc_d)
            rows.append(dict(model=label, group=g, n=int(mask.sum()), adhd=int(y.sum()),
                             f1=f1_pt, f1_lo=f_lo, f1_hi=f_hi,
                             auc=auc_pt, auc_lo=a_lo, auc_hi=a_hi))
        for name, g1, g2 in gap_defs:
            for metric, k in (("F1", 0), ("ROC-AUC", 1)):
                d = draws[g1][k] - draws[g2][k]
                est = draws[g1][2 + k] - draws[g2][2 + k]
                lo, hi = interval(d)
                gap_rows.append(dict(model=label, gap=name, metric=metric, estimate=est,
                                     ci_low=lo, ci_high=hi, excludes_zero=bool(lo > 0 or hi < 0)))

    subgroup = pd.DataFrame(rows)
    gaps = pd.DataFrame(gap_rows)
    subgroup.to_csv(OUT / "subgroup_ci.csv", index=False)
    gaps.to_csv(OUT / "subgroup_gaps.csv", index=False)

    report = ["=== Subgroup metrics with 95% bootstrap intervals ===", "",
              subgroup.round(3).to_string(index=False), "",
              "=== Gaps between groups ===", "", gaps.round(3).to_string(index=False), ""]
    (OUT / "subgroup_ci.txt").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()