"""
Complete-case check: does imputation of missing subscores inflate the results?

311 of the 947 participants have no Inattentive or Hyper/Impulsive score
(three whole sites never recorded them), and those values are filled with
the training median. Two questions follow.

1. Part 1: how do the models do on participants with real scores versus
   participants whose scores were imputed? Predictions come from 5-fold
   cross-validation and leave-one-site-out over all 947 rows, then are split
   by whether the score was real. Group results carry bootstrap intervals.
2. Part 2: if a model is trained and evaluated on complete cases only, how
   does it compare with the all-rows model on those same participants?
   (Sites 4, 7 and 8 drop out entirely, leaving four sites.)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
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
N_BOOT = 2000

SETS = {
    "B": dict(cols=MODEL_B_NUMERIC + MODEL_B_CATEGORICAL, preproc=build_preprocessor_b, xgb=False),
    "A": dict(cols=MODEL_A_NUMERIC + MODEL_A_CATEGORICAL + MODEL_A_BINARY, preproc=build_preprocessor_a, xgb=True),
}


def has_scores(df):
    return (df["Inattentive"].notna() & df["Hyper/Impulsive"].notna()).values


def boot(y, p, rng, n_boot=N_BOOT):
    f1s, aucs = [], []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        f1s.append(f1_score(yb, (p[idx] >= 0.5).astype(int)))
        aucs.append(roc_auc_score(yb, p[idx]))
    return np.array(f1s), np.array(aucs)


def splits_for(df):
    cv = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(df, df["target"]))
    sites = df["Site"].values
    loso = [(np.where(sites != s)[0], np.where(sites == s)[0]) for s in sorted(np.unique(sites))]
    return cv, loso


def group_row(label, scheme, group, y, p, rng):
    m = metric_dict(y, p)
    f1_d, auc_d = boot(y, p, rng)
    return dict(feature_set=label, scheme=scheme, group=group, n=len(y), adhd=int(y.sum()),
                prevalence=y.mean(), **m,
                f1_lo=np.percentile(f1_d, 2.5), f1_hi=np.percentile(f1_d, 97.5),
                auc_lo=np.percentile(auc_d, 2.5), auc_hi=np.percentile(auc_d, 97.5)), f1_d, auc_d


def main():
    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/val.csv")
    test = pd.read_csv("data/processed/test.csv")
    full = pd.concat([train, val, test], ignore_index=True)
    y_full = full["target"].astype(int).values
    real = has_scores(full)
    print(f"real scores: {real.sum()}  imputed: {(~real).sum()}  "
          f"ADHD rate real={y_full[real].mean():.3f} imputed={y_full[~real].mean():.3f}")

    rng = np.random.default_rng(SEED)
    part1, gap_rows = [], []
    part2 = []

    for label, cfg in SETS.items():
        cols = cfg["cols"]
        print(f"\n##### Feature set {label}: all-rows model #####")
        results, fitted = run_model_selection(train, val, cols, cfg["preproc"], cfg["xgb"])
        top3 = results.head(3)["model"].tolist()
        cv_splits, loso_splits = splits_for(full)
        p = {"5-fold CV": oof_predictions(full, cols, top3, fitted, cv_splits),
             "leave-one-site-out": oof_predictions(full, cols, top3, fitted, loso_splits)}

        for scheme, probs in p.items():
            draws = {}
            for group, mask in (("real scores", real), ("imputed scores", ~real), ("all rows", np.ones(len(full), bool))):
                row, f1_d, auc_d = group_row(label, scheme, group, y_full[mask], probs[mask], rng)
                part1.append(row)
                draws[group] = (f1_d, auc_d)
            for k, name in enumerate(("F1", "ROC-AUC")):
                a, b = draws["real scores"][k], draws["imputed scores"][k]
                m_len = min(len(a), len(b))  # the two groups are resampled independently
                d = a[:m_len] - b[:m_len]
                gap_rows.append(dict(feature_set=label, scheme=scheme, metric=name,
                                     ci_note="real minus imputed",
                                     ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5)))

        print(f"##### Feature set {label}: trained and evaluated on complete cases only #####")
        tr_c, va_c, te_c = train[has_scores(train)], val[has_scores(val)], test[has_scores(test)]
        full_c = full[real].reset_index(drop=True)
        res_c, fit_c = run_model_selection(tr_c, va_c, cols, cfg["preproc"], cfg["xgb"])
        top3_c = res_c.head(3)["model"].tolist()
        trv_c = pd.concat([tr_c, va_c], ignore_index=True)
        ens = fit_ensemble(top3_c, fit_c, trv_c[cols], trv_c["target"].astype(int))
        y_te = te_c["target"].astype(int).values
        part2.append(dict(feature_set=label, scheme="test, complete cases", n=len(te_c),
                          **metric_dict(y_te, ens.predict_proba(te_c[cols])[:, 1])))
        cv_c, loso_c = splits_for(full_c)
        y_c = full_c["target"].astype(int).values
        part2.append(dict(feature_set=label, scheme="5-fold CV, complete cases", n=len(full_c),
                          **metric_dict(y_c, oof_predictions(full_c, cols, top3_c, fit_c, cv_c))))
        part2.append(dict(feature_set=label, scheme="leave-one-site-out, complete cases (4 sites)", n=len(full_c),
                          **metric_dict(y_c, oof_predictions(full_c, cols, top3_c, fit_c, loso_c))))

    p1 = pd.DataFrame(part1)
    p2 = pd.DataFrame(part2)
    gaps = pd.DataFrame(gap_rows)
    p1.to_csv(OUT / "complete_case_by_group.csv", index=False)
    p2.to_csv(OUT / "complete_case_retrained.csv", index=False)
    gaps.to_csv(OUT / "complete_case_gaps.csv", index=False)

    report = ["=== Part 1: all-rows models, split by whether the score was real ===", "",
              p1.round(3).to_string(index=False), "",
              "=== Real-minus-imputed difference, 95% bootstrap interval ===", "",
              gaps.round(3).to_string(index=False), "",
              "=== Part 2: trained and evaluated on complete cases only ===", "",
              p2.round(3).to_string(index=False), ""]
    (OUT / "complete_case.txt").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()