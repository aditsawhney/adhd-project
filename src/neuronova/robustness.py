"""
Robustness checks for the Model A versus Model B comparison.

Three things the single random split cannot show:

1. Sampling error. Paired bootstrap over the 190 test participants gives
   95% intervals for each metric and for the A-minus-B difference.
2. Site shortcuts. Several inputs identify the acquisition site almost
   completely (IQ Measure, ADHD Measure) and sites differ sharply in ADHD
   prevalence. Variants of Model A without those fields show how much of its
   advantage depends on them.
3. Site generalisation. Leave-one-site-out (LOSO) evaluation trains on six
   sites and tests on the seventh. Comparing pooled LOSO scores with
   ordinary 5-fold cross-validation on the same rows shows how much the
   random split flatters each model.

Hyperparameters and the top-three ensemble members are chosen once per
variant on the locked train/validation split (same procedure as train.py) and
then reused, unchanged, in the cross-validation and LOSO folds.
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
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from features import (
    MODEL_A_BINARY,
    MODEL_A_CATEGORICAL,
    MODEL_A_NUMERIC,
    MODEL_B_CATEGORICAL,
    MODEL_B_NUMERIC,
    _categorical_pipeline,
    _numeric_pipeline,
)
from train import run_model_selection

OUT = Path("reports/robustness")
N_BOOT = 2000
SEED = 42
MIN_PER_CLASS_FOR_AUC = 5

A_CAT = MODEL_A_CATEGORICAL
VARIANTS = {
    "B": dict(numeric=MODEL_B_NUMERIC, categorical=MODEL_B_CATEGORICAL, binary=[], xgb=False),
    "A": dict(numeric=MODEL_A_NUMERIC, categorical=A_CAT, binary=MODEL_A_BINARY, xgb=True),
    "A - IQ Measure": dict(
        numeric=MODEL_A_NUMERIC,
        categorical=[c for c in A_CAT if c != "IQ Measure"],
        binary=MODEL_A_BINARY, xgb=True),
    "A - IQ Measure - ADHD Measure": dict(
        numeric=MODEL_A_NUMERIC,
        categorical=[c for c in A_CAT if c not in ("IQ Measure", "ADHD Measure")],
        binary=MODEL_A_BINARY, xgb=True),
}


def make_preproc(cfg):
    parts = [
        ("num", _numeric_pipeline(), cfg["numeric"]),
        ("cat", _categorical_pipeline(), cfg["categorical"]),
    ]
    if cfg["binary"]:
        parts.append(("bin", "passthrough", cfg["binary"]))
    return ColumnTransformer(parts)


def cols_of(cfg):
    return cfg["numeric"] + cfg["categorical"] + cfg["binary"]


def metric_dict(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, p),
        "avg_precision": average_precision_score(y, p),
        "accuracy": accuracy_score(y, pred),
    }


def fit_ensemble(top3, fitted, X, y):
    members = [
        (n, CalibratedClassifierCV(clone(fitted[n]), method="sigmoid", cv=5))
        for n in top3
    ]
    return VotingClassifier(members, voting="soft").fit(X, y)


def oof_predictions(df, cols, top3, fitted, splits):
    p = np.full(len(df), np.nan)
    y = df["target"].astype(int)
    for tr, te in splits:
        ens = fit_ensemble(top3, fitted, df.iloc[tr][cols], y.iloc[tr])
        p[te] = ens.predict_proba(df.iloc[te][cols])[:, 1]
    return p


def paired_bootstrap(y, probs, pairs, n_boot=N_BOOT, seed=SEED):
    """probs: name -> probability array on the same rows. Returns two frames:
    per-model metric intervals and paired-difference intervals."""
    rng = np.random.default_rng(seed)
    n = len(y)
    names = list(probs)
    keys = ["f1", "roc_auc", "avg_precision", "accuracy"]
    draws = {nm: {k: [] for k in keys} for nm in names}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        for nm in names:
            m = metric_dict(yb, probs[nm][idx])
            for k in keys:
                draws[nm][k].append(m[k])
    rows = []
    for nm in names:
        point = metric_dict(y, probs[nm])
        for k in keys:
            lo, hi = np.percentile(draws[nm][k], [2.5, 97.5])
            rows.append(dict(model=nm, metric=k, estimate=point[k], ci_low=lo, ci_high=hi))
    diff_rows = []
    for a, b in pairs:
        for k in keys:
            d = np.array(draws[a][k]) - np.array(draws[b][k])
            est = metric_dict(y, probs[a])[k] - metric_dict(y, probs[b])[k]
            lo, hi = np.percentile(d, [2.5, 97.5])
            diff_rows.append(dict(comparison=f"{a} minus {b}", metric=k, estimate=est,
                                  ci_low=lo, ci_high=hi, share_positive=(d > 0).mean()))
    return pd.DataFrame(rows), pd.DataFrame(diff_rows)


def per_site_table(df, p, label):
    rows = []
    y = df["target"].astype(int).values
    for s in sorted(df["Site"].unique()):
        mk = (df["Site"] == s).values
        ys, ps = y[mk], p[mk]
        pred = (ps >= 0.5).astype(int)
        pos, neg = int(ys.sum()), int((1 - ys).sum())
        rows.append(dict(
            variant=label, site=int(s), n=int(mk.sum()), adhd=pos,
            prevalence=pos / mk.sum(),
            recall=(pred[ys == 1].mean() if pos else np.nan),
            specificity=((1 - pred[ys == 0]).mean() if neg else np.nan),
            auc=(roc_auc_score(ys, ps)
                 if min(pos, neg) >= MIN_PER_CLASS_FOR_AUC else np.nan),
        ))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/val.csv")
    test = pd.read_csv("data/processed/test.csv")
    full = pd.concat([train, val, test], ignore_index=True)
    trv = pd.concat([train, val], ignore_index=True)
    y_test = test["target"].astype(int).values

    cv_splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                     .split(full, full["target"]))
    sites = full["Site"].values
    loso_splits = [(np.where(sites != s)[0], np.where(sites == s)[0])
                   for s in sorted(np.unique(sites))]

    test_probs, summary_rows, site_tables = {}, [], []
    for name, cfg in VARIANTS.items():
        print(f"\n##### Variant: {name} ({len(cols_of(cfg))} inputs) #####")
        cols = cols_of(cfg)
        res, fitted = run_model_selection(train, val, cols, lambda c=cfg: make_preproc(c), cfg["xgb"])
        top3 = res.head(3)["model"].tolist()
        print("members:", top3)

        ens = fit_ensemble(top3, fitted, trv[cols], trv["target"].astype(int))
        test_probs[name] = ens.predict_proba(test[cols])[:, 1]
        m_test = metric_dict(y_test, test_probs[name])

        print("running 5-fold CV and leave-one-site-out ...")
        p_cv = oof_predictions(full, cols, top3, fitted, cv_splits)
        p_loso = oof_predictions(full, cols, top3, fitted, loso_splits)
        y_full = full["target"].astype(int).values
        m_cv, m_loso = metric_dict(y_full, p_cv), metric_dict(y_full, p_loso)

        for scheme, m in (("random split, test (n=190)", m_test),
                          ("5-fold CV, all rows (n=947)", m_cv),
                          ("leave-one-site-out, all rows (n=947)", m_loso)):
            summary_rows.append(dict(variant=name, scheme=scheme, **m))
        site_tables.append(per_site_table(full, p_loso, name))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "cv_vs_loso.csv", index=False)
    pd.concat(site_tables).to_csv(OUT / "loso_by_site.csv", index=False)

    ci, diff = paired_bootstrap(
        y_test, test_probs,
        pairs=[("A", "B"), ("A - IQ Measure - ADHD Measure", "B"),
               ("A", "A - IQ Measure - ADHD Measure")])
    ci.to_csv(OUT / "bootstrap_ci.csv", index=False)
    diff.to_csv(OUT / "bootstrap_differences.csv", index=False)

    # figure: pooled ROC-AUC and F1 by evaluation scheme
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    schemes = list(summary["scheme"].unique())
    short = ["random split\n(test)", "5-fold CV", "leave-one-\nsite-out"]
    w = 0.2
    for ax, metric, title in zip(axes, ["roc_auc", "f1"], ["ROC-AUC", "F1"]):
        for i, v in enumerate(VARIANTS):
            vals = [summary[(summary.variant == v) & (summary.scheme == s)][metric].iloc[0]
                    for s in schemes]
            ax.bar(np.arange(3) + (i - 1.5) * w, vals, w, label=v)
        ax.set_xticks(range(3))
        ax.set_xticklabels(short)
        ax.set_title(title)
        ax.set_ylim(0.4, 1.0)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "site_robustness.png", dpi=150)
    plt.close(fig)

    lines = ["=== Test-set metrics with paired-bootstrap 95% intervals ===",
             ci.round(4).to_string(index=False), "",
             "=== Paired differences ===", diff.round(4).to_string(index=False), "",
             "=== Random split vs 5-fold CV vs leave-one-site-out ===",
             summary.round(4).to_string(index=False), "",
             "=== Leave-one-site-out, per held-out site ===",
             pd.concat(site_tables).round(3).to_string(index=False)]
    (OUT / "robustness_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()