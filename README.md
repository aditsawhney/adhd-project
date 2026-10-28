# ADHD Classification & Fairness Audit

This project looks at ADHD classification on the ADHD-200 phenotypic dataset, and tries to answer three things: how much predictive value full clinical data adds over what's known before any clinical contact, how fair each approach is across sex and age, and how well either approach actually generalizes to a site it wasn't trained on.

The pipeline is leakage-audited, fully reproducible, and evaluates two feature sets side by side.

A prototype web form for the pre-assessment model, built separately, is live at https://neuronova-alpha.vercel.app (code in [aditsawhney/Neuronova](https://github.com/aditsawhney/Neuronova)). The model it serves is not the exact artefact evaluated in the paper.

## The two models

| | Model B, pre-assessment | Model A, extended |
|---|---|---|
| **Features** | Age, sex, handedness, Inattentive score, Hyper/Impulsive score (5 total) | Adds IQ scores, IQ instrument used, medication status, comorbid-diagnosis flag, and assessment instrument used (13 raw inputs, 23 after encoding) |
| **Represents** | What a referrer (school counsellor, GP, parent) knows before any clinical evaluation | Everything classifiable from the full recorded clinical picture |
| **Test F1 / ROC-AUC** | 0.892 / 0.948 | 0.909 / 0.972 |
| **Ensemble** | ExtraTrees, RandomForest, SVM (calibrated, soft-voted) | XGBoost, RandomForest, ExtraTrees (calibrated, soft-voted) |

Full metrics, fairness breakdowns and plots for both models are in `reports/`.

## Leakage audit

Before any model is trained, `src/neuronova/data.py` checks each field for whether it can carry the label or stand in for it. The diagnosis is the target, and `ScanDir ID` and `Site` are excluded because they identify the row or the recruiting site and say nothing about the child. `ADHD Index` is excluded from both models because on its own it predicts the diagnosis with an AUC of 0.93, which makes it close to a proxy for the label. All preprocessing is fitted on training data only, every model is scored on the same F1 against one fixed validation fold, probabilities are calibrated before ensembling, and the classification threshold is the standard 0.5.

## The main finding: cross-site generalization

Random splits and 5-fold cross-validation both mix all seven acquisition sites into every train/test partition, so they only measure performance on sites the model has already seen data from. Leave-one-site-out evaluation, training on six sites and testing on the seventh, tells a different story:

| Evaluation | Model B ROC-AUC | Model A ROC-AUC |
|---|---:|---:|
| Random split | 0.948 | 0.972 |
| 5-fold CV | 0.949 | 0.975 |
| Leave-one-site-out | 0.757 | 0.764 |

Two sites fail outright, and for two different reasons. Site 1 (245 participants) used a different behavioral rating instrument, scored 9 to 36, than the other six sites, which used instruments scored 40 to 90, and both ranges end up in the same column. A model trained mostly on the 40-90 range reads site 1's scores as uniformly low risk, and recall there drops to about 0.02. Site 4 (73 participants) has no subscore data at all. When it's held out, every participant gets the same median-imputed value for what's normally the most important feature, so the model just predicts the majority class and recall goes to zero.

We also checked whether Model A's advantage comes from fields that mostly encode acquisition site (`IQ Measure`, `ADHD Measure`). Removing them doesn't meaningfully change performance, so that's not what's driving the gap. But the underlying generalization problem is real regardless of which feature set is used.

Full numbers, the per-site breakdown, and bootstrap confidence intervals are in `reports/robustness/`.

## Model selection was tested, not assumed

Two design choices, excluding XGBoost from Model B and soft-voting three models instead of using one, were each tested rather than assumed. Every check ran across validation, held-out test, 5-fold CV, and leave-one-site-out.

XGBoost for Model B (`src/neuronova/xgboost_check.py`) wins the validation fold, F1 of 0.889 against the retained roster's 0.870, but loses on every other evaluation, most clearly on leave-one-site-out ROC-AUC (0.712 vs. 0.757). That's the standard pattern of overfitting to the fold it was selected against, so excluding it was the right call, and now there's evidence behind it instead of an assumption.

Ensemble versus solo best model (`src/neuronova/ensemble_check.py`) showed something similar. For Model B the ensemble wins on every metric. For Model A, a solo XGBoost model actually wins on held-out test F1 (0.924 vs. the ensemble's 0.909), but loses badly on leave-one-site-out ROC-AUC (0.711 vs. 0.764). The same pattern, winning in-distribution and losing cross-site, shows up independently in both checks.

Full numbers: `reports/model_b/xgboost_check.csv` and `reports/ensemble_vs_solo.csv`.

## Repo structure

```
adhd-project/
├── data/
│   ├── raw/                       # original CSV, untouched by code
│   └── processed/                  # train/val/test splits (generated)
├── src/neuronova/
│   ├── data.py                     # cleaning + Phase 0 leakage/redundancy audit
│   ├── features.py                  # both preprocessors + the locked 60/20/20 split
│   ├── train.py                     # model selection (8 classifiers) + calibrated ensembles
│   ├── evaluate.py                  # test-set metrics, plots, permutation importance, fairness audit
│   ├── shap_analysis.py             # SHAP check for both models, each on its own top-ranked tree model
│   ├── robustness.py                # bootstrap CIs, ablations, 5-fold CV vs. leave-one-site-out
│   ├── xgboost_check.py             # tests whether XGBoost actually helps Model B (it doesn't)
│   └── ensemble_check.py            # tests soft-voted ensemble vs. solo best model, both feature sets
├── tests/
│   └── test_pipeline.py             # guards against leakage, bad splits, mis-coded fields
├── models/                         # trained .joblib artifacts (generated)
├── reports/                        # metrics, plots, audit text (generated)
│   ├── model_a/                     # includes shap_summary.png
│   ├── model_b/                     # includes shap_summary.png, xgboost_check.csv
│   ├── robustness/                  # bootstrap CIs, LOSO results, site-level breakdown
│   ├── comparison_report.csv        # the A-vs-B headline comparison
│   └── ensemble_vs_solo.csv         # ensemble vs. solo-best-model generalization check
├── pyproject.toml
└── uv.lock
```

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.12
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

Create a `.env` file in the repo root:
```
PYTHONPATH=src
```

If you're on macOS and hit a `libomp.dylib` load error, run `brew install libomp`. XGBoost's wheel needs the OpenMP runtime, which doesn't come bundled on macOS.

## Running the pipeline

Run these in order, since each phase depends on the previous one's output:

```bash
uv run --env-file .env python src/neuronova/data.py        # Phase 0: clean + audit
uv run --env-file .env python src/neuronova/features.py    # Phase 1: locked split + preprocessors
uv run --env-file .env python src/neuronova/train.py        # Phase 2: model selection + calibration
uv run --env-file .env python src/neuronova/evaluate.py     # Phase 4: test metrics + fairness audit
uv run --env-file .env python src/neuronova/shap_analysis.py   # Phase 6: SHAP, both models
uv run --env-file .env python src/neuronova/robustness.py      # bootstrap CIs + leave-one-site-out
uv run --env-file .env python src/neuronova/xgboost_check.py   # is XGBoost worth adding to Model B?
uv run --env-file .env python src/neuronova/ensemble_check.py  # is the ensemble worth it vs. solo?
```

`features.py`, `train.py`, `evaluate.py`, `robustness.py`, `xgboost_check.py`, and `ensemble_check.py` all regenerate their outputs under `data/processed/`, `models/`, and `reports/`, so it's safe to rerun any of them from scratch. The split is deterministic (`random_state=42` throughout), so results should match across machines.

To run the guard tests (leakage checks, split integrity, correct sex coding):
```bash
uv run --env-file .env python -m pytest -q tests
```

## Fairness audit

Both models perform worst in the 14+ age band (Model A F1 of 0.828, Model B F1 of 0.774), which lines up with the Conners behavioral subscales being normed mostly on younger children. Model A also shows a wider sex-based F1 gap than Model B (0.102 vs. 0.031 difference), but the female subgroup in the test set is only 72 people, so this needs replication before we'd call it a stable finding. Details are in `reports/model_a/audit_report.txt` and `reports/model_b/audit_report.txt`.

## Dataset

ADHD-200 Consortium, *ADHD-200 Sample*, The International Neuroimaging Data-Sharing Initiative (INDI), 2012. http://fcon1000.projects.nitrc.org/indi/adhd200/

## Authorship

All the code, modeling, and analysis here, the leakage audit, the dual feature-set design, the calibrated ensembles, the fairness audit, and the cross-site robustness work, was built independently. This repo is the technical basis for an in-progress research paper. Academic collaborators are contributing to the writing, framing, and review of that paper, but the codebase and results here are solely authored.