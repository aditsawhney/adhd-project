# ADHD Classification & Fairness Audit

Research project on ADHD classification using the ADHD-200 phenotypic
dataset, built around a specific question: **how much predictive value
does full clinical/phenotypic data add over what's known before any
clinical contact occurs**, and **how fair is each approach across sex and
age**?

This is the corrected, dual-model successor to an earlier team notebook
that — as documented below — had label leakage. The pipeline here is
leakage-safe, reproducible end to end, and evaluates two feature sets
side by side rather than one.

## The two models

| | Model B — pre-assessment | Model A — extended |
|---|---|---|
| **Features** | Age, Sex, Handedness, Inattentive score, Hyper/Impulsive score (5) | + IQ scores, IQ instrument used, medication status, comorbid-diagnosis flag, ADHD assessment instrument used (13 raw → 23 post-encoding) |
| **Represents** | What a referrer (school counsellor, GP, parent) knows before any clinical evaluation | What's classifiable using the full recorded phenotypic/clinical picture |
| **Test F1 / ROC-AUC** | 0.892 / 0.948 | 0.909 / 0.972 |
| **Ensemble** | ExtraTrees + RandomForest + SVM (calibrated, soft-voted) | XGBoost + RandomForest + ExtraTrees (calibrated, soft-voted) |

Full metrics, fairness breakdowns, and plots for both are in `reports/`.

## What changed from the original notebook

The original team implementation used `DX_3` (a one-hot slice of the
diagnosis label itself) and `ScanDir ID` (a row identifier) as model
inputs — both leak the label rather than predict it. It also mixed
scoring metrics across models (SVM on F1, RF/XGBoost on accuracy),
applied no probability calibration, and used an unjustified 0.3
classification threshold. `ADHD Index`, while not leaked in the same
mechanical sense, was found (Phase 0 audit, below) to be a near-proxy for
the label (AUC=0.93 alone) and is excluded from both models here.

This rebuild fixes all of the above — see `src/neuronova/data.py`'s
Phase 0 audit for the actual leakage/redundancy numbers, and
`src/neuronova/train.py` for the calibration and scoring fixes.

## Repo structure

```
adhd-project/
├── data/
│   ├── raw/                     # original CSV, untouched by code
│   └── processed/                # train/val/test splits (generated)
├── src/neuronova/
│   ├── data.py                   # cleaning + Phase 0 leakage/redundancy audit
│   ├── features.py                # both preprocessors + the locked 60/20/20 split
│   ├── train.py                   # model selection (8 classifiers) + calibrated ensembles
│   ├── evaluate.py                # test-set metrics, plots, permutation importance, fairness audit
│   └── shap_analysis.py           # secondary SHAP check on a standalone tree model
├── models/                       # trained .joblib artifacts (generated)
├── reports/                      # metrics, plots, audit text (generated)
│   ├── model_a/
│   ├── model_b/
│   └── comparison_report.csv     # the A-vs-B headline comparison
├── notebooks/                    # original team notebook (reference)
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

**macOS + XGBoost:** if you hit a `libomp.dylib` load error, run
`brew install libomp` — XGBoost's wheel needs the OpenMP runtime, which
isn't bundled on macOS.

## Running the pipeline

Run in order — each phase depends on the previous one's output:

```bash
uv run --env-file .env python src/neuronova/data.py       # Phase 0: clean + audit
uv run --env-file .env python src/neuronova/features.py   # Phase 1: locked split + preprocessors
uv run --env-file .env python src/neuronova/train.py       # Phase 2: model selection + calibration
uv run --env-file .env python src/neuronova/evaluate.py    # Phase 4: test metrics + fairness audit
uv run --env-file .env python src/neuronova/shap_analysis.py  # Phase 6: SHAP secondary check
```

`features.py`, `train.py`, and `evaluate.py` regenerate everything in
`data/processed/`, `models/`, and `reports/` — safe to re-run from
scratch at any time; the split is deterministic (`random_state=42`
throughout).

## Fairness audit — headline findings

- Both models perform weakest on the 14+ age band (Model A F1=0.828,
  Model B F1=0.774), consistent with the Conners behavioral subscales
  being normed predominantly on younger children.
- Model A shows a wider sex-based F1 gap than Model B (ΔF1=0.102 vs.
  0.031); given the small female subgroup (N=72) this needs replication
  before being treated as a stable finding — see `reports/model_a/audit_report.txt`.

Full breakdowns: `reports/model_a/audit_report.txt`, `reports/model_b/audit_report.txt`.

## Dataset

ADHD-200 Consortium, *ADHD-200 Sample*, The International Neuroimaging
Data-Sharing Initiative (INDI), 2012.
http://fcon1000.projects.nitrc.org/indi/adhd200/
