"""
Guards against the problems found in the earlier notebook: label-derived or
identifier columns among the inputs, preprocessing fitted on held-out rows,
overlapping splits, and mis-coded sex. Run from the repo root: pytest -q
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "neuronova"))

from data import clean, load_raw  # noqa: E402
from features import (  # noqa: E402
    MODEL_A_BINARY,
    MODEL_A_CATEGORICAL,
    MODEL_A_NUMERIC,
    MODEL_B_CATEGORICAL,
    MODEL_B_NUMERIC,
    build_preprocessor_a,
    build_preprocessor_b,
    locked_split,
)

FORBIDDEN = {"DX", "target", "ScanDir ID", "Site", "ADHD Index",
             "QC_Athena", "QC_NIAK", "Secondary Dx"}
FEATURES_A = MODEL_A_NUMERIC + MODEL_A_CATEGORICAL + MODEL_A_BINARY
FEATURES_B = MODEL_B_NUMERIC + MODEL_B_CATEGORICAL


@pytest.fixture(scope="module")
def data():
    cwd = Path.cwd()
    import os
    os.chdir(ROOT)
    try:
        df = clean(load_raw())
    finally:
        os.chdir(cwd)
    return df.dropna(subset=["target"]).reset_index(drop=True)


def test_no_forbidden_inputs():
    for cols in (FEATURES_A, FEATURES_B):
        assert not FORBIDDEN & set(cols)
        assert not any(c.startswith("DX") for c in cols)


def test_model_b_is_subset_of_model_a():
    assert set(FEATURES_B) <= set(FEATURES_A)


def test_target_counts(data):
    assert len(data) == 947
    assert int(data["target"].sum()) == 362
    assert set(data["target"].unique()) == {0.0, 1.0}


def test_split_is_disjoint_and_stratified(data):
    train, val, test = locked_split(data)
    ids = [set(d["ScanDir ID"]) for d in (train, val, test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert sum(map(len, ids)) == len(data)
    assert (len(train), len(val), len(test)) == (567, 190, 190)
    overall = data["target"].mean()
    for part in (train, val, test):
        assert abs(part["target"].mean() - overall) < 0.01


def test_preprocessor_uses_training_rows_only(data):
    train, _, test = locked_split(data)
    pre = build_preprocessor_b().fit(train[FEATURES_B])
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    age_idx = MODEL_B_NUMERIC.index("Age")
    assert scaler.mean_[age_idx] == pytest.approx(train["Age"].mean())
    assert scaler.mean_[age_idx] != pytest.approx(data["Age"].mean(), abs=1e-9)


def test_output_width(data):
    train, _, _ = locked_split(data)
    assert build_preprocessor_b().fit_transform(train[FEATURES_B]).shape[1] == 8
    assert build_preprocessor_a().fit_transform(train[FEATURES_A]).shape[1] == 23


def test_sex_coding_matches_dataset_key(data):
    """Dataset key: 0 = female, 1 = male. Boys are the majority and have the
    higher ADHD rate in ADHD-200, which this also checks."""
    counts = data["Gender"].value_counts()
    assert counts["Male"] > counts["Female"]
    rate = data.groupby("Gender")["target"].mean()
    assert rate["Male"] > rate["Female"]


@pytest.mark.parametrize("name,cols", [("model_a", FEATURES_A), ("model_b", FEATURES_B)])
def test_saved_models_predict_valid_probabilities(name, cols):
    path = ROOT / "models" / f"{name}.joblib"
    test_csv = ROOT / "data" / "processed" / "test.csv"
    if not (path.exists() and test_csv.exists()):
        pytest.skip("run train.py first")
    model = joblib.load(path)
    test = pd.read_csv(test_csv)
    p = model.predict_proba(test[cols])[:, 1]
    assert p.shape == (len(test),)
    assert np.all((p >= 0) & (p <= 1))
    row = test[cols].iloc[[0]].copy()
    row["Handedness"] = "Ambidextrous-unseen"
    assert 0 <= model.predict_proba(row)[0, 1] <= 1