"""
Phase 1: leakage-safe preprocessing for both feature sets.

Model B = pre-assessment-only (what's live in the deployed app)
Model A = extended clinical/phenotypic set, minus everything that's
          label-derived (DX_3, ScanDir ID, DX, Site) or a near-proxy
          for the label (ADHD Index — see reports/phase0 audit)
"""
import sys
from pathlib import Path

# make sibling modules importable regardless of how uv/PYTHONPATH/the
# editable install resolve (or fail to resolve) the 'neuronova' package
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_B_NUMERIC = ["Age", "Inattentive", "Hyper/Impulsive"]
MODEL_B_CATEGORICAL = ["Gender", "Handedness"]

MODEL_A_NUMERIC = [
    "Age", "Inattentive", "Hyper/Impulsive",
    "Verbal IQ", "Performance IQ", "Full2 IQ", "Full4 IQ",
]
MODEL_A_CATEGORICAL = [
    "Gender", "Handedness", "IQ Measure", "Med Status", "ADHD Measure",
]
MODEL_A_BINARY = ["has_secondary_dx"]  # already 0/1, no encoding needed


def _numeric_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])


def _categorical_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])


def build_preprocessor_b() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", _numeric_pipeline(), MODEL_B_NUMERIC),
        ("cat", _categorical_pipeline(), MODEL_B_CATEGORICAL),
    ])


def build_preprocessor_a() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", _numeric_pipeline(), MODEL_A_NUMERIC),
        ("cat", _categorical_pipeline(), MODEL_A_CATEGORICAL),
        ("bin", "passthrough", MODEL_A_BINARY),
    ])


def locked_split(df, target_col="target", test_size=0.2, val_size=0.2, random_state=42):
    """
    One stratified 60/20/20 split, shared by Model A and Model B so their
    results are directly comparable on the exact same rows. Split on rows
    only — feature selection happens downstream via the two preprocessors.
    """
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[target_col], random_state=random_state
    )
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val,
        stratify=train_val[target_col], random_state=random_state,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


if __name__ == "__main__":
    import pandas as pd
    from data import clean, load_raw

    df = clean(load_raw())
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    train, val, test = locked_split(df)
    print(f"train={len(train)} val={len(val)} test={len(test)}")
    print("class balance (train):", train["target"].value_counts(normalize=True).to_dict())
    print("class balance (val):  ", val["target"].value_counts(normalize=True).to_dict())
    print("class balance (test): ", test["target"].value_counts(normalize=True).to_dict())

    train.to_csv("data/processed/train.csv", index=False)
    val.to_csv("data/processed/val.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)

    # sanity: fit both preprocessors on train only, transform val, check shapes
    pre_a = build_preprocessor_a()
    pre_b = build_preprocessor_b()
    Xa_train = pre_a.fit_transform(train[MODEL_A_NUMERIC + MODEL_A_CATEGORICAL + MODEL_A_BINARY])
    Xb_train = pre_b.fit_transform(train[MODEL_B_NUMERIC + MODEL_B_CATEGORICAL])
    print(f"Model A transformed shape: {Xa_train.shape}")
    print(f"Model B transformed shape: {Xb_train.shape}")