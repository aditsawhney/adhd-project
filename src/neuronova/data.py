"""
Phase 0: load + clean the ADHD-200 phenotypic dataset, and audit the
extended (Model A) feature set for leakage/redundancy before anything
gets trained on it.
"""
import numpy as np
import pandas as pd

RAW_PATH = "data/raw/adhd200_preprocessed_phenotypics.csv"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # DX: mixed numeric + "pending" strings -> coerce, drop pending
    df["DX"] = pd.to_numeric(df["DX"], errors="coerce")

    # Handedness: EHI floats (-1..1) mixed with site categorical codes (0/1/2/L)
    handedness_map = {"0": "-1.0", "1": "1.0", "2": "0.0", "L": "-1.0"}
    df["Handedness"] = df["Handedness"].replace(handedness_map)
    df["Handedness"] = pd.to_numeric(df["Handedness"], errors="coerce")

    def bin_handedness(v):
        if pd.isna(v):
            return np.nan
        if v < -0.4:
            return "Left"
        if v > 0.4:
            return "Right"
        return "Mixed"

    df["Handedness"] = df["Handedness"].apply(bin_handedness)

    # Inattentive / Hyper-Impulsive: strings at some sites, -999 sentinel
    for col in ["Inattentive", "Hyper/Impulsive", "ADHD Index"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] == -999, col] = np.nan

    # IQ Measure: -999 sentinel, same pattern
    df["IQ Measure"] = pd.to_numeric(df["IQ Measure"], errors="coerce")
    df.loc[df["IQ Measure"] == -999, "IQ Measure"] = np.nan

    # Med Status / ADHD Measure: string columns with '-999' / 'pending' tokens
    # mixed in with real NaN -> normalize all to NaN
    missing_tokens = {"-999", "pending", "nan", "none", ""}

    def _normalize_missing(v):
        if pd.isna(v):
            return np.nan
        v = str(v).strip()
        return np.nan if v.lower() in missing_tokens else v

    for col in ["Med Status", "ADHD Measure"]:
        df[col] = df[col].apply(_normalize_missing)

    # Secondary Dx: 84% missing, 67 near-unique free-text categories among the
    # rest -> not usable as a categorical feature (would be one-hot noise).
    # Collapse to a binary comorbidity flag instead.
    df["has_secondary_dx"] = df["Secondary Dx"].notna().astype(int)

    # Gender: numeric 0/1 -> semantically categorical
    df["Gender"] = df["Gender"].map({0.0: "Female", 1.0: "Male"})

    # Target
    df["target"] = df["DX"].apply(lambda x: np.nan if pd.isna(x) else (0 if x == 0 else 1))

    return df


def audit_extended_features(df: pd.DataFrame) -> dict:
    """
    Phase 0 required check before Model A is allowed to use ADHD Measure /
    ADHD Index: is either of these a near-proxy for the label?
    """
    results = {}

    # Drop pending/NaN target rows for the audit, same as the modeling set will
    audited = df.dropna(subset=["target"]).copy()

    # 1. ADHD Measure missingness vs target
    audited["measure_missing"] = audited["ADHD Measure"].isna() | (
        audited["ADHD Measure"].astype(str).str.strip().str.lower().isin(["nan", "none", ""])
    )
    crosstab = pd.crosstab(audited["measure_missing"], audited["target"], normalize="index")
    results["adhd_measure_missingness_by_target"] = crosstab
    results["adhd_measure_missing_rate_control"] = crosstab.loc[True, 0] if True in crosstab.index else None
    results["adhd_measure_missing_rate_adhd"] = crosstab.loc[True, 1] if True in crosstab.index else None

    # 2. ADHD Index correlation with target (point-biserial ~ pearson on binary target)
    valid = audited.dropna(subset=["ADHD Index", "target"])
    results["adhd_index_n_valid"] = len(valid)
    results["adhd_index_corr_with_target"] = valid["ADHD Index"].corr(valid["target"])

    # 3. ADHD Index correlation with the subscales already in the 5-feature set
    #    (redundancy check, separate from the leakage check)
    valid2 = audited.dropna(subset=["ADHD Index", "Inattentive", "Hyper/Impulsive"])
    results["adhd_index_n_valid_for_redundancy"] = len(valid2)
    results["adhd_index_corr_with_inattentive"] = valid2["ADHD Index"].corr(valid2["Inattentive"])
    results["adhd_index_corr_with_hyperimpulsive"] = valid2["ADHD Index"].corr(valid2["Hyper/Impulsive"])

    # 4. Simple separability check: can ADHD Index alone split classes near-perfectly?
    #    (AUC of ADHD Index as a lone predictor of target)
    from sklearn.metrics import roc_auc_score
    if valid["ADHD Index"].nunique() > 1:
        results["adhd_index_solo_auc"] = roc_auc_score(valid["target"], valid["ADHD Index"])
    else:
        results["adhd_index_solo_auc"] = None

    return results


if __name__ == "__main__":
    raw = load_raw()
    cleaned = clean(raw)
    audit = audit_extended_features(cleaned)

    print("=== Phase 0 audit: ADHD Measure / ADHD Index leakage & redundancy check ===\n")
    print("ADHD Measure missingness by target class:")
    print(audit["adhd_measure_missingness_by_target"])
    print()
    print(f"ADHD Index -- target correlation (n={audit['adhd_index_n_valid']}): "
          f"{audit['adhd_index_corr_with_target']:.4f}")
    print(f"ADHD Index solo AUC as a predictor of target: {audit['adhd_index_solo_auc']:.4f}")
    print()
    print(f"ADHD Index -- Inattentive correlation (n={audit['adhd_index_n_valid_for_redundancy']}): "
          f"{audit['adhd_index_corr_with_inattentive']:.4f}")
    print(f"ADHD Index -- Hyper/Impulsive correlation: "
          f"{audit['adhd_index_corr_with_hyperimpulsive']:.4f}")

    cleaned.to_csv("data/processed/cleaned_full.csv", index=False)