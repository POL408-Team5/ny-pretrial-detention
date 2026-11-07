"""Feature sets, leakage check and model-ready feature frames (role A).

The three nested feature sets are defined in config.FEATURE_SETS:
    CASE              charge, prior record, pending cases, supervision, prior FTA, age, gender
    CASE_COURT        + county
    CASE_COURT_JUDGE  + anonymised judge

Public names (AGENTS.md section 3.5):
    FEATURE_SETS, NUMERIC, CATEGORICAL
    check_leakage(cols)
    get_X(df, feature_set, final=False)
"""
from __future__ import annotations

import pandas as pd

from src import config

FEATURE_SETS: dict[str, list[str]] = config.FEATURE_SETS

# Counts and 0/1 indicators are numeric; everything else is one-hot encoded downstream.
NUMERIC: list[str] = [
    "prior_vfo_cnt", "prior_nonvfo_cnt", "prior_misd_cnt",
    "pend_vfo", "pend_nonvfo", "pend_misd", "supervision",
    "dcjs_missing", "age_at_arrest",
]
ALL_FEATURES: list[str] = list(dict.fromkeys(f for cols in FEATURE_SETS.values() for f in cols))
CATEGORICAL: list[str] = [c for c in ALL_FEATURES if c not in NUMERIC]

MISSING_LABEL = "Missing"
# Columns where a missing value has a substantive meaning (data dictionary): blank = not an attempt.
FILL_VALUES = {"top_arraign_attempt_indicator": "Not attempt"}


def check_leakage(cols: list[str]) -> None:
    """Raise ValueError if any column may not be used as a feature.

    Forbidden: anything in config.LEAKAGE_BLOCKLIST, any column starting with "y_",
    and the fairness grouping variables (config.FAIRNESS_GROUPS).
    """
    blocked = set(config.LEAKAGE_BLOCKLIST) | set(config.FAIRNESS_GROUPS)
    bad = [c for c in cols if c in blocked or c.startswith("y_")]
    if bad:
        raise ValueError(f"Columns not allowed as features (leakage or fairness-only): {bad}")


def get_X(df: pd.DataFrame, feature_set: str, final: bool = False) -> pd.DataFrame:
    """Return the feature columns of `feature_set`, ready for a sklearn preprocessor.

    - Categorical columns: strings; missing -> "Missing" (or a meaningful fill, see FILL_VALUES).
    - Numeric columns: float; missing values are kept as NaN (impute inside the model pipeline,
      fitted on training data only). `dcjs_missing` marks rows whose prior-record fields are missing.
    - final=True uses config.JUDGE_COL_FINAL (coded on 2021-2023) in place of judge_id, keeping
      the column name "judge_id" so that a fitted pipeline sees the same columns.
    """
    if feature_set not in FEATURE_SETS:
        raise KeyError(f"Unknown feature set {feature_set!r}; choose from {list(FEATURE_SETS)}")
    cols = FEATURE_SETS[feature_set]
    check_leakage(cols)

    source = df.copy() if final and "judge_id" in cols else df
    if final and "judge_id" in cols:
        source["judge_id"] = df[config.JUDGE_COL_FINAL]

    missing_cols = [c for c in cols if c not in source.columns]
    if missing_cols:
        raise KeyError(f"Columns missing from the data: {missing_cols}")

    X = pd.DataFrame(index=source.index)
    for c in cols:
        if c in NUMERIC:
            X[c] = pd.to_numeric(source[c], errors="coerce").astype("float64")
        else:
            fill = FILL_VALUES.get(c, MISSING_LABEL)
            X[c] = source[c].astype("object").where(source[c].notna(), fill).astype(str)
    return X
