"""Tests for src/features.py (run: python -m pytest tests/)."""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import CATEGORICAL, FEATURE_SETS, NUMERIC, check_leakage, get_X


def _toy(n: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.choice(["A", "B"], n) for c in CATEGORICAL})
    for c in NUMERIC:
        df[c] = rng.integers(0, 3, n).astype(float)
    df["judge_id"] = ["J001", "J002", "other", "unseen", "J001", "J002"][:n]
    df[config.JUDGE_COL_FINAL] = ["J005", "J006", "J007", "J008", "J005", "J006"][:n]
    df.loc[0, "gender"] = np.nan
    df.loc[1, "top_arraign_attempt_indicator"] = np.nan
    df.loc[2, "prior_vfo_cnt"] = np.nan
    df["y_detain"] = 0
    df["race"] = "Black"
    return df


def test_feature_sets_are_nested_and_clean():
    a, b, c = (set(FEATURE_SETS[k]) for k in ["CASE", "CASE_COURT", "CASE_COURT_JUDGE"])
    assert a < b < c
    check_leakage(FEATURE_SETS["CASE_COURT_JUDGE"])  # must not raise


@pytest.mark.parametrize("col", ["race", "ethnicity", "representation_type", "y_detain",
                                 "release_decision_at_arraign", "judge_name", "y_anything"])
def test_check_leakage_rejects(col):
    with pytest.raises(ValueError):
        check_leakage(["age_at_arrest", col])


def test_get_X_columns_and_types():
    X = get_X(_toy(), "CASE_COURT_JUDGE")
    assert list(X.columns) == FEATURE_SETS["CASE_COURT_JUDGE"]
    assert "race" not in X.columns and "y_detain" not in X.columns
    for c in NUMERIC:
        assert X[c].dtype == "float64"
    for c in [c for c in X.columns if c not in NUMERIC]:
        assert X[c].map(type).eq(str).all()          # no NaN in categorical columns
    assert X.loc[0, "gender"] == "Missing"
    assert X.loc[1, "top_arraign_attempt_indicator"] == "Not attempt"
    assert np.isnan(X.loc[2, "prior_vfo_cnt"])        # numeric NaN kept for the pipeline


def test_get_X_final_uses_final_judge_codes():
    df = _toy()
    X = get_X(df, "CASE_COURT_JUDGE", final=True)
    assert X["judge_id"].tolist() == df[config.JUDGE_COL_FINAL].tolist()
    assert df["judge_id"].iloc[0] == "J001"           # input not modified


def test_get_X_unknown_set():
    with pytest.raises(KeyError):
        get_X(_toy(), "EVERYTHING")
