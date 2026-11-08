"""C-01 checks: temporal boundary, small cells and hand-calculated adjustment."""
import pandas as pd
import pytest

from src import config
from src.fairness import (
    county_rates, coverage_tables, judge_percentiles, judge_rates,
    load_court_judge_inputs,
)


@pytest.fixture
def training(monkeypatch):
    monkeypatch.setattr(config, "MIN_GROUP_N", 2)
    monkeypatch.setattr(config, "MIN_JUDGE_CASES", 2)
    return pd.DataFrame({
        config.YEAR_COL: [config.TRAIN_YEARS[0]] * 6,
        "county_name": ["New York"] * 6,
        "judge_id": ["J001"] * 2 + ["J002"] * 4,
        "top_severity_at_arraign": ["Misdemeanor"] * 4 + ["Felony"] * 2,
        "top_charge_weight_at_arraign": ["AM"] * 4 + ["BF"] * 2,
        "top_charge_at_arraign_violent_felony_ind": ["N"] * 4 + ["Y"] * 2,
        config.TARGET_Q1: [1, 0, 0, 0, 1, 1],
    })


def test_indirect_standardization_by_hand(training):
    result = judge_rates(training)
    overall = result.loc[result["severity"].eq("All")].set_index("judge_id")
    # Low-weight reference = 1/4; high-weight reference = 2/2.
    assert overall.loc["J001", "expected_rate"] == pytest.approx(.25)
    assert overall.loc["J001", "observed_minus_expected"] == pytest.approx(.25)
    assert overall.loc["J002", "expected_rate"] == pytest.approx(.625)
    assert overall.loc["J002", "observed_minus_expected"] == pytest.approx(-.125)
    # Weighted residuals sum to zero because these two judges cover the reference.
    assert (overall["n"] * overall["observed_minus_expected"]).sum() == pytest.approx(0)


def test_small_subgroups_suppressed_and_collapsed_judges_excluded(training):
    training.loc[0, "county_name"] = "Kings"
    assert "Kings" not in set(county_rates(training)["county_name"])
    training.loc[0, "top_severity_at_arraign"] = "Felony"
    training.loc[2:5, "judge_id"] = "other"
    result = judge_rates(training)
    assert set(result["judge_id"]) == {"J001"}
    assert set(result["severity"]) == {"All"}


def test_percentiles_are_unweighted_across_judges(training):
    result = judge_percentiles(judge_rates(training)).set_index("severity")
    assert result.loc["All", "n_judges"] == 2
    assert result.loc["All", "observed_minus_expected_p50"] == pytest.approx(.0625)


@pytest.mark.parametrize("function", [county_rates, judge_rates])
def test_outcome_eda_rejects_future_years(training, function):
    training.loc[0, config.YEAR_COL] = config.TEST_YEAR
    with pytest.raises(ValueError, match="training-year"):
        function(training)


def test_coverage_rejects_any_outcome_column(training):
    with pytest.raises(ValueError, match="outcome-free"):
        coverage_tables(training)


def test_parquet_projection_excludes_future_outcomes(monkeypatch, training):
    calls = []
    def fake_read(path, columns, filters):
        calls.append((columns, filters))
        if config.TARGET_Q1 not in columns:
            return pd.DataFrame({config.ID_COL: ["a", "b"]})
        return training
    monkeypatch.setattr(pd, "read_parquet", fake_read)
    load_court_judge_inputs()
    assert config.TARGET_Q1 not in calls[0][0]
    assert (config.YEAR_COL, "in", config.TRAIN_YEARS) in calls[1][1]


def test_coverage_denominator_before_small_cell_suppression(monkeypatch):
    monkeypatch.setattr(config, "MIN_GROUP_N", 2)
    metadata = pd.DataFrame({
        config.YEAR_COL: [config.TEST_YEAR] * 4,
        "arraign_month": [1] * 4,
        "judge_id": ["unseen", "unseen", "J001", "other"],
        config.JUDGE_COL_FINAL: ["unseen", "J001", "J001", "other"],
    })
    annual, _ = coverage_tables(metadata)
    row = annual.loc[annual["coding"].eq("judge_id")].iloc[0]
    assert row["n"] == 2
    assert row["n_total"] == 4
    assert row["case_share"] == pytest.approx(.5)
