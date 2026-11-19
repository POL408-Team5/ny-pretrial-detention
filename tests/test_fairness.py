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


# ---------------------------------------------------------------------------
# C-02 checks (hand-calculated)
# ---------------------------------------------------------------------------
import numpy as np

from src.fairness import choose_threshold, group_metrics, predict_label, residuals


def _preds(y, p, ids=None, model="logit", fs="CASE"):
    ids = ids or [f"c{i}" for i in range(len(y))]
    return pd.DataFrame({
        config.ID_COL: ids, config.YEAR_COL: config.VALID_YEAR, "model": model,
        "feature_set": fs, "target": config.TARGET_Q1, "split": "valid",
        "y_true": y, "y_prob": p,
    })


def test_threshold_matches_actual_share():
    # 2 of 5 actual positives -> the two highest scores are predicted 1.
    y = [1, 0, 1, 0, 0]
    p = [0.9, 0.8, 0.7, 0.2, 0.1]
    t = choose_threshold(p, y)
    assert t == 0.8
    assert predict_label(p, t).tolist() == [1, 1, 0, 0, 0]


def test_threshold_ties_pick_closest_share():
    # k = 2, but scores 0.8 are tied three times: >=0.8 gives 4 positives (gap 2),
    # >=0.9 gives 1 (gap 1) -> 0.9 is closer.
    y = [1, 1, 0, 0, 0]
    p = [0.9, 0.8, 0.8, 0.8, 0.1]
    assert choose_threshold(p, y) == 0.9
    # Exact tie in the gap: prefer the higher threshold. k = 2; >=0.9 -> 1, >=0.8 -> 3.
    y2 = [1, 1, 0, 0]
    p2 = [0.9, 0.8, 0.8, 0.1]
    assert choose_threshold(p2, y2) == 0.9


def test_group_metrics_fpr_fnr_by_hand(monkeypatch):
    monkeypatch.setattr(config, "MIN_GROUP_N", 2)
    y = [1, 1, 0, 0, 1, 0, 0, 0]
    p = [0.9, 0.3, 0.6, 0.1, 0.8, 0.7, 0.2, 0.1]
    pred = _preds(y, p)
    meta = pd.DataFrame({config.ID_COL: pred[config.ID_COL], "race": ["A"] * 4 + ["B"] * 4})
    out = group_metrics(pred, meta, "race", threshold=0.5).set_index("group")
    # A: yhat = 1,0,1,0 -> FP 1 of 2 negatives, FN 1 of 2 positives
    assert out.loc["A", "fpr"] == 0.5 and out.loc["A", "fnr"] == 0.5
    # B: y = 1,0,0,0, yhat = 1,1,0,0 -> FP 1/3; only one positive (< MIN_GROUP_N) -> FNR, AUC blank
    assert out.loc["B", "fpr"] == 1 / 3
    assert np.isnan(out.loc["B", "fnr"]) and np.isnan(out.loc["B", "roc_auc"])
    assert out.loc["A", "calibration_gap"] == np.mean([0.9, 0.3, 0.6, 0.1]) - 0.5


def test_group_metrics_suppresses_small_groups(monkeypatch):
    monkeypatch.setattr(config, "MIN_GROUP_N", 3)
    pred = _preds([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    meta = pd.DataFrame({config.ID_COL: pred[config.ID_COL], "race": ["A", "A", "A", None]})
    out = group_metrics(pred, meta, "race", threshold=0.5)
    assert out["group"].tolist() == ["A"]          # "Missing" has 1 case -> dropped


def test_group_metrics_rejects_mixed_specs():
    pred = pd.concat([_preds([1, 0], [0.9, 0.1]), _preds([1, 0], [0.8, 0.2], model="xgb")])
    meta = pd.DataFrame({config.ID_COL: ["c0", "c1"], "race": ["A", "A"]})
    with pytest.raises(ValueError):
        group_metrics(pred, meta, "race", threshold=0.5)


def test_residual_sign_and_interval(monkeypatch):
    monkeypatch.setattr(config, "MIN_GROUP_N", 2)
    pred = _preds([1, 1, 0, 0], [0.5, 0.5, 0.5, 0.5])
    meta = pd.DataFrame({config.ID_COL: pred[config.ID_COL], "county_name": ["X", "X", "Y", "Y"]})
    out = residuals(pred, meta, "county_name", n_boot=200, seed=1).set_index("group")
    assert out.loc["X", "residual"] == 0.5 and out.loc["Y", "residual"] == -0.5
    assert out.loc["X", "ci_low"] == out.loc["X", "ci_high"] == 0.5   # no variation within group


# --- C-05 (2026-11-19): one number checked by hand, as the Week 8 notebook asks ---

def test_disparity_rates_by_hand(monkeypatch):
    from src.fairness import disparity_rates, rate_comparison
    monkeypatch.setattr(config, "MIN_GROUP_N", 2)
    pred = _preds([1, 0, 1, 0, 1, 0, 0, 0], [0.9, 0.6, 0.3, 0.1, 0.8, 0.7, 0.2, 0.1])
    meta = pd.DataFrame({config.ID_COL: pred[config.ID_COL], "race": list("AAAABBBB")})
    out = disparity_rates(pred, meta, "race", threshold=0.5).set_index("group")
    # A: y 1,0,1,0 / yhat 1,1,0,0 -> TP 1 FP 1 FN 1 TN 1
    assert (out.loc["A", ["TP", "FP", "FN", "TN"]] == [1, 1, 1, 1]).all()
    assert out.loc["A", "flag_rate"] == 0.5 and out.loc["A", "fpr"] == 0.5 and out.loc["A", "precision"] == 0.5
    # B: y 1,0,0,0 / yhat 1,1,0,0 -> TP 1 FP 1 FN 0 TN 2 -> FPR 1/3; TPR blank (1 positive < MIN_GROUP_N)
    assert out.loc["B", "fpr"] == pytest.approx(1 / 3) and np.isnan(out.loc["B", "tpr"])
    cmp = rate_comparison(out.reset_index(), "A", "B").set_index("rate")
    assert cmp.loc["fpr", "gap"] == pytest.approx(0.5 - 1 / 3)
    assert cmp.loc["fpr", "ratio"] == pytest.approx((1 / 3) / 0.5)
    assert cmp.loc["flag_rate", "four_fifths"] == True   # both 0.5 -> ratio 1


def test_equalizing_threshold_reaches_target(monkeypatch):
    from src.fairness import equalizing_threshold, disparity_rates
    monkeypatch.setattr(config, "MIN_GROUP_N", 2)
    pred = _preds([0, 0, 0, 0, 1, 1], [0.6, 0.7, 0.2, 0.1, 0.9, 0.8])
    meta = pd.DataFrame({config.ID_COL: pred[config.ID_COL], "race": ["A"] * 6})
    t = equalizing_threshold(pred, meta, "race", "A", target_fpr=0.0)
    assert t > 0.7                                      # must clear both negatives at 0.6 and 0.7
    assert disparity_rates(pred, meta, "race", t).iloc[0]["fpr"] == 0
