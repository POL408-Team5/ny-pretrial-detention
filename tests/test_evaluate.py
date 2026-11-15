import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from src import config
from src.evaluate import (
    bootstrap_ci, calibration_table, comparison_table, load_predictions,
    metrics, paired_bootstrap_diff,
)


def test_metrics_have_expected_perfect_and_chance_values():
    y = [0, 1, 0, 1]
    assert metrics(y, y)["roc_auc"] == 1
    assert metrics(y, y)["avg_precision"] == 1
    assert metrics(y, [0.5] * 4)["roc_auc"] == 0.5
    p = [0.2, 0.9, 0.4, 0.8]
    assert metrics(y, p)["brier"] == pytest.approx(brier_score_loss(y, p))


def test_bad_probabilities_and_constant_score_calibration():
    y = [0, 1, 0, 1]
    with pytest.raises(ValueError, match="Probabilities"):
        metrics(y, [0, 1, np.nan, 2])
    table = calibration_table(y, [0.3] * 4)
    assert table["n"].tolist() == [4]
    assert table["observed_rate"].tolist() == [0.5]
    assert table["mean_pred"].tolist() == pytest.approx([0.3])


def test_paired_bootstrap_preserves_matching_and_direction():
    y = np.tile([0, 1], 32)
    perfect, reverse = y.astype(float), 1.0 - y
    assert paired_bootstrap_diff(y, reverse, perfect, roc_auc_score, n=25) == (1, 1, 1)
    assert paired_bootstrap_diff(y, perfect, perfect, roc_auc_score, n=25) == (0, 0, 0)
    assert bootstrap_ci(y, perfect, roc_auc_score, n=25) == (1, 1, 1)


def test_local_prediction_loader_and_split_specific_table(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.parquet"
    ensemble = tmp_path / "ensemble.parquet"
    monkeypatch.setattr(config, "PRED_BASELINE_FILE", baseline)
    monkeypatch.setattr(config, "PRED_ENSEMBLE_FILE", ensemble)
    rows = pd.DataFrame({
        config.ID_COL: ["0x1", "0x2", "0x3", "0x4"],
        config.YEAR_COL: [2021, 2021, 2023, 2023],
        "model": "logit",
        "feature_set": "CASE",
        "target": config.TARGET_Q1,
        "split": ["train_oof", "train_oof", "valid", "valid"],
        "y_true": [0, 1, 0, 1],
        "y_prob": [0.9, 0.1, 0.2, 0.8],
    })
    rows.to_parquet(baseline, index=False)
    rows.assign(model="rf").to_parquet(ensemble, index=False)
    table = comparison_table(load_predictions())
    assert table.loc["logit", ("CASE", "roc_auc")] == 1
    assert table.loc["rf", ("CASE", "n")] == 2
    rows.to_parquet(ensemble, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_predictions()


def _comparison_fixture(supplement=False):
    from tests.test_models import _development_fixture
    from src.models import _prediction_frame

    df = _development_fixture().iloc[:100].copy()
    df.loc[80:89, "judge_id"] = "J001"
    df.loc[90:93, "judge_id"] = "other"
    df.loc[94:96, "judge_id"] = "unseen"
    df.loc[97:99, "judge_id"] = "missing"
    df[config.JUDGE_COL_FINAL] = "J999"  # Must not define the BD-06 subset.
    keys = [("rule", "CASE")] + [(m, f) for m in ["logit", "lasso", "tree", "rf", "xgb"]
                                  for f in config.FEATURE_SETS]
    if supplement:
        keys.append(("logit", "CASE_PLUS_GENDER"))
    frames = []
    for j, (model, fs) in enumerate(keys):
        for split, part in [("train_oof", df.iloc[:80]), ("valid", df.iloc[80:])]:
            p = 0.2 + 0.6 * part[config.TARGET_Q1].to_numpy() + 0.02 * np.sin(np.arange(len(part)) + j)
            frames.append(_prediction_frame(part, model, fs, config.TARGET_Q1, split, p))
    return df, pd.concat(frames, ignore_index=True)


def test_weighted_auc_matches_sklearn_for_ties_and_nonunit_counts():
    from src.evaluate import _auc_rank_groups, _weighted_auc_columns

    rng = np.random.default_rng(config.RANDOM_STATE)
    y = np.tile([0, 1], 20)
    p = np.column_stack([rng.integers(0, 5, len(y))/4, rng.random(len(y)), np.full(len(y), 0.5)])
    prepared = _auc_rank_groups(y, p)
    for _ in range(10):
        weights = rng.integers(0, 4, len(y)).astype(float)
        expected = [roc_auc_score(y, p[:, j], sample_weight=weights) for j in range(p.shape[1])]
        np.testing.assert_allclose(_weighted_auc_columns(weights, prepared), expected, atol=1e-14, rtol=0)


def test_matrix_bootstrap_reproduces_same_resampled_rows_and_paired_interval():
    from src.evaluate import paired_auc_bootstrap_matrix

    y = np.tile([0, 1], 12)
    rng = np.random.default_rng(config.RANDOM_STATE)
    p = np.column_stack([np.round(rng.random(len(y)), 1), rng.random(len(y))])
    estimate, actual = paired_auc_bootstrap_matrix(y, p, n=50)
    np.testing.assert_allclose(estimate, [roc_auc_score(y, col) for col in p.T], atol=1e-14, rtol=0)
    rng = np.random.default_rng(config.RANDOM_STATE)
    expected = []
    for _ in range(50):
        idx = rng.integers(0, len(y), len(y))
        expected.append([roc_auc_score(y[idx], col[idx]) for col in p.T])
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
    reference = paired_bootstrap_diff(y, p[:, 0], p[:, 1], roc_auc_score, n=50)
    np.testing.assert_allclose(np.percentile(actual[:, 1]-actual[:, 0], [2.5, 97.5]), reference[1:], atol=1e-14, rtol=0)


@pytest.mark.parametrize("failure", ["missing_case", "duplicate", "label", "year", "probability", "test", "numeric_id"])
def test_prediction_audit_rejects_misaligned_or_forbidden_rows(failure):
    from src.evaluate import audit_development_predictions

    df, pred = _comparison_fixture()
    if failure == "missing_case":
        pred = pred.drop(index=0)
    elif failure == "duplicate":
        pred = pd.concat([pred, pred.iloc[:1]], ignore_index=True)
    elif failure == "label":
        pred.loc[0, "y_true"] = 1 - pred.loc[0, "y_true"]
    elif failure == "year":
        pred.loc[0, config.YEAR_COL] = config.VALID_YEAR
    elif failure == "probability":
        pred.loc[0, "y_prob"] = np.nan
    elif failure == "test":
        pred.loc[0, "split"] = "test"
        pred.loc[0, config.YEAR_COL] = config.TEST_YEAR
    else:
        pred[config.ID_COL] = np.arange(len(pred))
    with pytest.raises(ValueError):
        audit_development_predictions(pred, df)


def test_comparison_excludes_supplements_and_uses_training_judge_codes(monkeypatch):
    from src.evaluate import validation_comparison

    monkeypatch.setattr(config, "MIN_GROUP_N", 1)
    df, pred = _comparison_fixture(supplement=True)
    result = validation_comparison(pred, df, n=25)
    sizes = result["cohorts"].set_index("cohort")["n"].to_dict()
    assert sizes == {"all_valid": 20, "coded_judges": 10}
    assert len(result["metrics"]) == 32
    assert len(result["increments"]) == 20
    assert len(result["model_differences"]) == 12
    assert not result["metrics"]["feature_set"].eq("CASE_PLUS_GENDER").any()
    assert result["metrics"].loc[result["metrics"].cohort.eq("coded_judges"), "n"].eq(10).all()
    assert result["coverage"].feature_set.eq("CASE_PLUS_GENDER").any()


def test_paired_comparison_is_invariant_to_input_row_order(monkeypatch):
    from src.evaluate import validation_comparison

    monkeypatch.setattr(config, "MIN_GROUP_N", 1)
    df, pred = _comparison_fixture()
    expected = validation_comparison(pred, df, n=25)
    actual = validation_comparison(
        pred.sample(frac=1, random_state=config.RANDOM_STATE),
        df.sample(frac=1, random_state=config.RANDOM_STATE), n=25,
    )
    for key in ["metrics", "increments", "model_differences", "calibration", "cohorts"]:
        pd.testing.assert_frame_equal(actual[key], expected[key])


def test_missing_entire_model_pair_is_not_silently_accepted(monkeypatch):
    from src.evaluate import validation_comparison

    monkeypatch.setattr(config, "MIN_GROUP_N", 1)
    df, pred = _comparison_fixture()
    pred = pred.loc[~(pred.model.eq("rf") & pred.feature_set.eq("CASE"))]
    with pytest.raises(ValueError, match="all five"):
        validation_comparison(pred, df, n=25)
