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
