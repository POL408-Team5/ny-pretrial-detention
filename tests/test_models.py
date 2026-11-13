import joblib
import numpy as np
import pandas as pd
import pytest

from src import config, features
from src.models import BASELINE_MODELS, PREDICTION_COLUMNS, fit_and_predict, split_by_year
from src.rules import rule_scores


def _development_fixture():
    size = 102
    df = pd.DataFrame(index=range(size))
    for field in features.ALL_FEATURES:
        df[field] = 1.0 if field in features.NUMERIC else "common"
    df[config.ID_COL] = [f"0x{i:04x}" for i in range(size)]
    df[config.MAIN_SAMPLE_COL] = True
    df[config.YEAR_COL] = [config.TRAIN_YEARS[0]] * 40 + [config.TRAIN_YEARS[1]] * 40 + [
        config.VALID_YEAR,
    ] * 20 + [config.TEST_YEAR] * 2
    df[config.SPLIT_COL] = ["train"] * 80 + ["valid"] * 20 + ["test"] * 2
    df[config.TARGET_Q1] = [0, 1] * 50 + [9, 9]  # Sentinel: cannot be scored.
    df["age_at_arrest"] = np.r_[np.tile([22., 40.], 50), [9999., 9999.]]
    df.loc[0, "prior_vfo_cnt"] = np.nan
    df.loc[0, "dcjs_missing"] = 1
    df["gender"] = ["Female", "Male"] * 50 + ["test_only", "test_only"]
    df.loc[80:99, "arrest_type"] = "validation_only"
    df.loc[100:, "arrest_type"] = "test_only"
    return df


def test_main_sample_filter_and_split_year_consistency():
    df = _development_fixture()
    df.loc[0, config.MAIN_SAMPLE_COL] = False
    parts = split_by_year(df)
    assert {k: len(v) for k, v in parts.items()} == {"train": 79, "valid": 20, "test": 2}
    df.loc[3, config.SPLIT_COL] = "valid"
    with pytest.raises(ValueError, match="Year"):
        split_by_year(df)


@pytest.mark.parametrize("feature_set", list(features.ALL_SETS))
@pytest.mark.parametrize("model_name", list(BASELINE_MODELS))
def test_development_pipeline_does_not_extract_test_features(
    tmp_path, monkeypatch, feature_set, model_name,
):
    df = _development_fixture()
    get_x = features.get_X
    years_seen = set()

    def checked(part, name, **kwargs):
        years_seen.update(part[config.YEAR_COL])
        assert config.TEST_YEAR not in set(part[config.YEAR_COL])
        return get_x(part, name, **kwargs)

    monkeypatch.setattr(features, "get_X", checked)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    model, grid = BASELINE_MODELS[model_name]
    small_grid = {key: values[:1] for key, values in grid.items()}
    out = fit_and_predict(df, model_name, model, small_grid, feature_set)
    assert years_seen == set(config.TRAIN_YEARS) | {config.VALID_YEAR}
    assert out.columns.tolist() == PREDICTION_COLUMNS
    assert out.groupby("split").size().to_dict() == {"train_oof": 80, "valid": 20}
    assert out["y_prob"].between(0, 1).all()
    saved = joblib.load(tmp_path / f"{config.TARGET_Q1}_{feature_set}_{model_name}.joblib")
    names = saved.named_steps["preprocess"].get_feature_names_out()
    assert not any("validation_only" in name or "test_only" in name for name in names)
    assert any("gender_" in name for name in names) == (feature_set == "CASE_PLUS_GENDER")
    assert not any("race_" in name for name in names)
    assert any("county_name_" in name for name in names) == (feature_set in {"CASE_COURT", "CASE_COURT_JUDGE"})
    assert any("judge_id_" in name for name in names) == (feature_set == "CASE_COURT_JUDGE")
    valid = df.loc[df[config.SPLIT_COL].eq("valid")]
    np.testing.assert_allclose(
        saved.predict_proba(get_x(valid, feature_set))[:, 1],
        out.loc[out["split"].eq("valid"), "y_prob"],
    )


def test_validation_outcomes_never_change_tuning_or_predictions(tmp_path, monkeypatch):
    df = _development_fixture()
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    model, _ = BASELINE_MODELS["logit"]
    kwargs = dict(model_name="logit", estimator=model, param_grid={"clf__C": [0.1, 1]},
                  feature_set="CASE_COURT_JUDGE")
    before = fit_and_predict(df, **kwargs)
    df.loc[df[config.SPLIT_COL].eq("valid"), config.TARGET_Q1] = 0
    after = fit_and_predict(df, **kwargs)
    assert before.attrs["best_params"] == after.attrs["best_params"]
    np.testing.assert_array_equal(before["y_prob"], after["y_prob"])


def test_suite_saves_an_aligned_development_only_prediction_file(tmp_path, monkeypatch):
    from src import models

    df = _development_fixture()
    estimator, grid = BASELINE_MODELS["tree"]
    monkeypatch.setattr(models, "BASELINE_MODELS", {
        "tree": (estimator, {key: values[:1] for key, values in grid.items()})
    })
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "PRED_BASELINE_FILE", tmp_path / "predictions.parquet")
    out, diagnostics, tuning = models.run_baseline_suite(df)
    pd.testing.assert_frame_equal(pd.read_parquet(config.PRED_BASELINE_FILE), out)
    assert set(out["model"]) == {"tree", "rule"}
    assert len(out) == (len(features.ALL_SETS) + 1) * 100
    assert not out[config.YEAR_COL].eq(config.TEST_YEAR).any()
    assert diagnostics["n_train"].eq(80).all()
    assert diagnostics["n_valid"].eq(20).all()
    assert set(diagnostics["feature_set"]) == set(features.ALL_SETS)
    assert len(tuning) == len(features.ALL_SETS)


def test_final_test_path_is_disabled_before_access():
    model, grid = BASELINE_MODELS["logit"]
    with pytest.raises(RuntimeError, match="disabled"):
        fit_and_predict(None, "logit", model, grid, "CASE", run_test=True)


def test_charge_rule_fallback_and_no_validation_outcome_fit():
    grade = "top_charge_weight_at_arraign"
    violent = "top_charge_at_arraign_violent_felony_ind"
    df = pd.DataFrame({
        config.ID_COL: [f"0x{i:04x}" for i in range(143)],
        config.MAIN_SAMPLE_COL: True,
        config.YEAR_COL: [config.TRAIN_YEARS[0]] * 70 + [config.TRAIN_YEARS[1]] * 70
        + [config.VALID_YEAR] * 3,
        config.SPLIT_COL: ["train"] * 140 + ["valid"] * 3,
        grade: ["AF"] * 120 + ["BM"] * 20 + ["AF", "AF", "UNSEEN_GRADE"],
        violent: ["Y"] * 100 + ["N"] * 40 + ["Y", "N", "N"],
        config.TARGET_Q1: [1] * 80 + [0] * 40 + [1, 0] * 10 + [0, 0, 1],
    })
    out = rule_scores(df)
    expected = [0.8, 80 / 120, 90 / 140]
    np.testing.assert_allclose(out.loc[out["split"].eq("valid"), "y_prob"], expected)
    changed = df.copy()
    changed.loc[140:, config.TARGET_Q1] = [1, 1, 0]
    np.testing.assert_allclose(rule_scores(changed)["y_prob"], out["y_prob"])
    # Each training fold has fewer than 100 AF/Y cases, unlike the full fit.
    assert (out.loc[out["split"].eq("train_oof"), "y_prob"].iloc[:100] < 0.8).all()

