"""Train-only charge-grade and violent-felony rule for BD-02."""

import numpy as np
import pandas as pd

from src import config, features
from src.models import _development_splits, _outcome, _prediction_frame, _stratified_cv


_GRADE = "top_charge_weight_at_arraign"
_VIOLENT = "top_charge_at_arraign_violent_felony_ind"


def _keys(df):
    columns = [_GRADE, _VIOLENT]
    features.check_leakage(columns)
    return df[columns].astype("string").fillna(features.MISSING_LABEL).reset_index(drop=True)


def _fit_rule(x, y):
    sample = x.assign(outcome=y)
    cells = sample.groupby([_GRADE, _VIOLENT], observed=True)["outcome"].agg(["size", "mean"])
    # The current handoff uses the existing 100-case minimum for this rule.
    cells = cells.loc[cells["size"] >= config.MIN_GROUP_N, "mean"]
    grades = sample.groupby(_GRADE, observed=True)["outcome"].mean()
    return cells, grades, float(np.mean(y))


def _score_rule(x, fitted):
    cells, grades, overall = fitted
    cell_rate = cells.reindex(pd.MultiIndex.from_frame(x)).to_numpy(dtype=float)
    grade_rate = x[_GRADE].map(grades).fillna(overall).to_numpy(dtype=float)
    return np.where(np.isnan(cell_rate), grade_rate, cell_rate)


def rule_scores(df) -> pd.DataFrame:
    """Use fold-fitted rates for train OOF, full training rates for 2023."""
    train, valid = _development_splits(df)
    target = config.TARGET_Q1
    y_train = _outcome(train, target)
    x_train, x_valid = _keys(train), _keys(valid)
    oof = np.empty(len(train))
    for fit_idx, score_idx in _stratified_cv(y_train).split(x_train, y_train):
        fitted = _fit_rule(x_train.iloc[fit_idx].reset_index(drop=True), y_train[fit_idx])
        oof[score_idx] = _score_rule(x_train.iloc[score_idx], fitted)
    valid_p = _score_rule(x_valid, _fit_rule(x_train, y_train))
    return pd.concat([
        _prediction_frame(train, "rule", "CASE", target, "train_oof", oof),
        _prediction_frame(valid, "rule", "CASE", target, "valid", valid_p),
    ], ignore_index=True)
