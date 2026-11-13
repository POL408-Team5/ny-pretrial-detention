"""Shared development-only training workflow and BD-05 baseline suite."""

from __future__ import annotations

import json
import warnings
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import ConvergenceWarning

from src import config, features


def _lasso_logistic():
    """Keep true L1 regularization across old and new sklearn APIs."""
    modern = LogisticRegression().get_params().get("penalty", "deprecated") == "deprecated"
    regularization = {"l1_ratio": 1.0} if modern else {"penalty": "l1"}
    return LogisticRegression(
        solver="liblinear", max_iter=2000, random_state=config.RANDOM_STATE,
        **regularization,
    )


BASELINE_MODELS = {
    "logit": (LogisticRegression(max_iter=2000), {"clf__C": [0.01, 0.1, 1, 10]}),
    "lasso": (
        _lasso_logistic(),
        {"clf__C": [0.001, 0.01, 0.1, 1]},
    ),
    "tree": (
        DecisionTreeClassifier(random_state=config.RANDOM_STATE),
        {"clf__max_depth": [3, 5, 8, 12], "clf__min_samples_leaf": [50, 200]},
    ),
}

PREDICTION_COLUMNS = [
    config.ID_COL, config.YEAR_COL, "model", "feature_set", "target",
    "split", "y_true", "y_prob",
]


def split_by_year(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Use the published split tags and check their agreement with the years."""
    main = df.loc[df[config.MAIN_SAMPLE_COL].eq(True).fillna(False)]
    rules = {
        "train": config.TRAIN_YEARS,
        "valid": [config.VALID_YEAR],
        "test": [config.TEST_YEAR],
    }
    if main[config.ID_COL].isna().any() or main[config.ID_COL].duplicated().any():
        raise ValueError("Main-sample case IDs must be unique and nonmissing.")
    if not main[config.SPLIT_COL].isin(rules).all():
        raise ValueError("Main sample contains an unrecognized split.")
    parts = {}
    for label, years in rules.items():
        part = main.loc[main[config.SPLIT_COL].eq(label)]
        if not part[config.YEAR_COL].isin(years).all():
            raise ValueError(f"Year does not agree with the {label} split.")
        parts[label] = part
    return parts


def _development_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove test rows before feature extraction or access to outcome values."""
    split_tags = df[config.SPLIT_COL]
    years = df[config.YEAR_COL]
    if not split_tags.eq("test").fillna(False).eq(years.eq(config.TEST_YEAR).fillna(False)).all():
        raise ValueError("Test-year and test-split metadata disagree.")
    parts = split_by_year(df.loc[~split_tags.eq("test").fillna(False)])
    if parts["train"].empty or parts["valid"].empty:
        raise ValueError("Both the training and validation splits must be present.")
    return parts["train"], parts["valid"]


def make_preprocessor(feature_set: str, scale: bool = True) -> ColumnTransformer:
    """Create an unfitted transformer using the published feature types."""
    columns = features.ALL_SETS[feature_set]
    features.check_leakage(columns, allow=config.SUPPLEMENTARY_ALLOWED.get(feature_set, ()))
    numeric = [c for c in columns if c in features.NUMERIC]
    categorical = [c for c in columns if c in features.CATEGORICAL]
    numeric_steps = [
        ("impute", SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True)),
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


def _outcome(part: pd.DataFrame, target: str) -> np.ndarray:
    y = part[target]
    if not y.isin([0, 1]).all():
        raise ValueError(f"{target} must contain nonmissing binary labels.")
    return y.to_numpy(dtype=int)


def _stratified_cv(y: np.ndarray) -> StratifiedKFold:
    n_splits = 5  # Agreed in the BD-01 handoff.
    if np.bincount(y, minlength=2).min() < n_splits:
        raise ValueError("Five-fold ROC-AUC CV requires at least five cases in each class.")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)


def _prediction_frame(part, model_name, feature_set, target, split, probability):
    out = part[[config.ID_COL, config.YEAR_COL]].copy()
    out["model"] = model_name
    out["feature_set"] = feature_set
    out["target"] = target
    out["split"] = split
    out["y_true"] = _outcome(part, target)
    out["y_prob"] = probability
    return out[PREDICTION_COLUMNS].reset_index(drop=True)


def fit_and_predict(
    df, model_name, estimator, param_grid, feature_set,
    target=config.TARGET_Q1, run_test=False, scale=True,
) -> pd.DataFrame:
    """Tune in 2021–22, produce OOF scores and score the independent 2023 split.

    OOF fits use the parameters selected using these training data, so they
    are a development diagnostic rather than a nested-CV estimate. The
    validation year is not used to fit the model or choose its parameters.
    BD-07 will enable the final-test path after the separate model freeze.
    """
    if run_test:
        raise RuntimeError("Final testing is disabled in BD-01–04; use the approved BD-07 workflow.")
    if model_name not in {"logit", "lasso", "tree", "rf", "xgb"}:
        raise ValueError(f"Unsupported model name: {model_name!r}.")
    if target not in {config.TARGET_Q1, config.TARGET_Q3, config.TARGET_Q3_ALT}:
        raise ValueError(f"Unsupported outcome: {target!r}.")
    train, valid = _development_splits(df)
    x_train = features.get_X(train, feature_set)
    x_valid = features.get_X(valid, feature_set)
    y_train = _outcome(train, target)
    _outcome(valid, target)
    cv = _stratified_cv(y_train)
    clf = clone(estimator)
    if "random_state" in clf.get_params(deep=False) and clf.random_state is None:
        clf.set_params(random_state=config.RANDOM_STATE)
    pipeline = Pipeline([("preprocess", make_preprocessor(feature_set, scale)), ("clf", clf)])
    search = GridSearchCV(
        pipeline, param_grid or {}, scoring="roc_auc", cv=cv,
        refit=True, n_jobs=1, error_score="raise",
    )
    started = perf_counter()
    # Do not silently publish a grid search or an OOF fit that did not converge.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        search.fit(x_train, y_train)
        oof = cross_val_predict(
            clone(search.best_estimator_), x_train, y_train, cv=cv,
            method="predict_proba", n_jobs=1,
        )[:, 1]
    valid_p = search.best_estimator_.predict_proba(x_valid)[:, 1]
    result = pd.concat([
        _prediction_frame(train, model_name, feature_set, target, "train_oof", oof),
        _prediction_frame(valid, model_name, feature_set, target, "valid", valid_p),
    ], ignore_index=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / f"{target}_{feature_set}_{model_name}.joblib"
    joblib.dump(search.best_estimator_, model_path)
    result.attrs["best_params"] = search.best_params_
    result.attrs["fit_diagnostics"] = {
        "model": model_name, "feature_set": feature_set, "target": target,
        "n_train": len(train), "n_valid": len(valid),
        "n_raw_features": x_train.shape[1],
        "n_transformed_features": len(
            search.best_estimator_.named_steps["preprocess"].get_feature_names_out()
        ),
        "best_cv_auc": float(search.best_score_),
        "best_params": json.dumps(search.best_params_, sort_keys=True),
        "elapsed_seconds": perf_counter() - started,
    }
    result.attrs["tuning_results"] = [
        {"model": model_name, "feature_set": feature_set,
         "params": json.dumps(params, sort_keys=True),
         "mean_cv_auc": float(search.cv_results_["mean_test_score"][i]),
         "std_cv_auc": float(search.cv_results_["std_test_score"][i]),
         "rank": int(search.cv_results_["rank_test_score"][i])}
        for i, params in enumerate(search.cv_results_["params"])
    ]
    return result


def run_baseline_suite(df) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit all three baselines on every main and supplementary feature set.

    Only training OOF and validation predictions are produced. No test outcome
    is accessed, and existing published training function arguments are unchanged.
    The rule is included once under CASE because it uses only two charge fields.
    """
    from src.rules import rule_scores

    predictions = [rule_scores(df)]
    diagnostics, tuning = [], []
    for name, (estimator, grid) in BASELINE_MODELS.items():
        for feature_set in features.ALL_SETS:
            print(f"Fitting {name} / {feature_set}", flush=True)
            out = fit_and_predict(
                df, name, estimator, grid, feature_set,
                run_test=False, scale=name != "tree",
            )
            diagnostics.append(out.attrs["fit_diagnostics"])
            tuning.extend(out.attrs["tuning_results"])
            print(
                f"  selected {out.attrs['best_params']}; "
                f"CV AUC {out.attrs['fit_diagnostics']['best_cv_auc']:.6f}",
                flush=True,
            )
            out.attrs = {}
            predictions.append(out)
    combined = pd.concat(predictions, ignore_index=True)[PREDICTION_COLUMNS]
    keys = [config.ID_COL, "model", "feature_set", "target", "split"]
    if combined.duplicated(keys).any():
        raise ValueError("Duplicate prediction keys in the baseline suite.")
    if not combined["split"].isin(["train_oof", "valid"]).all():
        raise ValueError("The baseline suite must not contain test predictions.")
    if not np.isfinite(combined["y_prob"]).all() or not combined["y_prob"].between(0, 1).all():
        raise ValueError("The baseline suite contains invalid probabilities.")
    config.PRED_BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.PRED_BASELINE_FILE.with_suffix(".pending.parquet")
    combined.to_parquet(temporary, index=False)
    temporary.replace(config.PRED_BASELINE_FILE)
    return combined, pd.DataFrame(diagnostics), pd.DataFrame(tuning)

