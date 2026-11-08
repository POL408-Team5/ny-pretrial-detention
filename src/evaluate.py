"""Common prediction metrics and comparison tables for role D."""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src import config
from src.models import PREDICTION_COLUMNS


def _inputs(y_true, y_prob):
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_prob, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError("Labels and probabilities must be nonempty, aligned 1-D arrays.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary and nonmissing.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1].")
    return y.astype(int), p


def metrics(y_true, y_prob) -> dict:
    y, p = _inputs(y_true, y_prob)
    has_both_classes = y.min() != y.max()
    return {
        "roc_auc": float(roc_auc_score(y, p)) if has_both_classes else np.nan,
        "avg_precision": float(average_precision_score(y, p)) if has_both_classes else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "n": len(y),
        "base_rate": float(y.mean()),
    }


def calibration_table(y_true, y_prob, bins=10) -> pd.DataFrame:
    """Quantile bins without arbitrarily splitting ties."""
    y, p = _inputs(y_true, y_prob)
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer.")
    bucket = np.zeros(len(p), dtype=int) if np.ptp(p) == 0 else pd.qcut(
        p, q=bins, labels=False, duplicates="drop",
    )
    table = pd.DataFrame({"bin": bucket, "y": y, "p": p})
    return table.groupby("bin", as_index=False).agg(
        mean_pred=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size"),
    )


def _percentile_bootstrap(y, score, n, seed):
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer.")
    estimate = float(score(np.arange(len(y))))
    if not np.isfinite(estimate):
        return estimate, np.nan, np.nan
    rng = np.random.default_rng(seed)
    both_classes = y.min() != y.max()
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        try:
            value = float(score(idx))
        except ValueError:
            yy = y[idx]
            if both_classes and yy.min() == yy.max():
                continue  # Some ROC-AUC versions raise for single-class draws.
            raise
        if np.isfinite(value):
            values.append(value)
    if not values:
        return estimate, np.nan, np.nan
    lo, hi = np.percentile(values, [2.5, 97.5])
    return estimate, float(lo), float(hi)


def bootstrap_ci(
    y_true, y_prob, metric: Callable, n=config.N_BOOTSTRAP, seed=config.RANDOM_STATE,
) -> tuple:
    """Docket-level percentile CI for one set of fixed predictions."""
    y, p = _inputs(y_true, y_prob)
    return _percentile_bootstrap(y, lambda idx: metric(y[idx], p[idx]), n, seed)


def paired_bootstrap_diff(
    y_true, p_a, p_b, metric: Callable, n=config.N_BOOTSTRAP, seed=config.RANDOM_STATE,
) -> tuple:
    """CI for score(B) minus score(A); both scores use identical sampled cases."""
    y, a = _inputs(y_true, p_a)
    _, b = _inputs(y_true, p_b)
    return _percentile_bootstrap(
        y, lambda idx: metric(y[idx], b[idx]) - metric(y[idx], a[idx]), n, seed,
    )


def _validate_prediction_rows(pred_df):
    if set(pred_df.columns) != set(PREDICTION_COLUMNS) or not pred_df.columns.is_unique:
        raise ValueError("Prediction columns do not match the published schema.")
    keys = [config.ID_COL, "model", "feature_set", "target", "split"]
    if pred_df[keys].isna().any().any() or pred_df.duplicated(keys).any():
        raise ValueError("Prediction keys must be nonmissing and unique.")


def load_predictions() -> pd.DataFrame:
    """Combine available baseline and ensemble outputs without changing them."""
    paths = [config.PRED_BASELINE_FILE, config.PRED_ENSEMBLE_FILE]
    frames = [pd.read_parquet(path) for path in paths if path.is_file()]
    if not frames:
        raise FileNotFoundError("Neither baseline nor ensemble predictions are available.")
    for frame in frames:
        _validate_prediction_rows(frame)
    result = pd.concat(frames, ignore_index=True)[PREDICTION_COLUMNS]
    _validate_prediction_rows(result)
    return result


def comparison_table(pred_df, split="valid") -> pd.DataFrame:
    """Summarize one split, with feature set × metric columns."""
    if split not in {"train_oof", "valid", "test"}:
        raise ValueError(f"Unknown prediction split: {split!r}.")
    _validate_prediction_rows(pred_df)
    selected = pred_df.loc[pred_df["split"].eq(split)]
    if selected.empty:
        raise ValueError(f"No predictions for split {split!r}.")
    rows = []
    for (target, model, feature_set), g in selected.groupby(
        ["target", "model", "feature_set"], sort=False, observed=True,
    ):
        rows.append({
            "target": target, "model": model, "feature_set": feature_set,
            **metrics(g["y_true"], g["y_prob"]),
        })
    result = pd.DataFrame(rows).set_index(["target", "model", "feature_set"]).unstack()
    result.columns = result.columns.swaplevel(0, 1)
    result.columns.names = ["feature_set", "metric"]
    if len(result.index.get_level_values("target").unique()) == 1:
        result.index = result.index.droplevel("target")
    return result.sort_index(axis=1, level=0, sort_remaining=False)
