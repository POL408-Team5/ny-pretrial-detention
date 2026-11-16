"""E-03: SHAP and permutation importance for the frozen final models (written by Yiling, 2026-11-15).

Works on the BD-07 outputs (`src.final_test`): label "test" after the one-time 2024 run, or
"dryrun" (fit 2021-22, scored on 2023) before it. Importance is about the model's predictions
of historical decisions, not causal effects.

- shap_by_feature: TreeExplainer on the fitted XGBoost; one-hot columns are summed back to
  their original feature before taking mean |SHAP|.
- permutation_importance: drop in ROC-AUC when one original feature (or a predefined group of
  correlated features) is shuffled across cases; repeated with different shuffles.
- gain_by_feature: XGBoost "total gain", summed to original features (the impurity-style measure).
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config, features

# Groups fixed before looking at any importance result (so correlated inputs are judged together).
FEATURE_GROUPS = {
    "charge": ["top_severity_at_arraign", "top_charge_weight_at_arraign",
               "top_charge_at_arraign_violent_felony_ind", "top_arraign_attempt_indicator",
               "arraign_charge_category", "hate_crime_ind"],
    "record_counts (D14)": ["prior_vfo_cnt", "prior_nonvfo_cnt", "prior_misd_cnt",
                            "pend_vfo", "pend_nonvfo", "pend_misd"],
    "supervision_and_process": ["supervision", "dcjs_missing", "dat_wo_ws_prior_to_arraign",
                                "arrest_type", "case_type"],
    "age": ["age_at_arrest"],
    "county": ["county_name"],
    "judge": ["judge_id"],
}


def available_label() -> str:
    """'test' once the 2024 run has finished, otherwise 'dryrun'."""
    log = config.REPORTS / "bd07_test_run_log.json"
    if log.exists() and json.loads(log.read_text()).get("status") == "finished":
        return "test"
    if (config.DATA_PROCESSED / "predictions_dryrun.parquet").exists():
        return "dryrun"
    raise FileNotFoundError("Run `python -m src.final_test --dry-run` (or the test) first.")


def load_model(label: str, model: str, feature_set: str):
    folder = "final" if label == "test" else "dryrun"
    return joblib.load(config.MODELS_DIR / folder / f"{config.TARGET_Q1}_{feature_set}_{model}.joblib")


def scored_rows(label: str) -> pd.DataFrame:
    """Main-sample rows of the scored year (2024 for the test, 2023 for the dry run)."""
    year = config.TEST_YEAR if label == "test" else config.VALID_YEAR
    df = pd.read_parquet(config.ANALYTIC_FILE, filters=[(config.YEAR_COL, "==", year)])
    return df.loc[df[config.MAIN_SAMPLE_COL]].reset_index(drop=True)


def raw_X(rows: pd.DataFrame, feature_set: str, label: str) -> pd.DataFrame:
    return features.get_X(rows, feature_set, final=label == "test")


def _column_owner(pipe, raw_columns) -> list[str]:
    """Original feature for each transformed column (longest matching prefix)."""
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    owners = []
    for name in names:
        stripped = name.split("__", 1)[1]
        match = max((c for c in raw_columns if stripped == c or stripped.startswith(c + "_")), key=len)
        owners.append(match)
    return owners


def shap_by_feature(pipe, X: pd.DataFrame, n_sample: int = 5000, seed: int = config.RANDOM_STATE):
    """(summary table, per-case SHAP by original feature, the sampled raw rows)."""
    import shap

    sample = X.sample(n=min(n_sample, len(X)), random_state=seed)
    Z = pipe.named_steps["preprocess"].transform(sample)
    Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
    values = shap.TreeExplainer(pipe.named_steps["clf"]).shap_values(Z)
    owners = _column_owner(pipe, list(X.columns))
    per_case = pd.DataFrame(values).T.groupby(np.array(owners)).sum().T   # sum one-hot columns
    per_case.index = sample.index
    summary = (per_case.abs().mean().rename("mean_abs_shap").sort_values(ascending=False)
               .rename_axis("feature").reset_index())
    summary["rank"] = np.arange(1, len(summary) + 1)
    return summary, per_case, sample


def gain_by_feature(pipe, raw_columns) -> pd.DataFrame:
    owners = _column_owner(pipe, list(raw_columns))
    booster = pipe.named_steps["clf"].get_booster()
    gain = booster.get_score(importance_type="total_gain")          # keys f0, f1, ...
    per_col = pd.Series({owners[int(k[1:])]: 0.0 for k in gain})
    for k, v in gain.items():
        per_col[owners[int(k[1:])]] += v
    out = (per_col / per_col.sum()).rename("gain_share").sort_values(ascending=False)
    return out.rename_axis("feature").reset_index()


def permutation_importance(pipe, X: pd.DataFrame, y, groups: dict | None = None,
                           n_repeats: int = 5, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Mean drop in ROC-AUC when the columns of each item are shuffled together (same row order)."""
    y = np.asarray(y, dtype=int)
    base = roc_auc_score(y, pipe.predict_proba(X)[:, 1])
    items = groups or {c: [c] for c in X.columns}
    rng = np.random.default_rng(seed)
    rows = []
    for name, cols in items.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        drops = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            order = rng.permutation(len(X))
            shuffled[cols] = X[cols].iloc[order].to_numpy()
            drops.append(base - roc_auc_score(y, pipe.predict_proba(shuffled)[:, 1]))
        rows.append({"item": name, "columns": ", ".join(cols), "auc_drop_mean": float(np.mean(drops)),
                     "auc_drop_sd": float(np.std(drops, ddof=1)) if n_repeats > 1 else np.nan,
                     "base_auc": base, "n_repeats": n_repeats})
    return pd.DataFrame(rows).sort_values("auc_drop_mean", ascending=False).reset_index(drop=True)
