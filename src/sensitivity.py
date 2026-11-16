"""A1-06 sensitivity check: refit one model on each sensitivity version, development years only.

    python -m src.build_dataset --sensitivity     # builds the versions first
    python -m src.sensitivity                      # default: logit x three main feature sets
    python -m src.sensitivity --model tree         # any model in models.BASELINE_MODELS

For every version (main + config.SENSITIVITY_VARIANTS) the model is tuned on 2021-22 with
models.fit_and_predict and scored on 2023. Validation AUC / AP / Brier are reported twice:
on the version's own 2023 sample, and on the 2023 dockets shared with the main sample
(same rows for every version, so differences come from training, not from the rows scored).
No 2024 row is read. Fitted models go to models/sensitivity/{version}/, never over the main models.

Writes reports/a106_sensitivity_valid.csv (aggregates only).

D14 check (added 2026-11-15):

    python -m src.sensitivity --d14

Refits logit (full grid) and XGBoost (the hyperparameters E-02 selected for each set) on the main
sample without the DCJS prior-record and pending-case counts, whose reference date is not documented
(data/README D14). Writes reports/a106_d14_valid.csv.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager

import pandas as pd

from src import config, evaluate, features, models
from src.build_dataset import variant_files

OUT_FILE = config.REPORTS / "a106_sensitivity_valid.csv"
D14_FILE = config.REPORTS / "a106_d14_valid.csv"
# DCJS counts whose reference date (arraignment or later) is not documented. `supervision` is
# documented as "at the time of arraignment"; `dcjs_missing` only records whether a DCJS match exists.
D14_FIELDS = ["prior_vfo_cnt", "prior_nonvfo_cnt", "prior_misd_cnt", "pend_vfo", "pend_nonvfo", "pend_misd"]


@contextmanager
def _models_dir(path):
    """fit_and_predict saves to config.MODELS_DIR; point it elsewhere for these fits."""
    old = config.MODELS_DIR
    config.MODELS_DIR = path
    try:
        yield
    finally:
        config.MODELS_DIR = old


@contextmanager
def _extra_feature_sets(sets):
    """Register temporary feature sets for get_X / make_preprocessor without touching the main sets."""
    added = {k: v for k, v in sets.items() if k not in features.ALL_SETS}
    features.ALL_SETS.update(added)
    try:
        yield
    finally:
        for k in added:
            features.ALL_SETS.pop(k, None)


def _load(path):
    df = pd.read_parquet(path)
    return df[df[config.SPLIT_COL].isin(["train", "valid"])]   # the test year is never loaded


def run(model_name: str = "logit", feature_sets=None) -> pd.DataFrame:
    feature_sets = feature_sets or list(features.FEATURE_SETS)
    estimator, grid = models.BASELINE_MODELS[model_name]
    versions = {"main": config.ANALYTIC_FILE}
    versions.update({name: variant_files(name)[0] for name in config.SENSITIVITY_VARIANTS})

    main = _load(config.ANALYTIC_FILE)
    shared = set(main.loc[main[config.MAIN_SAMPLE_COL] & main[config.SPLIT_COL].eq("valid"), config.ID_COL])

    rows = []
    for version, path in versions.items():
        df = _load(path)
        n_train = int((df[config.MAIN_SAMPLE_COL] & df[config.SPLIT_COL].eq("train")).sum())
        for fs in feature_sets:
            print(f"{version} / {model_name} / {fs}", flush=True)
            with _models_dir(config.ROOT / "models" / "sensitivity" / version):
                pred = models.fit_and_predict(df, model_name, estimator, grid, fs,
                                              scale=model_name != "tree")
            valid = pred[pred["split"].eq("valid")]
            for scored_on, part in [("own_valid", valid), ("shared_valid", valid[valid[config.ID_COL].isin(shared)])]:
                m = evaluate.metrics(part["y_true"].to_numpy(), part["y_prob"].to_numpy())
                rows.append({"version": version, "model": model_name, "feature_set": fs,
                             "scored_on": scored_on, "n_train": n_train, **m,
                             "best_params": str(pred.attrs.get("best_params"))})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_FILE, index=False)
    print(f"Wrote {OUT_FILE.relative_to(config.ROOT)}")
    return out


def run_d14(model_names=("logit", "xgb")) -> pd.DataFrame:
    """Main sample, each nested set with and without the D14 fields; validation year only."""
    import json
    from src import models_ensemble

    df = _load(config.ANALYTIC_FILE)
    reduced = {f"{fs}_NO_D14": [c for c in cols if c not in D14_FIELDS] for fs, cols in features.FEATURE_SETS.items()}
    e02 = pd.read_csv(config.REPORTS / "e02_fit_diagnostics.csv").set_index(["model", "feature_set"])
    rows = []
    with _extra_feature_sets(reduced), _models_dir(config.ROOT / "models" / "sensitivity" / "d14"):
        for name in model_names:
            for fs in features.FEATURE_SETS:
                if name == "xgb":
                    estimator, _ = models_ensemble.ENSEMBLE_MODELS["xgb"]
                    grid = {k: [v] for k, v in json.loads(e02.loc[("xgb", fs), "best_params"]).items()}
                    scale = False
                else:
                    estimator, grid = models.BASELINE_MODELS[name]
                    scale = name != "tree"
                for spec in [fs, f"{fs}_NO_D14"]:
                    print(f"D14 / {name} / {spec}", flush=True)
                    pred = models.fit_and_predict(df, name, estimator, grid, spec, scale=scale)
                    valid = pred[pred["split"].eq("valid")]
                    m = evaluate.metrics(valid["y_true"].to_numpy(), valid["y_prob"].to_numpy())
                    rows.append({"model": name, "base_set": fs, "d14_fields": "dropped" if spec != fs else "kept",
                                 "n_features": len(features.ALL_SETS[spec]), **m,
                                 "best_params": str(pred.attrs.get("best_params"))})
    out = pd.DataFrame(rows)
    out.to_csv(D14_FILE, index=False)
    print(f"Wrote {D14_FILE.relative_to(config.ROOT)}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="logit", choices=list(models.BASELINE_MODELS))
    parser.add_argument("--d14", action="store_true", help="run the D14 record-timing check instead")
    args = parser.parse_args()
    run_d14() if args.d14 else run(args.model)
