"""BD-07: frozen final refit and the one-time 2024 test (written by Yiling, 2026-11-15).

Three separate steps, so the test year can only be used once and only after the freeze:

    python -m src.final_test --freeze [--threshold-mode carry_over|share_rule]
        Write reports/bd07_frozen_spec.json from the development results (models, hyperparameters,
        thresholds, threshold rule). Commit and push it BEFORE the test: this is the public record.

    python -m src.final_test --dry-run
        Same code path, but fit on 2021-22 and score 2023 with `judge_id`. Must reproduce the
        validation results; no 2024 row is read. Outputs are labelled "dryrun".

    python -m src.final_test --run --confirm-freeze
        Refuses unless the frozen spec is committed and unchanged and no test has been run.
        Refits each frozen specification on 2021-23 (judge_id_final), scores 2024 once, writes
        reports/bd07_test_*.csv and a run log. After this, no model or threshold changes.

Frozen choices (docs/冻结记录_2026-11-16.md): XGBoost and logit x three nested sets; hyperparameters
from E-02 / BD-05; XGBoost tree count chosen by the same training-only early-stopping rule; no
recalibration; thresholds from C-02 (y_prob >= t).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from src import config, evaluate, features, models
from src.fairness import predict_label

SPEC_FILE = config.REPORTS / "bd07_frozen_spec.json"
RUN_LOG = config.REPORTS / "bd07_test_run_log.json"
MAIN_SETS = list(features.FEATURE_SETS)              # CASE, CASE_COURT, CASE_COURT_JUDGE
FINAL_MODELS = ["xgb", "logit"]


# ---------------------------------------------------------------------------
# Step 1: freeze
# ---------------------------------------------------------------------------

def _git(*args) -> str:
    return subprocess.check_output(["git", *args], cwd=config.ROOT, text=True).strip()


def build_frozen_spec(threshold_mode: str = "carry_over") -> dict:
    if threshold_mode not in {"carry_over", "share_rule"}:
        raise ValueError("threshold_mode must be carry_over or share_rule")
    params = pd.concat([pd.read_csv(config.REPORTS / "bd05_fit_diagnostics.csv"),
                        pd.read_csv(config.REPORTS / "e02_fit_diagnostics.csv")])
    params = params.set_index(["model", "feature_set"])["best_params"]
    thresholds = pd.read_csv(config.REPORTS / "c02_thresholds.csv").set_index(["model", "feature_set"])
    specs = []
    for model in FINAL_MODELS:
        for fs in MAIN_SETS:
            if (model, fs) not in thresholds.index:
                raise ValueError(f"No C-02 threshold for {model} / {fs}; rerun notebook 08.")
            t = thresholds.loc[(model, fs)]
            specs.append({"model": model, "feature_set": fs,
                          "params": json.loads(params.loc[(model, fs)]),
                          "threshold_2023": float(t["threshold"]),
                          "actual_share_2023": float(t["actual_share"])})
    return {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": _git("rev-parse", "HEAD"),
        "refit_years": config.TRAIN_YEARS + [config.VALID_YEAR],
        "test_year": config.TEST_YEAR,
        "judge_column_for_test": config.JUDGE_COL_FINAL,
        "xgb_tree_count": "early-stopping rule inside each fit (TrainOnlyEarlyStoppingXGBClassifier, ceiling 2000, patience 50)",
        "recalibration": "none",
        "threshold_rule": "predict 1 if y_prob >= threshold",
        "threshold_mode": threshold_mode,
        "threshold_mode_meaning": {
            "carry_over": "use threshold_2023 unchanged on the refitted model's 2024 scores",
            "share_rule": "threshold = score at which the 2024 predicted share equals actual_share_2023 (uses 2024 scores, never 2024 labels)",
        }[threshold_mode],
        "specs": specs,
    }


def write_frozen_spec(threshold_mode: str) -> None:
    if RUN_LOG.exists():
        raise RuntimeError("The test has already been run; the frozen spec cannot be rewritten.")
    spec = build_frozen_spec(threshold_mode)
    SPEC_FILE.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Wrote {SPEC_FILE.relative_to(config.ROOT)}. Commit and push it before --run.")


def load_frozen_spec() -> dict:
    if not SPEC_FILE.exists():
        raise FileNotFoundError("Run --freeze first.")
    return json.loads(SPEC_FILE.read_text())


# ---------------------------------------------------------------------------
# Fitting with frozen hyperparameters (no search)
# ---------------------------------------------------------------------------

def _estimator(model: str, params: dict):
    if model == "xgb":
        from src.models_ensemble import ENSEMBLE_MODELS
        base, _ = ENSEMBLE_MODELS["xgb"]
    else:
        base, _ = models.BASELINE_MODELS[model]
    clf = clone(base)
    clf.set_params(**{k.removeprefix("clf__"): v for k, v in params.items()})
    if "random_state" in clf.get_params(deep=False) and clf.random_state is None:
        clf.set_params(random_state=config.RANDOM_STATE)
    return clf


def fit_score(fit_rows, score_rows, model, feature_set, params, final: bool):
    """Fit preprocessor + model on fit_rows, return (scores for score_rows, fitted pipeline)."""
    x_fit = features.get_X(fit_rows, feature_set, final=final)
    x_score = features.get_X(score_rows, feature_set, final=final)
    y_fit = fit_rows[config.TARGET_Q1].to_numpy(int)
    pipe = Pipeline([("preprocess", models.make_preprocessor(feature_set, scale=model != "xgb")),
                     ("clf", _estimator(model, params))])
    pipe.fit(x_fit, y_fit)
    return pipe.predict_proba(x_score)[:, 1], pipe


# ---------------------------------------------------------------------------
# Evaluation tables (shared by dry run and test)
# ---------------------------------------------------------------------------

def _thresholds(spec, scores: dict) -> dict:
    out = {}
    for s in spec["specs"]:
        key = (s["model"], s["feature_set"])
        if spec["threshold_mode"] == "carry_over":
            out[key] = s["threshold_2023"]
        else:   # share rule on the scored year's own scores; labels are not used
            p = np.sort(scores[key])[::-1]
            k = int(round(len(p) * s["actual_share_2023"]))
            out[key] = float(p[k - 1]) if k > 0 else float(np.nextafter(p.max(), np.inf))
    return out


def evaluate_scores(scored: pd.DataFrame, scores: dict, spec: dict, judge_col: str) -> dict:
    """All BD-07 tables for one scored year. `scored` holds y and judge codes, aligned with scores."""
    y = scored[config.TARGET_Q1].to_numpy(int)
    keys = [(s["model"], s["feature_set"]) for s in spec["specs"]]
    matrix = np.column_stack([scores[k] for k in keys])
    coded = scored[judge_col].astype("string").str.startswith("J").fillna(False).to_numpy()
    cohorts = {"all": np.ones(len(y), bool), "coded_judges": coded}

    metric_rows, inc_rows = [], []
    for cohort, mask in cohorts.items():
        est, draws = evaluate.paired_auc_bootstrap_matrix(y[mask], matrix[mask])
        lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
        for j, (model, fs) in enumerate(keys):
            m = evaluate.metrics(y[mask], matrix[mask, j])
            metric_rows.append({"cohort": cohort, "model": model, "feature_set": fs, **m,
                                "auc_ci_low": lo[j], "auc_ci_high": hi[j],
                                "mean_pred": float(matrix[mask, j].mean())})
        pairs = [(m, a, b) for m in FINAL_MODELS for a, b in
                 [("CASE", "CASE_COURT"), ("CASE_COURT", "CASE_COURT_JUDGE"), ("CASE", "CASE_COURT_JUDGE")]]
        pairs += [("xgb_minus_logit", fs, fs) for fs in MAIN_SETS]
        for name, a, b in pairs:
            if name == "xgb_minus_logit":
                ia, ib = keys.index(("logit", a)), keys.index(("xgb", b))
                label = f"xgb minus logit ({a})"
            else:
                ia, ib = keys.index((name, a)), keys.index((name, b))
                label = f"{name}: {b} minus {a}"
            d = draws[:, ib] - draws[:, ia]
            inc_rows.append({"cohort": cohort, "comparison": label, "delta_auc": est[ib] - est[ia],
                             "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
                             "n": int(mask.sum())})

    cuts = _thresholds(spec, scores)
    conf_rows, cal_rows = [], []
    for model, fs in keys:
        p = scores[(model, fs)]
        yhat = predict_label(p, cuts[(model, fs)])
        tp, fp = int(((yhat == 1) & (y == 1)).sum()), int(((yhat == 1) & (y == 0)).sum())
        fn, tn = int(((yhat == 0) & (y == 1)).sum()), int(((yhat == 0) & (y == 0)).sum())
        conf_rows.append({"model": model, "feature_set": fs, "threshold": cuts[(model, fs)],
                          "threshold_mode": spec["threshold_mode"], "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                          "fpr": fp / (fp + tn), "fnr": fn / (fn + tp), "precision": tp / max(tp + fp, 1),
                          "predicted_share": yhat.mean(), "actual_share": y.mean()})
        cal = evaluate.calibration_table(y, p)
        cal.insert(0, "feature_set", fs); cal.insert(0, "model", model)
        cal_rows.append(cal)
    coverage = (scored[judge_col].astype("string").fillna("missing")
                .map(lambda v: "coded" if v.startswith("J") else v).value_counts(normalize=True)
                .rename("share").rename_axis("judge_code").reset_index())
    return {"metrics": pd.DataFrame(metric_rows), "increments": pd.DataFrame(inc_rows),
            "confusion": pd.DataFrame(conf_rows), "calibration": pd.concat(cal_rows, ignore_index=True),
            "judge_coverage": coverage}


def _write(tables: dict, label: str) -> None:
    for name, table in tables.items():
        path = config.REPORTS / f"bd07_{label}_{name}.csv"
        table.to_csv(path, index=False)
        print("Wrote", path.relative_to(config.ROOT))


def _sha256(path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _run(spec: dict, fit_rows, score_rows, final: bool, label: str) -> dict:
    judge_col = config.JUDGE_COL_FINAL if final else "judge_id"
    scores, frames, fit_log = {}, [], []
    model_dir = config.MODELS_DIR / ("final" if final else "dryrun")
    model_dir.mkdir(parents=True, exist_ok=True)
    for s in spec["specs"]:
        key = (s["model"], s["feature_set"])
        print(f"{label}: fitting {key[0]} / {key[1]} on {len(fit_rows):,} cases", flush=True)
        p, pipe = fit_score(fit_rows, score_rows, *key, s["params"], final=final)
        scores[key] = p
        joblib.dump(pipe, model_dir / f"{config.TARGET_Q1}_{key[1]}_{key[0]}.joblib")
        clf = pipe.named_steps["clf"]
        fit_log.append({"model": key[0], "feature_set": key[1],
                        "selected_n_estimators": getattr(clf, "selected_n_estimators_", None)})
        out = score_rows[[config.ID_COL, config.YEAR_COL]].copy()
        out["model"], out["feature_set"], out["target"] = key[0], key[1], config.TARGET_Q1
        out["split"] = "test" if final else "valid_dryrun"
        out["y_true"], out["y_prob"] = score_rows[config.TARGET_Q1].to_numpy(int), p
        frames.append(out)
    pred = pd.concat(frames, ignore_index=True)
    pred_path = config.DATA_PROCESSED / f"predictions_{label}.parquet"
    pred.to_parquet(pred_path, index=False)
    tables = evaluate_scores(score_rows.reset_index(drop=True), scores, spec, judge_col)
    tables["fit_log"] = pd.DataFrame(fit_log)
    _write(tables, label)
    return {"predictions": str(pred_path.relative_to(config.ROOT)), "predictions_sha256": _sha256(pred_path),
            "tables": tables}


# ---------------------------------------------------------------------------
# Step 2: dry run on development years; Step 3: the one-time test
# ---------------------------------------------------------------------------

def dry_run() -> pd.DataFrame:
    spec = load_frozen_spec()
    df = pd.read_parquet(config.ANALYTIC_FILE, filters=[(config.YEAR_COL, "!=", config.TEST_YEAR)])
    main = df.loc[df[config.MAIN_SAMPLE_COL]]
    assert not main[config.YEAR_COL].eq(config.TEST_YEAR).any()
    fit_rows = main.loc[main[config.SPLIT_COL].eq("train")]
    score_rows = main.loc[main[config.SPLIT_COL].eq("valid")]
    result = _run(spec, fit_rows, score_rows, final=False, label="dryrun")
    # Reproduction check against the published validation metrics.
    ref = pd.concat([pd.read_csv(config.REPORTS / "bd05_metrics.csv"), pd.read_csv(config.REPORTS / "e02_metrics.csv")])
    ref = ref.loc[ref["split"].eq("valid")].set_index(["model", "feature_set"])["roc_auc"]
    got = result["tables"]["metrics"].query("cohort == 'all'").set_index(["model", "feature_set"])["roc_auc"]
    check = pd.DataFrame({"published_valid_auc": ref.reindex(got.index), "dryrun_auc": got})
    check["diff"] = check["dryrun_auc"] - check["published_valid_auc"]
    check.reset_index().to_csv(config.REPORTS / "bd07_dryrun_reproduction.csv", index=False)
    print(check.round(6).to_string())
    tol = {"logit": 1e-6, "xgb": 1e-3}
    bad = [k for k, d in check["diff"].items() if abs(d) > tol[k[0]]]
    if bad:
        raise AssertionError(f"Dry run does not reproduce the validation results for {bad}")
    print("Dry run reproduces the validation results (logit exactly, XGBoost within 0.001).")
    return check


def run_test(confirm_freeze: bool) -> None:
    if not confirm_freeze:
        raise RuntimeError("Pass --confirm-freeze after the group has confirmed the freeze record.")
    if RUN_LOG.exists():
        raise RuntimeError(f"The 2024 test has already been run ({RUN_LOG.name}). It is run only once.")
    spec = load_frozen_spec()
    rel = str(SPEC_FILE.relative_to(config.ROOT))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=config.ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=config.ROOT).returncode == 0
    if not (tracked and clean):
        raise RuntimeError("Commit and push the frozen spec (unchanged) before running the test.")
    df = pd.read_parquet(config.ANALYTIC_FILE)
    main = df.loc[df[config.MAIN_SAMPLE_COL]]
    fit_rows = main.loc[main[config.YEAR_COL].isin(spec["refit_years"])]
    score_rows = main.loc[main[config.YEAR_COL].eq(spec["test_year"])]
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    RUN_LOG.write_text(json.dumps({"started_utc": started, "status": "running"}, indent=2) + "\n")  # blocks a rerun
    result = _run(spec, fit_rows, score_rows, final=True, label="test")
    RUN_LOG.write_text(json.dumps({
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "finished", "code_commit": _git("rev-parse", "HEAD"),
        "frozen_spec_sha256": _sha256(SPEC_FILE), "n_refit": len(fit_rows), "n_test": len(score_rows),
        "predictions": result["predictions"], "predictions_sha256": result["predictions_sha256"],
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ["pandas", "numpy", "scikit-learn", "xgboost"]},
    }, indent=2) + "\n")
    print(f"Test finished. Wrote {RUN_LOG.name}. Do not change models or thresholds after this.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    step = parser.add_mutually_exclusive_group(required=True)
    step.add_argument("--freeze", action="store_true")
    step.add_argument("--dry-run", action="store_true")
    step.add_argument("--run", action="store_true")
    parser.add_argument("--threshold-mode", default="carry_over", choices=["carry_over", "share_rule"])
    parser.add_argument("--confirm-freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        write_frozen_spec(args.threshold_mode)
    elif args.dry_run:
        dry_run()
    else:
        run_test(args.confirm_freeze)
