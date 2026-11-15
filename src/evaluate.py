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


def audit_development_predictions(pred_df, development) -> pd.DataFrame:
    """Require identical development-case coverage before paired comparisons.

    Audit all supplied prediction rows, including any prespecified gender
    supplements. The caller separately excludes supplements from main results.
    Test-year rows are rejected rather than silently dropped.
    """
    from src.models import split_by_year

    _validate_prediction_rows(pred_df)
    if not pred_df["target"].eq(config.TARGET_Q1).all():
        raise ValueError("BD-06 accepts the Q1 target only.")
    if not pred_df["split"].isin(["train_oof", "valid"]).all():
        raise ValueError("BD-06 must not contain test or apparent-training predictions.")
    allowed_years = [*config.TRAIN_YEARS, config.VALID_YEAR]
    if not development[config.YEAR_COL].isin(allowed_years).all():
        raise ValueError("Comparison metadata must contain development years only.")
    if not pred_df[config.YEAR_COL].isin(allowed_years).all():
        raise ValueError("Predictions contain a non-development year.")
    if not pred_df[config.ID_COL].map(lambda x: isinstance(x, str)).all():
        raise ValueError("Case identifiers must remain strings.")
    if not set(pred_df["model"]).issubset({"rule", "logit", "lasso", "tree", "rf", "xgb"}):
        raise ValueError("Unrecognized model in the prediction file.")
    allowed_sets = set(config.FEATURE_SETS) | set(config.SUPPLEMENTARY_SETS)
    if not set(pred_df["feature_set"]).issubset(allowed_sets):
        raise ValueError("Unrecognized feature set in the prediction file.")
    if not pred_df.loc[pred_df["model"].eq("rule"), "feature_set"].eq("CASE").all():
        raise ValueError("The two-field rule is reported once under CASE.")
    _inputs(pred_df["y_true"], pred_df["y_prob"])
    parts = split_by_year(development)
    if not parts["test"].empty:
        raise ValueError("Test metadata reached the comparison workflow.")
    rows = []
    for (model, feature_set), g in pred_df.groupby(["model", "feature_set"], sort=False):
        if set(g["split"]) != {"train_oof", "valid"}:
            raise ValueError("Every model-feature pair needs both development splits.")
        for split, reference in [("train_oof", parts["train"]), ("valid", parts["valid"])]:
            ref = reference.set_index(config.ID_COL).sort_index()
            got = g.loc[g["split"].eq(split)].set_index(config.ID_COL).sort_index()
            if not got.index.equals(ref.index):
                raise ValueError(f"Case coverage differs: {model}/{feature_set}/{split}.")
            if not np.array_equal(got[config.YEAR_COL], ref[config.YEAR_COL]):
                raise ValueError("Case years disagree with analytic metadata.")
            if not np.array_equal(got["y_true"], ref[config.TARGET_Q1]):
                raise ValueError("Case labels disagree with analytic metadata.")
            rows.append({"model": model, "feature_set": feature_set, "split": split, "n": len(got)})
    return pd.DataFrame(rows)


def _auc_rank_groups(y, probabilities):
    """Cache tied-score group IDs; no model or threshold is fitted."""
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    groups = []
    for probability in probabilities.T:
        order = np.argsort(probability, kind="stable")
        sorted_p = probability[order]
        starts = np.r_[0, np.flatnonzero(sorted_p[1:] != sorted_p[:-1]) + 1]
        sizes = np.diff(np.r_[starts, len(y)])
        group = np.empty(len(y), dtype=np.intp)
        group[order] = np.repeat(np.arange(len(starts)), sizes)
        groups.append((group[positive], group[negative], len(starts)))
    return positive, negative, groups


def _weighted_auc_columns(weights, prepared):
    """Exact ROC-AUC for bootstrap multiplicities, with half credit for ties."""
    positive, negative, groups = prepared
    pos_weight, neg_weight = weights[positive], weights[negative]
    denominator = pos_weight.sum() * neg_weight.sum()
    if denominator == 0:
        return np.full(len(groups), np.nan)
    result = []
    for pos_group, neg_group, count in groups:
        pos = np.bincount(pos_group, weights=pos_weight, minlength=count)
        neg = np.bincount(neg_group, weights=neg_weight, minlength=count)
        result.append(np.dot(pos, np.cumsum(neg) - 0.5 * neg) / denominator)
    return np.asarray(result)


def paired_auc_bootstrap_matrix(
    y_true, probabilities, n=config.N_BOOTSTRAP, seed=config.RANDOM_STATE, progress=False,
):
    """Paired docket bootstrap for all fixed prediction columns in one cohort.

    Each draw samples n_case docket indices with replacement. Multiplicity
    weights give the exact same AUC as scoring those repeated rows, including
    ties. All columns share each draw. Intervals are conditional on fitted
    models; this is not refitting, a judge-cluster bootstrap or selection-adjusted.
    """
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("Probabilities must be a nonempty two-dimensional matrix.")
    y, _ = _inputs(y_true, matrix[:, 0])
    if matrix.shape[0] != len(y) or not np.isfinite(matrix).all() or not ((matrix >= 0) & (matrix <= 1)).all():
        raise ValueError("Invalid probability matrix.")
    if len(np.unique(y)) != 2:
        raise ValueError("Both classes are required for AUC intervals.")
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer.")
    prepared = _auc_rank_groups(y, matrix)
    estimate = _weighted_auc_columns(np.ones(len(y)), prepared)
    rng = np.random.default_rng(seed)
    draws = np.empty((n, matrix.shape[1]))
    for i in range(n):
        weights = np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y)).astype(float)
        draws[i] = _weighted_auc_columns(weights, prepared)
        if progress and (i + 1) % max(1, n // 5) == 0:
            print(f"  paired bootstrap: {i + 1}/{n}", flush=True)
    return estimate, draws


def validation_comparison(
    pred_df, development, n=config.N_BOOTSTRAP, seed=config.RANDOM_STATE, progress=False,
):
    """BD-06: all main model sets, paired gains, coded-judge subset and calibration.

    This is deliberately development-only. No model is fitted or recalibrated,
    no fairness threshold is selected, and gender supplements are excluded.
    """
    coverage = audit_development_predictions(pred_df, development)
    models = ["logit", "lasso", "tree", "rf", "xgb"]
    feature_sets = list(config.FEATURE_SETS)
    keys = [("rule", "CASE")] + [(model, fs) for model in models for fs in feature_sets]
    valid = pred_df.loc[pred_df["split"].eq("valid") & pred_df["feature_set"].isin(feature_sets)]
    available = set(zip(valid["model"], valid["feature_set"]))
    if available != set(keys):
        raise ValueError("Need all five models x main feature sets and one CASE rule.")
    reference = development.loc[
        development[config.MAIN_SAMPLE_COL].eq(True) & development[config.SPLIT_COL].eq("valid")
    ].set_index(config.ID_COL).sort_index()
    judge_col = config.FEATURES_JUDGE[0]
    if judge_col not in reference:
        raise ValueError("Training-year judge IDs are required for the coded subset.")
    pivot = valid.pivot(index=config.ID_COL, columns=["model", "feature_set"], values="y_prob")
    pivot = pivot.reindex(index=reference.index, columns=pd.MultiIndex.from_tuples(keys))
    matrix = pivot.to_numpy(dtype=float)
    labels = reference[config.TARGET_Q1].to_numpy(dtype=int)
    masks = {
        "all_valid": np.ones(len(reference), dtype=bool),
        "coded_judges": reference[judge_col].astype("string").str.startswith("J").fillna(False).to_numpy(dtype=bool),
    }
    summary, increments, comparisons, calibration, cohorts = [], [], [], [], []
    bootstrap = {}

    def interval(values):
        finite = values[np.isfinite(values)]
        if not len(finite):
            return np.nan, np.nan, 0
        lo, hi = np.percentile(finite, [2.5, 97.5])
        return float(lo), float(hi), len(finite)

    for cohort, mask in masks.items():
        y, p = labels[mask], matrix[mask]
        if len(y) < config.MIN_GROUP_N:
            raise ValueError("Cohort is below the reporting minimum.")
        if progress:
            print(f"Cohort {cohort}: n={len(y):,}", flush=True)
        auc, draws = paired_auc_bootstrap_matrix(y, p, n=n, seed=seed, progress=progress)
        bootstrap[cohort] = draws
        cohorts.append({"cohort": cohort, "n": len(y), "n_positive": int(y.sum()),
                        "base_rate": float(y.mean()), "share_of_valid": float(mask.mean())})
        point = []
        for i, (model, fs) in enumerate(keys):
            m = metrics(y, p[:, i])
            if not np.isclose(m["roc_auc"], auc[i], atol=1e-12, rtol=0):
                raise AssertionError("Weighted AUC does not reproduce sklearn.")
            point.append(m)
            lo, hi, successful = interval(draws[:, i])
            summary.append({"cohort": cohort, "model": model, "feature_set": fs, **m,
                            "mean_pred": float(p[:, i].mean()),
                            "calibration_gap": float(p[:, i].mean() - y.mean()),
                            "auc_ci_low": lo, "auc_ci_high": hi, "bootstrap_draws": successful})
            table = calibration_table(y, p[:, i])
            table = table.loc[table["n"] >= config.MIN_GROUP_N].copy()
            table.insert(0, "feature_set", fs)
            table.insert(0, "model", model)
            table.insert(0, "cohort", cohort)
            calibration.append(table)

        def contrast(before, after, **metadata):
            a, b = keys.index(before), keys.index(after)
            lo, hi, successful = interval(draws[:, b] - draws[:, a])
            return {"cohort": cohort, **metadata, "n": len(y),
                    "delta_roc_auc": float(auc[b] - auc[a]),
                    "ci_low": lo, "ci_high": hi, "bootstrap_draws": successful,
                    "delta_avg_precision": point[b]["avg_precision"] - point[a]["avg_precision"],
                    "delta_brier": point[b]["brier"] - point[a]["brier"]}

        for model in models:
            for before, after in zip(feature_sets, feature_sets[1:]):
                increments.append(contrast((model, before), (model, after), model=model,
                    comparison=f"{after} minus {before}"))
        # Candidate diagnostics, not a multiplicity- or model-selection-adjusted claim.
        for fs in feature_sets:
            for baseline in ["logit", "lasso"]:
                comparisons.append(contrast((baseline, fs), ("xgb", fs),
                    model="xgb", reference_model=baseline, feature_set=fs))
    return {"metrics": pd.DataFrame(summary), "increments": pd.DataFrame(increments),
            "model_differences": pd.DataFrame(comparisons),
            "calibration": pd.concat(calibration, ignore_index=True),
            "cohorts": pd.DataFrame(cohorts), "coverage": coverage,
            "bootstrap": bootstrap, "prediction_keys": keys}


def plot_validation_comparison(results, output_dir=None):
    """Save standalone BD-06 figures; do not train or tune any model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = output_dir or config.FIGURES
    directory.mkdir(parents=True, exist_ok=True)
    models = ["logit", "lasso", "tree", "rf", "xgb"]
    names = ["Logistic", "LASSO logistic", "Decision tree", "Random forest", "XGBoost"]
    feature_sets = list(config.FEATURE_SETS)
    cohorts = ["all_valid", "coded_judges"]
    titles = {"all_valid": "All validation cases", "coded_judges": "Individually coded judges"}
    colors = ["#31688e", "#7b3294", "#d18f30", "#449a75", "#ba4b4b"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    source = f"Source: NY UCS, NYC; validation {config.VALID_YEAR}. Historical bail/remand decisions; no causal interpretation."

    def save(fig, name):
        fig.savefig(directory / f"{name}.png", dpi=180)
        fig.savefig(directory / f"{name}.svg")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    image = None
    for ax, cohort in zip(axes, cohorts):
        data = results["metrics"].query("cohort == @cohort and model != 'rule'")
        table = data.pivot(index="model", columns="feature_set", values="roc_auc").reindex(index=models, columns=feature_sets)
        image = ax.imshow(table.to_numpy(), vmin=0.90, vmax=0.96, cmap="Blues", aspect="auto")
        for i in range(len(models)):
            for j in range(len(feature_sets)):
                ax.text(j, i, f"{table.iloc[i, j]:.4f}", ha="center", va="center", color="#111111")
        count = int(results["cohorts"].set_index("cohort").loc[cohort, "n"])
        ax.set(title=f"{titles[cohort]} (n={count:,})", xticks=range(3),
               xticklabels=["Case", "+ County", "+ Judge"], yticks=range(5), yticklabels=names)
    fig.suptitle("Five-model comparison on the same cases", fontsize=14)
    fig.text(0.02, 0.02, source, fontsize=8)
    fig.subplots_adjust(bottom=0.15, top=0.83, left=0.14, right=0.85, wspace=0.18)
    color_axis = fig.add_axes([0.88, 0.19, 0.018, 0.60])
    fig.colorbar(image, cax=color_axis, label="Validation ROC-AUC")
    save(fig, "bd06_validation_auc")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    descriptions = [f"{b} minus {a}" for a, b in zip(feature_sets, feature_sets[1:])]
    for ax, description, title in zip(axes, descriptions, ["Added county information", "Added judge information"]):
        for cohort, color, offset, label in [
            ("all_valid", "#31688e", -0.13, "All validation cases"),
            ("coded_judges", "#d18f30", 0.13, "Individually coded judges"),
        ]:
            rows = results["increments"].loc[
                results["increments"].cohort.eq(cohort) & results["increments"].comparison.eq(description)
            ].set_index("model").reindex(models)
            yy = np.arange(len(models)) + offset
            ax.hlines(yy, rows.ci_low, rows.ci_high, color=color, linewidth=2)
            ax.plot(rows.delta_roc_auc, yy, "o", color=color, label=label)
        ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
        ax.set(title=title, xlabel="Change in validation AUC (95% paired interval)",
               yticks=range(5), yticklabels=names)
        ax.grid(axis="x", alpha=0.2)
    axes[0].invert_yaxis()
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle(f"Nested feature gains: {config.N_BOOTSTRAP:,} paired docket resamples", fontsize=14)
    fig.text(0.02, 0.02, source + " Intervals condition on the fitted predictions.", fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    save(fig, "bd06_nested_auc_gains")

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    for row, cohort in enumerate(cohorts):
        for col, fs in enumerate(feature_sets):
            ax = axes[row, col]
            ax.plot([0, 1], [0, 1], "--", color="#777777", linewidth=1)
            for model, name, color in zip(models, names, colors):
                data = results["calibration"].loc[
                    results["calibration"].cohort.eq(cohort) & results["calibration"].feature_set.eq(fs)
                    & results["calibration"].model.eq(model)
                ]
                ax.plot(data.mean_pred, data.observed_rate, "o-", color=color, label=name, markersize=3)
            ax.set(title=f"{titles[cohort]} | {['Case', '+ County', '+ Judge'][col]}", xlim=(0, 1), ylim=(0, 1))
            ax.grid(alpha=0.15)
            if row == 1:
                ax.set_xlabel("Mean predicted probability")
            if col == 0:
                ax.set_ylabel("Observed positive-label proportion")
    axes[0, 2].legend(loc="upper left", fontsize=8)
    fig.suptitle("Calibration on the validation year (no recalibration fitted)", fontsize=14)
    fig.text(0.02, 0.015, source + f" Quantile bins; bins below n={config.MIN_GROUP_N} are omitted.", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save(fig, "bd06_calibration")


def write_validation_report(results, provenance, output_dir=None):
    """Write BD-06 aggregate artifacts and the group model-selection handoff."""
    import json

    directory = output_dir or config.REPORTS
    directory.mkdir(parents=True, exist_ok=True)
    for name in ["metrics", "increments", "model_differences", "calibration", "cohorts", "coverage"]:
        results[name].to_csv(directory / f"bd06_{name}.csv", index=False)
    (directory / "bd06_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n",
    )
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config.DATA_PROCESSED / "bd06_auc_bootstrap.npz", **results["bootstrap"])

    def table(frame, columns):
        lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"]*len(columns)) + "|"]
        for _, row in frame.iterrows():
            lines.append("| " + " | ".join(
                f"{row[c]:.6f}" if isinstance(row[c], (float, np.floating)) else str(row[c])
                for c in columns) + " |")
        return "\n".join(lines)

    full = results["metrics"].loc[results["metrics"].cohort.eq("all_valid")]
    learned = full.loc[~full.model.eq("rule")]
    winner = learned.sort_values("roc_auc", ascending=False).iloc[0]
    cohorts = results["cohorts"].set_index("cohort")
    text = f'''# BD-06 validation comparison and model recommendation

Analysis date: {provenance["analysis_date"]}. Validation year: {config.VALID_YEAR}.

**Status:** Shuyi's comparison is complete. The validation-AUC candidate is **{winner["model"]} / {winner["feature_set"]}**, AUC **{winner["roc_auc"]:.6f}**, AP **{winner["avg_precision"]:.6f}**, Brier **{winner["brier"]:.6f}**. This is a recommendation for the group decision, not a claim that the model, thresholds or final-test workflow have already been frozen.

## Inputs and boundaries

- Main source commit: `{provenance["main_commit"]}`. Ensemble predictions: PR #7 commit `{provenance["ensemble_source_commit"]}`, final training-only early-stopping [Release]({provenance["ensemble_release"]}). The PR is a separate review/merge step.
- The ensemble file and companion manifest match the published Release SHA-256 digests. All twelve ensemble metric rows were independently recomputed against the manifest; main baseline validation metrics were checked against the committed main report.
- Training years: {config.TRAIN_YEARS}; validation: {config.VALID_YEAR}. Every supplied model-feature pair has the same case IDs, years and labels. Test/apparent-training rows, nondevelopment years, duplicates, missing cases and invalid probabilities are rejected before comparison.
- Five models x three main feature sets; the two-field charge rule appears once under CASE. Gender supplements are audited for alignment but excluded from all primary comparisons. Race, ethnicity and gender are not added to main predictors.
- No model was retrained, recalibrated or tuned in BD-06. No fairness threshold was selected. No final-test raw file or test prediction was read or generated, and `run_test=True` remains disabled.

## Full validation comparison

{table(full, ["model", "feature_set", "roc_auc", "auc_ci_low", "auc_ci_high", "avg_precision", "brier", "mean_pred", "calibration_gap", "n"])}

AUC is ranking, not accuracy. AP is interpreted alongside positive-label prevalence; lower Brier indicates better probability accuracy. Calibration gap is mean prediction minus observed label prevalence. The rule's CASE label does not mean it uses all CASE inputs. The SVM screening decision is documented in the role handoff; no extra family is trained here.

![Validation model comparison](figures/bd06_validation_auc.png)

## Nested information gains: paired intervals

{table(results["increments"], ["cohort", "model", "comparison", "delta_roc_auc", "ci_low", "ci_high", "delta_avg_precision", "delta_brier", "n"])}

Differences are larger minus smaller feature set on the same cases. Positive AUC/AP and negative Brier differences favor the larger set. Include negative results: the direction of a judge gain is not assumed to be positive for every algorithm. An AUC difference is not the proportion of decisions caused or explained by a judge.

![Paired information gains](figures/bd06_nested_auc_gains.png)

## Individually coded judge subset

The all-case cohort contains **{int(cohorts.loc["all_valid", "n"]):,}** cases. The diagnostic subset contains **{int(cohorts.loc["coded_judges", "n"]):,}** cases (**{cohorts.loc["coded_judges", "share_of_valid"]:.2%}** of validation), selected only by training-year `judge_id` starting with J. `other`, `unseen` and missing IDs are excluded; `judge_id_final` is never used for this validation subset. Every model is rescored on this same subset, not just the judge model.

{table(results["metrics"].loc[results["metrics"].cohort.eq("coded_judges")], ["model", "feature_set", "roc_auc", "auc_ci_low", "auc_ci_high", "avg_precision", "brier", "n"])}

This subset answers a coverage diagnostic. It is not a replacement for the all-case result, a randomized judge comparison or proof of the effect of unfamiliar judges. Its case mix may differ from the whole validation sample.

## Probability calibration

![Calibration](figures/bd06_calibration.png)

Quantile-bin mean predictions are compared with observed positive-label proportions, separately in each cohort. Ties are not artificially split; bins below `config.MIN_GROUP_N` are omitted. No recalibration was fitted. Mean gaps and Brier scores complement the plots; these are overall/coded-judge diagnostics, not the protected-group fairness analysis assigned to C.

## Candidate comparisons against logistic baselines

{table(results["model_differences"], ["cohort", "model", "reference_model", "feature_set", "delta_roc_auc", "ci_low", "ci_high", "delta_avg_precision", "delta_brier"])}

These candidate diagnostics are conditional on the fitted models and validation selection. They are not multiple-comparison- or model-selection-adjusted significance claims.

## Uncertainty method and limits

- Ordinary docket-level percentile bootstrap, {config.N_BOOTSTRAP:,} draws, seed {config.RANDOM_STATE}. Each draw resamples the same docket indices for every prediction column in a cohort. Exact multiplicity-weighted AUC includes half credit for tied scores and is checked against sklearn and the existing paired-bootstrap helper. No tuning or model refitting occurs inside this bootstrap.
- Marginal AUC and paired difference intervals use the 2.5th/97.5th percentiles. Finite-draw counts are retained. Full/subset cohorts are analyzed separately; their intervals are not an interval for the difference between cohorts.
- These intervals condition on fitted predictions and do not account for model-selection uncertainty, retraining, within-judge correlation or unlinked repeated defendants. They are not judge-cluster intervals or causal estimates. Small deltas should be assessed for magnitude as well as their interval.
- Training OOF uses training-selected parameters and is a diagnostic rather than nested-CV performance. Final model choice uses validation; generalization to the next year remains untested until the separately authorized freeze and BD-07.
- Prior/pending-record reference dates remain undocumented (data/README.md D14). Court/judge assignment is not randomized. Predicting historical bail/remand decisions does not establish appropriate decisions, actual detention or future offending.

## Group decision and handoff

1. Review the candidate's AUC/AP/Brier and calibration, including negative judge gains in other model families and the individually coded-judge diagnostic.
2. C supplies the validation-year policy thresholds and protected-group error audit; tied probabilities need an explicit threshold convention. This report does not replace C's analysis.
3. Confirm and record the final family, three feature sets, selected parameters, preprocessing, package versions, threshold(s), and XGBoost stopping/refit policy before using the test year. The current 2000 rounds is a search ceiling, not the saved model's selected tree count.
4. BD-07 must separately implement/test the final refit path on expanded training years, use `get_X(..., final=True)` for the final judge mapping, preserve frozen choices and run the final test once after the 11/16 freeze. Policy-threshold confusion matrices belong to that stage.
5. E owns SHAP/permutation and C owns test fairness/residuals. Test-set explanations must not feed back into model/feature/threshold retuning. A owns final-model sensitivity checks and source/literature limitations. Q3 and extra clustering are not part of BD-06.

Only aggregate tables/figures, the notebook, evaluation code/tests and Shuyi's status/records are review candidates. Case predictions, raw files, bootstrap draws and fitted models are local-only. No teammate source file or data definition was changed.
'''
    (directory / "bd06_validation_report.md").write_text(text, encoding="utf-8", newline="\n")
    return directory / "bd06_validation_report.md"
