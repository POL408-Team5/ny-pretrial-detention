"""Role C: court/judge EDA (C-01) and the threshold / group-error audit (C-02).

C-01: training-only court/judge EDA and outcome-free coverage diagnostics.
C-02: choose_threshold, predict_label, group_metrics, residuals (reuse src.evaluate).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np
import pandas as pd

from src import config
from src.sample import to_snake


def load_court_judge_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project training outcomes and all-year identity metadata separately.

    Future outcomes are never selected from parquet, even into an intermediate
    DataFrame. Main-sample membership is supplied by the existing A pipeline.
    """
    metadata = [config.ID_COL, config.YEAR_COL, "arraign_month", "judge_id",
                config.JUDGE_COL_FINAL]
    coverage = pd.read_parquet(
        config.ANALYTIC_FILE, columns=metadata,
        filters=[(config.MAIN_SAMPLE_COL, "==", True)],
    )
    training = pd.read_parquet(
        config.ANALYTIC_FILE,
        columns=metadata + ["county_name", "top_severity_at_arraign",
                            "top_charge_weight_at_arraign",
                            "top_charge_at_arraign_violent_felony_ind",
                            config.TARGET_Q1],
        filters=[(config.MAIN_SAMPLE_COL, "==", True),
                 (config.YEAR_COL, "in", config.TRAIN_YEARS)],
    )
    _require_training(training)
    if not coverage[config.ID_COL].is_unique:
        raise ValueError("Coverage case IDs must be unique.")
    return training, coverage


def _require_training(df: pd.DataFrame) -> None:
    if df.empty or not df[config.YEAR_COL].isin(config.TRAIN_YEARS).all():
        raise ValueError("Outcome EDA accepts nonempty training-year data only.")
    if not df[config.TARGET_Q1].isin([0, 1]).all():
        raise ValueError("Training outcomes must be observed binary values.")


def _publishable(table: pd.DataFrame) -> pd.DataFrame:
    """Suppress entire cells, including counts, below the reporting minimum."""
    return table.loc[table["n"] >= config.MIN_GROUP_N].reset_index(drop=True)


def category_counts(training: pd.DataFrame) -> pd.DataFrame:
    """Observed categories, with small cells removed before any display."""
    _require_training(training)
    parts = []
    for col in ["county_name", "top_severity_at_arraign",
                "top_charge_weight_at_arraign",
                "top_charge_at_arraign_violent_felony_ind"]:
        counts = training[col].fillna("Missing").value_counts()
        parts.append(pd.DataFrame({"field": col, "value": counts.index,
                                   "n": counts.values}))
    return _publishable(pd.concat(parts, ignore_index=True))


def county_rates(training: pd.DataFrame) -> pd.DataFrame:
    """County outcome rates, pooled and within each observed severity level."""
    _require_training(training)
    pooled = training.groupby("county_name")[config.TARGET_Q1].agg(
        n="size", observed_rate="mean").reset_index()
    pooled["severity"] = "All"
    stratified = training.groupby(
        ["county_name", "top_severity_at_arraign"], observed=True,
        dropna=False)[config.TARGET_Q1].agg(n="size", observed_rate="mean").reset_index()
    stratified = stratified.rename(columns={"top_severity_at_arraign": "severity"})
    return _publishable(pd.concat([pooled, stratified], ignore_index=True))


def judge_rates(training: pd.DataFrame) -> pd.DataFrame:
    """Rates and indirect standardization for individually coded judges.

    Reference cell means use all training main-sample cases (including other
    judges). Each case receives the pooled mean for its charge-weight x violent
    indicator cell; expected_rate is the judge-specific average of those means.
    These are descriptive in-sample references, not model predictions or causal
    effects. Small reference cells may contribute internally but are not output.
    Within-severity rows also require MIN_GROUP_N cases in that severity.
    """
    _require_training(training)
    frame = training.copy()
    grid = ["top_charge_weight_at_arraign",
            "top_charge_at_arraign_violent_felony_ind"]
    frame[grid] = frame[grid].astype("string").fillna("Missing")
    frame["expected"] = frame.groupby(grid, observed=True)[config.TARGET_Q1].transform("mean")
    totals = frame.groupby("judge_id").size()
    eligible = totals[totals >= config.MIN_JUDGE_CASES].index
    frame = frame.loc[frame["judge_id"].str.fullmatch(r"J\d+").fillna(False)
                      & frame["judge_id"].isin(eligible)]
    pooled = frame.groupby("judge_id").agg(
        n=(config.TARGET_Q1, "size"), observed_rate=(config.TARGET_Q1, "mean"),
        expected_rate=("expected", "mean")).reset_index()
    pooled["severity"] = "All"
    within = frame.groupby(["judge_id", "top_severity_at_arraign"], observed=True).agg(
        n=(config.TARGET_Q1, "size"), observed_rate=(config.TARGET_Q1, "mean"),
        expected_rate=("expected", "mean")).reset_index().rename(
            columns={"top_severity_at_arraign": "severity"})
    result = _publishable(pd.concat([pooled, within], ignore_index=True))
    result["observed_minus_expected"] = result["observed_rate"] - result["expected_rate"]
    return result.sort_values(["severity", "judge_id"]).reset_index(drop=True)


def judge_percentiles(rates: pd.DataFrame) -> pd.DataFrame:
    """Unweighted percentiles across eligible judges, not across cases."""
    rows = []
    for severity, group in rates.groupby("severity", observed=True):
        row = {"severity": severity, "n_judges": len(group), "n": int(group["n"].sum())}
        for metric in ["observed_rate", "observed_minus_expected"]:
            for quantile in [0.1, 0.5, 0.9]:
                row[f"{metric}_p{int(quantile * 100)}"] = group[metric].quantile(quantile)
        rows.append(row)
    return _publishable(pd.DataFrame(rows))


def coverage_tables(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Outcome-free yearly and monthly shares of encoded/other/unseen judges.

    final coding is reported for the test year only as an encoding diagnostic;
    it is not used to change training thresholds or choose a model.
    """
    if config.TARGET_Q1 in metadata:
        raise ValueError("Coverage functions must receive outcome-free metadata.")
    yearly, monthly = [], []
    for col in ["judge_id", config.JUDGE_COL_FINAL]:
        frame = metadata.copy()
        if col == config.JUDGE_COL_FINAL:
            frame = frame.loc[frame[config.YEAR_COL].eq(config.TEST_YEAR)]
        frame["status"] = frame[col].where(
            ~frame[col].str.fullmatch(r"J\d+").fillna(False), "coded")
        for keys, destination in [([config.YEAR_COL], yearly),
                                   ([config.YEAR_COL, "arraign_month"], monthly)]:
            table = frame.groupby(keys + ["status"], observed=True).size().rename("n").reset_index()
            table["n_total"] = table.groupby(keys)["n"].transform("sum")
            table["case_share"] = table["n"] / table["n_total"]
            table["coding"] = col
            destination.append(_publishable(table))
    return pd.concat(yearly, ignore_index=True), pd.concat(monthly, ignore_index=True)


def identity_caseload_summaries(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count actual judges without exposing names or changing published IDs.

    The analytic file has collapsed low-frequency and unseen identities. Recover
    counts from raw name + case-ID columns only, matched to main-sample metadata.
    Names exist only transiently inside this function. No outcome column is read;
    no name-to-code mapping, row-level identity, or individual small cell returns.
    New means absent from the *training main sample*, matching A's coding rule;
    it does not necessarily mean a newly appointed judge.
    """
    if config.TARGET_Q1 in metadata:
        raise ValueError("Identity diagnostics require outcome-free metadata.")
    counts_by_year = {}
    for year in config.YEARS:
        sub = metadata.loc[metadata[config.YEAR_COL].eq(year), [config.ID_COL, "judge_id"]]
        if sub.empty:
            continue
        path = config.DATA_RAW / config.RAW_FILE_PATTERN.format(year=year)
        if not path.exists():
            path = path.with_suffix("")
        raw = pd.read_csv(path, usecols=lambda c: to_snake(c) in
                          {"internal_case_id", "judge_name"},
                          encoding="utf-8-sig", dtype="string")
        raw.columns = [to_snake(c) for c in raw.columns]
        raw = raw.rename(columns={"internal_case_id": config.ID_COL})
        raw["judge_name"] = raw["judge_name"].str.strip().replace("", pd.NA)
        joined = sub.merge(raw, on=config.ID_COL, how="left", validate="one_to_one", indicator=True)
        if not joined["_merge"].eq("both").all():
            raise ValueError("Raw identity lookup did not match every main-sample case.")
        counts_by_year[year] = joined["judge_name"].value_counts()
        del raw, joined

    train_counts = pd.concat([counts_by_year[y] for y in config.TRAIN_YEARS], axis=1).fillna(0).sum(axis=1)
    total_train = int(metadata[config.YEAR_COL].isin(config.TRAIN_YEARS).sum())
    threshold_rows = []
    for threshold in [config.MIN_JUDGE_CASES, 2 * config.MIN_JUDGE_CASES]:
        selected = train_counts.loc[train_counts >= threshold]
        threshold_rows.append({"min_cases": threshold, "n_judges": len(selected),
                               "n": int(selected.sum()), "n_total": total_train,
                               "case_share": selected.sum() / total_train,
                               "n_judges_below_threshold": int((train_counts < threshold).sum())})

    distributions = []
    def summarize(counts, year, cohort):
        if counts.sum() < config.MIN_GROUP_N:
            return
        distributions.append({"period": str(year), "cohort": cohort,
                              "n_judges": len(counts), "n": int(counts.sum()),
                              "caseload_p10": counts.quantile(0.1),
                              "caseload_p50": counts.quantile(0.5),
                              "caseload_p90": counts.quantile(0.9),
                              "n_judges_ge_min": int((counts >= config.MIN_JUDGE_CASES).sum())})
    summarize(train_counts, "Training", "All named identities")
    for year in [config.VALID_YEAR, config.TEST_YEAR]:
        counts = counts_by_year[year]
        new = counts.loc[~counts.index.isin(train_counts.index)]
        expected_n = metadata.loc[metadata[config.YEAR_COL].eq(year), "judge_id"].eq("unseen").sum()
        if int(new.sum()) != int(expected_n):
            raise ValueError("New-identity counts disagree with A's unseen coding.")
        summarize(new, year, "Unseen in training main sample")
        summarize(counts.loc[counts.index.isin(train_counts.index)], year, "Seen in training main sample")
    return _publishable(pd.DataFrame(threshold_rows)), _publishable(pd.DataFrame(distributions))


def plot_county_rates(table: pd.DataFrame):
    """Deck figure: common percentage scale across pooled and severity panels."""
    levels = ["All"] + sorted(set(table["severity"]) - {"All"})
    fig, axes = plt.subplots(1, len(levels), figsize=(12, 4.7), sharex=True, sharey=True)
    colors = ["#1E5367", "#A85D35", "#447D6D"]
    for ax, severity, color in zip(np.atleast_1d(axes), levels, colors):
        group = table.loc[table["severity"].eq(severity)].set_index("county_name").reindex(config.NYC_COUNTIES)
        y = np.arange(len(group))
        ax.scatter(group["observed_rate"], y, s=65, color=color, zorder=3)
        for pos, rate in enumerate(group["observed_rate"]):
            if pd.notna(rate):
                ax.annotate(f"{rate:.1%}", (rate, pos), xytext=(7, 0), textcoords="offset points", va="center", fontsize=10)
        ax.set_title(severity, fontsize=13, loc="left")
        ax.set_yticks(y, config.NYC_COUNTIES)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="x", color="#E2E6E8", zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_xlabel("Bail-set or remanded")
    axes[0].invert_yaxis()
    max_rate = table["observed_rate"].max()
    axes[0].set_xlim(0, max_rate * 1.3)
    years = "–".join(map(str, [min(config.TRAIN_YEARS), max(config.TRAIN_YEARS)]))
    fig.suptitle("County differences depend on charge severity", x=.07, ha="left", fontsize=17, weight="bold")
    fig.text(.07, .025, f"Source: NY UCS; NYC main sample, {years}. Descriptive rates; not actual detention or causal effects.", fontsize=9, color="#53616A")
    fig.tight_layout(rect=[.03, .08, 1, .9])
    return fig


def plot_judge_spread(rates: pd.DataFrame):
    """Deck figure: all eligible judges as anonymous dots, without a ranking."""
    levels = sorted(set(rates["severity"]) - {"All"})
    fig, axes = plt.subplots(len(levels), 2, figsize=(12, 7), squeeze=False)
    for row, severity in enumerate(levels):
        group = rates.loc[rates["severity"].eq(severity)].sort_values("judge_id")
        for col, metric in enumerate(["observed_rate", "observed_minus_expected"]):
            ax = axes[row, col]
            y = np.arange(len(group))
            # Deterministic ID order, not outcome ranking; points are judges.
            ax.scatter(group[metric], y, s=18, color="#1E5367" if col == 0 else "#A85D35", alpha=.65)
            ax.set_yticks([])
            if col:
                ax.axvline(0, color="#667580", lw=1)
            for q, style in [(0.1, ":"), (0.5, "--"), (0.9, ":")]:
                ax.axvline(group[metric].quantile(q), color="#667580", ls=style, lw=.8)
            ax.xaxis.set_major_formatter(
                PercentFormatter(1) if col == 0
                else FuncFormatter(lambda x, _: f"{x * 100:+.0f}"))
            ax.grid(axis="x", color="#E2E6E8")
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.set_title(f"{severity} · {len(group)} judges", loc="left", fontsize=12)
            ax.set_xlabel("Bail-set or remanded" if col == 0 else "Observed − expected (percentage points)")
    years = "–".join(map(str, [min(config.TRAIN_YEARS), max(config.TRAIN_YEARS)]))
    fig.suptitle("Judge differences remain after a coarse case-mix adjustment", x=.06, ha="left", fontsize=16, weight="bold")
    fig.text(.06, .04, f"Each dot = one anonymous judge in ID order; ≥{config.MIN_GROUP_N} cases within severity. Lines: judge P10 / P50 / P90.\nExpected = pooled charge-weight × violent-indicator cell rate; NYC main sample, {years}, NY UCS. No causal interpretation.", fontsize=9, color="#53616A")
    fig.tight_layout(rect=[.02, .12, 1, .93])
    return fig


# ---------------------------------------------------------------------------
# C-02: policy threshold, group error audit and residuals (written by Yiling at
# merge, 2026-11-15, because C-02 was due before the freeze). Validation year only
# until the BD-07 test run; the functions themselves do not look at the year.
# ---------------------------------------------------------------------------

def choose_threshold(y_prob_valid, y_true_valid) -> float:
    """Policy threshold: predicted share of 1s in 2023 = actual 2023 share of y_detain.

    Rule (frozen 2026-11-15): predict 1 when y_prob >= threshold. With k = round(n * actual
    share), the threshold is the k-th highest score. If tied scores make an exact k
    impossible, take whichever of the two neighbouring distinct scores gives a predicted
    share closer to the actual share (the higher score on an exact tie). Each model and
    feature set gets its own threshold by this same rule; the numbers are not shared.
    """
    from src.evaluate import _inputs

    y, p = _inputs(y_true_valid, y_prob_valid)
    k = int(round(len(y) * y.mean()))
    if k == 0:
        return float(np.nextafter(p.max(), np.inf))   # nothing predicted positive
    ordered = np.sort(p)[::-1]
    candidates = np.unique(ordered)[::-1]               # distinct scores, high to low
    t_k = ordered[k - 1]
    i = int(np.searchsorted(-candidates, -t_k))        # position of t_k among candidates
    options = [candidates[i]] + ([candidates[i - 1]] if i > 0 else [])
    def gap(t):
        return abs(int((p >= t).sum()) - k)
    return float(min(options, key=lambda t: (gap(t), -t)))


def predict_label(y_prob, threshold: float) -> np.ndarray:
    """0/1 decision under the frozen rule (>= threshold)."""
    return (np.asarray(y_prob, dtype=float) >= threshold).astype(int)


def _one_spec(pred_df: pd.DataFrame, split: str) -> pd.DataFrame:
    part = pred_df.loc[pred_df["split"].eq(split)]
    specs = part[["model", "feature_set", "target"]].drop_duplicates()
    if len(specs) != 1:
        raise ValueError("Pass predictions for exactly one model, feature set and target.")
    if part[config.ID_COL].duplicated().any():
        raise ValueError("Each case must appear once per split.")
    return part


def _attach(pred_df: pd.DataFrame, df: pd.DataFrame, col: str, split: str) -> pd.DataFrame:
    part = _one_spec(pred_df, split)
    meta = df[[config.ID_COL, col]].drop_duplicates(config.ID_COL)
    out = part.merge(meta, on=config.ID_COL, how="left", validate="one_to_one")
    if len(out) != len(part):
        raise ValueError("Case identifiers did not align.")
    out[col] = out[col].astype("string").fillna("Missing")
    return out


def group_metrics(pred_df, df, group_col, threshold, split="valid") -> pd.DataFrame:
    """Per-group calibration and errors at a fixed threshold.

    Errors are relative to the historical decision label, not to a "correct" decision.
    A group is dropped if n < MIN_GROUP_N. FPR (FNR) is blanked when the group has fewer
    than MIN_GROUP_N actual 0s (1s), and AUC when either class is below that size.
    """
    from src.evaluate import metrics

    data = _attach(pred_df, df, group_col, split)
    rows = []
    for group, g in data.groupby(group_col, sort=True):
        y, p = g["y_true"].to_numpy(int), g["y_prob"].to_numpy(float)
        if len(y) < config.MIN_GROUP_N:
            continue
        yhat = predict_label(p, threshold)
        n_pos, n_neg = int(y.sum()), int((1 - y).sum())
        fp, fn = int(((yhat == 1) & (y == 0)).sum()), int(((yhat == 0) & (y == 1)).sum())
        enough_pos, enough_neg = n_pos >= config.MIN_GROUP_N, n_neg >= config.MIN_GROUP_N
        m = metrics(y, p)
        rows.append({
            "group_col": group_col, "group": group, "n": len(y), "n_pos": n_pos, "n_neg": n_neg,
            "base_rate": float(y.mean()), "mean_pred": float(p.mean()),
            "calibration_gap": float(p.mean() - y.mean()),
            "roc_auc": m["roc_auc"] if enough_pos and enough_neg else np.nan,
            "pred_pos_rate": float(yhat.mean()),
            "fpr": fp / n_neg if enough_neg else np.nan,
            "fnr": fn / n_pos if enough_pos else np.nan,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        spec = _one_spec(pred_df, split).iloc[0]
        out.insert(0, "feature_set", spec["feature_set"])
        out.insert(0, "model", spec["model"])
        out["threshold"] = threshold
    return out


def residuals(pred_df, df, by, split="valid", n_boot=None, seed=None) -> pd.DataFrame:
    """Observed rate minus mean prediction by county or judge code, with a bootstrap interval.

    Cases are resampled within each group (percentile interval). A positive residual means
    more bail/remand than the model predicts for those cases. This is a prediction residual,
    not a causal judge effect. Groups with n < MIN_GROUP_N are dropped.
    """
    n_boot = config.N_BOOTSTRAP if n_boot is None else n_boot
    rng = np.random.default_rng(config.RANDOM_STATE if seed is None else seed)
    data = _attach(pred_df, df, by, split)
    rows = []
    for group, g in data.groupby(by, sort=True):
        if len(g) < config.MIN_GROUP_N:
            continue
        diff = g["y_true"].to_numpy(float) - g["y_prob"].to_numpy(float)
        boot = np.concatenate([diff[rng.integers(0, len(diff), size=(b, len(diff)))].mean(axis=1)
                               for b in np.diff(np.r_[0:n_boot:50, n_boot])])   # 50 draws at a time
        rows.append({"by": by, "group": group, "n": len(g),
                     "observed_rate": float(g["y_true"].mean()), "mean_pred": float(g["y_prob"].mean()),
                     "residual": float(diff.mean()),
                     "ci_low": float(np.quantile(boot, 0.025)), "ci_high": float(np.quantile(boot, 0.975))})
    out = pd.DataFrame(rows)
    if not out.empty:
        spec = _one_spec(pred_df, split).iloc[0]
        out.insert(0, "feature_set", spec["feature_set"])
        out.insert(0, "model", spec["model"])
    return out
