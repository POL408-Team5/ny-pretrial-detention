"""Load the raw UCS Pretrial Release files and build the analytic sample (role A).

Steps (plan v2, section 3; AGENTS.md section 3.3):
    S0  all rows in the raw files
    S1  NYC counties
    S2  drop "Disposed at arraign"           (no release decision)
    S3  drop "Unknown"                       -> every row from here on is kept in analytic.parquet
    S4  case type in CASE_TYPES_MAIN          (flag: excl_case_type)
    S5  drop nominal bail                     (flag: nominal_bail; only if EXCLUDE_NOMINAL_BAIL)
    S6  one docket per arrest cycle           (flag: dup_cycle; only if DEDUP_ARREST_CYCLE)
        -> main_sample

Public functions:
    load_raw(years=None) -> DataFrame
    build_sample(raw) -> (analytic DataFrame, sample_flow DataFrame)
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def to_snake(name: str) -> str:
    """'Release Decision at Arraign' -> 'release_decision_at_arraign'."""
    name = name.replace("﻿", "").strip()
    return re.sub(r"[\s\-]+", "_", name).lower()


def _raw_path(year: int) -> Path | None:
    """Return the gzip file if present, else the plain CSV, else None."""
    gz = config.DATA_RAW / config.RAW_FILE_PATTERN.format(year=year)
    if gz.exists():
        return gz
    plain = gz.with_suffix("") if gz.suffix == ".gz" else gz
    return plain if plain.exists() else None


def _clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip text columns; empty strings become missing."""
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            s = df[col].astype("string").str.strip()
            df[col] = s.mask(s == "")
    return df


def load_raw(years: list[int] | None = None, nyc_only: bool = True) -> pd.DataFrame:
    """Read the raw yearly files, convert column names to snake_case and add year columns.

    nyc_only=True keeps only NYC rows (step S1) while reading, to save memory; the
    all-row counts per year (step S0) are kept in `raw.attrs["s0_counts"]`.

    Adds:
        file_year     year in the file name
        arraign_year  year of first_arraign_date (format M/1/YYYY)
    Raises if the files do not share the same columns.
    """
    years = years or config.YEARS
    frames, columns, s0_counts = [], {}, {}
    for year in years:
        path = _raw_path(year)
        if path is None:
            warnings.warn(f"No raw file for {year} in {config.DATA_RAW}; skipping that year.")
            continue
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        df.columns = [to_snake(c) for c in df.columns]
        columns[year] = list(df.columns)
        s0_counts[year] = len(df)
        if nyc_only:
            df = df[df["county_name"].isin(config.NYC_COUNTIES)]
        df["file_year"] = year
        frames.append(df)
        print(f"  read {path.name}: {s0_counts[year]:,} rows" + (f" ({len(df):,} NYC)" if nyc_only else ""))
    if not frames:
        raise FileNotFoundError(f"No raw files found in {config.DATA_RAW}")

    ref_year = next(iter(columns))
    for year, cols in columns.items():
        if cols != columns[ref_year]:
            only_ref = sorted(set(columns[ref_year]) - set(cols))
            only_this = sorted(set(cols) - set(columns[ref_year]))
            raise ValueError(
                f"Columns differ between {ref_year} and {year}: "
                f"only in {ref_year}: {only_ref}; only in {year}: {only_this}"
            )

    raw = pd.concat(frames, ignore_index=True)
    del frames
    raw = _clean_strings(raw)
    dates = pd.to_datetime(raw["first_arraign_date"], format="%m/%d/%Y", errors="coerce")
    raw[config.YEAR_COL] = dates.dt.year.astype("Int64")
    raw["arraign_month"] = dates.dt.month.astype("Int64")

    mismatch = (raw[config.YEAR_COL] != raw["file_year"]).fillna(True)
    if mismatch.any():
        warnings.warn(
            f"{int(mismatch.sum()):,} rows have arraign_year different from the file year "
            "or a missing arraignment date; arraign_year is set to the file year for these rows."
        )
        raw.loc[mismatch, config.YEAR_COL] = raw.loc[mismatch, "file_year"]
    raw.attrs["s0_counts"] = s0_counts
    return raw


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------
def _flow_row(step: str, rule: str, df: pd.DataFrame, years: list[int]) -> dict:
    counts = df[config.YEAR_COL].value_counts()
    row = {"step": step, "rule": rule}
    for y in years:
        row[str(y)] = int(counts.get(y, 0))
    row["total"] = int(len(df))
    return row


def _nominal_bail(df: pd.DataFrame) -> pd.Series:
    """Bail set and every non-missing first bail amount <= NOMINAL_BAIL_MAX (at least one amount)."""
    amounts = df[config.BAIL_AMOUNT_COLS].apply(pd.to_numeric, errors="coerce")
    has_amount = amounts.notna().any(axis=1)
    max_amount = amounts.max(axis=1, skipna=True)
    bail_set = df[config.RELEASE_COL].eq("Bail-set").fillna(False)
    return (bail_set & has_amount & (max_amount <= config.NOMINAL_BAIL_MAX)).astype(int)


def _dedup_cycles(df: pd.DataFrame, eligible: pd.Series) -> pd.Series:
    """Flag all but one docket within each arr_cycle_id (among eligible rows).

    Kept docket: earliest arraignment year first (so a later year never changes an earlier
    year's sample), then most serious top charge (SEVERITY_ORDER; unknown last), then smallest
    case_id. Rows without arr_cycle_id are never flagged.
    """
    rank = {w: i for i, w in enumerate(config.SEVERITY_ORDER)}
    sub = df.loc[eligible & df[config.CYCLE_COL].notna(), [config.CYCLE_COL, config.ID_COL, config.YEAR_COL]].copy()
    sub["_sev"] = (
        df.loc[sub.index, "top_charge_weight_at_arraign"].map(rank).fillna(len(rank)).astype(int)
    )
    sub = sub.sort_values([config.CYCLE_COL, config.YEAR_COL, "_sev", config.ID_COL])
    dup = sub.groupby(config.CYCLE_COL).cumcount() > 0
    out = pd.Series(0, index=df.index)
    out.loc[dup[dup].index] = 1
    return out


def _judge_ids(df: pd.DataFrame, fit_years: list[int]) -> pd.Series:
    """Anonymous judge codes from main-sample caseloads in `fit_years`.

    J001, J002, ... in descending training caseload (>= MIN_JUDGE_CASES);
    'other'  = judge seen in fit_years with fewer cases;
    'unseen' = judge not seen in fit_years (later years only);
    'missing' = no judge name. The name-to-code table is not saved.
    """
    train = df[config.YEAR_COL].isin(fit_years) & df[config.MAIN_SAMPLE_COL]
    counts = df.loc[train, "judge_name"].value_counts()
    counts = counts.sort_index().sort_values(ascending=False, kind="stable")  # deterministic ties
    frequent = counts[counts >= config.MIN_JUDGE_CASES].index
    code = {name: f"J{i:03d}" for i, name in enumerate(frequent, start=1)}
    seen = set(counts.index)

    names = df["judge_name"]
    ids = names.map(code)
    ids = ids.where(ids.notna(), np.where(names.isin(seen), "other", "unseen"))
    ids = ids.where(names.notna(), "missing")
    return ids.astype("string")


def build_sample(
    raw: pd.DataFrame,
    exclude_nominal_bail: bool | None = None,
    dedup_arrest_cycle: bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply S1-S6, build outcomes, flags, judge_id, age_group and split.

    Returns (analytic, sample_flow). `analytic` keeps every row after S3;
    use `main_sample` to select the modeling sample. The two switches default to
    config.EXCLUDE_NOMINAL_BAIL / config.DEDUP_ARREST_CYCLE; the A1-06 sensitivity
    versions (config.SENSITIVITY_VARIANTS) turn one of them off.
    """
    if exclude_nominal_bail is None:
        exclude_nominal_bail = config.EXCLUDE_NOMINAL_BAIL
    if dedup_arrest_cycle is None:
        dedup_arrest_cycle = config.DEDUP_ARREST_CYCLE
    years = sorted(raw[config.YEAR_COL].dropna().unique().tolist())
    s0 = raw.attrs.get("s0_counts")
    if s0:  # load_raw(nyc_only=True) already applied S1
        row = {"step": "S0", "rule": "All rows in the raw files"}
        row.update({str(y): int(s0.get(y, 0)) for y in years})
        row["total"] = int(sum(s0.get(y, 0) for y in years))
        flow = [row]
    else:
        flow = [_flow_row("S0", "All rows in the raw files", raw, years)]

    df = raw[raw["county_name"].isin(config.NYC_COUNTIES)]
    flow.append(_flow_row("S1", "NYC counties (New York, Kings, Bronx, Queens, Richmond)", df, years))

    df = df[df[config.RELEASE_COL] != config.RELEASE_DISPOSED]
    flow.append(_flow_row("S2", "Drop cases disposed at arraignment (no release decision)", df, years))

    df = df[df[config.RELEASE_COL] != config.RELEASE_UNKNOWN].copy()
    flow.append(_flow_row("S3", "Drop unknown release decisions (= rows in analytic.parquet)", df, years))

    expected = set(config.RELEASE_DETAIN + config.RELEASE_FREE)
    unexpected = set(df[config.RELEASE_COL].dropna().unique()) - expected
    if unexpected or df[config.RELEASE_COL].isna().any():
        raise ValueError(f"Unexpected release decision values: {unexpected} (or missing values)")

    # --- identifiers and outcomes ---
    df[config.ID_COL] = df["internal_case_id"].astype("string")  # hex string, e.g. 0xF799...; not numeric
    if not df[config.ID_COL].is_unique:
        raise ValueError("case_id is not unique")
    release = df[config.RELEASE_COL]
    df[config.TARGET_Q1] = release.isin(config.RELEASE_DETAIN).astype(int)
    df["nominal_bail"] = _nominal_bail(df)
    df["bail_posted_at_arraign"] = (
        release.eq("Bail-set") & df["bail_set_and_posted_at_arraign"].eq("Y").fillna(False)
    ).astype(int)
    df[config.RELEASED_COL] = (
        release.isin(config.RELEASE_FREE) | df["bail_posted_at_arraign"].eq(1)
    ).astype(int)
    df[config.TARGET_Q3] = df["warrant_ordered_btw_arraign_and_dispo"].map({"Y": 1, "N": 0}).astype("Int64")
    rearrest = df["rearrest"]
    df[config.TARGET_Q3_ALT] = (
        pd.Series(np.where(rearrest.eq("No Arrest").fillna(False), 0, 1), index=df.index)
        .astype("Int64")
        .mask(rearrest.isna())
    )
    df["dcjs_missing"] = df["prior_vfo_cnt"].isna().astype(int)
    # age_at_arrest == 0 is a placeholder (~0.4% of rows): use age_at_crime when it is positive, else missing.
    age = pd.to_numeric(df["age_at_arrest"], errors="coerce")
    crime_age = pd.to_numeric(df["age_at_crime"], errors="coerce")
    df["age_fixed"] = age.eq(0).astype(int)
    df["age_at_arrest"] = age.mask(age.eq(0), crime_age.where(crime_age > 0))

    # --- S4-S6 ---
    df["excl_case_type"] = (~df["case_type"].isin(config.CASE_TYPES_MAIN)).astype(int)
    keep = df["excl_case_type"].eq(0)
    flow.append(_flow_row("S4", f"Case type in {config.CASE_TYPES_MAIN}", df[keep], years))

    if exclude_nominal_bail:
        keep &= df["nominal_bail"].eq(0)
        flow.append(_flow_row("S5", f"Drop nominal bail (all first bail amounts <= ${config.NOMINAL_BAIL_MAX:g})",
                              df[keep], years))

    df["dup_cycle"] = _dedup_cycles(df, keep)
    if dedup_arrest_cycle:
        keep &= df["dup_cycle"].eq(0)
        flow.append(_flow_row("S6", "One docket per arr_cycle_id (earliest year, then most serious charge)", df[keep], years))

    df[config.MAIN_SAMPLE_COL] = keep.astype(bool)
    flow.append(_flow_row("MAIN", "Main sample", df[keep], years))

    # --- judge, age group, split ---
    # judge_id: coded on training years (model selection on 2023);
    # judge_id_final: coded on training + validation years (final refit before the 2024 test).
    df["judge_id"] = _judge_ids(df, config.TRAIN_YEARS)
    df[config.JUDGE_COL_FINAL] = _judge_ids(df, config.TRAIN_YEARS + [config.VALID_YEAR])
    df = df.drop(columns=["judge_name"])
    df["age_group"] = pd.cut(
        df["age_at_arrest"], bins=config.AGE_BINS, labels=config.AGE_LABELS, right=False
    ).astype("string")
    split_map = {y: "train" for y in config.TRAIN_YEARS}
    split_map[config.VALID_YEAR] = "valid"
    if config.TEST_YEAR is not None:
        split_map[config.TEST_YEAR] = "test"
    df[config.SPLIT_COL] = df[config.YEAR_COL].map(split_map).astype("string")

    front = [config.ID_COL, config.CYCLE_COL, config.YEAR_COL, "arraign_month", config.SPLIT_COL,
             config.MAIN_SAMPLE_COL, config.TARGET_Q1, "nominal_bail", "bail_posted_at_arraign",
             config.RELEASED_COL, config.TARGET_Q3, config.TARGET_Q3_ALT,
             "excl_case_type", "dup_cycle", "dcjs_missing", "age_fixed", "judge_id", config.JUDGE_COL_FINAL, "age_group"]
    df = df[front + [c for c in df.columns if c not in front]].reset_index(drop=True)
    return df, pd.DataFrame(flow)


# ---------------------------------------------------------------------------
# Diagnostics used by build_dataset and notebook 01
# ---------------------------------------------------------------------------
def diagnostics(df: pd.DataFrame) -> dict:
    """Counts that should be reported alongside the sample flow (aggregates only)."""
    main = df[df[config.MAIN_SAMPLE_COL]]
    cyc = df.loc[df[config.CYCLE_COL].notna(), [config.CYCLE_COL, config.YEAR_COL]]
    span = cyc.groupby(config.CYCLE_COL)[config.YEAR_COL].nunique()
    bail_set = df[config.RELEASE_COL].eq("Bail-set")
    jid = main["judge_id"]
    return {
        "rows_analytic": len(df),
        "rows_main": len(main),
        # outcome rates only for training / validation years: the test year stays unseen until the freeze
        "y_detain_rate_main_by_year (train/valid only)": main[main[config.YEAR_COL] != config.TEST_YEAR]
        .groupby(config.YEAR_COL)[config.TARGET_Q1].mean().round(4).to_dict(),
        "nominal_bail_share_of_bail_set (train/valid only)": round(
            df.loc[bail_set & (df[config.YEAR_COL] != config.TEST_YEAR), "nominal_bail"].mean(), 4),
        "bail_set_without_amount": int((bail_set & df[config.BAIL_AMOUNT_COLS].isna().all(axis=1)).sum()),
        "cycles_spanning_years": int((span > 1).sum()),
        "rows_in_cycles_spanning_years": int(cyc[config.CYCLE_COL].isin(span[span > 1].index).sum()),
        "n_coded_judges": int(jid[jid.str.startswith("J").fillna(False)].nunique()),
        "judge_id_share_by_year": (
            main.assign(k=jid.where(~jid.str.startswith("J").fillna(False), "coded"))
            .groupby(config.YEAR_COL)["k"].value_counts(normalize=True).round(4).to_dict()
        ),
    }
