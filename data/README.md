# Data

The raw data are public. To keep everyone on the same release, the raw files are committed **gzip-compressed** (`data/raw/*.csv.gz`, 23–31 MB each; pandas reads them directly). Uncompressed CSVs (160–210 MB) exceed GitHub's 100 MB file limit and are ignored. `data/processed/` is not committed: build it locally with `python -m src.build_dataset`. Only role A updates the raw files.

## Source

New York State Unified Court System, Division of Technology & Court Research. *Pretrial Release Data* (state-paid courts).
https://www.nycourts.gov/division-technology-court-research/pretrial-release-data

Per the official documentation (`docs/Pretrial Release Data documentation- for website.pdf`):

- One row per de-identified **defendant-docket**. Co-defendants and multiple dockets of the same person cannot be linked; dockets from the same arrest event can be linked through `arr_cycle_id`, which is present only when the docket matched a DCJS arrest record.
- Only dockets with a top arraignment charge of a felony or misdemeanor are included.
- Data are refreshed every six months, and **cases from earlier periods are updated** (e.g., a case pending in one release may be disposed in the next). Always record the download date.
- Missing DCJS (criminal history) fields mean no matching DCJS record, **not** no criminal history.
- NYC community courts (Midtown, Red Hook) are complete only for arraignments after 2021-03-28; superior courts only after each court's `UCMSLiveDate`. In the NYC files, community courts are not listed separately from the county Criminal Courts.

## Files in `data/raw/`

| File | Rows | Downloaded | sha256 of the uncompressed CSV (first 16) |
|---|---:|---|---|
| `NYS for Web 2021.csv.gz` | 246,202 | 2026-11-06 | `b676e4f9f4e30d62` |
| `NYS for Web 2022.csv.gz` | 286,588 | 2026-11-06 | `b745ab9e902b687d` |
| `NYS for Web 2023.csv.gz` | 324,619 | 2026-11-06 | `01dddf25180b9bad` |
| `NYS for Web 2024.csv.gz` | — | **to download** | check columns against 2021–2023 |

All 131 columns are identical across 2021–2023. Check a file: `gunzip -c "data/raw/NYS for Web 2021.csv.gz" | shasum -a 256`.

Documentation in `docs/`: `PretrialReleaseDataDictionaryWeb.xlsx` (downloaded 2026-11-06) and the documentation PDF. A newer dictionary appears to be online (it reportedly lists a "Nominal bail-set" category, a DCJS extract date and fields that apply only after 2024-11-19); download it as `docs/PretrialReleaseDataDictionaryWeb_<date>.xlsx` and compare.

## Build

```bash
python -m src.build_dataset
```

Outputs `data/processed/analytic.parquet` (one row per docket after the NYC / decision filters, with sample flags) and `reports/sample_flow.csv` (counts only).

## Processing decisions

Status: **proposed**, to be confirmed at the 2026-11-07 team meeting. Evidence: quick checks on the 2022 NYC file (117,294 dockets after dropping cases disposed at arraignment and unknown decisions).

| # | Decision | Reason |
|---|---|---|
| D1 | Main scope: the five NYC counties | One court system; conclusions limited to NYC |
| D2 | Drop `Disposed at arraign` and `Unknown` decisions | No release decision |
| D3 | Keep `Docket` and `Felony Youth Complaint` case types | Other types are superior-court first securing orders with completeness caveats (~8% of NYC rows) |
| D4 | `y_detain` = `Remanded` or `Bail-set`; 0 = `ROR` or `Nonmonetary release` | The decision itself; named "bail set or remanded", not "detained" |
| D5 | Flag **nominal bail** (bail set and every first bail amount ≤ $1); exclude from the main sample, keep for sensitivity | The dictionary notes $1 bail is often set to credit time served when the person is remanded on another case; ~20% of NYC bail-set dockets in 2022 |
| D6 | One docket per `arr_cycle_id` (most serious top charge); dockets without `arr_cycle_id` kept | 11% of rows are in multi-docket cycles; decisions differ within 13% of those cycles. Also check cycles that span years |
| D7 | `representation_type` excluded from features | Defined as the most recent attorney type, so it includes counsel retained after arraignment |
| D8 | Prior-record fields: missing kept as a separate indicator (`dcjs_missing`), not set to 0 | Missing = no DCJS match |
| D9 | Judge names replaced by `judge_id`; names never shown in outputs | Privacy and course plan |
| D10 | Raw column names converted to snake_case on load | Consistent code across the team |
