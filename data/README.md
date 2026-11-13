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

| File | Rows | Downloaded | sha256 of the uncompressed CSV (first 16) | Use |
|---|---:|---|---|---|
| `NYS for Web 2021.csv.gz` | 246,202 | 2026-11-06 | `b676e4f9f4e30d62` | train |
| `NYS for Web 2022.csv.gz` | 286,588 | 2026-11-06 | `b745ab9e902b687d` | train |
| `NYS for Web 2023.csv.gz` | 324,619 | 2026-11-06 | `01dddf25180b9bad` | validation |
| `NYS for Web 2024.csv.gz` | 350,051 | 2026-11-07 | `8cc811152bf623da` | test (run once after the freeze) |
| `NYS for Web 2025.csv.gz` | 353,403 | 2026-11-07 | `294616173357af08` | not in `config.YEARS`; possible extra out-of-time check only (13.5% of dockets still pending) |

2021–2025 share the same 131 columns and the same six release-decision categories. `NYS for Web 2020.csv` (130 columns, no `rearrest_180`; reform's first year and COVID) is kept locally only and not used.

Check a file: `gunzip -c "data/raw/NYS for Web 2024.csv.gz" | shasum -a 256`.

Documentation in `docs/`: `PretrialReleaseDataDictionaryWeb.xlsx` and the documentation PDF were downloaded again on 2026-11-07 and are **byte-identical** to the 2026-11-06 copies, so the current official dictionary has no separate "Nominal bail-set" category; nominal bail is identified from the bail amounts (D5). Also in `docs/`: `Disposition Hierarchy.pdf` and `Pretrial Release Data Dashboard Overview.pdf`.

## Build

```bash
python -m src.build_dataset
```

Outputs `data/processed/analytic.parquet` (one row per docket after the NYC / decision filters, with sample flags) and `reports/sample_flow.csv` (counts only).

## Test-year protocol

2024 is the test year. Before the 11/16 freeze (moved from 11/15 on 11/13), no outcome statistic (`y_detain`, release decisions, nominal-bail share) is computed for 2024: `src/sample.diagnostics()`, notebook 01 and Table 1 exclude it. Sample counts, feature distributions and judge-coding shares for 2024 are shown because they do not involve the outcome. Disclosure: on 2026-11-07, before this rule was added, the build printed the 2024 main-sample `y_detain` rate once (about 12%); no model was fit or evaluated on 2024.

## Processing decisions

Status: **proposed**, to be confirmed at the 2026-11-07 team meeting. Evidence: quick checks on the 2022 NYC file (117,294 dockets after dropping cases disposed at arraignment and unknown decisions).

| # | Decision | Reason |
|---|---|---|
| D1 | Main scope: the five NYC counties | One court system; conclusions limited to NYC |
| D2 | Drop `Disposed at arraign` and `Unknown` decisions | No release decision |
| D3 | Keep `Docket` and `Felony Youth Complaint` case types | Other types are superior-court first securing orders with completeness caveats (~8% of NYC rows) |
| D4 | `y_detain` = `Remanded` or `Bail-set`; 0 = `ROR` or `Nonmonetary release` | The decision itself; named "bail set or remanded", not "detained" |
| D5 | Flag **nominal bail** (bail set and every first bail amount ≤ $1); exclude from the main sample, keep for sensitivity | The dictionary notes $1 bail is often set to credit time served when the person is remanded on another case; ~20% of NYC bail-set dockets in 2022 |
| D6 | One docket per `arr_cycle_id`: earliest arraignment year, then most serious top charge; dockets without `arr_cycle_id` kept | Multi-docket cycles are almost all a local-court docket plus a superior-court case for the same arrest, which D3 already removes; D6 drops only 81 further rows (2021–2024). Keeping the earliest year means a later year never changes an earlier year's sample; afterwards no main-sample cycle spans two years |
| D7 | `representation_type` excluded from features | Defined as the most recent attorney type, so it includes counsel retained after arraignment |
| D8 | Prior-record fields: missing kept as a separate indicator (`dcjs_missing`), not set to 0 | Missing = no DCJS match |
| D9 | Judge names replaced by `judge_id`; names never shown in outputs | Privacy and course plan |
| D10 | Raw column names converted to snake_case on load | Consistent code across the team |
| D11 | `case_id` = `internal_case_id`, kept as a string | It is a hexadecimal identifier (e.g. `0xF7…`), not a number |
| D12 | `judge_id` coded on training-year (2021–22) main-sample caseloads (≥ 100 cases → `J001`…; fewer → `other`; not seen → `unseen`); `judge_id_final` coded the same way on 2021–2023 for the final refit | `unseen` share: 30.5% of 2023 and 52.9% of 2024 cases under `judge_id`; 28.2% of 2024 cases under `judge_id_final` |
| D13 | `age_at_arrest` = 0 replaced by `age_at_crime` when positive, else missing; flagged `age_fixed` | 0 is a placeholder (1,898 main-sample dockets, ~0.4%); 56 remain missing |
| D14 | Prior-record and pending-case fields used as provided; the reference date is **not documented** | The UCS dictionary and the DCJS supplemental source notes do not state whether `prior_*` and `pend_*` are computed as of arrest/arraignment or as of the extract date (only `supervision` says "at the time of arraignment"). Listed as a limitation; question sent to UCS/DCJS if possible |

Sample flow (2021–2024): 1,207,460 raw rows → 647,564 NYC → 502,989 after dropping disposed/unknown (`analytic.parquet`) → 449,474 main sample (train 193,196; validation 120,457; test 135,821). See `reports/sample_flow.csv` and `notebooks/01_build_sample.ipynb`.
