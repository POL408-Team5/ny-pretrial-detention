# Arraignment Bail and Remand Decisions in New York City

SE_POL 408 (Applied Data Science and Application in Social Policy), Northwestern University, Fall 2026 — Final Project

## Question

At arraignment, a New York City judge decides whether to release the defendant, set non-monetary conditions, set bail, or remand. **How predictable are decisions to set bail or remand from the case and the defendant's record, how much do the court and the judge add, and are the model's errors similar across racial groups?**

- **Decision maker and time:** the arraignment judge, usually within about 24 hours of arrest. Features are limited to information available at arraignment.
- **Outcome (`y_detain`):** bail set or remanded (1) vs. released on recognizance or non-monetary conditions (0). This is the judge's decision, not whether the person was actually held (many people post bail).
- **Scope:** the five NYC counties, arraignment years 2021–2024. We describe the predictability of historical decisions and differences across courts, judges and groups; we make no causal claims and do not evaluate the bail reform.

## Data

New York State Unified Court System, Division of Technology & Court Research, *Pretrial Release Data* (state-paid courts). Public, de-identified, one row per defendant-docket, one CSV per arraignment year. Download: https://www.nycourts.gov/division-technology-court-research/pretrial-release-data

The files are refreshed every six months and earlier cases are updated, so results correspond to a specific download. See [`data/README.md`](data/README.md) for files, download dates and processing decisions.

## Design

| Step | Choice |
|---|---|
| Sample | NYC counties; drop cases disposed at arraignment and unknown decisions; local-court dockets and felony youth complaints; nominal ($1) bail flagged; one docket per arrest cycle |
| Feature sets | `CASE` (charge, prior record, pending cases, supervision, age) ⊂ `CASE_COURT` (+ county) ⊂ `CASE_COURT_JUDGE` (+ anonymised judge) |
| Models | Logistic, LASSO logistic, decision tree, random forest, XGBoost; charge-only rule baseline |
| Split | Train 2021–2022 (cross-validated tuning) · validate 2023 (model selection) · test 2024 (run once) |
| Evaluation | ROC-AUC, average precision, Brier score, calibration; bootstrap intervals on the test year |
| Interpretation | SHAP; calibration and error rates by race; court/judge residuals (anonymised) |

Full plan (team working document, Chinese): [`docs/项目方案_v2.md`](docs/项目方案_v2.md).

## Repository structure

```
ny-pretrial-detention/
├── data/
│   ├── raw/          # NYS for Web <year>.csv.gz (public data, tracked)
│   ├── processed/    # analytic.parquet, predictions (not tracked)
│   └── external/     # not used yet
├── docs/             # plan, task handoffs (docs/handoff/), UCS data dictionary and documentation
├── notebooks/        # numbered analysis notebooks
├── src/              # config, sample, features, build_dataset, models, evaluate, fairness, explain
├── reports/          # aggregate tables and figures only (tracked)
├── models/           # fitted models (not tracked)
├── slides/
├── AGENTS.md         # team conventions and instructions for AI assistants
├── CHANGELOG.md
└── AI_Use_Statement.md
```

## Reproducing

1. `pip install -r requirements.txt` (Python 3.11)
2. The raw files are included as `data/raw/NYS for Web <year>.csv.gz` (public data, release listed in `data/README.md`).
3. `python -m src.build_dataset` → `data/processed/analytic.parquet` and `reports/sample_flow.csv`
4. Run the notebooks in numeric order.

## Team

| Role | Members |
|---|---|
| A — Data and features | Yiling Long, Xinhui Zong |
| B/D — Modeling framework, baseline models, evaluation | Shuyi Niu |
| C — Courts, judges and fairness | Yixuan Wen |
| E — Ensemble models, SHAP, EDA | Yanqi Wu |

## Status

- 2026-11-06: project started; quick profiling of the 2022 file.
- 2026-11-07: plan v2; repository set up; `src/config.py` updated. Next: sample and feature code (A, 11/8–9).
