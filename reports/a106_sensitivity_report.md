# A1-06 Sensitivity of the sample rules (validation year only)

Yiling Long, 2026-11-13. Reproduce with:

```
python -m src.build_dataset --sensitivity   # ~1 min; writes analytic_{version}.parquet and reports/sample_flow_{version}.csv
python -m src.sensitivity                    # ~15 min; logit x three main feature sets x three versions
```

Each version switches off one main-sample rule and rebuilds everything downstream (including judge codes). Logit is tuned on 2021–22 with `models.fit_and_predict` and scored on 2023. No 2024 row is read. Models are saved to `models/sensitivity/`, so the main models are untouched. Full table: `a106_sensitivity_valid.csv`.

## Sample size

| Version | Rule switched off | Main sample | vs main |
|---|---|---|---|
| main | — | 449,474 | — |
| `nominal_as_detain` | S5: nominal bail kept, coded `y_detain = 1` | 466,677 | +17,204 dockets, −1 (de-duplication kept a nominal-bail docket of the same arrest) |
| `no_dedup` | S6: every docket of an arrest cycle kept | 449,555 | +81 dockets |

## Validation AUC (2023)

"Own" = the version's own 2023 sample. "Shared" = the 120,457 dockets in the main 2023 sample, the same rows for every version.

| Version | Scored on | 2023 detain rate | CASE | CASE_COURT | CASE_COURT_JUDGE | Judge increment |
|---|---|---|---|---|---|---|
| main | own = shared | 12.8% | 0.9203 | 0.9208 | 0.9236 | +0.0028 |
| `nominal_as_detain` | own | 15.9% | 0.8822 | 0.8826 | 0.8851 | +0.0025 |
| `nominal_as_detain` | shared | 12.8% | 0.9189 | 0.9193 | 0.9218 | +0.0025 |
| `no_dedup` | own | 12.8% | 0.9202 | 0.9207 | 0.9236 | +0.0029 |

## What this means

- **De-duplication does not matter.** Only 81 dockets share an arrest cycle with another main-sample docket, and every number moves by 0.0001 or less.
- **Excluding nominal bail changes the level but not the conclusions.** With nominal bail counted as detention, AUC on the full 2023 sample drops by about 0.04: nominal bail is hard to predict from charge and record. The data dictionary notes that $1 bail is often set to credit time served when the person is already held on another case, so it reflects that other case, not this one. This supports excluding it (D5). Training on those rows costs little on the main rows (−0.0015 to −0.0018). The order of the three feature sets and the size of the judge increment (about +0.003) are the same in every version.
- Deck wording: "Our results do not depend on de-duplication. Counting $1 bail as detention lowers AUC because such bail is hard to predict, but the ranking of feature sets and the judge increment are unchanged."

These are validation-year checks with logit. After the freeze, BD-07 can rerun `python -m src.sensitivity --model <final>` if the final model is another baseline. RF/XGBoost would need the ensemble estimators passed in.
