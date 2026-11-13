# BD-05 baseline models: development-year results

Status: locally completed and independently verified; awaiting upload/review. No final model is selected here.

## Design and provenance

- Upstream main: `2b0f2cdd847f5d0174e366896f17ff958efd2373`. Existing published function arguments, config, sample rules and feature definitions are unchanged.
- NYC main sample: 89,189 cases in the first training year, 104,007 in the second, 120,457 in validation. Every development-year sample-flow step matches the published report.
- Only development-year raw files were available/read in this run; their uncompressed SHA-256 hashes match the team release. The test-year raw file was absent. No test-year outcome statistics or predictions were computed.
- The current local copy had no analytic.parquet. Notebook 05 rebuilt only the development years using the unchanged upstream sample functions rather than writing a partial file under the canonical analytic filename. If the full analytic file exists, notebook 05 reads only development-year partitions.
- Full training sample, five-fold stratified CV with the configured seed, handoff parameter grids and ROC-AUC tuning. No subsampling, reweighting or oversampling. Preprocessing is fitted within each fold; trees are unscaled. Judge grouping follows the upstream outcome-free training-year mapping; judge_id_final is not used.
- Numeric gaps are computationally zero-filled with dcjs_missing retained; unknown categories are ignored by training-fitted one-hot encoders. This is the published BD-01 behavior, not a claim that missing records mean no record. Residual missing age is also zero-filled under that specification.
- Missingness audit: 1,714 training and 1,488 validation cases have missing prior-record fields; every missing prior/pending/supervision value is covered by dcjs_missing. Age remains missing for 27 training and 21 validation cases after upstream corrections.
- L1 regularization uses a version-compatible constructor; the solver and C grid are unchanged. Nonconvergence is an error. The existing fit_and_predict arguments are unchanged, so E can continue calling them.

## Main validation comparison

| model | feature_set | roc_auc | avg_precision | brier | n |
|---|---|---|---|---|---|
| logit | CASE | 0.920289 | 0.662465 | 0.067290 | 120457 |
| logit | CASE_COURT | 0.920785 | 0.665713 | 0.067026 | 120457 |
| logit | CASE_COURT_JUDGE | 0.923637 | 0.677013 | 0.065788 | 120457 |
| lasso | CASE | 0.920337 | 0.662595 | 0.067270 | 120457 |
| lasso | CASE_COURT | 0.920855 | 0.665698 | 0.067029 | 120457 |
| lasso | CASE_COURT_JUDGE | 0.923789 | 0.677448 | 0.065698 | 120457 |
| tree | CASE | 0.922505 | 0.655781 | 0.067615 | 120457 |
| tree | CASE_COURT | 0.922019 | 0.656373 | 0.067652 | 120457 |
| tree | CASE_COURT_JUDGE | 0.921855 | 0.656324 | 0.067652 | 120457 |

Charge-grade/violent-flag rule: AUC 0.846492, AP 0.478021, Brier 0.080927.
The highest validation AUC among these nine main fits is lasso / CASE_COURT_JUDGE (0.923789). This is provisional: ensemble models and the BD-06 paired comparisons remain to be combined.

For logistic regression, adding county alone changes AUC by about +0.0005; adding judge after county gives a further +0.0029. The corresponding LASSO pattern is similar. The tree does not improve with county/judge inputs on validation, even though its training CV score rises slightly. More inputs therefore do not guarantee better future-year performance. LASSO and L2 logistic with judges are very close; uncertainty and the ensemble results are needed before selecting a winner.

AUC measures ranking, not percent accuracy. AP is compared with the validation positive-label rate (12.8179%); lower Brier indicates better probability accuracy. Training OOF values appear in bd05_metrics.csv but reuse training-selected hyperparameters, so they are not nested-CV estimates. Validation is the primary comparison.

## Supplementary gender diagnostic

| model | delta_roc_auc | delta_avg_precision | delta_brier |
|---|---|---|---|
| logit | 0.001058 | 0.005523 | -0.000651 |
| lasso | 0.000811 | 0.004833 | -0.000569 |
| tree | -0.000350 | 0.000193 | 0.000088 |

These are CASE_PLUS_GENDER minus CASE, evaluated on the same cases. The main BD-05 requirement is nine fits; the LASSO/tree gender variants are additional diagnostics beyond the earlier logistic gender check, not extra main models. The logistic paired-bootstrap intervals are in bd05_logit_gender_ci_valid.csv. They condition on fitted predictions, do not incorporate retraining or judge-level dependence, and do not establish discriminatory treatment. Gender remains outside main feature sets; race and ethnicity are never predictors.

## Actual training matrix dimensions

| feature_set | n_train | n_valid | n_raw_features | n_transformed_features |
|---|---|---|---|---|
| CASE | 193196 | 120457 | 18 | 52 |
| CASE_COURT | 193196 | 120457 | 19 | 57 |
| CASE_COURT_JUDGE | 193196 | 120457 | 20 | 242 |
| CASE_PLUS_GENDER | 193196 | 120457 | 19 | 55 |

Encoded column counts describe the full-training fit; fold-specific encoders may have fewer categories. All model-feature combinations have identical case coverage.

## Artifacts and verification

- Local-only: data/processed/predictions_baseline.parquet (4,077,489 rows; 13 combinations: nine main, three gender supplements, one rule); 12 fitted joblib models. These must not be uploaded.
- Review artifacts: notebook 05, models.py, test_models.py and the bd05 aggregate CSV/report files. Existing BD-04 results are preserved separately.
- All 42 tests passed without warnings after the L1 compatibility correction. Every notebook code cell executed in a fresh Python 3.11 IPython process; this check did not run the full-team run_all.sh.
- Independent verification recomputed all 26 metric rows, checked every prediction's year/label alignment, reloaded all 12 models and reproduced their validation probabilities to 1e-12, verified training-only encoder categories and excluded audit/final-judge columns.

## Handoff and limits

BD-06: combine E's ensemble predictions; compare nested gains with paired bootstrap intervals; report individually coded judges separately and show calibration. The large unseen-judge share can dilute the apparent benefit of judge information. C owns threshold selection and group fairness metrics; decision-tree probability ties need an explicit convention because a single threshold may not match the observed positive proportion exactly. Final refitting/testing remains disabled until the separate freeze and BD-07; Q3 is not started.

Prior-record and pending-case reference dates remain undocumented (data/README.md D14). Court/judge gains are predictive associations, not causal effects; this outcome is a historical bail/remand decision, not actual detention or future offending.
