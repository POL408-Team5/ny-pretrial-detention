"""Project-wide constants and paths. Edit here, not inside notebooks or other modules.

Project: arraignment bail/remand decisions in New York City (SE_POL 408, Fall 2026).
Data: NY State Unified Court System, Pretrial Release Data (public CSV, one file per
arraignment year). https://www.nycourts.gov/division-technology-court-research/pretrial-release-data

Column names: src/sample.py converts every raw column name to snake_case on load
(lower case; spaces and hyphens -> "_"), e.g. "Release Decision at Arraign" ->
"release_decision_at_arraign". All names below use the snake_case form.

Items marked [decide] are proposals in docs/项目方案_v2.md section 11, pending the team meeting.
"""
from pathlib import Path

# ---- Paths ----
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"            # NYS for Web <year>.csv (not tracked)
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_EXTERNAL = ROOT / "data" / "external"
REPORTS = ROOT / "reports"                  # aggregate tables only (tracked)
FIGURES = REPORTS / "figures"
MODELS_DIR = ROOT / "models"                # not tracked
DOCS = ROOT / "docs"

RAW_FILE_PATTERN = "NYS for Web {year}.csv.gz"   # gzip, tracked in git; load_raw falls back to .csv
ANALYTIC_FILE = DATA_PROCESSED / "analytic.parquet"
SAMPLE_FLOW_FILE = REPORTS / "sample_flow.csv"
PRED_BASELINE_FILE = DATA_PROCESSED / "predictions_baseline.parquet"   # B
PRED_ENSEMBLE_FILE = DATA_PROCESSED / "predictions_ensemble.parquet"   # E

# ---- Years and split (plan v2, section 6) ----
YEARS = [2021, 2022, 2023, 2024]   # a missing year file is skipped with a warning
TRAIN_YEARS = [2021, 2022]
VALID_YEAR = 2023
TEST_YEAR = 2024                   # run once, after the 11/15 freeze
SPLIT_COL = "split"                # "train" / "valid" / "test"
YEAR_COL = "arraign_year"          # from first_arraign_date (M/1/YYYY)

# ---- Sample (plan v2, section 3) ----
MAIN_SCOPE = "nyc"
NYC_COUNTIES = ["New York", "Kings", "Bronx", "Queens", "Richmond"]
RELEASE_COL = "release_decision_at_arraign"
RELEASE_DISPOSED = "Disposed at arraign"    # no release decision -> dropped
RELEASE_UNKNOWN = "Unknown"                 # dropped
RELEASE_DETAIN = ["Remanded", "Bail-set"]   # y_detain = 1
RELEASE_FREE = ["ROR", "Nonmonetary release"]  # y_detain = 0
CASE_TYPES_MAIN = ["Docket", "Felony Youth Complaint"]   # [decide] others = superior-court first securing orders
NOMINAL_BAIL_MAX = 1.0             # bail set with every first amount <= $1 -> nominal_bail [decide]
BAIL_AMOUNT_COLS = [
    "first_bail_set_cash", "first_bail_set_credit", "first_insurance_company_bail_bond",
    "first_secured_surety_bond", "first_secured_app_bond", "first_unsecured_surety_bond",
    "first_unsecured_app_bond", "first_partially_secured_surety_bond",
    "first_partially_secured_app_bond",
]
EXCLUDE_NOMINAL_BAIL = True        # [decide] main sample drops nominal bail; sensitivity keeps it as 1
DEDUP_ARREST_CYCLE = True          # [decide] keep the most serious docket per arr_cycle_id
CYCLE_COL = "arr_cycle_id"
SEVERITY_ORDER = ["AF", "BF", "CF", "DF", "EF", "AM", "BM", "UM"]   # top_charge_weight_at_arraign, most serious first

# ---- Identifiers and outcomes (plan v2, section 4) ----
ID_COL = "case_id"                 # = internal_case_id (hex string, not numeric)
TARGET_Q1 = "y_detain"             # bail set or remanded at arraignment (NOT "actually detained")
TARGET_Q3 = "y_warrant"            # warrant ordered between arraignment and disposition (released only)
TARGET_Q3_ALT = "y_rearrest"       # supplementary only
RELEASED_COL = "released"          # ROR / NMR / bail set and posted at arraignment
MAIN_SAMPLE_COL = "main_sample"    # bool: passes every main-sample rule

# ---- Feature groups (plan v2, section 5) ----
FEATURES_CASE = [
    "top_severity_at_arraign", "top_charge_weight_at_arraign",
    "top_charge_at_arraign_violent_felony_ind", "top_arraign_attempt_indicator",
    "arraign_charge_category", "hate_crime_ind", "arrest_type", "case_type",
    "prior_vfo_cnt", "prior_nonvfo_cnt", "prior_misd_cnt",
    "pend_vfo", "pend_nonvfo", "pend_misd", "supervision", "dcjs_missing",
    "dat_wo_ws_prior_to_arraign",  # [decide] failed to appear for a DAT before arraignment
    "age_at_arrest",
]
FEATURES_COURT = ["county_name"]   # docket cases: one Criminal Court per county
FEATURES_JUDGE = ["judge_id"]      # anonymised in src/sample.py; coded on TRAIN_YEARS
JUDGE_COL_FINAL = "judge_id_final"  # coded on TRAIN_YEARS + VALID_YEAR; use in place of judge_id for the final refit / 2024 test
FEATURE_SETS = {                    # main analysis: nested sets
    "CASE": FEATURES_CASE,
    "CASE_COURT": FEATURES_CASE + FEATURES_COURT,
    "CASE_COURT_JUDGE": FEATURES_CASE + FEATURES_COURT + FEATURES_JUDGE,
}
# Supplementary set (team decision 2026-11-07): gender is an audit variable like race, but we also
# report how much it adds beyond case and record. Not part of the nested comparison.
SUPPLEMENTARY_SETS = {
    "CASE_PLUS_GENDER": FEATURES_CASE + ["gender"],
}
SUPPLEMENTARY_ALLOWED = {"CASE_PLUS_GENDER": ["gender"]}   # audit variables allowed in that set only
FAIRNESS_GROUPS = ["race", "ethnicity", "gender", "age_group"]   # not features in the main sets
AGE_BINS = [0, 18, 25, 35, 45, 55, 200]
AGE_LABELS = ["<18", "18-24", "25-34", "35-44", "45-54", "55+"]

# Columns that must never be features (plan v2, section 5.2). Checked by features.check_leakage().
LEAKAGE_BLOCKLIST = [
    # the decision and its components
    "release_decision_at_arraign", "remanded_to_jail_at_arraign", "ror_at_arraign",
    "bail_set_and_posted_at_arraign", "bail_set_and_not_posted_at_arraign", "nmr_at_arraign",
    "pretrial_supervision_at_arraign", "contact_pretrial_service_agency", "electronic_monitoring",
    "travel_restrictions", "passport_surrender", "no_firearms_or_weapons", "maintain_employment",
    "maintain_housing", "maintain_school", "placement_in_mandatory_program", "removal_to_hospital",
    "obey_order_of_protection", "obey_court_conditions_family_offense", "other_nmr",
    "order_of_protection", *BAIL_AMOUNT_COLS,
    "partially_secured_surety_bond_perc", "partially_secured_app_bond_perc", "bail_made_indicator",
    # prosecutor requests (empty in NYC; some fields only after 2024-11-19)
    "notrequestedflag", "remandrequestedflag", "nmrrequestedflag", "rorrequestedflag",
    "unspecifiedtyperequestedamount", "cashrequestedamount", "creditrequestedamount",
    "insurancecompanyrequestedamount", "securedsuretyrequestedamount", "securedapprequestedamount",
    "unsecuredsuretyrequestedamount", "unsecuredapprequestedamount",
    "partiallysecuredsuretyrequestedamount", "partiallysecuredapprequestedamount",
    "unspecifiedbondtyperequestedamount",
    # after arraignment
    "representation_type",         # "most recent attorney type": includes post-arraignment counsel
    "representation_at_securing_order",
    "app_count_arraign_to_dispo_released", "app_count_arraign_to_dispo_detained",
    "app_count_arraign_to_dispo_total", "def_attended_sched_pretrials",
    "warrant_ordered_btw_arraign_and_dispo", "first_bench_warrant_month", "first_bench_warrant_year",
    "non_stayed_wo", "num_of_stayed_wo", "num_of_row", "docket_status", "disposition_type",
    "disposition_detail", "dismissal_reason", "disposition_date", "most_severe_sentence",
    "top_conviction_law", "top_conviction_article_section", "top_conviction_attempt_indicator",
    "top_charge_at_conviction", "top_charge_severity_at_conviction",
    "top_charge_weight_at_conviction", "top_charge_at_conviction_violent_felony_ind",
    "days_arraign_remand_first_released", "known_days_in_custody",
    "days_arraign_bail_set_to_first_posted", "days_arraign_bail_set_to_first_release",
    "days_arraign_to_dispo", "minimptopconvdays", "maximptopconvdays",
    "rearrest", "rearrest_180", "rearrest_date", "rearrest_firearm", "rearrest_date_firearm",
    # fairness-only and identifying
    "race", "ethnicity", "age_group", "judge_name",
    # outcomes and sample flags
    "y_detain", "y_warrant", "y_rearrest", "released", "nominal_bail", "bail_posted_at_arraign",
    "main_sample", "dup_cycle", "excl_case_type", "judge_id_final", "age_fixed",
]

# ---- Modeling / evaluation ----
RANDOM_STATE = 408
MIN_JUDGE_CASES = 100      # judges with fewer training cases -> "other"; unseen in valid/test -> "unseen" [decide]
MIN_GROUP_N = 100          # fairness / judge cells below this are not reported
N_BOOTSTRAP = 1000         # test-set confidence intervals
