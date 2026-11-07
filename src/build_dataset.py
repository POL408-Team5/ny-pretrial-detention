"""Build the analytic dataset (role A).

    python -m src.build_dataset

Writes:
    data/processed/analytic.parquet   one row per docket after S3, with sample flags (not tracked)
    reports/sample_flow.csv           row counts per step and year (tracked)
"""
from __future__ import annotations

from pprint import pprint

from src import config
from src.sample import build_sample, diagnostics, load_raw


def main() -> None:
    print("Loading raw files ...")
    raw = load_raw()
    print("Building sample ...")
    analytic, flow = build_sample(raw)
    del raw

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    analytic.to_parquet(config.ANALYTIC_FILE, index=False)
    flow.to_csv(config.SAMPLE_FLOW_FILE, index=False)

    print(f"\nWrote {config.ANALYTIC_FILE.relative_to(config.ROOT)} "
          f"({len(analytic):,} rows, {analytic.shape[1]} columns)")
    print(f"Wrote {config.SAMPLE_FLOW_FILE.relative_to(config.ROOT)}\n")
    print(flow.to_string(index=False))
    print("\nDiagnostics:")
    pprint(diagnostics(analytic), sort_dicts=False, width=110)


if __name__ == "__main__":
    main()
