"""Build the analytic dataset (role A).

    python -m src.build_dataset                  # main version
    python -m src.build_dataset --sensitivity    # main + A1-06 sensitivity versions

Writes:
    data/processed/analytic.parquet   one row per docket after S3, with sample flags (not tracked)
    reports/sample_flow.csv           row counts per step and year (tracked)
With --sensitivity, also for each name in config.SENSITIVITY_VARIANTS:
    data/processed/analytic_{name}.parquet   (not tracked)
    reports/sample_flow_{name}.csv           (tracked)
"""
from __future__ import annotations

import argparse
from pprint import pprint

from src import config
from src.sample import build_sample, diagnostics, load_raw


def variant_files(name: str):
    """(analytic parquet, sample flow csv) for a sensitivity version."""
    return (config.DATA_PROCESSED / f"analytic_{name}.parquet",
            config.REPORTS / f"sample_flow_{name}.csv")


def main(sensitivity: bool = False) -> None:
    print("Loading raw files ...")
    raw = load_raw()
    print("Building sample ...")
    analytic, flow = build_sample(raw)

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

    if sensitivity:
        main_ids = set(analytic.loc[analytic[config.MAIN_SAMPLE_COL], config.ID_COL])
        del analytic
        for name, switches in config.SENSITIVITY_VARIANTS.items():
            out, out_flow = build_sample(raw, **switches)
            ids = set(out.loc[out[config.MAIN_SAMPLE_COL], config.ID_COL])
            # Relaxing one rule mostly adds dockets. With nominal bail kept, de-duplication can
            # also keep a different docket of the same arrest cycle, so a few main dockets drop out.
            added, dropped = len(ids - main_ids), len(main_ids - ids)
            pq, csv = variant_files(name)
            out.to_parquet(pq, index=False)
            out_flow.to_csv(csv, index=False)
            n_main = out[config.MAIN_SAMPLE_COL].sum()
            print(f"\n[{name}] {switches}: main sample {n_main:,} rows "
                  f"({added:,} dockets added, {dropped:,} dropped vs main); wrote {pq.relative_to(config.ROOT)}, "
                  f"{csv.relative_to(config.ROOT)}")
            print(out_flow.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sensitivity", action="store_true", help="also build the A1-06 sensitivity versions")
    main(sensitivity=parser.parse_args().sensitivity)
