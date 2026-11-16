#!/usr/bin/env bash
# Rebuild the data, run the tests and re-execute every analysis notebook in order.
# Usage (from the repository root, in the pol408 environment):  bash scripts/run_all.sh
#
# The one-time 2024 test (python -m src.final_test --run --confirm-freeze) is never run here.
# Before it has run, step 4 does the BD-07 dry run (fit 2021-22, score 2023) so notebooks 09-10
# can execute; after it has run, notebooks 09-10 use the saved test outputs.
set -euo pipefail
cd "$(dirname "$0")/.."

run_nb() {
  echo "   $1"
  python -m jupyter nbconvert --to notebook --execute --inplace \
         --ExecutePreprocessor.timeout=14400 \
         --ExecutePreprocessor.kernel_name=python3 "$1"
}

echo "== 1/5 Build analytic dataset"
python -m src.build_dataset

echo "== 2/5 Tests"
python -m pytest -q tests/

echo "== 3/5 Development notebooks 01-08"
for nb in notebooks/0[1-8]_*.ipynb; do run_nb "$nb"; done

echo "== 4/5 BD-07"
if [ -f reports/bd07_test_run_log.json ]; then
  echo "   2024 test already run once; using its saved outputs (not re-run)."
elif [ -f reports/bd07_frozen_spec.json ]; then
  python -m src.final_test --dry-run
else
  echo "   No frozen spec yet: run 'python -m src.final_test --freeze' first. Skipping 09-10."
  exit 0
fi

echo "== 5/5 Notebooks 09+ (explanation and test-year fairness)"
for nb in notebooks/09_*.ipynb notebooks/1[0-9]_*.ipynb; do
  [ -f "$nb" ] && run_nb "$nb"
done
echo "All steps finished."
