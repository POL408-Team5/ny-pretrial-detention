#!/usr/bin/env bash
# Rebuild the data, run the tests and re-execute every analysis notebook in order.
# Usage (from the repository root, in the pol408 environment):  bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/3 Build analytic dataset"
python -m src.build_dataset

echo "== 2/3 Tests"
python -m pytest -q tests/

echo "== 3/3 Notebooks (executed in place, numeric order)"
for nb in notebooks/[0-9][0-9]_*.ipynb; do
  echo "   $nb"
  python -m jupyter nbconvert --to notebook --execute --inplace \
         --ExecutePreprocessor.timeout=3600 \
         --ExecutePreprocessor.kernel_name=python3 "$nb"
done
echo "All steps finished."
