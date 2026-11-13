#!/usr/bin/env bash
# ML extension: sequence deep-learning challenger + explainability pack.
# Requires the core pipeline (run_pipeline.sh) to have run first,
# and: pip install -r requirements-ml.txt
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
$PY src/07_sequence_model.py
$PY src/08_explainability.py
echo "ML extension complete. See outputs/ml_extension_results.md."
