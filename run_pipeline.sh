#!/usr/bin/env bash
# Run the full pipeline end to end: data -> PD model -> LGD/EAD -> validation -> stress
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
$PY src/01_data_prep.py
$PY src/02_pd_model.py
$PY src/03_lgd_ead.py
$PY src/04_validation.py
$PY src/05_cecl_stress.py
echo "Pipeline complete. See outputs/ and docs/."
