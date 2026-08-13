#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"

python3 "$MOTIF_PROJECT_ROOT/calibration/build_calibration.py" \
  --out "$MOTIF_PROJECT_ROOT/calibration/calibration.txt" \
  --tokenizer "${MOTIF_FINAL_METADATA_DIR:-/workspace/motif3-work/source/final}" \
  --normalizer "${MOTIF_REFERENCE_NORMALIZER:-/workspace/motif3-work/reference-healing-mix/scripts/solar_format.py}" \
  --total-tokens "${MOTIF_CALIBRATION_TOKENS:-4000000}" \
  --seed "${MOTIF_CALIBRATION_SEED:-1234}"
