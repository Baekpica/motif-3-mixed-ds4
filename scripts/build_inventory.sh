#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"

python3 "$MOTIF_PROJECT_ROOT/converter/src/build_inventory.py" \
  --config "${MOTIF_FINAL_METADATA_DIR:-/workspace/motif3-work/source/final}/config.json" \
  --headers "$MOTIF_PROJECT_ROOT/manifests/safetensors-headers.json" \
  --out-dir "$MOTIF_PROJECT_ROOT/manifests"
