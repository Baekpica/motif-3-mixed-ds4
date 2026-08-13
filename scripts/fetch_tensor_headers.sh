#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"

python3 "$MOTIF_PROJECT_ROOT/converter/tools/fetch_st_headers.py" \
  --repo "$MOTIF_SOURCE_REPO" \
  --revision "$MOTIF_SOURCE_REV" \
  --index "${MOTIF_FINAL_METADATA_DIR:-/workspace/motif3-work/source/final}/model.safetensors.index.json" \
  --out "$MOTIF_PROJECT_ROOT/manifests/safetensors-headers.json" \
  --workers "${MOTIF_HEADER_WORKERS:-20}"
