#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"

GGUF_TEMPLATE=${GGUF_TEMPLATE:-/motif3/artifacts/Motif-3-MQ87-FIT-STRUCTURAL.gguf}
python3 "$MOTIF_PROJECT_ROOT/converter/src/build_structural_gguf.py" \
  --model-dir "${MOTIF_FINAL_METADATA_DIR:-/workspace/motif3-work/source/final}" \
  --inventory "$MOTIF_PROJECT_ROOT/manifests/tensor-inventory.json" \
  --out "$GGUF_TEMPLATE" \
  --source-map "$MOTIF_PROJECT_ROOT/manifests/gguf-source-map.json" \
  --gguf-py "${GGUF_PY_DIR:-/motif3/llama.cpp/gguf-py}"
