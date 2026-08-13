#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"
mkdir -p "$MOTIF_HF_CACHE" "$MOTIF_SOURCE_SNAPSHOT"

mode="${1:-weights}"
case "$mode" in
  metadata)
    hf download "$MOTIF_SOURCE_REPO" \
      --revision "$MOTIF_SOURCE_REV" \
      --local-dir "$MOTIF_SOURCE_SNAPSHOT" \
      --include '*.json' --include '*.py' --include '*.jinja' \
      --include '*.md' --include '*.pdf' --include '*.yaml'
    ;;
  weights)
    # The cache is on Runpod's fast local root volume. The returned snapshot
    # path is the canonical source; do not make a second local-dir copy.
    hf download "$MOTIF_SOURCE_REPO" \
      --revision "$MOTIF_SOURCE_REV" \
      --cache-dir "$MOTIF_HF_CACHE" \
      --include 'model-*.safetensors' \
      --include 'model.safetensors.index.json' \
      --max-workers "${MOTIF_DOWNLOAD_WORKERS:-16}"
    ;;
  *)
    echo "usage: $0 {metadata|weights}" >&2
    exit 2
    ;;
esac
