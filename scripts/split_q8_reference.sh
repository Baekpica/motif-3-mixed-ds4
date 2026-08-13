#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SPLITTER=${GGUF_SPLITTER:-/motif3/llama-build/bin/llama-gguf-split}
INPUT=${Q8_INPUT:-/motif3/artifacts/Motif-3-Q8_0.gguf}
OUT_DIR=${Q8_SHARD_DIR:-/motif3/artifacts/Motif-3-Q8_0-shards}
PREFIX=$OUT_DIR/Motif-3-Q8_0

[[ -x $SPLITTER ]] || { echo "missing splitter: $SPLITTER" >&2; exit 1; }
[[ -f $INPUT ]] || { echo "missing Q8 GGUF: $INPUT" >&2; exit 1; }
mkdir -p "$OUT_DIR"
if find "$OUT_DIR" -maxdepth 1 -type f -name '*.gguf' -print -quit | grep -q .; then
    echo "refusing to overwrite existing GGUF shards in $OUT_DIR" >&2
    exit 1
fi

"$SPLITTER" --split-max-size 31G "$INPUT" "$PREFIX"
count=$(find "$OUT_DIR" -maxdepth 1 -type f -name 'Motif-3-Q8_0-*-of-00011.gguf' | wc -l)
[[ $count -eq 11 ]] || { echo "expected 11 shards, found $count" >&2; exit 1; }

PYTHONPATH=/motif3/llama.cpp/gguf-py python3 "$ROOT/scripts/verify_motif3_gguf.py" \
    "$OUT_DIR" \
    --source-map "$ROOT/manifests/gguf-source-map-q8.json" \
    --variant q8-reference \
    --output "$ROOT/manifests/verify-q8.json"
