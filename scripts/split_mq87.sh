#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SPLITTER=${GGUF_SPLITTER:-/motif3/llama-build/bin/llama-gguf-split}
INPUT=${MQ87_INPUT:-/motif3/artifacts/Motif-3-MQ87-FIT.gguf}
OUT_DIR=${MQ87_SHARD_DIR:-/motif3/artifacts/Motif-3-MQ87-FIT-shards}
PREFIX=$OUT_DIR/Motif-3-MQ87-88-FIT
EXPECTED=${MQ87_EXPECTED_SHARDS:-11}

[[ -x $SPLITTER ]] || { echo "missing splitter: $SPLITTER" >&2; exit 1; }
[[ -f $INPUT ]] || { echo "missing MQ87 GGUF: $INPUT" >&2; exit 1; }

PYTHONPATH=/motif3/llama.cpp/gguf-py python3 "$ROOT/scripts/verify_motif3_gguf.py" \
    "$INPUT" \
    --source-map "$ROOT/manifests/gguf-source-map.json" \
    --variant mq87 \
    --output "$ROOT/manifests/verify-mq87-unsharded.json"

status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$ROOT/manifests/verify-mq87-unsharded.json")
[[ $status == ok ]] || { echo "unsharded MQ87 verification did not pass" >&2; exit 1; }

mkdir -p "$OUT_DIR"
if find "$OUT_DIR" -maxdepth 1 -type f -name '*.gguf' -print -quit | grep -q .; then
    echo "refusing to overwrite existing GGUF shards in $OUT_DIR" >&2
    exit 1
fi

"$SPLITTER" --split-max-size 9G "$INPUT" "$PREFIX"
count=$(find "$OUT_DIR" -maxdepth 1 -type f \
    -name 'Motif-3-MQ87-88-FIT-*.gguf' | wc -l)
[[ $count -eq $EXPECTED ]] || {
    echo "expected $EXPECTED shards, found $count" >&2
    exit 1
}

PYTHONPATH=/motif3/llama.cpp/gguf-py python3 "$ROOT/scripts/verify_motif3_gguf.py" \
    "$OUT_DIR" \
    --source-map "$ROOT/manifests/gguf-source-map.json" \
    --variant mq87 \
    --output "$ROOT/manifests/verify-mq87.json"
