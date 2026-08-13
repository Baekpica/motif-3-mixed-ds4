#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
QUANTIZER=${QUANTIZER:-/workspace/ds4/gguf-tools/deepseek4-quantize}
SOURCE=${MOTIF3_SOURCE:-/motif3/hf-cache/models--Motif-Technologies--Motif-3/snapshots/ccceb1a5fd7b5eb32e47841216b3caf5666c07bc}
TEMPLATE=${MQ87_TEMPLATE:-/motif3/artifacts/Motif-3-MQ87-FIT-STRUCTURAL.gguf}
IMATRIX=${MOTIF3_IMATRIX:-/motif3/calibration/Motif-3-Q8_0-imatrix.dat}
OUTPUT=${MQ87_OUTPUT:-/motif3/artifacts/Motif-3-MQ87-FIT.gguf}
THREADS=${MQ87_THREADS:-192}

[[ -x $QUANTIZER ]] || { echo "missing quantizer: $QUANTIZER" >&2; exit 1; }
[[ -d $SOURCE ]] || { echo "missing source snapshot: $SOURCE" >&2; exit 1; }
[[ -f $TEMPLATE ]] || { echo "missing structural template: $TEMPLATE" >&2; exit 1; }
[[ -f $IMATRIX ]] || { echo "missing Q8 imatrix: $IMATRIX" >&2; exit 1; }
[[ ! -e $OUTPUT ]] || { echo "refusing to overwrite existing output: $OUTPUT" >&2; exit 1; }

"$QUANTIZER" \
    --hf "$SOURCE" \
    --template "$TEMPLATE" \
    --out "$OUTPUT" \
    --imatrix "$IMATRIX" \
    --imatrix-strict \
    --threads "$THREADS"

PYTHONPATH=/motif3/llama.cpp/gguf-py python3 \
    "$ROOT/scripts/verify_motif3_gguf.py" \
    "$OUTPUT" \
    --source-map "$ROOT/manifests/gguf-source-map.json" \
    --variant mq87 \
    --output "$ROOT/manifests/verify-mq87-unsharded.json"

PYTHONPATH=/motif3/llama.cpp/gguf-py python3 \
    "$ROOT/scripts/verify_quantized_samples.py" \
    --gguf "$OUTPUT" \
    --source "$SOURCE" \
    --source-map "$ROOT/manifests/gguf-source-map.json" \
    --output "$ROOT/manifests/verify-mq87-samples.json"
