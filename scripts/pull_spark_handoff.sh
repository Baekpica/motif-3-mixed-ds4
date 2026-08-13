#!/usr/bin/env bash
set -euo pipefail

BUCKET=${BUCKET:-Baekpica/motif-3-spark-handoff}
HANDOFF_DIR=${1:-/workspace/motif-3-spark-handoff}
MODEL_DIR=${MODEL_DIR:-/workspace/motif-3-model}
EXPECTED=${EXPECTED_SHARDS:-11}
MERGED_MODEL=${MERGED_MODEL:-}
GGUF_SPLITTER=${GGUF_SPLITTER:-llama-gguf-split}

mkdir -p -- "$HANDOFF_DIR" "$MODEL_DIR"
hf buckets sync "hf://buckets/$BUCKET" "$HANDOFF_DIR"

(
    cd -- "$HANDOFF_DIR"
    sha256sum --check manifests/SHA256SUMS
)

repo=$(<"$HANDOFF_DIR/model/repo.txt")
revision=$(<"$HANDOFF_DIR/model/revision.txt")
[[ $revision =~ ^[0-9a-f]{40}$ ]] || {
    echo "handoff model revision is not a 40-character commit SHA: $revision" >&2
    exit 1
}

mapfile -t filenames <"$HANDOFF_DIR/model/filenames.txt"
[[ ${#filenames[@]} -eq $EXPECTED ]] || {
    echo "expected $EXPECTED mixed GGUF filenames, found ${#filenames[@]}" >&2
    exit 1
}
expected_first=$(printf 'Motif-3-MQ87-88-FIT-%05d-of-%05d.gguf' 1 "$EXPECTED")
[[ ${filenames[0]} == "$expected_first" ]] || {
    echo "unexpected first mixed GGUF filename: ${filenames[0]}" >&2
    exit 1
}

HF_XET_HIGH_PERFORMANCE=1 hf download \
    "$repo" \
    "${filenames[@]}" \
    --revision "$revision" \
    --local-dir "$MODEL_DIR"

(
    cd -- "$MODEL_DIR"
    sha256sum --check "$HANDOFF_DIR/model/sha256.txt"
)

if [[ -n $MERGED_MODEL ]]; then
    command -v "$GGUF_SPLITTER" >/dev/null 2>&1 || {
        echo "missing GGUF merge tool: $GGUF_SPLITTER" >&2
        exit 1
    }
    [[ ! -e $MERGED_MODEL ]] || {
        echo "refusing to overwrite merged GGUF: $MERGED_MODEL" >&2
        exit 1
    }
    mkdir -p -- "$(dirname -- "$MERGED_MODEL")"
    "$GGUF_SPLITTER" --merge "$MODEL_DIR/${filenames[0]}" "$MERGED_MODEL"
    merged_bytes=$(<"$HANDOFF_DIR/model/merged-bytes.txt")
    merged_sha256=$(<"$HANDOFF_DIR/model/merged-sha256.txt")
    [[ $(stat -c %s -- "$MERGED_MODEL") == "$merged_bytes" ]] || {
        echo "merged GGUF byte count mismatch" >&2
        exit 1
    }
    printf '%s  %s\n' "$merged_sha256" "$MERGED_MODEL" | sha256sum --check -
    printf 'merged model: %s (%s bytes)\n' \
        "$MERGED_MODEL" "$(stat -c %s "$MERGED_MODEL")"
fi

printf 'verified model: %s@%s\n' "$repo" "$revision"
printf 'handoff directory: %s\n' "$HANDOFF_DIR"
printf 'model directory: %s\n' "$MODEL_DIR"
