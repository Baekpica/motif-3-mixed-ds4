#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
REPO=${MQ87_REPO:-Baekpica/Motif-3-Mixed-Quant-GGUF}
SHARD_DIR=${MQ87_SHARD_DIR:-/motif3/artifacts/Motif-3-MQ87-FIT-shards}
EXPECTED=${MQ87_EXPECTED_SHARDS:-11}

status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$ROOT/manifests/verify-mq87.json")
[[ $status == ok ]] || { echo "MQ87 verification did not pass" >&2; exit 1; }

mapfile -t shards < <(find "$SHARD_DIR" -maxdepth 1 -type f \
    -name 'Motif-3-MQ87-88-FIT-*.gguf' -print | sort)
[[ ${#shards[@]} -eq $EXPECTED ]] || {
    echo "expected $EXPECTED shards, found ${#shards[@]}" >&2
    exit 1
}

start=${START_SHARD:-1}
end=${END_SHARD:-$EXPECTED}
for ((index = start; index <= end; index++)); do
    shard=${shards[index - 1]}
    remote=$(basename -- "$shard")
    echo "[MQ87 $index/$EXPECTED] uploading $remote ($(stat -c %s "$shard") bytes)"
    HF_XET_HIGH_PERFORMANCE=1 hf upload "$REPO" "$shard" "$remote" \
        --repo-type model --commit-message "Upload MQ87-88-FIT shard $index/$EXPECTED"
done
