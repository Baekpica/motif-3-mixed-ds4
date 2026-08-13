#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/scripts/env.sh"

snapshot="${MOTIF_HF_CACHE}/models--Motif-Technologies--Motif-3/snapshots/${MOTIF_SOURCE_REV}"
python3 "${repo_root}/fixtures/generate_reference_fixtures.py" \
  --snapshot "${snapshot}" \
  --source-metadata "${MOTIF_SOURCE_SNAPSHOT}" \
  --output "${repo_root}/fixtures/official-final" \
  "$@"
