#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
snapshot=${MOTIF3_SNAPSHOT:-/motif3/hf-cache/models--Motif-Technologies--Motif-3/snapshots/ccceb1a5fd7b5eb32e47841216b3caf5666c07bc}

python3 "$repo_root/fixtures/generate_tokenizer_fixtures.py" \
  --snapshot "$snapshot" \
  --output "$repo_root/fixtures/official-final"
