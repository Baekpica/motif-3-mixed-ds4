#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

Q8_REPO=Baekpica/Motif-3-GGUF
Q8_REV=5c266c95bf8c8d822d50e5e1cce9d108eaadb2af
MIXED_REPO=Baekpica/Motif-3-Mixed-Quant-GGUF
MIXED_REV=efd6044e25e7f8e3b459a737d021091e2e69b6c6
REPRO_REPO=Baekpica/motif-3-mixed-ds4
REPRO_BRANCH=feature/h200-mixed-handoff
DS4_REPO=https://github.com/Baekpica/ds4.git
DS4_BRANCH=feature/motif-3-model-loader
DS4_REV=d878ea1a1d67bc0f0bd60e20e75b4a011aa2d8d9
BUCKET=Baekpica/motif-3-spark-handoff

q8_info=$(hf models info "$Q8_REPO" --revision "$Q8_REV" \
  --expand sha,private,siblings)
jq -e --arg sha "$Q8_REV" '
  .sha == $sha and .private == false and
  ([.siblings[].rfilename | select(test("Motif-3-Q8_0-[0-9]{5}-of-00011\\.gguf$"))] | length) == 11
' <<<"$q8_info" >/dev/null

mixed_info=$(hf models info "$MIXED_REPO" --revision "$MIXED_REV" \
  --expand sha,private,siblings)
jq -e --arg sha "$MIXED_REV" '
  .sha == $sha and .private == false and
  ([.siblings[].rfilename | select(test("Motif-3-MQ87-88-FIT-[0-9]{5}-of-00011\\.gguf$"))] | length) == 11 and
  ([.siblings[].rfilename | select(test("MQ95|MQ97"))] | length) == 0
' <<<"$mixed_info" >/dev/null

remote_q8_manifest=$(hf download "$Q8_REPO" Q8_0-SHA256SUMS \
  --revision "$Q8_REV" --quiet)
remote_mixed_manifest=$(hf download "$MIXED_REPO" \
  MQ87-88-FIT-SHA256SUMS --revision "$MIXED_REV" --quiet)
cmp "$ROOT/publish/q8/Q8_0-SHA256SUMS" "$remote_q8_manifest"
cmp "$ROOT/publish/mixed/MQ87-88-FIT-SHA256SUMS" "$remote_mixed_manifest"

# Current cards and H200 evidence must match the final local publication
# staging tree, while the large weight checks above remain pinned to their
# immutable earlier revisions.
remote_q8_card=$(hf download "$Q8_REPO" README.md --quiet)
remote_mixed_card=$(hf download "$MIXED_REPO" README.md --quiet)
remote_h200_report=$(hf download "$MIXED_REPO" H200-DEVELOPMENT.md --quiet)
remote_mixed_report=$(hf download "$MIXED_REPO" MIXED-QUANT.md --quiet)
cmp "$ROOT/publish/q8/README.md" "$remote_q8_card"
cmp "$ROOT/publish/mixed/README.md" "$remote_mixed_card"
cmp "$ROOT/reports/H200-DEVELOPMENT.md" "$remote_h200_report"
cmp "$ROOT/reports/MIXED-QUANT.md" "$remote_mixed_report"

for evidence in \
  H200-VALIDATION.json \
  H200-COMPLETION-AUDIT.md \
  H200-256K-SM90-FULL-QUESTION.txt; do
  remote_evidence=$(hf download "$MIXED_REPO" "reports/$evidence" --quiet)
  cmp "$ROOT/reports/$evidence" "$remote_evidence"
done

if hf models info "$REPRO_REPO" >/dev/null 2>&1; then
  echo "unexpected HF model repository exists: $REPRO_REPO" >&2
  exit 1
fi

repro_info=$(gh repo view "$REPRO_REPO" \
  --json isPrivate,url,defaultBranchRef,homepageUrl)
jq -e --arg branch "$REPRO_BRANCH" --arg model "https://huggingface.co/$MIXED_REPO" '
  .isPrivate == false and .defaultBranchRef.name == $branch and
  .homepageUrl == $model
' <<<"$repro_info" >/dev/null
local_repro=$(git -C "$ROOT" rev-parse HEAD)
remote_repro=$(git ls-remote "https://github.com/$REPRO_REPO.git" \
  "refs/heads/$REPRO_BRANCH" | awk '{print $1}')
test "$remote_repro" = "$local_repro"

remote_ds4=$(git ls-remote "$DS4_REPO" "refs/heads/$DS4_BRANCH" | awk '{print $1}')
test "$remote_ds4" = "$DS4_REV"

bucket_info=$(hf buckets info "$BUCKET")
jq -e --arg id "$BUCKET" '.id == $id and .private == true' \
  <<<"$bucket_info" >/dev/null

printf 'public Q8 model: %s@%s (11 shards)\n' "$Q8_REPO" "$Q8_REV"
printf 'public mixed model: %s@%s (11 shards)\n' "$MIXED_REPO" "$MIXED_REV"
printf 'current Q8 documentation: %s\n' \
  "$(hf models info "$Q8_REPO" --expand sha | jq -r .sha)"
printf 'current mixed documentation: %s\n' \
  "$(hf models info "$MIXED_REPO" --expand sha | jq -r .sha)"
printf 'public GitHub reproduction: https://github.com/%s\n' "$REPRO_REPO"
printf 'reproduction revision: %s\n' "$remote_repro"
printf 'public ds4 runtime: %s@%s\n' "$DS4_BRANCH" "$DS4_REV"
printf 'private Spark handoff: hf://buckets/%s\n' "$BUCKET"
printf 'wrong HF model repository absent: %s\n' "$REPRO_REPO"
