# Motif-3 H200 → DGX Spark handoff

This private bucket preserves the expensive four-H200 development state for
the full-topology Motif-3 MQ87-88-FIT artifact.

There is intentionally no separate Hugging Face *model* repository for the
reproduction tree. The public reproduction repository is
`https://github.com/Baekpica/motif-3-mixed-ds4`, while the native runtime is
`https://github.com/Baekpica/ds4/tree/feature/motif-3-model-loader`. Public Hub
model repositories are limited to the Q8_0 and MQ87-88 GGUF artifacts;
expensive calibration state and offline Spark handoff material live in this
private bucket.

- Public model: `Baekpica/Motif-3-Mixed-Quant-GGUF`
- Native runtime: `Baekpica/ds4:feature/motif-3-model-loader` at
  `d878ea1a1d67bc0f0bd60e20e75b4a011aa2d8d9`
- Exact public revision: `model/revision.txt`
- Mixed shard filenames and hashes: `model/filenames.txt` and
  `model/sha256.txt`
- Canonical merged byte count/hash: `model/merged-bytes.txt` and
  `model/merged-sha256.txt`
- Q8-derived imatrix: `calibration/Motif-3-Q8_0-imatrix.dat`
- Rank-local imatrix accumulators: `calibration/partials/`
- Official-final numerical fixtures: `fixtures/official-final/`
- Exact 32K/64K/128K/256K native inputs plus the 262,080-token
  decode-reserved OpenAI fixture: `fixtures/long-context/`
- ds4 commit metadata, patch audit, and complete Motif-touched source snapshot:
  `ds4/`
- Converter, calibration, fixture, verification, quantization, split, and
  upload sources with tests: `reproduction/`
- H200 evidence and Spark gates: `reports/`

The H200 state includes a strict explicit full-image CUDA residency path,
native Motif sessions, production latent-KV/SWA-ring state, the official
Motif tokenizer/chat/tool protocol, an OpenAI-compatible server path, and the
`test-motif3-resident` plus `test_motif3_long` gates.  The engine refuses a
Motif session if weights are streamed/offloaded or if the resident CUDA graph
cannot own the model; it never falls through to the generic DeepSeek graph.

The public GGUF shards are intentionally not duplicated in this bucket. On
the Spark host, use `scripts/pull_spark_handoff.sh` from the reproduction
repository. It validates the complete bucket checksum manifest, downloads all
11 shards from the exact 40-character Hub revision, and verifies every model
SHA-256 before runtime work begins.

The current ds4 development loader consumes one GGUF. Set `MERGED_MODEL` and,
if needed, `GGUF_SPLITTER` when running the pull script to perform a standard
one-time `llama-gguf-split --merge` after shard verification. Do not use
`--delete-splits`; retain the verified public inputs until the merged file has
also passed the native loader/structure checks.

Do not requantize or silently change the artifact during Spark kernel, cache,
or server work. The H200 record contains native short/long inference evidence,
with correctness completed through 128K. Its two 256K attempts were stopped at
245,760 and 106,496 completed prefill tokens before decode when the user moved
remaining execution and optimization to Spark. This handoff does not itself
claim a completed 256K gate or single-GB10 262,144-token serving; resident
Spark prefill/decode, unified-memory headroom, and OpenAI-compatible server
validation on the target machine remain explicit release gates.
