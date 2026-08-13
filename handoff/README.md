# Motif-3 H200 → DGX Spark handoff

This private bucket preserves the expensive four-H200 development state for
the full-topology Motif-3 MQ87-88-FIT artifact.

There is intentionally no separate Hugging Face model repository for the
reproduction tree. Public Hub model repositories are limited to the Q8_0 and
MQ87-88 GGUF artifacts; private code, calibration state, fixtures, and ds4
handoff material live in this bucket.

- Public model: `Baekpica/Motif-3-Mixed-Quant-GGUF`
- Exact public revision: `model/revision.txt`
- Mixed shard filenames and hashes: `model/filenames.txt` and
  `model/sha256.txt`
- Q8-derived imatrix: `calibration/Motif-3-Q8_0-imatrix.dat`
- Rank-local imatrix accumulators: `calibration/partials/`
- Official-final numerical fixtures: `fixtures/official-final/`
- Exact 32K/64K/128K/256K inputs: `fixtures/long-context/`
- ds4 base, patch, and complete Motif-touched sources: `ds4/`
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
but this handoff does not itself claim single-GB10 262,144-token serving;
resident Spark prefill/decode, unified-memory headroom, and OpenAI-compatible
server validation on the target machine remain explicit release gates.
