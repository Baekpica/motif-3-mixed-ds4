# DGX Spark handoff: Motif-3 MQ87-88-FIT

## Immutable inputs

- Public model repository: `Baekpica/Motif-3-Mixed-Quant-GGUF`
- Exact Hub revision: see `model/revision.txt`
- Exact split total and per-shard hashes: see `model/sha256.txt` and the
  public repository's `MQ87-88-FIT-SHA256SUMS`
- Expensive development state:
  `hf://buckets/Baekpica/motif-3-spark-handoff`
- Public reproduction repository:
  `https://github.com/Baekpica/motif-3-mixed-ds4`
- The private bucket additionally freezes `reproduction/` and `ds4/` offline
  snapshots; no separate HF model repository is used for private handoff data
- ds4 branch: `feature/motif-3-model-loader`
- ds4 base: `b0309611041655f4e45671cfd9c9886aff161406`
- ds4 Motif implementation commit:
  `0a360dbe46dd50d26e170300adf18993ac3ab1a0`
- Source model revision:
  `Motif-Technologies/Motif-3@ccceb1a5fd7b5eb32e47841216b3caf5666c07bc`

Do not rebuild or requantize the GGUF during Spark runtime work. Kernel,
packing, latent-KV, and server experiments must retain the exact model
revision and all shard hashes. Return to the H200 pipeline only if a genuinely
new weight recipe is required.

## Receive and verify

```bash
HANDOFF_DIR=/workspace/motif-3-spark-handoff
mkdir -p "$HANDOFF_DIR/scripts"
hf buckets cp \
  hf://buckets/Baekpica/motif-3-spark-handoff/scripts/pull_spark_handoff.sh \
  "$HANDOFF_DIR/scripts/pull_spark_handoff.sh"
chmod +x "$HANDOFF_DIR/scripts/pull_spark_handoff.sh"

BUCKET=Baekpica/motif-3-spark-handoff \
MODEL_DIR=/workspace/motif-3-model \
MERGED_MODEL=/workspace/motif-3-model/Motif-3-MQ87-88-FIT.gguf \
GGUF_SPLITTER=/workspace/llama.cpp/build/bin/llama-gguf-split \
"$HANDOFF_DIR/scripts/pull_spark_handoff.sh" "$HANDOFF_DIR"
```

The merge is one-time artifact assembly because the current ds4 development
loader accepts one GGUF. It does not authorize SSD streaming during serving.
Keep the verified shards until the merged file passes the native loader and
structural checks, then ensure duplicate raw/repacked mappings are not both
physically resident during runtime admission.

Then obtain the pinned ds4 branch/state and build specifically for GB10. The
build log must show `sm_121a`; do not reuse an H200 `sm_90` binary or object.
Before serving, confirm that raw GGUF mappings and any aligned/repacked
artifact are not simultaneously physically resident.

## What the H200 stage established

- Official final checkpoint only; full 53-layer target topology and complete
  one-layer MTP preserved.
- All 51 sparse layers retain 384 routed experts, top-8 routing, and the shared
  expert; no pruning, merging, or dropping.
- Full Q8_0 reference generated, split, strictly verified, and published.
- Real Q8_0 forward activation collection processed 302,080 tokens and
  123,248,640 routed observations with zero uncovered layer/expert cells.
- Official-final router, Expert-Specific PolyNorm, modified mHC, YaRN,
  expanded GDLA, tokenizer, and chat fixtures passed on CPU/H200 CUDA.
- Deterministic UTF-8 inputs re-tokenize exactly to 32K, 64K, 128K, and 256K
  token arrays and include beginning/middle/end retrieval answers.
- The `sm_90` ds4 CLI/server build completed and a strict full-image copy made
  the 87.70 GiB mixed GGUF resident on one H200 without SSD streaming or CPU
  weight offload; CUDA free-memory delta was 97,991,524,352 bytes.
- The native Motif graph now owns GDLA, mHC, PolyNorm, dense/shared/routed MoE,
  latent cache, official chat/tool semantics, and the MTP weight path. It never
  falls through to the generic DeepSeek graph.
- A physical 262,144-token latent-cache allocation measured 4,236,751,872
  tensor bytes (3.946 GiB) and a 4,334,813,184-byte CUDA free-memory delta
  (4.037 GiB) with the model still resident.
- After resident CUDA preparation, the raw GGUF tensor mapping is explicitly
  discarded. H200 `/proc` evidence reduced that mapping from 91,955,608 kB RSS
  to 9,416 kB after copy and 29,512 kB after the complete resident regression.
  Recheck this behavior and final `MemAvailable` on GB10; do not set
  `DS4_CUDA_KEEP_MODEL_PAGES` for the capacity run.
- Short-context expanded/latent parity, chunk/ring lifecycle, OpenAI 2K,
  structured tool continuation with live-prefix reuse, two-session continuous
  batching, and native 32K/64K/128K retrieval passed on H200. See the H200
  report for the final 256K row.

See `reports/H200-DEVELOPMENT.md` and `reports/MIXED-QUANT.md` for exact
artifact and numerical records. H200 results are development evidence only.
The 128K/256K numerical processes began before the host source-page discard
fix; the final overlay separately passed the complete resident graph/cache
regression. The Spark gate must combine both properties in one target-host
server run.

## Initial capacity projection

The completed artifact and H200 cache allocation supply measured rows below;
unified-memory overhead and runtime pools remain target-host measurements:

| Pool | Projected size |
|---|---:|
| Public split GGUF aggregate | 87.69570 GiB |
| H200 resident model/runtime initialization delta | 97,991,524,352 bytes (91.26171875 GiB) measured |
| Latent KV + RoPE key + bounded SWA ring payload at 262,144 | 4,236,751,872 bytes (3.946 GiB) measured |
| H200 CUDA allocation delta for that session | 4,334,813,184 bytes (4.037 GiB) measured |
| H200 combined model/runtime + 256K-session delta | 102,326,337,536 bytes (95.298828125 GiB) measured |
| Steady CUDA workspace budget | up to 6 GiB |
| Server/session budget | up to 3 GiB |
| Required final `MemAvailable` | at least 8 GiB |

The latent-cache projection assumes only 14 full-attention layers retain
full-history latent KV and decoupled RoPE keys; the other 39 layers retain a
bounded 129-token SWA ring. Full-history expanded K/V is forbidden in the
production cache. Actual unified-memory residency on the Spark is
authoritative; GGUF file size is not an admission result.

## Required Spark sequence

1. **S1 — native load and smoke.** Build for `sm_121a`; validate all 2,287
   tensors, the 14-full/39-SWA schedule, 384E top-8 router, shared experts,
   PolyNorm, mHC, and MTP presence. Reject SSD streaming and CPU weight
   offload. Record every memory pool and OS `MemAvailable`.
2. **S2 — revalidate expanded-GDLA correctness on GB10.** At context ≤2K,
   repeat the supplied official expanded/KV comparison, including differential
   signal/noise heads, input-dependent lambda, output gate, YaRN, mHC, router
   selections, and top logits. H200 evidence is a reference, not an `sm_121a`
   substitute.
3. **S3 — revalidate latent-KV lifecycle on GB10.** Exercise the implemented
   latent KV plus RoPE key state across chunk boundaries, continuation, rewind,
   reset, prefix reuse, session isolation, cancellation, and cache identity.
   Block-local expansion is allowed; full-history expanded K/V materialization
   is not.
4. **S4 — resident context admission.** Execute exact 32,768, 65,536,
   131,072, then 262,144-token inputs. For every frontier record requested and
   admitted tokens, cold/chunked prefill, TTFT, post-prefill decode, model,
   cache, workspace, server memory, and final `MemAvailable`.
5. **S5 — semantics and quality.** Run the supplied beginning/middle/end
   retrieval fixtures, Korean long-context QA, structured output, continued
   generation, and short-context Q8/mixed comparisons. Allocation without a
   correct answer and successful decode is not a pass.
6. **S6 — API and concurrency.** Start `ds4-server` with context 262,144 and
   validate `/v1/models`, streaming and non-streaming
   `/v1/chat/completions`, reasoning/tool rendering, long requests,
   continuous batching, prefix/session reuse, cancellation, malformed
   requests, and resident-server memory. Integrate MTP only after latent-KV
   correctness is stable.

After correctness, profile in the order `nsys → identify → modify → ncu`.
Initial Motif-specific suspects are GDLA, mHC, PolyNorm, and 384E routing/sort;
do not rewrite the existing ds4 routed GEMM fast path without profiling
evidence.

## Release rule

Do not call this a single-DGX-Spark 256K release until one GB10 completes
native 262,144-token prefill followed by decode through the OpenAI-compatible
server, with model + latent KV + workspace + server resident, no SSD
streaming, no CPU weight offload, required headroom, and the short/32K/64K/
128K/256K correctness gates. Record the exact ds4 commit and model revision in
the final Spark report.
