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
  `d878ea1a1d67bc0f0bd60e20e75b4a011aa2d8d9`
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

BUCKET=Baekpica/motif-3-spark-handoff
MODEL_DIR=/workspace/motif-3-model
MERGED_MODEL=/workspace/motif-3-model/Motif-3-MQ87-88-FIT.gguf
GGUF_SPLITTER=/workspace/llama.cpp/build/bin/llama-gguf-split
export BUCKET MODEL_DIR MERGED_MODEL GGUF_SPLITTER
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

The H200 handoff stage already completed a clean compile/link-only rehearsal
of this exact revision: all five runtime programs and the Motif CUDA,
resident, and long-test binaries contained only `sm_121a` code objects. That
does not replace this clean rebuild or any execution gate on the actual GB10.

```bash
DS4_DIR=/workspace/motif-3-ds4
DS4_REV=d878ea1a1d67bc0f0bd60e20e75b4a011aa2d8d9
git clone --branch feature/motif-3-model-loader \
  https://github.com/Baekpica/ds4.git "$DS4_DIR"
git -C "$DS4_DIR" checkout --detach "$DS4_REV"
test "$(git -C "$DS4_DIR" rev-parse HEAD)" = "$DS4_REV"

make -C "$DS4_DIR" clean
make -C "$DS4_DIR" cuda-spark 2>&1 | tee "$HANDOFF_DIR/reports/spark-build.log"
grep -F -- '-gencode arch=compute_121a,code=sm_121a' \
  "$HANDOFF_DIR/reports/spark-build.log"
```

Build and run the supplied target-host fixtures with that same architecture;
do not reuse any binary from the H200 bundle:

```bash
make -C "$DS4_DIR" \
  tests/test_motif3_loader tests/test_motif3_reference \
  tests/test_motif3_tokenizer tests/test_motif3_cuda \
  tests/test_motif3_resident tests/test_motif3_long \
  CUDA_ARCH=sm_121

"$DS4_DIR/tests/test_motif3_loader" "$MERGED_MODEL"
"$DS4_DIR/tests/test_motif3_reference" \
  "$HANDOFF_DIR/fixtures/official-final"
"$DS4_DIR/tests/test_motif3_tokenizer" "$MERGED_MODEL" \
  "$HANDOFF_DIR/fixtures/official-final/tokenizer-chat.ds4tok"
"$DS4_DIR/tests/test_motif3_cuda" \
  "$HANDOFF_DIR/fixtures/official-final"
"$DS4_DIR/tests/test_motif3_resident" "$MERGED_MODEL"

for TOKENS in 32768 65536 131072; do
  "$DS4_DIR/tests/test_motif3_long" "$MERGED_MODEL" \
    "$HANDOFF_DIR/fixtures/long-context/context-${TOKENS}.tokens.npy"
done

# The authoritative decode-reserved 256K input is already exactly 262,080
# tokens and preserves the complete question. test_motif3_long admits 64
# additional decode positions, yielding the native 262,144-token context.
"$DS4_DIR/tests/test_motif3_long" "$MERGED_MODEL" \
  "$HANDOFF_DIR/fixtures/long-context/context-262144-server.tokens.npy"
```

For the release server, deliberately omit both `--ssd-streaming` and
`--kv-disk-dir`:

```bash
CUDA_VISIBLE_DEVICES=0 "$DS4_DIR/ds4-server" \
  --cuda --model "$MERGED_MODEL" --ctx 262144 \
  --prefill-chunk 256 --batched-session 1 --tokens 64 \
  --host 127.0.0.1 --port 8000 2>&1 | \
  tee "$HANDOFF_DIR/reports/spark-server.log"
```

In another shell, send the decode-reserved API fixture. It retains the full
25-token question/generation tail and all three records while removing exactly
64 one-token filler repetitions, so the official rendered prompt is 262,080
tokens inside the native 262,144-token admission. The validator requires the
exact model ID, API token accounting, JSON array, record order, stop reason,
and non-empty decode.

```bash
python3 "$HANDOFF_DIR/reproduction/scripts/run_openai_long_gate.py" \
  --text "$HANDOFF_DIR/fixtures/long-context/context-262144-server.txt" \
  --answer "$HANDOFF_DIR/fixtures/long-context/context-262144-server.answer.json" \
  --output "$HANDOFF_DIR/reports/spark-openai-256k-result.json"
```

Capture the same process once after startup and again immediately after the
256K request has completed prefill and decode. The server log preserves the
prefill/decode timing and allocator diagnostics. The post-decode capture is
the release measurement; it must retain at least 8 GiB `MemAvailable`, show
process `VmSwap: 0`, and keep the GGUF mapping RSS small after the CUDA-owned
image is prepared.

```bash
SERVER_PID=$(pgrep -n -x ds4-server)
"$HANDOFF_DIR/reproduction/scripts/capture_spark_memory.sh" \
  "$SERVER_PID" "$MERGED_MODEL" \
  "$HANDOFF_DIR/reports/spark-memory-startup.txt"

# Run the strict OpenAI request above, then capture the still-resident server.
"$HANDOFF_DIR/reproduction/scripts/capture_spark_memory.sh" \
  "$SERVER_PID" "$MERGED_MODEL" \
  "$HANDOFF_DIR/reports/spark-memory-post-decode.txt"
```

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
  weight offload; two native-`sm_90` repeats measured 97,438,334,976–
  97,991,524,352 bytes, and capacity accounting uses the higher result.
- The native Motif graph now owns GDLA, mHC, PolyNorm, dense/shared/routed MoE,
  latent cache, official chat/tool semantics, and the MTP weight path. It never
  falls through to the generic DeepSeek graph.
- A physical 262,144-token latent-cache allocation measured 4,236,751,872
  tensor bytes (3.946 GiB) and a 4,334,813,184-byte CUDA free-memory delta
  (4.037 GiB) with the model still resident.
- After resident CUDA preparation, the raw GGUF tensor mapping is explicitly
  discarded. H200 `/proc` evidence reduced that mapping from 91,955,608 kB RSS
  to 9,416 kB after copy and 29,512 kB after the latest native-`sm_90`
  resident regression.
  Recheck this behavior and final `MemAvailable` on GB10; do not set
  `DS4_CUDA_KEEP_MODEL_PAGES` for the capacity run.
- Short-context expanded/latent parity, chunk/ring lifecycle, OpenAI 2K,
  structured tool continuation with live-prefix reuse, two-session continuous
  batching, and native 32K/64K/128K retrieval passed on H200. See the H200
  report for the final 256K row.

See `reports/H200-DEVELOPMENT.md` and `reports/MIXED-QUANT.md` for exact
artifact and numerical records. H200 results are development evidence only.
One separately disclosed legacy-trim 256K process began before the host
source-page discard fix. The final all-`sm_90` overlay passed the resident
graph/cache regression and exact 32K/64K/128K gates, and its corrected
full-question 256K gate is the authoritative H200 row. The Spark gate must
still combine these properties in one target-host OpenAI server run.

## Initial capacity projection

The completed artifact and H200 cache allocation supply measured rows below;
unified-memory overhead and runtime pools remain target-host measurements:

| Pool | Projected size |
|---|---:|
| Public split GGUF aggregate | 87.69570 GiB |
| H200 resident model/runtime initialization delta | 97,991,524,352 bytes (91.26171875 GiB), conservative higher of two native-`sm_90` repeats |
| Latent KV + RoPE key + bounded SWA ring payload at 262,144 | 4,236,751,872 bytes (3.946 GiB) measured |
| H200 CUDA allocation delta for that session | 4,334,813,184 bytes (4.037 GiB) measured |
| H200 combined model/runtime + 256K-session delta | 102,326,337,536 bytes (95.298828125 GiB), using the higher resident repeat |
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
