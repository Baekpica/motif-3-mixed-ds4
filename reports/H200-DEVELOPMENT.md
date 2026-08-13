# Motif-3 four-H200 development record

This report records only work completed on the Runpod four-H200 development
host. It is evidence for artifact construction and numerical bring-up, not a
claim of single-DGX-Spark serving.

## Immutable inputs

- Official final model: `Motif-Technologies/Motif-3`
- Revision: `ccceb1a5fd7b5eb32e47841216b3caf5666c07bc`
- Source parameters: `314,841,775,750`
- Source tensors: `2,236`
- Official implementation oracle:
  `MotifTechnologies/vllm@4cd9eb4129883565e69d508038d783d59ee01867`
- llama.cpp conversion base:
  `ggml-org/llama.cpp@1d2869c6e54d5003f3927a79efbca0fefa034a6d`
- ds4 base: `Baekpica/ds4@b0309611041655f4e45671cfd9c9886aff161406`
- ds4 Motif implementation:
  `Baekpica/ds4@bbce7eecf54703ae315328d4e240531c5a9f1a22`
- Public reproduction:
  `https://github.com/Baekpica/motif-3-mixed-ds4`

Motif-3-Beta was not downloaded or used as a source, oracle, fixture input,
or fallback.

## Host

- 4 × NVIDIA H200, `sm_90`, 143,771 MiB reported per device
- CUDA 13 development environment
- 3.0 TiB system memory
- 192 logical CPU workers used by the quantizer

## Q8_0 reference

The full-topology Q8_0 reference was generated before activation collection.
The unsharded file is exactly `334,810,733,184` bytes. Static validation found
2,287 tensors: 696 Q8_0, 318 BF16, and 1,273 F32. The source's 51 fused routed
gate/up tensors were split into separate gate and up tensors without dropping
values, accounting for the 51-tensor count increase.

The 11 split shards total `334,810,734,464` bytes. Unsharded and split
verification both checked exact tensor names, shapes, types, payload bytes,
source revision, all 53 target layers, all sparse gate/up/down groups, and the
complete MTP block.

The public Q8 reference is fixed at
`Baekpica/Motif-3-GGUF@5c266c95bf8c8d822d50e5e1cce9d108eaadb2af`.
A 57-row source comparison sampled embeddings, LM head, GDLA, dense/shared
paths, sparse layers 2/26/52, fused gate/up split halves, protected controls,
and MTP. Protected F32/BF16 rows were source-exact; the minimum Q8 cosine was
`0.9999740124` and the maximum sampled relative RMSE was `0.007201264`.

## Real Q8 activation collection

The calibration runtime memory-mapped and directly dequantized the generated
Q8_0 GGUF. Its layer-major forward pass implemented the pinned final model's
mHC, expanded GDLA, YaRN, interleaved 128-token SWA/full attention,
differential signal/noise heads, output gate, sigmoid top-8 router and route
normalization, shared expert, and Expert-Specific PolyNorm.

- Corpus: 1,539 rendered documents, 4,079,555 official-tokenizer tokens
- Sampled pass: 590 × 512 = 302,080 tokens
- Total sparse routes: 123,248,640
- Imatrix entries: 153
- Sparse coverage cells: 51 × 384
- Zero-coverage cells: 0
- Per-cell route count: min 313, median 5,934, p95 9,986.7, max 85,011
- Output bytes: 742,004,501
- SHA-256:
  `54fcc4d6d1fe96a3fc12bd24869eff128c724737ac07ce866ce08018a5e3cfbc`
- Four-rank elapsed time: approximately 2m50s

This was not a uniform, random, proxy-model, or source-BF16 imatrix.

## MQ87-88-FIT mixed artifact

The strict-imatrix mixed quantizer completed all 2,287 tensors in 35m27.980s
with 192 CPU workers. The exact unsharded result is 94,162,541,472 bytes
(87.69570056 GiB). Its type inventory is 1,273 F32, 318 BF16, 543 Q8_0,
102 IQ2_XXS, and 51 Q2_K tensors.

The 11 release shards total 94,162,542,816 bytes. Both unsharded and split
source-map verification passed. A 57-row source comparison found minimum
cosines of 0.99997401 for Q8_0, 0.94179320 for IQ2_XXS, and 0.95804620 for
Q2_K; protected F32/BF16 values were source-exact. The native ds4 Motif-3
binder accepted the completed artifact with the official-final topology and
MTP present.

## Native CUDA runtime and strict residency

The complete ds4 CLI/server suite was rebuilt for `sm_90`. Motif-3 has an
explicit tensor binder and a separate CUDA execution graph; it never falls
through to the generic DeepSeek graph. The native path implements the first
two dense layers, 51 sparse layers, sigmoid top-8 routing and scaling, shared
experts, per-expert PolyNorm, modified mHC, YaRN, interleaved full/SWA GDLA,
differential signal/noise heads, the input-dependent lambda, output gate,
latent attention state, and the MTP weight graph.

The final rebuild used `make -B ... CUDA_ARCH=sm_90`; `cuobjdump --list-elf`
then reported only `sm_90` code objects for `ds4`, `ds4-server`, the CUDA
fixture binary, and the resident regression binary. This explicit check caught
and replaced an earlier cached default-codegen object before the final resident
measurements below were frozen.

Non-streaming Motif startup now requires a complete device copy. A server
launch fails if the 87.70 GiB image cannot become device-resident instead of
silently retaining a host-mapped/no-copy weight path. The final explicitly
rebuilt `sm_90` resident gate copied the model in `9.560` seconds and measured a
`97,438,334,976`-byte (`90.746520996094 GiB`) CUDA free-memory delta for the model plus runtime
initialization versus the `94,162,541,472`-byte GGUF. Engine close returned to
within the test's release tolerance. SSD streaming, CPU weight offload,
multi-tier weight caching, and distributed placement were disabled.

After all optional CUDA mappings/caches are prepared, the runtime now releases
the original tensor payload pages with `MADV_DONTNEED` while retaining only
the small GGUF metadata/tokenizer mapping. This fixed a missing `sys/mman.h`
include that had made the intended source-page release a compile-time no-op.
On the live server, the GGUF mapping RSS fell from `91,955,608 kB` to
`9,416 kB`; total process RSS after a successful inference was
`1,028,528 kB`. The mapping did not refault during inference. Thus the raw
GGUF is not a second steady physical weight image beside the CUDA-owned copy,
which is required before unified-memory admission on Spark.

The dedicated resident regression now enforces a `262,144 kB` ceiling both
immediately after CUDA copy and after sparse, expanded/latent, MTP, chunked
decode, and 256K-cache exercises. Its final native-`sm_90` run measured
`9,416 kB` after copy and `29,640 kB` after all inference work, passing both
checks.

The earlier OpenAI 32K and native 64K run, plus the isolated 128K/256K
processes, predate the explicit native-`sm_90` rebuild; the 128K/256K jobs also
started before the source-page discard fix. Those runs execute the same source
graph on H200 through the toolkit-compatible default CUDA code object, while
their MMQ objects are `sm_90`; their timings are therefore correctness bring-up
data, not native-`sm_90` performance claims. The final overlay was rebuilt with
every CUDA code object verified as `sm_90` by `cuobjdump` and separately passed
the complete resident native graph/cache gate. A separately linked all-`sm_90`
long binary then passed the exact 32K retrieval gate at 125.34 tok/s prefill
and 1.942 tok/s decode while the isolated 256K process continued on GPU 3.

## Numerical and structural fixtures

The following completed against fixtures generated from the pinned official
final checkpoint:

- router selected IDs exact; selected-weight maximum error `2.98e-8`
- Expert-Specific PolyNorm BF16 result bit-exact
- mHC pre/post/Sinkhorn/reduced/residual results bit-exact
- YaRN inverse frequencies bit-exact; attention scale exact
- CUDA BF16 conversion, router, PolyNorm, mHC, expanded GDLA, and
  differential-output fixtures passed on H200
- native binder validated 53 target layers, 14 full-attention plus 39 SWA
  layers, 384E top-8, shared experts, and the complete MTP block
- Motif tokenizer/chat parity passed 16 raw and 5 official rendered fixtures
- reproduction Python tests: 11 passed, including exact long-fixture hashes
  and token lengths
- ds4 server unit suite, including Motif rendering, tool parsing, and the
  generation/history prefix asymmetry: passed

The real mixed-weight sparse-layer diagnostic selected the exact top-8 router
experts, measured PolyNorm NRMSE `2.63e-11`, Q2 down-projection cosine
`0.9996071`, and combined sparse-output cosine `0.9998363`.

## Expanded-path bring-up and production latent state

The official-style expanded historical K/V executor remains an independent
short-context oracle. On the final `sm_90` build, a 21-token prompt ran at
78.23 tok/s. The production latent executor ran at 127.71 tok/s; both selected the same first
token and all top-8 logits overlapped. Full-logit cosine was `0.99490164` with
NRMSE `0.10087`.

Production sessions persist normalized `kv_latent`, rotated `k_pe`, bounded
SWA rings, position/cache identity, and a separate MTP cache frontier. They do
not persist expanded historical K/V. A direct 256-row pass and chunked prefix
extension agreed on the first token and top-8 logits; an identical 64-row
suffix replay produced logit cosine `1.0` and NRMSE `0`. The final `sm_90` gate
measured 388.13 tok/s for a one-chunk pass, 220.28 tok/s for prefix extension,
and 19.380 tok/s for the following decode token.

With the model still resident, creating a native 262,144-token session
allocated `4,236,751,872` bytes (`3.946 GiB`) of tensor payload and caused a
measured CUDA free-memory delta of `4,334,813,184` bytes (`4.037 GiB`),
including allocator overhead. The allocation includes
full-history latent/RoPE state only for the 14 full-attention layers; the other
39 target layers and MTP retain bounded rings. The residency/session gate
passed without SSD streaming or CPU weight offload.

Adding that session delta to the final native-`sm_90` model/runtime
initialization delta gives `101,773,148,160` bytes
(`94.783630371094 GiB`) on H200. This is useful
capacity evidence, but it is not a substitute for GB10 unified-memory and
`MemAvailable` measurement.

## MTP

All MTP tensors are present and bound. A real-weight teacher-forced diagnostic
evaluated 19 rows, produced finite logits, and completed in 0.002 seconds. Its
sample draft token was rank 11 under the teacher target distribution. This is
a weight-path/graph diagnostic; speculative acceptance-rate optimization is
left for the Spark phase and is not used to inflate the H200 decode results.

## OpenAI-compatible server

`ds4-server` identifies the model as `motif-3`, accepts the
`Motif-Technologies/Motif-3` alias, and renders the official final
start/end-of-turn chat protocol. It supports thinking and no-thinking modes,
official JSON `<tool_call>`/`<tool_response>` records, streaming and
non-streaming chat, and resident-session prefix reuse.

The exact 2,048-token official-template retrieval request completed through
`/v1/chat/completions` with the expected three-code JSON array. Measured prefill
was 346.72 tok/s and 52-token decode was 12.64 tok/s. The API reported exactly
2,048 prompt tokens.

A real `get_weather` tool round trip returned a structured OpenAI tool call,
accepted the result, and generated `The weather in Seoul is currently 27°C
with clear skies.` The no-thinking follow-up reused all 165 live tokens and
evaluated only the 51-token tool-result/new-assistant suffix. This specifically
validates Motif's official template asymmetry: `<think></think>` is present in
the generation prompt but omitted when that assistant tool turn becomes
history.

Two resident sessions were then exercised with simultaneous non-streaming
OpenAI requests. Both identical deterministic requests returned the expected
sentinel; server logs show overlapping prefill/decode scheduling and
independent 20-token prompt accounting.

## Deterministic long-context validation

Official-tokenizer fixture v2 produces exact rendered-chat token arrays at
32,768, 65,536, 131,072, and 262,144 tokens. Each fixture places independent
retrieval codes near the beginning, middle, and end and ends with a strict
JSON-only question. Saved UTF-8 user text plus the pinned system/generation
template re-tokenizes to the exact authoritative token array.

The 262,144 fixture reserves 64 native-context tokens for decode by removing
only filler immediately before the fixed 20-token question; its actual prompt
is therefore 262,080 tokens and the admitted context remains exactly 262,144.

| Gate | Interface | Prefill | Decode | Result |
|---:|---|---:|---:|---|
| 2K | OpenAI chat | 346.72 tok/s | 12.64 tok/s | exact three-code JSON |
| 32K | native standalone, all-`sm_90` | 125.34 tok/s | 1.942 tok/s | exact three-code JSON; 43-token decode |
| 32K | OpenAI chat | 124.92 tok/s | 1.95 tok/s | exact three-code JSON; 32,768 prompt tokens |
| 64K | native standalone | 68.54 tok/s | 1.023 tok/s | exact three-code JSON |
| 128K | native standalone | 36.34 tok/s | 0.525 tok/s | exact three-code JSON; 131,072-token prompt + 49-token decode |
| 256K | native standalone | running | running | running on an isolated H200 |

These are correctness measurements from the current unfused bring-up graph,
not release speed claims. The expected decline at long context identifies GDLA
as the primary optimization target. Precision, topology, and context were not
reduced to improve the figures. NVIDIA Nsight Systems/Compute are not installed
on this Runpod image, so no synthetic profiler claim is made; focused kernel
profiling remains in the Spark handoff.

## Scope boundary

This H200 stage establishes native resident execution, latent-cache lifecycle,
short/long correctness gates, and an OpenAI-compatible server on a discrete
H200. It does **not** establish physical unified-memory residency, available
OS headroom, `sm_121a` kernel behavior, or 262,144-token serving on one GB10.
Those measurements remain mandatory on the target DGX Spark and cannot be
inferred from the GGUF size or H200 VRAM results.
