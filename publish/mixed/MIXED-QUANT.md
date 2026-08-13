# Motif-3 MQ87-88-FIT artifact record

This report records the completed public weight artifact built on the Runpod
four-H200 development host. It is an artifact and component-correctness record,
not a claim of single-DGX-Spark serving.

## Recipe and provenance

- Source: `Motif-Technologies/Motif-3`
- Source revision: `ccceb1a5fd7b5eb32e47841216b3caf5666c07bc`
- Official implementation oracle:
  `MotifTechnologies/vllm@4cd9eb4129883565e69d508038d783d59ee01867`
- Full Q8_0 reference:
  `Baekpica/Motif-3-GGUF@5c266c95bf8c8d822d50e5e1cce9d108eaadb2af`
- Imatrix SHA-256:
  `54fcc4d6d1fe96a3fc12bd24869eff128c724737ac07ce866ce08018a5e3cfbc`
- Quantizer mode: strict external imatrix, 192 CPU workers
- Quantization elapsed: 35m27.980s

Only the official final Motif-3 release was used. Motif-3-Beta was not used as
a source, oracle, calibration input, fixture input, or fallback.

The routed gate/up tensors for all 51 sparse layers use `IQ2_XXS`; their routed
down tensors use `Q2_K`. Embedding/head, GDLA, dense MLP, shared expert, and MTP
projections use `Q8_0`. Routers, RMSNorm, Expert-Specific PolyNorm, and mHC
control tensors remain F32; mHC small projections remain BF16. There is no Q4
promotion in this MQ87-88 release.

## Exact artifact

| Property | Result |
|---|---:|
| Unsharded bytes | 94,162,541,472 |
| Unsharded GiB | 87.69570056 |
| Split files | 11 |
| Split aggregate bytes | 94,162,542,816 |
| Split aggregate GiB | 87.69570181 |
| Tensor payload bytes | 94,152,882,712 |
| Unsharded SHA-256 | `15755a735753bc1396e5ffa539e65a779a4fd769e8833360a4d743c4c60c2f25` |
| Tensors | 2,287 |
| F32 | 1,273 |
| BF16 | 318 |
| Q8_0 | 543 |
| IQ2_XXS | 102 |
| Q2_K | 51 |

The split metadata adds 1,344 bytes to the aggregate. No tensor payload is
duplicated across shards. The nominal class is descriptive, not a cosmetic
hard gate; this exact 87.6957 GiB result was accepted without another
precision-reducing pass.

## Structural verification

Both the unsharded file and complete split set passed the locked source-map
verifier. It checked exact source revision and architecture metadata, 53 target
layers, native context 262,144, 384 routed experts with top-8 selection, all
sparse gate/up/down groups for layers 2–52, complete tensor names, shapes,
types, byte sizes and bounds, shared experts, and the one-layer MTP block.

The split result contains every expected tensor exactly once. There were no
missing, duplicate, unexpected, mistyped, misshaped, or out-of-bounds tensors.
Per-shard SHA-256 digests are in `MQ87-88-FIT-SHA256SUMS` in the public model
repository.

## Numerical source-row verification

A 57-row comparison against pinned BF16 source tensors sampled embeddings,
LM head, GDLA, protected controls, dense/shared paths, routed experts at layers
2, 26 and 52, both losslessly split halves of the checkpoint's fused gate/up
tensors, and MTP.

| Type | Minimum cosine | Maximum relative RMSE |
|---|---:|---:|
| F32 | 0.99999988 (source-exact values) | 0 |
| BF16 | 1.0 (source-exact values) | 0 |
| Q8_0 | 0.99997401 | 0.00720126 |
| IQ2_XXS | 0.94179320 | 0.34932663 |
| Q2_K | 0.95804620 | 0.28694226 |

All rows were finite and nonzero. The tiny reported F32 cosine rounding below
one comes from the float dot-product metric; its elementwise maximum absolute
difference is exactly zero.

The native ds4 Motif-3 binder accepted the final unsharded artifact and
reported the official-final 53-layer topology, 14 full-attention plus 39 SWA
layers, 384E top-8 routing, shared experts, and complete MTP weights.

## Native H200 execution evidence

The final explicitly rebuilt `sm_90` CUDA path copied the complete 87.70 GiB
image into one H200 in two strict repeats. Copy time was 9.560–12.070 seconds;
the model/runtime CUDA free-memory delta was 97,438,334,976–97,991,524,352
bytes (90.747–91.262 GiB). Capacity accounting uses the higher repeat. Strict
Motif residency rejects a failed device copy rather than continuing through a
host-mapped/no-copy path. No SSD streaming, CPU weight offload, or multi-tier
expert cache was enabled.

The production Motif session stores normalized latent KV and rotated RoPE keys
for the 14 full-attention layers and bounded SWA rings for the remaining
layers; it does not persist expanded historical K/V. A native 262,144-token
session allocated 3.946 GiB of cache tensors and a measured 4.037 GiB including
CUDA allocator overhead while the model remained resident. The exact values
were 4,236,751,872 payload bytes and a 4,334,813,184-byte CUDA delta.

Short-context expanded/latent bring-up selected the same first token and all
top-8 logits, with full-logit cosine `0.99490164`. Direct/chunked cache replay
of the same suffix produced cosine `1.0`. The real mixed sparse-layer check
measured Q2 down cosine `0.9996071` and final sparse-output cosine `0.9998363`.
The MTP real-weight teacher-forced diagnostic produced finite logits over 19
rows.

The OpenAI-compatible server returned the exact 2,048-token retrieval answer
at 346.72 tok/s prefill and 12.64 tok/s decode, completed a structured tool
call/result loop with full live-prefix reuse, and served two simultaneous
resident sessions. Native 32K, 64K, and 128K beginning/middle/end retrieval
gates also passed. Two 256K attempts reached 245,760 and 106,496 completed
prefill tokens but were stopped before decode when the user moved remaining
execution and optimization to Spark; neither is a 256K correctness pass.

## Remaining target-hardware gates

H200 results do not establish resident GB10 unified-memory headroom or
`sm_121a` behavior. The final single-DGX-Spark claim still requires the exact
artifact to complete native 262,144-token prefill followed by decode through
the OpenAI-compatible server with model, latent KV, workspace, and server
resident, no SSD streaming or CPU weight offload, and the required OS memory
headroom. Those gates remain explicit in the private Spark handoff.
