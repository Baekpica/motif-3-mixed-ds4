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
  `Baekpica/Motif-3-GGUF@590e3f608b5a41c87016c9d3a6b994284bdcb02f`
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
layers, 384E top-8 routing, and MTP present.

The `sm_90` CUDA build also completed a loader-only full-residency smoke on one
H200 with no SSD streaming or CPU weight offload. The 87.70 GiB image copied in
9.3–9.5 seconds and produced a 97,991,524,352-byte CUDA free-memory delta while
open; close returned to within 2 MiB of the starting free-memory level. Native
session creation remains guarded until the Motif execution graph is connected.

## Remaining release gates

This H200 result does not establish end-to-end native generation, production
latent-KV lifecycle correctness, resident GB10 memory, OpenAI-compatible 256K
serving, or 256K prefill followed by decode. Those gates remain explicit in the
private DGX Spark handoff and must be measured on the target machine.
