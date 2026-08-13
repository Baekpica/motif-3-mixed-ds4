---
base_model: Motif-Technologies/Motif-3
base_model_relation: quantized
license: mit
language:
  - en
  - ko
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - motif
  - motif-3
  - mixture-of-experts
  - mixed-quantization
  - iq2_xxs
  - dgx-spark
  - long-context
---

# Motif 3 — Mixed-Quant GGUF

**A structurally intact 314.8B-parameter Motif-3 MoE compressed to 87.70 GiB.
Native H200 residency, latent-KV execution, OpenAI serving, tools, continuous
batching, and retrieval through 128K are validated; the single-GB10 256K
release gate remains open.**

Full-topology mixed-quant conversion of
[`Motif-Technologies/Motif-3`](https://huggingface.co/Motif-Technologies/Motif-3),
designed as the resident-weight baseline for one 128 GB-class DGX Spark.
This is an independent conversion, not an official Motif Technologies release.

Nothing is pruned, merged, expert-dropped, or layer-dropped. The artifact
retains all 53 target-model layers, the first two dense layers, all 51 sparse
layers, all 384 routed experts per sparse layer with top-8 routing, the shared
expert, Grouped Differential Latent Attention (GDLA), Expert-Specific
PolyNorm, modified mHC, and the complete one-layer MTP predictor.

## Artifact

| Variant | Split | Exact size | Purpose |
|---|---:|---:|---|
| **MQ87-88-FIT** | 11 files | 94,162,542,816 bytes (87.6957 GiB) | one-Spark capacity/release baseline |

Start with `Motif-3-MQ87-88-FIT-00001-of-00011.gguf`; compatible split-aware
runtimes discover the remaining shards automatically. Exact per-shard hashes
are published in `MQ87-88-FIT-SHA256SUMS`.

The weight files are fixed at Hub revision
`efd6044e25e7f8e3b459a737d021091e2e69b6c6`. Later model-card/report commits
do not change shard bytes or hashes.

```bash
hf download Baekpica/Motif-3-Mixed-Quant-GGUF \
  --revision efd6044e25e7f8e3b459a737d021091e2e69b6c6 \
  --include 'Motif-3-MQ87-88-FIT-*.gguf' \
  --include MQ87-88-FIT-SHA256SUMS \
  --local-dir ./Motif-3-MQ87-88-FIT
```

The name is a nominal capacity class, not a hard file-size gate. The actual
artifact is accepted when its measured aligned/repacked resident form fits the
target Spark together with latent KV, workspace, and server state. It is not
repeatedly reduced merely to force a cosmetic 87–88 GiB filename claim.

## Precision recipe

Precision is assigned by module role. Always-active and control paths stay
conservative; the 51×384 routed-expert stack carries the compression burden.

| Tensor group | Type | Rationale |
|---|---|---|
| Token embedding and LM head | `Q8_0` | token/logit fidelity |
| GDLA projections and differential/output-gate paths | `Q8_0` | attention and long-context stability |
| Dense MLP, shared expert, MTP projections | `Q8_0` | always active |
| Routed expert gate/up, layers 2–52 | `IQ2_XXS` + Q8 imatrix | dominant parameter mass |
| Routed expert down, layers 2–52 | `Q2_K` + Q8 imatrix | capacity baseline |
| Router weights | `F32` | top-8 decision stability |
| RMSNorm, Expert PolyNorm coefficients/biases | `F32` | normalization/activation stability |
| mHC controls/scalars | `F32` | sigmoid, clamp, Sinkhorn stability |
| mHC small projections | `BF16` | small protected matrices |

The completed GGUF contains 2,287 tensors: **1,273 F32, 318 BF16, 543
Q8_0, 102 IQ2_XXS, and 51 Q2_K**. The 102 IQ2 tensors are separately stored
gate and up matrices. They originate from 51 fused checkpoint gate/up tensors;
the split is lossless before quantization and retains every expert value.

No Q4 edge-layer promotion is included in this release. MQ95/MQ97 are outside
the scope of this capacity-first artifact.

## Q8_0 activation calibration

Calibration was collected only after producing and freezing the separate
full-topology Q8_0 reference in
[`Baekpica/Motif-3-GGUF`](https://huggingface.co/Baekpica/Motif-3-GGUF).

The corpus reuses the checksum-pinned normalization and target shares audited
for the preceding Solar Open 2 mixed-quant release, but every record is
rendered and counted again with Motif-3's official final tokenizer and chat
template. It contains 1,539 documents, 4,079,555 official-tokenizer tokens,
and 17,011,959 bytes.

| Calibration bucket | Target token share |
|---|---:|
| Instruction-following chat | 22% |
| Cascade stage 1 reasoning | 16% |
| Cascade stage 2 reasoning/tools | 16% |
| Korean | 16% |
| Other multilingual | 12% |
| Finance | 6% |
| SWE agentic | 6% |
| Algorithmic code | 6% |

The H200 calibration runtime directly dequantized the Q8_0 GGUF and executed
the official Motif-3 equations with a layer-major schedule: mHC, expanded
historical K/V GDLA, interleaved 128-token SWA/full attention, YaRN, the
differential signal/noise heads, elementwise attention output gate, sigmoid
top-8 routing with correction-bias selection and route renormalization, shared
experts, and per-expert PolyNorm.

| | |
|---|---:|
| Calibration chunks | 590 × 512 tokens |
| Total official-tokenizer tokens | 302,080 |
| Routed observations | 123,248,640 |
| Sparse layer/expert cells | 51 × 384 |
| Zero-coverage cells | **0** |
| Routes per layer/expert | min 313, median 5,934, p95 9,986.7, max 85,011 |
| Imatrix size | 742,004,501 bytes |
| Imatrix SHA-256 | `54fcc4d6d1fe96a3fc12bd24869eff128c724737ac07ce866ce08018a5e3cfbc` |

Gate/up importance observes the exact FFN-normalized input. Down importance
observes the Expert-Specific PolyNorm output after route weighting, matching
the ds4 routed-MoE execution order. No uniform, random, proxy-model, or
source-BF16 activation matrix was substituted.

The machine-readable activation report is published here as
`Q8_0-IMATRIX-REPORT.json`. Corpus construction is included in the public
reproduction materials; the exact rendered corpus, final imatrix, and
rank-local accumulators are preserved in the private Spark handoff.

## Provenance

| | |
|---|---|
| Source model | `Motif-Technologies/Motif-3` |
| Exact source revision | `ccceb1a5fd7b5eb32e47841216b3caf5666c07bc` |
| Source parameters | 314,841,775,750 |
| Source tensors | 2,236 |
| GGUF tensors | 2,287 |
| Native context metadata | 262,144 tokens |
| Full Q8_0 reference | `Baekpica/Motif-3-GGUF@5c266c95bf8c8d822d50e5e1cce9d108eaadb2af` |
| Fixed mixed-weight revision | `efd6044e25e7f8e3b459a737d021091e2e69b6c6` |
| Official implementation oracle | `MotifTechnologies/vllm@4cd9eb4129883565e69d508038d783d59ee01867` |
| Conversion base | `ggml-org/llama.cpp@1d2869c6e54d5003f3927a79efbca0fefa034a6d` |
| ds4 base | `Baekpica/ds4@b0309611041655f4e45671cfd9c9886aff161406` |
| Native ds4 implementation | `Baekpica/ds4:feature/motif-3-model-loader@0a360dbe46dd50d26e170300adf18993ac3ab1a0` |
| Public reproduction | [`Baekpica/motif-3-mixed-ds4`](https://github.com/Baekpica/motif-3-mixed-ds4) |
| Private Spark handoff | Expensive calibration state plus offline reproduction/runtime snapshots are preserved in `hf://buckets/Baekpica/motif-3-spark-handoff` |

Only the official final Motif-3 checkpoint was used. Motif-3-Beta was not used
as a source, calibration input, implementation oracle, or fallback.

## Model structure and context memory

Layers whose index is divisible by four use full attention (14 layers); the
other 39 layers use a bounded 128-token sliding window. The production target
does not retain expanded historical K/V. Its persistent context state is
latent KV plus the decoupled RoPE key for full-attention layers, bounded SWA
ring state, and cache identity/position state.

On H200, creating a native 262,144-token session while the model remained
resident allocated 4,236,751,872 bytes (3.946 GiB) of cache tensor payload and
produced a 4,334,813,184-byte (4.037 GiB) CUDA free-memory delta including
allocator overhead. This is a physical H200 measurement, not a Spark claim.
Physical unified-memory residency and OS headroom on the target GB10 remain
authoritative.

The H200 model/runtime initialization delta plus the 256K-session delta was
102,326,337,536 bytes (95.298828125 GiB). This supports the capacity design;
it does not predict GB10 driver, allocator, or OS overhead.

## Runtime compatibility

Motif 3 is not a Llama-family graph. A compatible runtime must implement its
384E sigmoid router, route normalization/scale, shared expert,
Expert-Specific PolyNorm, modified mHC, GDLA/differential heads and output
gate, interleaved SWA/full attention, YaRN, latent KV semantics, and MTP.

The target runtime is
[`Baekpica/ds4`](https://github.com/Baekpica/ds4) on the H200 development
branch named `feature/motif-3-model-loader`, pinned above to its exact public
implementation commit. The private Spark handoff also carries an offline
source snapshot and commit metadata.
Stock GGUF runtimes should not be assumed to execute this architecture merely
because they can parse the container.

The current native branch has an explicit Motif tensor binder and CUDA graph,
production latent-KV/SWA-ring sessions, strict device-resident model loading,
the official tokenizer/chat/reasoning/tool protocol, and an OpenAI-compatible
`ds4-server` path. Motif sessions refuse streaming/offloaded weights and never
fall through to the generic DeepSeek graph.

The public files use standard GGUF splitting. The current ds4 development
loader consumes one merged GGUF, so merge from the first shard with
`llama-gguf-split --merge` before launch. This is a one-time artifact assembly,
not SSD weight streaming; production admission still requires the merged or
repacked weights to be resident and forbids simultaneous physical residency of
duplicate raw/repacked mappings.

## Validation status

The unsharded artifact and the complete 11-file split set independently passed
strict source-map validation. The verifier checked the pinned revision,
architecture metadata, native 262,144-token context, all 53 target layers,
all routed gate/up/down tensors for sparse layers 2–52, all 384 experts, the
shared experts, and the complete MTP block. It found exactly 2,287 unique
tensors with no missing, duplicate, unexpected, mistyped, misshaped, or
out-of-bounds payloads. The split set totals 94,162,542,816 bytes; every shard
also has a published SHA-256 digest.

A separate 57-row numerical comparison sampled the embedding and LM head,
GDLA/control paths, dense/shared paths, routed experts at layers 2, 26, and 52,
both halves of the checkpoint's fused gate/up weights, and MTP. Protected
F32/BF16 rows were source-exact. Minimum sampled cosine was `0.9999740` for
Q8_0, `0.9417932` for IQ2_XXS, and `0.9580462` for Q2_K. The native ds4
Motif-3 binder also accepted the completed mixed artifact as the official-final
53-layer, 14-full/39-SWA, 384E top-8 topology with MTP present.

The rebuilt `sm_90` runtime copied the full 87.70 GiB image into one H200 in
9.315–10.924 seconds without SSD streaming or CPU weight offload. The measured
CUDA free-memory delta for model and runtime initialization was
97,991,524,352 bytes. Strict residency fails startup instead of silently using
host-mapped weights.

Once optional CUDA preparation finishes, ds4 discards the raw GGUF tensor
pages while retaining only metadata/tokenizer mapping. Measured GGUF mapping
RSS fell from 91,955,608 kB to 9,416 kB and remained low through inference, so
the raw file is not kept as a second steady physical weight image beside the
CUDA-owned model copy.

The automated resident gate caps this mapping at 262,144 kB both after copy
and after native graph/cache execution. Its final H200 run measured 9,416 kB
and 29,512 kB respectively.

The native expanded-path oracle and production latent path selected the same
first token and all top-8 logits on the short fixture; full-logit cosine was
`0.99490164`. Direct/chunked cache replay produced cosine `1.0`. The real
mixed sparse-layer diagnostic measured Q2 down cosine `0.9996071` and final
sparse-output cosine `0.9998363`. A real-weight MTP diagnostic evaluated 19
teacher-forced rows with finite logits.

### H200 end-to-end evidence

| Gate | Interface | Prefill | Decode | Correctness |
|---:|---|---:|---:|---|
| 2K | OpenAI chat | 346.72 tok/s | 12.64 tok/s | exact beginning/middle/end JSON |
| 32K | native | 125.25 tok/s | 1.946 tok/s | exact beginning/middle/end JSON |
| 32K | OpenAI chat | 124.92 tok/s | 1.95 tok/s | exact JSON; 32,768 prompt tokens |
| 64K | native | 68.54 tok/s | 1.023 tok/s | exact beginning/middle/end JSON |
| 128K | native | 36.34 tok/s | 0.525 tok/s | exact JSON; 131,072-token prompt + 49-token decode |
| 256K | native | running | running | isolated H200 gate in progress |

The OpenAI server also completed a structured `get_weather` tool-call/result
loop. Its no-thinking continuation reused the full 165-token live prefix and
evaluated only the 51-token tool-result/new-assistant suffix. Two simultaneous
deterministic requests on two resident sessions both returned the expected
output, exercising continuous batching and session isolation.

Machine-readable structural and numerical reports are included as
`MQ87-88-FIT-VERIFY.json` and `MQ87-88-FIT-SAMPLE-VERIFY.json`.
The human-readable artifact and host records are included as
`MIXED-QUANT.md` and `H200-DEVELOPMENT.md`.

H200 development has validated the pinned source inventory, official router,
PolyNorm, mHC, tokenizer/chat/tool handling, expanded and latent GDLA paths,
the full Q8_0 GGUF, the 302,080-token Q8 activation-collection pass, strict
residency, short/long native generation, and the OpenAI server path above.

This card does **not** claim completed single-DGX-Spark 262,144-token serving.
The native graph and server now exist, but the release gate still requires the
exact artifact to complete 256K prefill followed by decode on GB10 with model,
latent KV, workspace, and server resident, no SSD streaming or CPU weight
offload, measured unified-memory/OS headroom, and short/32K/64K/128K/256K
correctness. Those target-machine measurements must be reported separately.

## Limitations

- This is a hardware-oriented, very-low-bit routed-expert quantization. Quality
  must be evaluated for the intended languages, reasoning, code, tools, and
  long-context workloads.
- The public GGUF is a weight artifact, not a guarantee that an unrelated
  runtime implements Motif-3 correctly.
- The 262,144-token metadata comes from the source architecture. It is not a
  substitute for measured resident prefill/decode validation.
- DGX Spark uses coherent unified memory; conventional host-RAM plus discrete
  VRAM accounting describes a different deployment.

## Acknowledgements

- **[Motif Technologies](https://huggingface.co/Motif-Technologies)** — the
  Motif-3 model and official implementation.
- **[antirez/ds4](https://github.com/antirez/ds4)** — the original engine,
  loader, server, session machinery, and routed-MoE foundation.
- **[Entrpi/ds4-on-spark](https://github.com/Entrpi/ds4-on-spark)** — DGX Spark
  CUDA and unified-memory groundwork used by downstream ds4 development.
- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** — GGUF and the
  quantization formats used by this artifact.

Errors in this conversion, runtime port, calibration, or measurements are
ours, not theirs.

## License and attribution

The source model identifies its license as MIT. See the official
[`Motif-Technologies/Motif-3` model card](https://huggingface.co/Motif-Technologies/Motif-3)
for intended use, evaluation, citation, and license context.
