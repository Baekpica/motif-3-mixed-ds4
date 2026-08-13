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
`Baekpica/Motif-3-GGUF@590e3f608b5a41c87016c9d3a6b994284bdcb02f`.
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

## H200 CUDA build and resident loader

The complete ds4 CLI/server suite was rebuilt for `sm_90`. The final mixed
GGUF was then opened through the non-streaming CUDA engine with an explicit
full-image chunk copy on one H200. This exposed and fixed a loader bug where a
successful temporary host registration caused `DS4_CUDA_COPY_MODEL_CHUNKED`
to be treated as if device residency had already completed.

After the fix, the engine copied the 87.70 GiB image in 9.3–9.5 seconds. The
measured CUDA free-memory delta while open was 97,991,524,352 bytes for the
model plus runtime initialization, versus a 94,162,541,472-byte GGUF. The
engine closed cleanly and returned all but 2 MiB of the pre-open free-memory
level. No SSD streaming or CPU weight offload was enabled.

The residency smoke deliberately stops before session execution. Session
creation is explicitly guarded for Motif-3 until the native graph is connected,
preventing the generic DeepSeek attention/cache graph from running against
GDLA/mHC tensors and returning invalid but plausible text.

## Numerical and structural fixtures

The following completed on H200/CPU with the official final checkpoint
fixtures:

- router selected IDs exact; selected-weight max error `2.98e-8`
- Expert-Specific PolyNorm BF16 result bit-exact
- mHC pre/post/Sinkhorn/reduced/residual results bit-exact
- YaRN inverse frequencies bit-exact; attention scale exact
- CUDA BF16 conversion, router, PolyNorm, mHC, expanded GDLA, and
  differential-output fixtures passed on H200
- native ds4 mixed tensor binder validated 53 layers, 14 full-attention plus
  39 SWA layers, 384E top-8, shared experts, and MTP
- Motif tokenizer/chat parity passed 16 raw and 5 rendered fixtures
- reproduction tests: 9 passed

## Deterministic long-context inputs

Official-tokenizer fixtures were generated at exactly 32,768, 65,536,
131,072, and 262,144 tokens. Each places independent retrieval codes near the
beginning, middle, and end and finishes with a strict JSON-only question.
For every fixture, the saved UTF-8 text re-tokenizes to the exact saved token
array. SHA-256 values and expected answers are in the long-context manifest.

## Explicitly not established here

This H200 stage does not yet establish native ds4 end-to-end generation,
latent-KV lifecycle correctness, resident GB10 memory, 256K prefill followed
by decode, or OpenAI-compatible serving. H200 full-weight residency is a loader
gate only. The remaining items are Spark handoff gates and must not be inferred
from artifact size, component fixtures, or the H200 copy measurement.
