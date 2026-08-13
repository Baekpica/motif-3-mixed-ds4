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
  - q8_0
  - long-context
---

# Motif 3 — Q8_0 GGUF

Full-topology Q8_0 reference conversion of
[`Motif-Technologies/Motif-3`](https://huggingface.co/Motif-Technologies/Motif-3).
This is an independent conversion, not an official Motif Technologies release.

The conversion retains all 53 transformer layers, including the first two
dense layers and all 51 sparse layers; all 384 routed experts per sparse layer
(top-8 routing); the shared expert; Grouped Differential Latent Attention
(GDLA); Expert-Specific PolyNorm; modified mHC; and the one-layer MTP head.
Nothing is pruned, merged, expert-dropped, or layer-dropped.

## Artifact

| Variant | Split | Exact size | Purpose |
|---|---:|---:|---|
| Q8_0 reference | 11 shards | 334,810,734,464 bytes (311.817 GiB) | imatrix calibration and full-model correctness reference |

Start with `Motif-3-Q8_0-00001-of-00011.gguf`; split-aware runtimes discover
the remaining shards automatically. Exact hashes are in `Q8_0-SHA256SUMS`.

The weight files are fixed at Hub revision
`5c266c95bf8c8d822d50e5e1cce9d108eaadb2af`. Later model-card commits do
not change shard bytes or hashes.

```bash
hf download Baekpica/Motif-3-GGUF \
  --revision 5c266c95bf8c8d822d50e5e1cce9d108eaadb2af \
  --include 'Motif-3-Q8_0-*.gguf' \
  --include Q8_0-SHA256SUMS \
  --local-dir ./Motif-3-Q8_0
```

This reference follows the control-path protection policy used by the mixed
release. Its 2,287 GGUF tensors comprise:

| Tensor type | Count | Role |
|---|---:|---|
| `Q8_0` | 696 | embeddings, LM head, GDLA/dense/shared/MTP projections, all routed expert gate/up/down matrices |
| `BF16` | 318 | small mHC projection matrices |
| `F32` | 1,273 | routers, norms, PolyNorm coefficients/biases, and mHC controls/scalars |

The protected BF16/F32 tensors are intentional. “Q8_0 reference” means every
large quantizable weight matrix, including every routed expert matrix, is
Q8_0; decision-sensitive controls remain at their locked higher precision.

## Provenance

| | |
|---|---|
| Source model | `Motif-Technologies/Motif-3` |
| Exact source revision | `ccceb1a5fd7b5eb32e47841216b3caf5666c07bc` |
| Source parameters | 314,841,775,750 |
| Source tensors | 2,236 |
| GGUF tensors | 2,287 (51 fused gate/up tensors are losslessly separated) |
| GGUF architecture | `motif3` |
| Native context metadata | 262,144 tokens |
| Fixed Q8-weight revision | `5c266c95bf8c8d822d50e5e1cce9d108eaadb2af` |
| Official implementation oracle | `MotifTechnologies/vllm@4cd9eb4129883565e69d508038d783d59ee01867` |
| Conversion base | `ggml-org/llama.cpp@1d2869c6e54d5003f3927a79efbca0fefa034a6d` |
| Native runtime | [`Baekpica/ds4:feature/motif-3-model-loader@d878ea1`](https://github.com/Baekpica/ds4/tree/feature/motif-3-model-loader) |
| Public reproduction | [`Baekpica/motif-3-mixed-ds4`](https://github.com/Baekpica/motif-3-mixed-ds4) |
| Private Spark handoff | Expensive state is preserved in `hf://buckets/Baekpica/motif-3-spark-handoff` |

Only the official final Motif-3 checkpoint above was used. Motif-3-Beta was
not used as a source, calibration input, implementation oracle, or fallback.

## Validation status

The completed 11-shard artifact passed a strict source-map verification:

- all 2,287 expected GGUF tensors are present exactly once;
- every shape, type, byte length, and split index/count matches the locked map;
- all 53 layers are present, including routed gate/up/down tensors for sparse
  layers 2–52 and the full MTP block;
- `general.source.revision` is the pinned official revision above;
- the aggregate verified tensor payload is 334,801,074,712 bytes;
- no structural-template marker remains in the release files.

A separate source-row sanity pass dequantized 57 representative rows spanning
the embedding, LM head, GDLA, dense/shared paths, routed gate/up/down matrices
at sparse layers 2/26/52, protected controls, and MTP. It also samples both
halves of the fused source gate/up tensors across experts 0/173/383. F32 and
BF16 protected rows were source-exact; Q8_0 cosine was at least
`0.9999740124`, with maximum sampled relative RMSE `0.007201264`.

The machine-readable report and complete per-shard SHA-256 list are included
as `Q8_0-VERIFY.json`, `Q8_0-SAMPLE-VERIFY.json`, and
`Q8_0-SHA256SUMS`.

The 262,144-token value is source architecture metadata. This Q8 repository
does not by itself claim completed single-DGX-Spark 256K serving. That release
gate belongs to the native `motif3` ds4 runtime and requires measured resident
weights, latent KV, workspace/server memory, prefill, decode, and semantic
checks on GB10.

## Runtime compatibility

Motif 3 is not a Llama-family graph. A runtime must implement its 384E sigmoid
router and route normalization, shared expert, Expert-Specific PolyNorm,
GDLA/differential heads and output gate, modified mHC, interleaved SWA/full
attention with YaRN, latent KV semantics, and MTP. Do not assume that a stock
GGUF runtime recognizes `motif3` merely because it can parse the container.

The H200 development branch is publicly available as
[`Baekpica/ds4:feature/motif-3-model-loader`](https://github.com/Baekpica/ds4/tree/feature/motif-3-model-loader)
at exact implementation commit
`d878ea1a1d67bc0f0bd60e20e75b4a011aa2d8d9`. The private Spark handoff also
preserves an offline source snapshot and commit metadata.

## Intended use in the mixed-quant pipeline

This artifact is produced first and frozen as the full-model Q8_0 reference.
The next stage collects an activation importance matrix from the pinned
calibration corpus, after which routed gate/up weights are quantized to
`IQ2_XXS` with that imatrix and routed down weights to `Q2_K`. The resulting
MQ87–88 artifact is published separately at
[`Baekpica/Motif-3-Mixed-Quant-GGUF`](https://huggingface.co/Baekpica/Motif-3-Mixed-Quant-GGUF).

## License and attribution

The source model identifies its license as MIT. See the official
[`Motif-Technologies/Motif-3` model card](https://huggingface.co/Motif-Technologies/Motif-3)
for the architecture, intended-use, evaluation, citation, and license context.
