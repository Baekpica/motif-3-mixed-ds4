# Motif 3 MQ87-88-FIT and native ds4 reproduction

This repository records the official-final-only conversion and four-H200
bring-up of the full-topology Motif-3 `MQ87-88-FIT` GGUF. The immutable mixed
weights are public at
[`Baekpica/Motif-3-Mixed-Quant-GGUF`](https://huggingface.co/Baekpica/Motif-3-Mixed-Quant-GGUF),
and the preceding full Q8_0 reference is public at
[`Baekpica/Motif-3-GGUF`](https://huggingface.co/Baekpica/Motif-3-GGUF).

The recipe retains all 53 target layers, the first two dense layers, all 51
sparse layers and all 384 routed experts per layer, top-8 routing, the shared
expert, Expert-Specific PolyNorm, GDLA, modified mHC, and the complete
one-layer MTP block. Nothing is pruned, merged, expert-dropped, layer-dropped,
or distilled.

## Completed pipeline

| Stage | Result |
|---|---|
| Official source pin and inventory | `Motif-Technologies/Motif-3@ccceb1a5fd7b5eb32e47841216b3caf5666c07bc`; 2,236 tensors and 314,841,775,750 parameters |
| Full Q8_0 reference | 334,810,733,184-byte unsharded GGUF; 11 public shards; strict structure and sampled source verification passed |
| Real Q8 activation calibration | 302,080 tokens, 123,248,640 routed observations, zero missing cells across 51×384 layer/experts |
| MQ87-88-FIT | 94,162,541,472-byte unsharded GGUF; 87.6957 GiB; 11 public shards; all 2,287 tensors verified |
| Native ds4 H200 execution | Router/shared/dense/PolyNorm/mHC/GDLA/latent KV/MTP/tokenizer/tools/OpenAI server implemented and exercised |
| Context correctness | 2K, 32K, 64K, and 128K passed; the isolated 256K gate is active |
| Target DGX Spark | Deliberately pending; exact code, fixtures, model hashes, and expensive calibration state are in the private handoff |

`87-88 GiB` is a nominal capacity class, not a cosmetic hard cutoff. This
artifact was accepted at its measured 87.6957 GiB without another
precision-reducing pass. On the final native-`sm_90` H200 build, the resident
model/runtime plus a 262,144-token latent-cache session produced a combined
94.783630371094 GiB CUDA allocation delta. Actual GB10 unified-memory residency and OS headroom remain
target-host measurements.

## Pinned inputs

- Motif 3 final: `ccceb1a5fd7b5eb32e47841216b3caf5666c07bc`
- Official Motif vLLM fork: `4cd9eb4129883565e69d508038d783d59ee01867`
- Official training example: `f2cad4115bde073bc61dead4dce8db9711db0ab6`
- llama.cpp conversion base: `1d2869c6e54d5003f3927a79efbca0fefa034a6d`
- Baekpica/ds4 base: `b0309611041655f4e45671cfd9c9886aff161406`
- Baekpica/ds4 Motif implementation:
  `feature/motif-3-model-loader@bbce7eecf54703ae315328d4e240531c5a9f1a22`
- Mixed-weight revision: `efd6044e25e7f8e3b459a737d021091e2e69b6c6`
- Q8 reference revision: `5c266c95bf8c8d822d50e5e1cce9d108eaadb2af`

Motif-3-Beta was not used as a source, oracle, fixture input, calibration
input, or fallback.

## Repository map

- `manifests/`: source hashes, full tensor inventory, locked topology, source
  maps, quant policy, and verification outputs
- `converter/`: structural GGUF and source-inventory tooling
- `calibration/`: corpus construction and real Q8 activation collector; the
  public tree includes composition but not redistributed corpus text
- `fixtures/official-final/`: official-equation router, PolyNorm, mHC, GDLA,
  tokenizer, and chat fixtures
- `fixtures/long-context/`: exact deterministic 32K/64K/128K/256K inputs
  published with the final H200 bundle
- Native runtime: public
  [`Baekpica/ds4:feature/motif-3-model-loader`](https://github.com/Baekpica/ds4/tree/feature/motif-3-model-loader)
  at the pinned implementation commit above
- `reports/`: artifact, H200 validation, and DGX Spark handoff records
- `publish/`: public Q8 and mixed model cards plus verification manifests

The private `hf://buckets/Baekpica/motif-3-spark-handoff` snapshot additionally
contains the exact rendered calibration corpus, the 742,004,501-byte imatrix,
all four rank-local accumulators, checksum manifest, and Spark pull script.
The public GGUF shards are referenced by immutable revision and SHA-256 rather
than duplicated in that bucket.

## Audit and reproduce

Use Python 3.11 or newer. Lightweight manifest/fixture tests run with:

```bash
python3 -m pip install -e '.[test]'
pytest -q
```

The H200 build used `/motif3` on Runpod's faster root-backed volume for source
weights, GGUFs, imatrix state, and temporary conversion I/O. `scripts/env.sh`
contains the exact default layout. Pipeline entry points are deliberately
separate so each irreversible large artifact can be verified before the next
stage:

```bash
bash scripts/download_sources.sh metadata
bash scripts/download_sources.sh weights
bash scripts/fetch_tensor_headers.sh
bash scripts/build_inventory.sh
bash scripts/build_reference_fixtures.sh
bash scripts/build_tokenizer_fixtures.sh
```

Q8 conversion, activation collection, mixed quantization, split, and
verification entry points are retained under `converter/`, `calibration/`,
and `scripts/`. They require the pinned source snapshot, llama.cpp conversion
base, and ds4 quantizer recorded above. The private handoff supplies the exact
expensive calibration outputs so Spark work must not requantize the model.

For the native runtime, check out the pinned public ds4 implementation commit.
The private handoff additionally preserves `ds4/HEAD`, status, patch metadata,
and complete Motif-touched sources as an offline audit snapshot. Build H200
with the recorded `sm_90` configuration.
On DGX Spark, rebuild from clean objects with `make cuda-spark` and confirm the
log emits `compute_121a`/`sm_121a`; do not reuse H200 binaries.

## Scope boundary

The H200 evidence covers strict CUDA weight residency, source-page release,
production latent cache allocation/lifecycle, native generation, the
OpenAI-compatible server, tools, continuous batching, and the context gates
listed above. It does not claim completed single-DGX-Spark 262,144-token
serving. That claim requires the immutable artifact to pass the supplied
32K/64K/128K/256K and API gates on one GB10 with model, cache, workspace,
server, and required `MemAvailable` resident, without SSD streaming or CPU
weight offload.
