# Motif-3 MQ87-88 H200 completion audit

This audit covers the authorized four-H200 development and publication scope.
It does not convert H200 evidence into a claim about physical unified-memory
residency or `sm_121a` execution on a DGX Spark. Those target-host gates remain
explicit in `DGX-SPARK-HANDOFF.md`.

| Requirement | Authoritative evidence | H200-scope result |
|---|---|---|
| Official final checkpoint only | `manifests/source.json` pins `Motif-Technologies/Motif-3@ccceb1a5fd7b5eb32e47841216b3caf5666c07bc` and hashes config/modeling/tokenizer/template/generation files; `excluded_sources` names Beta | Passed |
| Complete topology and MTP, no pruning/dropping/merging | `manifests/architecture.json`, `tensor-inventory.json`, and both MQ verifiers cover 53 layers, 51 sparse layers, 384 experts, shared expert, and one MTP block | Passed |
| Q8_0 → real activation imatrix → MQ87-88 order | Public Q8 revision `5c266c95bf8c8d822d50e5e1cce9d108eaadb2af`; 302,080-token/123,248,640-route imatrix report; fixed mixed-weight revision `efd6044e25e7f8e3b459a737d021091e2e69b6c6` | Passed |
| Preceding Solar calibration reference | Private Solar handoff composition re-read; seed, requested tokens, dataset/revision/normalizer, and all eight bucket share/repo/revision/file/token-quota specs match before Motif tokenizer/template rerender | Passed |
| Locked mixed-quant policy | `manifests/quant-recipe-mq87-fit.yaml` and `verify-mq87-unsharded.json`: F32 1,273; BF16 318; Q8_0 543; IQ2_XXS 102; Q2_K 51 | Passed |
| Spark-fit artifact without size-chasing degradation | 94,162,542,816 split bytes = 87.69570181 GiB; accepted without another precision-reducing pass | Passed |
| Native motif3 loader and execution graph | `Baekpica/ds4:feature/motif-3-model-loader@d878ea1a1d67bc0f0bd60e20e75b4a011aa2d8d9`; loader/reference/tokenizer/CUDA suites; `cuobjdump` verifies every code object in final `ds4`, `ds4-server`, CUDA-fixture, and resident-fixture binaries as `sm_90` | Passed |
| Spark CUDA-architecture build readiness | Separate clean `cuda-spark` compile/link at the pinned revision; runtime and Motif test binaries contain `sm_121a` code objects only | Passed as x86-host cross-build evidence; GB10 rebuild/execution remain open |
| Router/shared/dense/PolyNorm/mHC/GDLA/YaRN/latent KV/MTP | Official fixtures, H200 CUDA fixtures, short expanded/latent comparison, cache replay, real sparse diagnostic, and MTP teacher-forced diagnostic in `H200-DEVELOPMENT.md` | Passed for H200 bring-up |
| No SSD weight streaming or CPU weight offload | Strict full-image CUDA loader and `test_motif3_resident`; Motif admission rejects incomplete residency | Passed |
| No duplicate steady raw GGUF weight image | Source mapping RSS 91,955,608 KiB before release, 9,416 KiB after copy, and 29,512 KiB after the latest full native-`sm_90` resident regression | Passed on final runtime overlay |
| 262,144-token production cache allocation | 4,236,751,872-byte latent/RoPE/SWA payload; 4,334,813,184-byte CUDA delta; no persistent expanded historical K/V | Passed |
| OpenAI-compatible server and continuous batching | Exact 2K and 32K OpenAI retrieval, streaming/non-streaming unit coverage, structured tool continuation, full live-prefix reuse, and two resident sessions | Passed |
| Target 256K OpenAI gate reproducibility | Hash-pinned 262,080-token fixture preserves full 25-token question plus all records; strict stdlib API client validates model ID, token accounting, ordered JSON, stop reason, and decode | Prepared and locally integrity-tested; GB10 execution remains open |
| Short/32K/64K/128K/256K correctness | `H200-VALIDATION.json` and exact deterministic long fixtures | Short/32K/64K/128K passed. The two 256K attempts stopped before decode at 245,760 and 106,496 completed prefill tokens; 256K is explicitly not passed and is transferred to Spark per user direction. |
| Kernel optimization status | H200 timing rows and model-card caution; Nsight tools absent on this host | Correctness-first graph only. No optimized release-speed claim; profiling and prefill/decode optimization transferred to Spark per user direction. |
| Public Q8 and mixed artifacts | `Baekpica/Motif-3-GGUF` and `Baekpica/Motif-3-Mixed-Quant-GGUF`; fixed shard revisions and SHA-256 manifests | Passed |
| Public reproduction and runtime | `Baekpica/motif-3-mixed-ds4` and the pinned public ds4 branch above | Passed |
| Offline ds4 patch reproducibility | Binary patch from pinned main base `b030961…` to Motif head `d878ea1…` passes `git apply --check` in a clean detached base checkout | Passed |
| Artifact placement separation | The two GGUF repositories are public HF models, reproduction is the public GitHub repository, expensive state is the private bucket, and `Baekpica/motif-3-mixed-ds4` is absent from HF model repositories | Passed |
| Private Spark handoff | `hf://buckets/Baekpica/motif-3-spark-handoff`, checksum manifest, exact calibration/imatrix partials, fixtures, sources, commands, and immutable model references | PENDING_FINAL_SYNC |
| MQ95/MQ97 excluded | No MQ95/MQ97 artifact or precision promotion is published | Passed |

NVIDIA Nsight Systems and Compute are absent from the Runpod image. No
invented profiler result is claimed. Per the handoff scope, focused kernel
profiling remains `nsys → identify → modify → ncu` on the Spark phase; the
current timing evidence identifies GDLA as the first target without changing
precision, topology, or context.

The remaining release claim after this H200 scope is narrow but mandatory: a
single GB10 must rebuild the pinned runtime for `sm_121a`, complete native
262,144-token prefill followed by decode through the OpenAI-compatible server,
and show model + latent KV + workspace + server resident with sufficient OS
`MemAvailable`, without SSD weight streaming or CPU weight offload.
