# Native Motif-3 feature audit

Runtime: `Baekpica/ds4:feature/motif-3-model-loader` at
`d878ea1a1d67bc0f0bd60e20e75b4a011aa2d8d9`.

This matrix ties each required Motif-specific behavior to committed source and
an executed gate. It is an H200 implementation audit, not a GB10 result.

| Requirement | Committed implementation | Executed evidence |
|---|---|---|
| Explicit family/config admission | `config_validate_motif3_model` requires the pinned final config SHA, 53 layers, 262,144 context, 384E/top-8, official SWA pattern, YaRN, PolyNorm, mHC, and one MTP block | loader topology gate; malformed/missing tensors fail admission |
| Complete tensor binder | `weights_bind_motif3_layer`, `weights_bind_motif3_mtp`, and `weights_validate_motif3_layout` bind dense, all sparse/shared, attention/control, output, and MTP tensors | 2,287-tensor mixed GGUF accepted; 53 layers, 14 full + 39 SWA, MTP reported present |
| 384E sigmoid top-8 routing | `ds4_gpu_motif3_router_select_batch_model_tensor` implements correction-bias selection plus normalized/scaled weights in F32 | official CPU/CUDA fixture exact IDs; max selected-weight error 2.98e-8; real layer-2 IDs exact |
| Existing ds4 routed-MoE reuse | `ds4_gpu_motif3_routed_moe_batch_tensor` dispatches the existing IQ2_XXS gate/up and Q2_K down MMQ paths, then performs Motif weighting | real mixed layer: Q2 down cosine 0.9996071 and combined output cosine 0.9998363 |
| Shared and first dense experts | Motif native FFN graph binds Q8 dense/shared projections separately from routed experts | full native forward, expanded/latent parity, resident regression, and long gates |
| Expert-Specific PolyNorm | per-assignment moments and independent expert coefficient/bias lookup between gate/up and down MMQ; F32 moments/apply | official BF16 result bit-exact; real mixed layer NRMSE 2.63e-11 |
| Modified mHC | F32 control accumulation, sigmoid/clamp/Sinkhorn, pre/residual application and combine kernels | official reference and H200 CUDA fixtures bit-exact |
| Expanded GDLA oracle | Q/K/V expansion, RoPE, full/window attention, differential signal/noise, lambda, output gate | official expanded fixture; 21-token real-weight oracle finite |
| Production latent GDLA | normalized latent KV, rotated `k_pe`, Q/K absorption, latent attention/value accumulation, bounded SWA rings, identity/frontier checks | no persistent expanded history; direct/chunk top-1/top-8 match; identical suffix replay cosine 1.0 |
| Interleaved attention and YaRN | layer schedule is admitted only as 14 full plus 39 128-token SWA layers; separate full/SWA RoPE bases and pinned YaRN metadata | topology gate; official RoPE fixture bit-exact; 32K/64K/128K and active 256K long execution |
| MTP structure/path | complete MTP block bound; separate frontier; real-weight teacher-forced diagnostic executes its projections | 19 rows, finite logits, draft rank 11; speculative acceptance optimization intentionally deferred |
| Official tokenizer/chat/reasoning/tools | native Motif byte-BPE pretokenizer and added tokens; exact final template, think/no-think asymmetry, tool-call/result parsing | 16 raw + 5 rendered fixtures; live structured tool round trip and cached continuation |
| OpenAI-compatible serving | native model aliases and advertised `motif-3` ID; chat streaming/non-streaming and resident sessions | live `/v1/models`; exact 2K and all-sm90 32K chat; two-session continuous scheduling |
| Strict resident contract | Motif requires a complete CUDA-owned image and rejects streaming/offloaded admission; source GGUF pages discarded after preparation | two all-sm90 repeats; conservative 91.262 GiB model/runtime, 4.037 GiB 256K session, 29,512 KiB source mapping RSS |

The full-history latent attention kernel is an unfused correctness-first path,
which explains the long-context throughput decline. The required optimization
order remains `nsys → identify → modify → ncu` on Spark; this does not justify
changing topology, precision, or context. MTP weights and diagnostics are
complete for handoff, but MTP is not used to inflate the reported decode rate.
