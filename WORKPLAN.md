# Work plan

1. **Complete** — Pin the official final checkpoint and all implementation
   oracles; explicitly exclude Motif-3-Beta.
2. **Complete** — Fetch safetensors headers and validate the complete tensor
   topology.
3. **Complete** — Build official-reference fixtures for PolyNorm, mHC,
   routing, GDLA, MTP, tokenizer, template, reasoning, and tools.
4. **Complete** — Add native `motif3` GGUF conversion and the locked
   MQ87-88 policy.
5. **Complete** — Generate, strictly verify, split, and publicly upload the
   full-topology Q8_0 reference.
6. **Complete** — Run real Q8 activation collection on four H200s with zero
   uncovered 51×384 sparse layer/expert cells.
7. **Complete** — Generate, strictly verify, split, and publicly upload the
   immutable 87.6957 GiB MQ87-88-FIT artifact.
8. **Complete** — Add native `motif3` loading and execution to the local
   `Baekpica/ds4:feature/motif-3-model-loader` state: dense/shared/routed MoE,
   PolyNorm, mHC, GDLA, tokenizer/tools, MTP, and OpenAI server.
9. **Complete** — Match the expanded official GDLA path at short context and
   implement production latent KV plus RoPE key and bounded SWA-ring state.
10. **Complete** — Pass H200 structural, numerical, resident-memory, API,
    tool, continuous-batching, short, 32K, 64K, and 128K gates.
11. **Running** — Complete the isolated immutable 256K H200 prefill/decode
    gate, then finalize the public Q8/Mixed reports and checksum-complete
    private Spark handoff bucket.
12. **Target-host handoff** — On one GB10, rebuild for `sm_121a` and validate
    resident 32K/64K/128K/256K prefill/decode, OpenAI-compatible serving, and
    physical unified-memory/OS headroom.

NVIDIA Nsight Systems and Compute are not installed on the Runpod image, so
no synthetic profiling result is claimed. The private handoff preserves the
required `nsys → identify → modify → ncu` sequence for the Spark phase.

MQ95 and MQ97 artifacts are intentionally outside this release.
