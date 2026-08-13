#!/usr/bin/env python3
"""Collect a routed-expert imatrix from the full-topology Motif-3 Q8_0 GGUF.

This is a calibration runtime, not a serving runtime.  It deliberately uses a
layer-major schedule so one Q8_0 layer is dequantized on each H200, applied to
all calibration chunks assigned to that rank, and then released.  The forward
path follows the pinned official Motif-3 implementation: mHC, expanded GDLA,
interleaved SWA/full attention, sigmoid top-8 routing, shared experts, and
expert-specific PolyNorm.  No source BF16 routed matrix participates in the
calibration forward pass.

Launch with torchrun, for example:

  torchrun --standalone --nproc-per-node=4 calibration/run_q8_imatrix.py \
      --gguf /motif3/artifacts/Motif-3-Q8_0.gguf \
      --dataset calibration/calibration.txt \
      --tokenizer /motif3/hf-cache/models--Motif-Technologies--Motif-3/snapshots/REV \
      --work-dir /motif3/calibration/motif3-q8-imatrix \
      --output /motif3/calibration/Motif-3-Q8_0-imatrix.dat
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


PINNED_REVISION = "ccceb1a5fd7b5eb32e47841216b3caf5666c07bc"
N_LAYER = 53
N_DENSE = 2
N_EXPERT = 384
TOP_K = 8
HIDDEN = 4096
MHC_EXPANSION = 4
Q_LORA = 1024
KV_LORA = 512
N_HEAD = 80
N_KV_HEAD = 16
N_NOISE_HEAD = 16
N_SIGNAL_HEAD = 64
HEAD_DIM = 192
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
VALUE_DIM = 128
EXPERT_FF = 1280
DENSE_FF = 12288
SWA_LEFT = 128
POLYNORM_SCALE = 0.5
RMS_EPS = 1.0e-5
MHC_EPS = 1.0e-6
POLYNORM_EPS = 1.0e-6
HIDDEN_CLAMP = 1_000_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--max-chunks", type=int, default=590)
    parser.add_argument("--max-layers", type=int, default=N_LAYER,
                        help="pilot/debug only; release collection uses 53")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--workspace-mib", type=int, default=512)
    parser.add_argument("--source-map", type=Path,
                        default=Path("manifests/gguf-source-map-q8.json"))
    return parser.parse_args()


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def distributed_init() -> tuple[int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    return rank, world, device


def barrier(world: int) -> None:
    if world > 1:
        dist.barrier()


def rank_log(rank: int, message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp} rank {rank}] {message}", flush=True)


def prepare_token_cache(args: argparse.Namespace, rank: int, world: int) -> Path:
    cache = args.work_dir / f"tokens-{args.chunk_size}x{args.max_chunks}.npz"
    meta_path = cache.with_suffix(".json")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        dataset_sha = sha256_file(args.dataset)
        wanted = {
            "dataset": str(args.dataset.resolve()),
            "dataset_sha256": dataset_sha,
            "chunk_size": args.chunk_size,
            "max_chunks": args.max_chunks,
            "tokenizer": str(args.tokenizer.resolve()),
            "source_revision": PINNED_REVISION,
        }
        reuse = False
        if cache.is_file() and meta_path.is_file():
            try:
                reuse = json.loads(meta_path.read_text(encoding="utf-8")) == wanted
            except (OSError, ValueError):
                reuse = False
        if not reuse:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer, trust_remote_code=True, local_files_only=True
            )
            text = args.dataset.read_text(encoding="utf-8")
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            need = args.chunk_size * args.max_chunks
            if len(token_ids) < need:
                raise RuntimeError(
                    f"calibration corpus has {len(token_ids)} tokens; need {need}"
                )
            chunks = np.asarray(token_ids[:need], dtype=np.int32).reshape(
                args.max_chunks, args.chunk_size
            )
            np.savez(cache, tokens=chunks)
            meta_path.write_text(
                json.dumps(wanted, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            rank_log(rank, f"tokenized {need} official tokens -> {cache}")
        else:
            rank_log(rank, f"reusing token cache {cache}")
    barrier(world)
    if not cache.is_file():
        raise RuntimeError(f"token cache was not created: {cache}")
    return cache


class Q8GGUF:
    """Memory-mapped GGUF tensor reader with GPU Q8_0 dequantization."""

    def __init__(self, path: Path, device: torch.device):
        gguf_py = os.environ.get("GGUF_PYTHON_PATH", "/motif3/llama.cpp/gguf-py")
        if gguf_py not in sys.path:
            sys.path.insert(0, gguf_py)
        from gguf import GGMLQuantizationType, GGUFReader

        self._qtype = GGMLQuantizationType
        self.reader = GGUFReader(path, "r")
        self.tensors = {tensor.name: tensor for tensor in self.reader.tensors}
        self.device = device

    def _type_name(self, tensor) -> str:
        return self._qtype(tensor.tensor_type).name

    def load(self, name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        tensor = self.tensors.get(name)
        if tensor is None:
            raise KeyError(f"missing GGUF tensor {name}")
        kind = self._type_name(tensor)
        target = dtype or torch.bfloat16
        array = np.asarray(tensor.data)
        # The mapped array is intentionally read-only; .to(device) performs
        # the owned copy before any in-place operation.
        with np.errstate(all="ignore"):
            host = torch.from_numpy(array)
        raw = host.to(self.device)
        del host

        if kind == "Q8_0":
            blocks = raw.reshape(*raw.shape[:-1], -1, 34)
            scale_bytes = blocks[..., :2].contiguous()
            scales = scale_bytes.view(torch.float16).to(target)
            q = blocks[..., 2:].view(torch.int8).to(target)
            q.mul_(scales)
            out = q.reshape(*raw.shape[:-1], int(tensor.shape[0]))
            del blocks, scale_bytes, scales, raw
            return out
        if kind == "BF16":
            out = raw.contiguous().view(torch.bfloat16)
            return out.to(target) if target != torch.bfloat16 else out
        if kind == "F16":
            out = raw.contiguous().view(torch.float16)
            return out.to(target) if target != torch.float16 else out
        if kind == "F32":
            out = raw.to(torch.float32)
            return out.to(target) if target != torch.float32 else out
        raise RuntimeError(f"unsupported calibration tensor type {kind}: {name}")


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.float().square().mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


def polynorm_mul(
    gate: torch.Tensor,
    up: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    output_scale: float = POLYNORM_SCALE,
) -> torch.Tensor:
    gate = gate.clamp(-HIDDEN_CLAMP, HIDDEN_CLAMP)
    up = up.clamp(-HIDDEN_CLAMP, HIDDEN_CLAMP)
    g = gate.float()
    g2 = g * g
    g3 = g2 * g
    inv2 = torch.rsqrt(g2.mean(-1, keepdim=True) + POLYNORM_EPS)
    inv4 = torch.rsqrt((g2 * g2).mean(-1, keepdim=True) + POLYNORM_EPS)
    inv6 = torch.rsqrt((g3 * g3).mean(-1, keepdim=True) + POLYNORM_EPS)
    w = torch.sigmoid(weight.float())
    poly = (
        w[..., 0] * g3 * inv6
        + w[..., 1] * g2 * inv4
        + w[..., 2] * g * inv2
        + bias.float()
    )
    return (poly * up.float()).to(gate.dtype) * output_scale


@dataclass
class MHCControls:
    h_post: torch.Tensor
    h_res: torch.Tensor
    layer_input: torch.Tensor


def mhc_pre(
    x: torch.Tensor,
    rms_weight: torch.Tensor,
    proj_pre: torch.Tensor,
    proj_post: torch.Tensor,
    proj_res: torch.Tensor,
    alpha_pre: torch.Tensor,
    alpha_post: torch.Tensor,
    alpha_res: torch.Tensor,
    bias_pre: torch.Tensor,
    bias_post: torch.Tensor,
    bias_res: torch.Tensor,
) -> MHCControls:
    n_tokens = x.shape[0]
    flat = x.reshape(n_tokens, MHC_EXPANSION * HIDDEN)
    norm = rms_norm(flat, rms_weight, MHC_EPS)
    pre = F.linear(norm, proj_pre).float()
    post = F.linear(norm, proj_post).float()
    res = F.linear(norm, proj_res).float().reshape(
        n_tokens, MHC_EXPANSION, MHC_EXPANSION
    )
    del flat, norm

    h_pre = torch.sigmoid(
        (alpha_pre.float() * pre + bias_pre.float()).clamp(-10.0, 10.0)
    )
    h_post = torch.sigmoid(
        (alpha_post.float() * post + bias_post.float()).clamp(-10.0, 10.0)
    )
    h_res = (alpha_res.float() * res + bias_res.float()).clamp(-20.0, 20.0).exp()
    for _ in range(20):
        h_res.div_(h_res.sum(-1, keepdim=True).clamp_min_(1.0e-8))
        h_res.div_(h_res.sum(-2, keepdim=True).clamp_min_(1.0e-8))
    layer_input = (x.float() * h_pre.unsqueeze(-1)).sum(1).to(x.dtype)
    return MHCControls(h_post=h_post, h_res=h_res, layer_input=layer_input)


def mhc_post(x: torch.Tensor, branch: torch.Tensor, controls: MHCControls) -> torch.Tensor:
    residual = torch.einsum("nij,njd->nid", controls.h_res, x.float())
    residual.add_(controls.h_post.unsqueeze(-1) * branch.float().unsqueeze(1))
    return residual.to(x.dtype)


def load_mhc(model: Q8GGUF, layer: int, which: str) -> dict[str, torch.Tensor]:
    prefix = f"blk.{layer}.mhc_{which}"
    return {
        "rms_weight": model.load(f"{prefix}.rms_norm.weight", torch.float32),
        "proj_pre": model.load(f"{prefix}.proj_pre.weight", torch.bfloat16),
        "proj_post": model.load(f"{prefix}.proj_post.weight", torch.bfloat16),
        "proj_res": model.load(f"{prefix}.proj_res.weight", torch.bfloat16),
        "alpha_pre": model.load(f"{prefix}.alpha_pre", torch.float32),
        "alpha_post": model.load(f"{prefix}.alpha_post", torch.float32),
        "alpha_res": model.load(f"{prefix}.alpha_res", torch.float32),
        "bias_pre": model.load(f"{prefix}.bias_pre", torch.float32),
        "bias_post": model.load(f"{prefix}.bias_post", torch.float32),
        "bias_res": model.load(f"{prefix}.bias_res", torch.float32),
    }


def neox_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1.float() * cos - x2.float() * sin,
                      x2.float() * cos + x1.float() * sin), dim=-1).to(x.dtype)


def yarn_inv_freq(device: torch.device) -> torch.Tensor:
    dim = ROPE_DIM
    base = 10000.0
    factor = 64.0
    original = 4096
    beta_fast = 32
    beta_slow = 1

    def correction_dim(rotations: int) -> float:
        return (dim * math.log(original / (rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(dim // 2, device=device, dtype=torch.float32) - low)
            / (high - low)).clamp_(0.0, 1.0)
    extrapolation_mask = 1.0 - ramp
    pos_freqs = base ** (
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
    )
    inv_extrapolation = 1.0 / pos_freqs
    inv_interpolation = 1.0 / (factor * pos_freqs)
    return inv_interpolation * (1.0 - extrapolation_mask) + inv_extrapolation * extrapolation_mask


def rope_tables(
    positions: torch.Tensor, full_attention: bool, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if full_attention:
        inv = yarn_inv_freq(device)
    else:
        inv = 1.0 / (
            10000.0 ** (
                torch.arange(0, ROPE_DIM, 2, device=device, dtype=torch.float32)
                / ROPE_DIM
            )
        )
    freq = positions.float().unsqueeze(-1) * inv.unsqueeze(0)
    return freq.cos().unsqueeze(1), freq.sin().unsqueeze(1)


class RaggedAttention:
    def __init__(
        self,
        device: torch.device,
        indptr: torch.Tensor,
        workspace_mib: int,
    ) -> None:
        import flashinfer

        self.indptr = indptr
        self.workspace = torch.empty(
            workspace_mib << 20, dtype=torch.uint8, device=device
        )
        self.wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace, kv_layout="NHD", backend="auto"
        )

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        full_attention: bool,
    ) -> torch.Tensor:
        mscale = 0.1 * math.log(64.0) + 1.0
        scale = HEAD_DIM ** -0.5
        if full_attention:
            scale *= mscale * mscale
        self.wrapper.plan(
            self.indptr,
            self.indptr,
            num_qo_heads=N_HEAD,
            num_kv_heads=N_KV_HEAD,
            head_dim_qk=HEAD_DIM,
            head_dim_vo=VALUE_DIM,
            causal=True,
            window_left=-1 if full_attention else SWA_LEFT,
            sm_scale=scale,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            o_data_type=torch.bfloat16,
            non_blocking=False,
        )
        return self.wrapper.run(q, k, v)


def warm_flashinfer(
    rank: int,
    world: int,
    device: torch.device,
    workspace_mib: int,
) -> None:
    """Compile the two attention plans once before all ranks enter layer 0."""
    if rank == 0:
        indptr = torch.tensor([0, 8], dtype=torch.int32, device=device)
        runner = RaggedAttention(device, indptr, min(workspace_mib, 128))
        q = torch.zeros(8, N_HEAD, HEAD_DIM, dtype=torch.bfloat16, device=device)
        k = torch.zeros(8, N_KV_HEAD, HEAD_DIM, dtype=torch.bfloat16, device=device)
        v = torch.zeros(8, N_KV_HEAD, VALUE_DIM, dtype=torch.bfloat16, device=device)
        runner.run(q, k, v, True)
        runner.run(q, k, v, False)
        torch.cuda.synchronize(device)
        del runner, q, k, v, indptr
        torch.cuda.empty_cache()
        rank_log(rank, "FlashInfer expanded-GDLA plans are warm")
    barrier(world)


def attention_forward(
    model: Q8GGUF,
    layer: int,
    x: torch.Tensor,
    positions: torch.Tensor,
    attention: RaggedAttention,
) -> torch.Tensor:
    prefix = f"blk.{layer}"
    norm_weight = model.load(f"{prefix}.attn_norm.weight", torch.float32)
    q_a = model.load(f"{prefix}.attn_q_a.weight", torch.bfloat16)
    q_a_norm = model.load(f"{prefix}.attn_q_a_norm.weight", torch.float32)
    q_b = model.load(f"{prefix}.attn_q_b.weight", torch.bfloat16)
    q_gate = model.load(f"{prefix}.attn_q_gate.weight", torch.bfloat16)
    kv_a = model.load(f"{prefix}.attn_kv_a.weight", torch.bfloat16)
    kv_a_norm = model.load(f"{prefix}.attn_kv_a_norm.weight", torch.float32)
    kv_b = model.load(f"{prefix}.attn_kv_b.weight", torch.bfloat16)
    lambda_weight = model.load(f"{prefix}.attn_lambda.weight", torch.bfloat16)
    output_weight = model.load(f"{prefix}.attn_output.weight", torch.bfloat16)

    attn_input = rms_norm(x, norm_weight, RMS_EPS)
    q_latent = rms_norm(F.linear(attn_input, q_a), q_a_norm, RMS_EPS)
    q = F.linear(q_latent, q_b).reshape(-1, N_HEAD, HEAD_DIM)
    gate = F.linear(q_latent, q_gate).reshape(-1, N_SIGNAL_HEAD, VALUE_DIM)

    kv_raw = F.linear(attn_input, kv_a)
    kv_latent = rms_norm(kv_raw[:, :KV_LORA], kv_a_norm, RMS_EPS)
    k_pe = kv_raw[:, KV_LORA:].reshape(-1, 1, ROPE_DIM)
    kv = F.linear(kv_latent, kv_b).reshape(
        -1, N_KV_HEAD, NOPE_DIM + VALUE_DIM
    )
    k_nope, value = kv.split((NOPE_DIM, VALUE_DIM), dim=-1)

    q_nope, q_pe = q.split((NOPE_DIM, ROPE_DIM), dim=-1)
    full_attention = layer % 4 == 0
    cos, sin = rope_tables(positions, full_attention, x.device)
    q_pe = neox_rope(q_pe, cos, sin)
    k_pe = neox_rope(k_pe, cos, sin)
    query = torch.cat((q_nope, q_pe), dim=-1).contiguous()
    key = torch.cat((k_nope, k_pe.expand(-1, N_KV_HEAD, -1)), dim=-1).contiguous()
    value = value.contiguous()

    attn_out = attention.run(query, key, value, full_attention)
    groups = attn_out.reshape(-1, N_NOISE_HEAD, 5, VALUE_DIM)
    signal = groups[:, :, :4, :].reshape(-1, N_SIGNAL_HEAD, VALUE_DIM)
    noise = groups[:, :, 4:, :].reshape(-1, N_NOISE_HEAD, VALUE_DIM)
    noise = noise.repeat_interleave(4, dim=1)
    # Lambda uses the pre-attention normalized hidden input and FP32 math.
    lambda_values = F.linear(attn_input.float(), lambda_weight.float())
    differential = signal - torch.sigmoid(lambda_values).to(signal.dtype).unsqueeze(-1) * noise
    differential.mul_(torch.sigmoid(gate))
    out = F.linear(differential.reshape(-1, N_SIGNAL_HEAD * VALUE_DIM), output_weight)

    del (
        norm_weight, q_a, q_a_norm, q_b, q_gate, kv_a, kv_a_norm, kv_b,
        lambda_weight, output_weight, attn_input, q_latent, q, gate, kv_raw,
        kv_latent, k_pe, kv, k_nope, q_nope, q_pe, cos, sin, query, key,
        value, attn_out, groups, signal, noise, lambda_values, differential,
    )
    return out


def dense_ffn(model: Q8GGUF, layer: int, x: torch.Tensor) -> torch.Tensor:
    prefix = f"blk.{layer}"
    gate_w = model.load(f"{prefix}.ffn_gate.weight", torch.bfloat16)
    up_w = model.load(f"{prefix}.ffn_up.weight", torch.bfloat16)
    down_w = model.load(f"{prefix}.ffn_down.weight", torch.bfloat16)
    poly_w = model.load(f"{prefix}.ffn_polynorm.weight", torch.float32)
    poly_b = model.load(f"{prefix}.ffn_polynorm.bias", torch.float32)
    gate = F.linear(x, gate_w)
    up = F.linear(x, up_w)
    mid = polynorm_mul(gate, up, poly_w, poly_b)
    out = F.linear(mid, down_w)
    del gate_w, up_w, down_w, poly_w, poly_b, gate, up, mid
    return out


def shared_ffn(model: Q8GGUF, layer: int, x: torch.Tensor) -> torch.Tensor:
    prefix = f"blk.{layer}"
    gate_w = model.load(f"{prefix}.ffn_gate_shexp.weight", torch.bfloat16)
    up_w = model.load(f"{prefix}.ffn_up_shexp.weight", torch.bfloat16)
    down_w = model.load(f"{prefix}.ffn_down_shexp.weight", torch.bfloat16)
    poly_w = model.load(f"{prefix}.ffn_polynorm_shexp.weight", torch.float32)
    poly_b = model.load(f"{prefix}.ffn_polynorm_shexp.bias", torch.float32)
    gate = F.linear(x, gate_w)
    up = F.linear(x, up_w)
    mid = polynorm_mul(gate, up, poly_w, poly_b)
    out = F.linear(mid, down_w)
    del gate_w, up_w, down_w, poly_w, poly_b, gate, up, mid
    return out


@dataclass
class SparseStats:
    gate_sum2: np.ndarray
    down_sum2: np.ndarray
    counts: np.ndarray


def sparse_ffn(
    model: Q8GGUF,
    layer: int,
    x: torch.Tensor,
) -> tuple[torch.Tensor, SparseStats]:
    prefix = f"blk.{layer}"
    router_w = model.load(f"{prefix}.ffn_gate_inp.weight", torch.float32)
    correction = model.load(f"{prefix}.exp_probs_b.bias", torch.float32)
    router_logits = F.linear(x.float(), router_w)
    scores = torch.sigmoid(router_logits)
    selected = torch.topk(scores + correction.unsqueeze(0), TOP_K, dim=-1,
                          sorted=True).indices
    route_weights = scores.gather(1, selected)
    route_weights.div_(route_weights.sum(-1, keepdim=True)).mul_(2.0)
    del router_w, correction, router_logits, scores

    flat_experts = selected.reshape(-1)
    flat_tokens = torch.arange(x.shape[0], device=x.device).repeat_interleave(TOP_K)
    flat_weights = route_weights.reshape(-1)
    order = torch.argsort(flat_experts, stable=True)
    flat_experts = flat_experts[order]
    flat_tokens = flat_tokens[order]
    flat_weights = flat_weights[order]
    counts = torch.bincount(flat_experts, minlength=N_EXPERT)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    del selected, route_weights, order

    gate_w = model.load(f"{prefix}.ffn_gate_exps.weight", torch.bfloat16)
    up_w = model.load(f"{prefix}.ffn_up_exps.weight", torch.bfloat16)
    down_w = model.load(f"{prefix}.ffn_down_exps.weight", torch.bfloat16)
    poly_w = model.load(f"{prefix}.ffn_polynorm_exps.weight", torch.float32)
    poly_b = model.load(f"{prefix}.ffn_polynorm_exps.bias", torch.float32)
    poly_b = poly_b.clamp(-0.5, 0.5)

    gate_sum2 = torch.zeros(N_EXPERT, HIDDEN, dtype=torch.float32, device=x.device)
    down_sum2 = torch.zeros(N_EXPERT, EXPERT_FF, dtype=torch.float32, device=x.device)
    routed = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float32, device=x.device)

    offsets_cpu = offsets.cpu().tolist()
    for expert in range(N_EXPERT):
        start, end = offsets_cpu[expert], offsets_cpu[expert + 1]
        if start == end:
            continue
        token_index = flat_tokens[start:end]
        expert_x = x.index_select(0, token_index)
        gate_sum2[expert] = expert_x.float().square().sum(0)
        gate = F.linear(expert_x, gate_w[expert])
        up = F.linear(expert_x, up_w[expert])
        mid = polynorm_mul(gate, up, poly_w[expert], poly_b[expert])
        # ds4 applies the route weight before the down GEMM; the imatrix must
        # therefore observe this exact down-projection input.
        mid.mul_(flat_weights[start:end].to(mid.dtype).unsqueeze(-1))
        down_sum2[expert] = mid.float().square().sum(0)
        expert_out = F.linear(mid, down_w[expert]).float()
        routed.index_add_(0, token_index, expert_out)
        del token_index, expert_x, gate, up, mid, expert_out

    shared = shared_ffn(model, layer, x)
    out = (routed + shared.float()).to(torch.bfloat16)
    stats = SparseStats(
        gate_sum2=gate_sum2.cpu().numpy(),
        down_sum2=down_sum2.cpu().numpy(),
        counts=counts.to(torch.int64).cpu().numpy(),
    )
    del (
        flat_experts, flat_tokens, flat_weights, counts, offsets, gate_w, up_w,
        down_w, poly_w, poly_b, gate_sum2, down_sum2, routed, shared,
    )
    return out, stats


def run_rank(
    args: argparse.Namespace,
    rank: int,
    world: int,
    device: torch.device,
    token_cache: Path,
) -> Path:
    source_map = json.loads(args.source_map.read_text(encoding="utf-8"))
    if source_map.get("variant") != "q8-reference":
        raise RuntimeError("imatrix collection requires the q8-reference source map")
    if source_map.get("source_revision") != PINNED_REVISION:
        raise RuntimeError("source map is not pinned to the official Motif-3 revision")

    token_matrix = np.load(token_cache, allow_pickle=False)["tokens"]
    local_tokens = np.ascontiguousarray(token_matrix[rank::world])
    if local_tokens.size == 0:
        raise RuntimeError(f"rank {rank} received no calibration chunks")
    n_sequence, chunk_size = local_tokens.shape
    n_tokens = int(local_tokens.size)
    tokens = torch.from_numpy(local_tokens.reshape(-1)).to(device=device, dtype=torch.long)
    positions = torch.arange(chunk_size, device=device, dtype=torch.int64).repeat(n_sequence)
    indptr = torch.arange(
        0, n_tokens + 1, chunk_size, device=device, dtype=torch.int32
    )
    attention = RaggedAttention(device, indptr, args.workspace_mib)
    model = Q8GGUF(args.gguf, device)
    if len(model.tensors) != 2287:
        raise RuntimeError(f"Q8 GGUF tensor count is {len(model.tensors)}, expected 2287")

    rank_log(rank, f"loading Q8_0 embedding for {n_sequence}x{chunk_size} tokens")
    embedding = model.load("token_embd.weight", torch.bfloat16)
    hidden = F.embedding(tokens, embedding)
    hidden = hidden.unsqueeze(1).expand(-1, MHC_EXPANSION, -1).contiguous()
    del embedding, tokens
    torch.cuda.empty_cache()

    sparse_layers = max(0, min(args.max_layers, N_LAYER) - N_DENSE)
    gate_sum2 = np.zeros((sparse_layers, N_EXPERT, HIDDEN), dtype=np.float32)
    down_sum2 = np.zeros((sparse_layers, N_EXPERT, EXPERT_FF), dtype=np.float32)
    counts = np.zeros((sparse_layers, N_EXPERT), dtype=np.int64)
    layer_seconds: list[float] = []

    for layer in range(min(args.max_layers, N_LAYER)):
        layer_start = time.perf_counter()
        mhc_attn = load_mhc(model, layer, "attn")
        attn_controls = mhc_pre(hidden, **mhc_attn)
        attn_out = attention_forward(
            model, layer, attn_controls.layer_input, positions, attention
        )
        after_attn = mhc_post(hidden, attn_out, attn_controls)
        del hidden, attn_out, attn_controls, mhc_attn

        mhc_ffn = load_mhc(model, layer, "ffn")
        ffn_controls = mhc_pre(after_attn, **mhc_ffn)
        ffn_norm_weight = model.load(f"blk.{layer}.ffn_norm.weight", torch.float32)
        ffn_input = rms_norm(ffn_controls.layer_input, ffn_norm_weight, RMS_EPS)
        del ffn_norm_weight

        if layer < N_DENSE:
            ffn_out = dense_ffn(model, layer, ffn_input)
        else:
            ffn_out, stats = sparse_ffn(model, layer, ffn_input)
            slot = layer - N_DENSE
            gate_sum2[slot] = stats.gate_sum2
            down_sum2[slot] = stats.down_sum2
            counts[slot] = stats.counts
            del stats

        hidden = mhc_post(after_attn, ffn_out, ffn_controls)
        del after_attn, ffn_out, ffn_input, ffn_controls, mhc_ffn
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - layer_start
        layer_seconds.append(elapsed)
        allocated = torch.cuda.memory_allocated(device) / 2**30
        reserved = torch.cuda.memory_reserved(device) / 2**30
        rank_log(
            rank,
            f"layer {layer:02d}/{min(args.max_layers, N_LAYER)-1:02d} "
            f"{elapsed:.2f}s cuda={allocated:.1f}/{reserved:.1f} GiB",
        )

    digest = hashlib.sha256(
        hidden[: min(128, hidden.shape[0])].float().cpu().numpy().tobytes()
    ).hexdigest()
    part = args.work_dir / f"imatrix-rank-{rank:02d}-of-{world:02d}.npz"
    np.savez(
        part,
        gate_sum2=gate_sum2,
        down_sum2=down_sum2,
        counts=counts,
        chunks=np.asarray([n_sequence], dtype=np.int64),
        tokens=np.asarray([n_tokens], dtype=np.int64),
        final_hidden_sha256=np.asarray([digest]),
        layer_seconds=np.asarray(layer_seconds, dtype=np.float64),
    )
    rank_log(rank, f"wrote partial sums {part}")
    return part


def write_i32(handle, value: int) -> None:
    handle.write(struct.pack("<i", int(value)))


def write_imatrix_entry(
    handle,
    name: str,
    sum2: np.ndarray,
    counts: np.ndarray,
) -> None:
    encoded = name.encode("utf-8")
    write_i32(handle, len(encoded))
    handle.write(encoded)
    write_i32(handle, 1)
    write_i32(handle, sum2.size)
    averages = np.ones_like(sum2, dtype=np.float32)
    np.divide(
        sum2,
        counts[:, None],
        out=averages,
        where=counts[:, None] != 0,
    )
    averages.astype("<f4", copy=False).tofile(handle)


def merge_partials(
    args: argparse.Namespace,
    world: int,
    rank: int,
) -> dict:
    parts = [args.work_dir / f"imatrix-rank-{i:02d}-of-{world:02d}.npz"
             for i in range(world)]
    missing = [str(path) for path in parts if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing imatrix partial(s): {missing}")

    gate_sum2 = None
    down_sum2 = None
    counts = None
    chunks = 0
    tokens = 0
    hidden_digests: list[str] = []
    timings: list[list[float]] = []
    for path in parts:
        data = np.load(path, allow_pickle=False)
        if gate_sum2 is None:
            gate_sum2 = data["gate_sum2"].copy()
            down_sum2 = data["down_sum2"].copy()
            counts = data["counts"].copy()
        else:
            gate_sum2 += data["gate_sum2"]
            down_sum2 += data["down_sum2"]
            counts += data["counts"]
        chunks += int(data["chunks"][0])
        tokens += int(data["tokens"][0])
        hidden_digests.append(str(data["final_hidden_sha256"][0]))
        timings.append(data["layer_seconds"].tolist())

    assert gate_sum2 is not None and down_sum2 is not None and counts is not None
    processed_sparse = gate_sum2.shape[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        write_i32(handle, processed_sparse * 3)
        for slot in range(processed_sparse):
            layer = slot + N_DENSE
            write_imatrix_entry(
                handle, f"blk.{layer}.ffn_gate_exps.weight",
                gate_sum2[slot], counts[slot],
            )
            write_imatrix_entry(
                handle, f"blk.{layer}.ffn_up_exps.weight",
                gate_sum2[slot], counts[slot],
            )
            write_imatrix_entry(
                handle, f"blk.{layer}.ffn_down_exps.weight",
                down_sum2[slot], counts[slot],
            )
        # llama.cpp's legacy footer: number of observed layer chunks and a
        # free-form dataset identifier.
        write_i32(handle, chunks * processed_sparse)
        dataset_name = str(args.dataset.resolve()).encode("utf-8")
        write_i32(handle, len(dataset_name))
        handle.write(dataset_name)

    nonzero = counts[counts > 0]
    zero_experts = int((counts == 0).sum())
    report = {
        "schema_version": 1,
        "status": "ok" if processed_sparse == 51 and zero_experts == 0 else "pilot",
        "source_model": "Motif-Technologies/Motif-3",
        "source_revision": PINNED_REVISION,
        "q8_gguf": str(args.gguf.resolve()),
        "q8_gguf_bytes": args.gguf.stat().st_size,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "chunks": chunks,
        "chunk_size": args.chunk_size,
        "tokens": tokens,
        "ranks": world,
        "processed_layers": min(args.max_layers, N_LAYER),
        "processed_sparse_layers": processed_sparse,
        "entries": processed_sparse * 3,
        "zero_coverage_layer_experts": zero_experts,
        "route_count_min": int(nonzero.min()) if nonzero.size else 0,
        "route_count_p50": float(np.median(nonzero)) if nonzero.size else 0,
        "route_count_p95": float(np.percentile(nonzero, 95)) if nonzero.size else 0,
        "route_count_max": int(nonzero.max()) if nonzero.size else 0,
        "total_routes": int(counts.sum()),
        "output": str(args.output.resolve()),
        "output_bytes": args.output.stat().st_size,
        "output_sha256": sha256_file(args.output),
        "final_hidden_rank_sha256": hidden_digests,
        "layer_seconds_by_rank": timings,
    }
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rank_log(rank, f"merged {args.output} ({report['output_bytes'] / 2**30:.3f} GiB)")
    rank_log(rank, f"coverage zeros={zero_experts}, min={report['route_count_min']}, "
                   f"p50={report['route_count_p50']:.0f}, max={report['route_count_max']}")
    return report


def main() -> int:
    args = parse_args()
    if args.chunk_size <= 0 or args.max_chunks <= 0:
        raise SystemExit("chunk size and count must be positive")
    if args.max_layers <= 0 or args.max_layers > N_LAYER:
        raise SystemExit(f"--max-layers must be between 1 and {N_LAYER}")
    if not args.gguf.is_file() or not args.dataset.is_file() or not args.tokenizer.is_dir():
        raise SystemExit("GGUF, dataset, or tokenizer path is missing")

    rank, world, device = distributed_init()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    token_cache = prepare_token_cache(args, rank, world)
    warm_flashinfer(rank, world, device, args.workspace_mib)
    run_rank(args, rank, world, device, token_cache)
    barrier(world)
    if rank == 0:
        report = merge_partials(args, world, rank)
        if args.max_layers == N_LAYER and report["status"] != "ok":
            raise RuntimeError("release imatrix failed full expert coverage")
    barrier(world)
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
