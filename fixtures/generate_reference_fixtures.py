#!/usr/bin/env python3
"""Generate small, deterministic fixtures from the pinned official Motif-3.

The fixture payload deliberately contains projected intermediates rather than
checkpoint weights.  It is therefore small enough to keep with the runtime,
while still locking the operations that are new to ds4: biased sigmoid routing,
expert-specific PolyNorm, and modified Hyper-Connections.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


PINNED_REVISION = "ccceb1a5fd7b5eb32e47841216b3caf5666c07bc"
PINNED_CONFIG_SHA256 = (
    "30f14b635d3258a18c3ff7e69829f8fbfa775e87477ffabb59a79115bba820a5"
)
ROUTER_LAYER = 2
MHC_LAYER = 0
TOKENS = 8
EXPERT = 173
EPS = 1e-6
HIDDEN_CLAMP = 1_000_000.0
POLYNORM_OUTPUT_SCALE = 0.5
POLYNORM_BIAS_CLAMP = 0.5
ROUTE_SCALE = 2.0
TOP_K = 8
SINKHORN_ITERS = 20
H_POST_COEFF = 1.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().contiguous().cpu()
    if value.dtype == torch.bfloat16:
        raw = value.view(torch.uint16).numpy().tobytes()
    else:
        raw = value.numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def f32(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def write_ds4_fixture(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a tiny dependency-free container consumed by ds4 C tests."""
    with path.open("wb") as stream:
        stream.write(struct.pack("<8sII", b"DS4FX1\0\0", 1, len(arrays)))
        for name, value in arrays.items():
            value = np.ascontiguousarray(value)
            if value.dtype == np.float32:
                dtype = 1
                value = value.astype("<f4", copy=False)
            elif value.dtype == np.int32:
                dtype = 2
                value = value.astype("<i4", copy=False)
            else:
                raise TypeError(f"unsupported ds4 fixture dtype for {name}: {value.dtype}")
            if value.ndim > 4:
                raise ValueError(f"ds4 fixture array has more than four dimensions: {name}")
            dims = list(value.shape) + [0] * (4 - value.ndim)
            encoded = name.encode("utf-8")
            stream.write(
                struct.pack(
                    "<IIIIQQQQQ",
                    len(encoded),
                    dtype,
                    value.ndim,
                    0,
                    *dims,
                    value.nbytes,
                )
            )
            stream.write(encoded)
            stream.write(value.tobytes())


class OfficialWeights:
    def __init__(self, snapshot: Path, index_path: Path):
        self.snapshot = snapshot
        self.weight_map = json.loads(index_path.read_text())["weight_map"]
        self.sources: dict[str, str] = {}

    def _path(self, name: str) -> Path:
        shard = self.weight_map[name]
        path = self.snapshot / shard
        if not path.exists():
            raise FileNotFoundError(f"missing official shard for {name}: {path}")
        self.sources[name] = shard
        return path

    def tensor(self, name: str) -> torch.Tensor:
        with safe_open(self._path(name), framework="pt", device="cpu") as reader:
            return reader.get_tensor(name)

    def expert_slice(self, name: str, expert: int) -> torch.Tensor:
        with safe_open(self._path(name), framework="pt", device="cpu") as reader:
            return reader.get_slice(name)[expert]


def make_input(shape: tuple[int, ...], seed: int, scale: float = 0.5) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Runtime hidden states arrive at these control paths as BF16.  Store their
    # exact BF16 values in an interoperable float32 NumPy array.
    return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16)


def router_fixture(weights: OfficialWeights, device: torch.device) -> dict[str, Any]:
    prefix = f"model.layers.{ROUTER_LAYER}.moe"
    gate_name = f"{prefix}.router.gate.weight"
    bias_name = f"{prefix}.expert_bias"
    gate = weights.tensor(gate_name).float()
    bias = weights.tensor(bias_name).float()
    hidden = make_input((TOKENS, gate.shape[1]), 0x4D4F5449, 0.625).float()

    # Canonical fixture uses full FP32, matching the control-path contract.
    logits = F.linear(hidden.to(device), gate.to(device)).float().cpu()
    scores = torch.sigmoid(logits)
    adjusted = scores + bias
    _, selected = torch.topk(adjusted, k=TOP_K, dim=-1, sorted=True)
    selected_scores = torch.gather(scores, 1, selected)
    route_weights = selected_scores / selected_scores.sum(-1, keepdim=True)
    route_weights = route_weights * ROUTE_SCALE

    return {
        "arrays": {
            "hidden": f32(hidden),
            "logits": f32(logits),
            "expert_bias": f32(bias),
            "selected_experts": selected.numpy().astype(np.int32),
            "route_weights": f32(route_weights),
        },
        "source_tensors": {
            gate_name: tensor_sha256(gate),
            bias_name: tensor_sha256(bias),
        },
    }


def polynorm_fixture(weights: OfficialWeights, device: torch.device) -> dict[str, Any]:
    prefix = f"model.layers.{ROUTER_LAYER}.moe.experts"
    gate_up_name = f"{prefix}.gate_up_proj"
    down_name = f"{prefix}.down_proj"
    coeff_name = f"{prefix}.act_fn.weight"
    bias_name = f"{prefix}.act_fn.bias"

    gate_up = weights.expert_slice(gate_up_name, EXPERT).to(torch.bfloat16)
    down = weights.expert_slice(down_name, EXPERT).to(torch.bfloat16)
    raw_coeff = weights.tensor(coeff_name)[EXPERT].float()
    raw_bias = weights.tensor(bias_name)[EXPERT].float()
    hidden = make_input((4, gate_up.shape[1]), 0x504F4C59, 0.375)

    hidden_gpu = hidden.to(device)
    gate_up_gpu = gate_up.to(device)
    down_gpu = down.to(device)
    projected = F.linear(hidden_gpu, gate_up_gpu)
    gate, up = projected.chunk(2, dim=-1)
    gate = gate.clamp(-HIDDEN_CLAMP, HIDDEN_CLAMP)
    up = up.clamp(-HIDDEN_CLAMP, HIDDEN_CLAMP)

    g = gate.float()
    g2 = g * g
    g3 = g2 * g
    inv2 = torch.rsqrt(g2.mean(-1, keepdim=True) + EPS)
    inv4 = torch.rsqrt((g2 * g2).mean(-1, keepdim=True) + EPS)
    inv6 = torch.rsqrt((g3 * g3).mean(-1, keepdim=True) + EPS)
    coeff = torch.sigmoid(raw_coeff).to(device)
    bias = raw_bias.clamp(-POLYNORM_BIAS_CLAMP, POLYNORM_BIAS_CLAMP).to(device)
    poly = coeff[0] * g3 * inv6 + coeff[1] * g2 * inv4 + coeff[2] * g * inv2 + bias

    # The official implementation performs PolyNorm and multiply in FP32,
    # casts once to the projection dtype, then applies output_scale.
    activated = (poly * up.float()).to(torch.bfloat16)
    activated = activated * POLYNORM_OUTPUT_SCALE
    output = F.linear(activated, down_gpu)

    # Also retain the pre-down-projection FP32 contract.  This is the most
    # useful fixture for testing ds4's activation independently of IQ2/Q2 error.
    activated_fp32 = (poly * up.float()) * POLYNORM_OUTPUT_SCALE
    return {
        "arrays": {
            "hidden": f32(hidden),
            "gate": f32(gate),
            "up": f32(up),
            "raw_coeff": f32(raw_coeff),
            "raw_bias": f32(raw_bias),
            "activated_fp32": f32(activated_fp32),
            "activated_bf16": f32(activated),
            "down_output_bf16": f32(output),
        },
        "source_tensors": {
            f"{gate_up_name}[{EXPERT}]": tensor_sha256(gate_up),
            f"{down_name}[{EXPERT}]": tensor_sha256(down),
            f"{coeff_name}[{EXPERT}]": tensor_sha256(raw_coeff),
            f"{bias_name}[{EXPERT}]": tensor_sha256(raw_bias),
        },
    }


def sinkhorn(value: torch.Tensor) -> torch.Tensor:
    value = value.float().clamp(-20.0, 20.0).exp()
    for _ in range(SINKHORN_ITERS):
        value = value / value.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        value = value / value.sum(dim=-2, keepdim=True).clamp(min=1e-8)
    return value


def mhc_fixture(weights: OfficialWeights, device: torch.device) -> dict[str, Any]:
    prefix = f"model.layers.{MHC_LAYER}.mhc_attn"
    names = {
        key: f"{prefix}.{key}"
        for key in (
            "rms_norm.weight",
            "proj_pre.weight",
            "proj_post.weight",
            "proj_res.weight",
            "alpha_pre",
            "alpha_post",
            "alpha_res",
            "bias_pre",
            "bias_post",
            "bias_res",
        )
    }
    tensors = {key: weights.tensor(name) for key, name in names.items()}
    expansion = tensors["bias_pre"].numel()
    hidden_size = tensors["rms_norm.weight"].numel() // expansion
    hidden = make_input((1, 4, expansion, hidden_size), 0x4D484330, 0.4375)
    flat = hidden.reshape(4, expansion * hidden_size).to(device)

    # RMS accumulation and all mHC control arithmetic remain FP32.  Projection
    # weights and inputs are BF16, as prescribed for the small projections.
    variance = flat.float().pow(2).mean(-1, keepdim=True)
    normalized = flat.float() * torch.rsqrt(variance + EPS)
    normalized = (normalized * tensors["rms_norm.weight"].float().to(device)).to(
        torch.bfloat16
    )
    pre = F.linear(normalized, tensors["proj_pre.weight"].to(device)).float()
    post = F.linear(normalized, tensors["proj_post.weight"].to(device)).float()
    residual = F.linear(normalized, tensors["proj_res.weight"].to(device)).float()
    residual = residual.reshape(1, 4, expansion, expansion)
    pre = pre.reshape(1, 4, expansion)
    post = post.reshape(1, 4, expansion)

    h_pre = torch.sigmoid(
        (
            tensors["alpha_pre"].float().to(device) * pre
            + tensors["bias_pre"].float().to(device)
        ).clamp(-10.0, 10.0)
    )
    h_post = H_POST_COEFF * torch.sigmoid(
        (
            tensors["alpha_post"].float().to(device) * post
            + tensors["bias_post"].float().to(device)
        ).clamp(-10.0, 10.0)
    )
    h_res = sinkhorn(
        tensors["alpha_res"].float().to(device) * residual
        + tensors["bias_res"].float().to(device)
    )
    hidden_fp32 = hidden.float().to(device)
    reduced = (hidden_fp32 * h_pre.unsqueeze(-1)).sum(dim=2)
    residual_mixed = torch.einsum("btij,btjd->btid", h_res, hidden_fp32)

    return {
        "arrays": {
            "hidden": f32(hidden),
            "projected_pre": f32(pre),
            "projected_post": f32(post),
            "projected_res": f32(residual),
            "alpha_pre": f32(tensors["alpha_pre"]),
            "alpha_post": f32(tensors["alpha_post"]),
            "alpha_res": f32(tensors["alpha_res"]),
            "bias_pre": f32(tensors["bias_pre"]),
            "bias_post": f32(tensors["bias_post"]),
            "bias_res": f32(tensors["bias_res"]),
            "h_pre": f32(h_pre),
            "h_post": f32(h_post),
            "h_res": f32(h_res),
            "reduced_input": f32(reduced),
            "residual_mixed": f32(residual_mixed),
        },
        "source_tensors": {
            name: tensor_sha256(tensors[key]) for key, name in names.items()
        },
    }


def yarn_inv_freq(
    dim: int,
    end: int,
    theta: float,
    original_length: int,
    factor: float,
    beta_fast: float,
    beta_slow: float,
    device: torch.device,
) -> torch.Tensor:
    def correction_dim(rotations: float) -> float:
        return dim * np.log(original_length / (rotations * 2.0 * np.pi)) / (
            2.0 * np.log(theta)
        )

    frequency = 1.0 / (
        theta
        ** (
            torch.arange(0, dim, 2, dtype=torch.float32, device=device)
            / float(dim)
        )
    )
    if end > original_length:
        low = max(int(np.floor(correction_dim(beta_fast))), 0)
        high = min(int(np.ceil(correction_dim(beta_slow))), dim - 1)
        if low == high:
            high += 0.001
        ramp = (
            torch.arange(dim // 2, dtype=torch.float32, device=device) - low
        ) / (high - low)
        smooth = 1.0 - ramp.clamp(0.0, 1.0)
        frequency = frequency / factor * (1.0 - smooth) + frequency * smooth
    return frequency


def apply_neox_rope(
    value: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor
) -> torch.Tensor:
    frequency = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    cosine = torch.cat([frequency.cos(), frequency.cos()], dim=-1)
    sine = torch.cat([frequency.sin(), frequency.sin()], dim=-1)
    while cosine.ndim < value.ndim:
        cosine = cosine.unsqueeze(1)
        sine = sine.unsqueeze(1)
    first, second = value.float().chunk(2, dim=-1)
    rotated = torch.cat([-second, first], dim=-1)
    return value.float() * cosine + rotated * sine


def gdla_fixture(weights: OfficialWeights, device: torch.device) -> dict[str, Any]:
    layer = 0
    prefix = f"model.layers.{layer}.self_attn"
    names = {
        "q_a": f"{prefix}.wq_a.weight",
        "q_a_norm": f"{prefix}.q_norm.weight",
        "q_b": f"{prefix}.wq_b.weight",
        "q_gate": f"{prefix}.wq_b_gate.weight",
        "kv_a": f"{prefix}.wkv_a.weight",
        "kv_a_norm": f"{prefix}.kv_norm.weight",
        "kv_b": f"{prefix}.wkv_b.weight",
        "lambda": f"{prefix}.lambda_proj.weight",
        "output": f"{prefix}.wo.weight",
    }
    tensors = {key: weights.tensor(name) for key, name in names.items()}
    hidden = make_input((8, 4096), 0x47444C41, 0.3125)
    hidden_gpu = hidden.to(device)

    # Official HF semantics: q-a/q-b and q RMS stay FP32 for parity, while
    # the stored attention activation and remaining projections are BF16.
    q_latent_raw = F.linear(hidden_gpu.float(), tensors["q_a"].float().to(device))
    q_latent = q_latent_raw * torch.rsqrt(
        q_latent_raw.pow(2).mean(-1, keepdim=True) + 1.0e-5
    )
    q_latent = q_latent * tensors["q_a_norm"].float().to(device)
    q = F.linear(q_latent, tensors["q_b"].float().to(device)).to(torch.bfloat16)
    q = q.reshape(8, 80, 192)
    q_latent_bf16 = q_latent.to(torch.bfloat16)
    gate_score = F.linear(q_latent_bf16, tensors["q_gate"].to(device))
    gate_score = gate_score.reshape(8, 64, 128)

    kv_raw = F.linear(hidden_gpu, tensors["kv_a"].to(device))
    kv_latent, k_pe = kv_raw.split([512, 64], dim=-1)
    kv_norm = kv_latent.float() * torch.rsqrt(
        kv_latent.float().pow(2).mean(-1, keepdim=True) + 1.0e-5
    )
    kv_norm = (kv_norm * tensors["kv_a_norm"].float().to(device)).to(torch.bfloat16)
    kv_proj = F.linear(kv_norm, tensors["kv_b"].to(device)).reshape(8, 16, 256)
    k_nope, value = kv_proj.split([128, 128], dim=-1)

    positions = torch.arange(8, dtype=torch.int32, device=device)
    inv_freq = yarn_inv_freq(
        64, 262144, 10000.0, 4096, 64.0, 32.0, 1.0, device
    )
    q_nope, q_pe_before = q.split([128, 64], dim=-1)
    q_pe_fp32 = apply_neox_rope(q_pe_before, positions, inv_freq)
    k_pe_before = k_pe.unsqueeze(1)
    k_pe_fp32 = apply_neox_rope(k_pe_before, positions, inv_freq)
    q_pe = q_pe_fp32.to(torch.bfloat16)
    k_pe = k_pe_fp32.to(torch.bfloat16)
    q_full = torch.cat([q_nope, q_pe], dim=-1)
    k_full = torch.cat([k_nope, k_pe.expand(-1, 16, -1)], dim=-1)

    # Expanded K/V reference. Head h consumes KV head floor(h/5); causal
    # attention is accumulated and softmaxed in FP32.
    qh = q_full.transpose(0, 1).float()
    kh = k_full.transpose(0, 1).float().repeat_interleave(5, dim=0)
    vh = value.transpose(0, 1).float().repeat_interleave(5, dim=0)
    mscale = 0.1 * np.log(64.0) + 1.0
    attention_scale = (192.0**-0.5) * mscale * mscale
    score = torch.matmul(qh, kh.transpose(-1, -2)) * attention_scale
    causal = torch.triu(
        torch.ones((8, 8), dtype=torch.bool, device=device), diagonal=1
    )
    score = score.masked_fill(causal, float("-inf"))
    probability = torch.softmax(score, dim=-1)
    attention_fp32 = torch.matmul(probability, vh).transpose(0, 1)

    lambda_value = F.linear(hidden_gpu, tensors["lambda"].to(device))
    grouped = attention_fp32.reshape(8, 16, 5, 128)
    signal = grouped[:, :, :4, :].reshape(8, 64, 128)
    noise = grouped[:, :, 4:, :].expand(-1, -1, 4, -1).reshape(8, 64, 128)
    diff_fp32 = signal - torch.sigmoid(lambda_value.float()).unsqueeze(-1) * noise
    diff_fp32 = diff_fp32 * torch.sigmoid(gate_score.float())

    # BF16 checkpoint execution result, retained for end-to-end tolerance.
    attention_bf16 = attention_fp32.to(torch.bfloat16)
    grouped_bf16 = attention_bf16.reshape(8, 16, 5, 128)
    signal_bf16 = grouped_bf16[:, :, :4, :].reshape(8, 64, 128)
    noise_bf16 = grouped_bf16[:, :, 4:, :].expand(-1, -1, 4, -1).reshape(
        8, 64, 128
    )
    lambda_scale = torch.sigmoid(lambda_value.float()).to(torch.bfloat16).unsqueeze(-1)
    diff_bf16 = signal_bf16 - lambda_scale * noise_bf16
    diff_bf16 = diff_bf16 * torch.sigmoid(gate_score).to(torch.bfloat16)
    output_bf16 = F.linear(
        diff_bf16.reshape(8, 64 * 128), tensors["output"].to(device)
    )

    # A second RoPE-only probe covers the native 256K position boundary.
    probe_positions = torch.tensor(
        [0, 1, 127, 128, 4095, 65535, 131071, 262143],
        dtype=torch.int32,
        device=device,
    )
    q_probe_fp32 = apply_neox_rope(q_pe_before, probe_positions, inv_freq)
    k_probe_fp32 = apply_neox_rope(k_pe_before, probe_positions, inv_freq)

    return {
        "arrays": {
            "hidden": f32(hidden),
            "positions": positions.cpu().numpy().astype(np.int32),
            "probe_positions": probe_positions.cpu().numpy().astype(np.int32),
            "yarn_inv_freq": f32(inv_freq),
            "attention_scale": np.asarray([attention_scale], dtype=np.float32),
            "q_pe_before": f32(q_pe_before),
            "k_pe_before": f32(k_pe_before),
            "q_pe_after_fp32": f32(q_pe_fp32),
            "k_pe_after_fp32": f32(k_pe_fp32),
            "q_pe_probe_fp32": f32(q_probe_fp32),
            "k_pe_probe_fp32": f32(k_probe_fp32),
            "q_full": f32(q_full),
            "k_full": f32(k_full),
            "value": f32(value),
            "lambda": f32(lambda_value),
            "gate_score": f32(gate_score),
            "attention_fp32": f32(attention_fp32),
            "diff_attention_fp32": f32(diff_fp32),
            "attention_bf16": f32(attention_bf16),
            "diff_attention_bf16": f32(diff_bf16),
            "output_bf16": f32(output_bf16),
        },
        "source_tensors": {
            name: tensor_sha256(tensors[key]) for key, name in names.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config_path = args.source_metadata / "config.json"
    index_path = args.source_metadata / "model.safetensors.index.json"
    if sha256_file(config_path) != PINNED_CONFIG_SHA256:
        raise RuntimeError("official config hash does not match pinned final release")
    config = json.loads(config_path.read_text())
    required = {
        "num_experts": 384,
        "experts_top_k": TOP_K,
        "score_func": "sigmoid",
        "route_norm": True,
        "route_scale": ROUTE_SCALE,
        "polynorm_output_scale": POLYNORM_OUTPUT_SCALE,
        "polynorm_bias_clamp": POLYNORM_BIAS_CLAMP,
        "mhc_expansion_rate": 4,
        "mhc_sinkhorn_iters": SINKHORN_ITERS,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise RuntimeError(f"official config {key}={config.get(key)!r}, expected {expected!r}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    weights = OfficialWeights(args.snapshot, index_path)
    args.output.mkdir(parents=True, exist_ok=True)

    generated: dict[str, dict[str, Any]] = {}
    jobs = {
        "router-layer2.npz": router_fixture,
        "polynorm-layer2-expert173.npz": polynorm_fixture,
        "mhc-layer0-attn.npz": mhc_fixture,
        "gdla-expanded-layer0.npz": gdla_fixture,
    }
    with torch.inference_mode():
        for filename, function in jobs.items():
            fixture = function(weights, device)
            path = args.output / filename
            np.savez_compressed(path, **fixture["arrays"])
            runtime_path = args.output / (Path(filename).stem + ".ds4fx")
            write_ds4_fixture(runtime_path, fixture["arrays"])
            generated[filename] = {
                "sha256": sha256_file(path),
                "runtime_fixture": {
                    "name": runtime_path.name,
                    "sha256": sha256_file(runtime_path),
                },
                "arrays": {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in fixture["arrays"].items()
                },
                "source_tensors_sha256": fixture["source_tensors"],
            }

    manifest = {
        "schema_version": 1,
        "source_policy": "official-final-only",
        "source": {
            "repo": "Motif-Technologies/Motif-3",
            "revision": PINNED_REVISION,
            "config_sha256": PINNED_CONFIG_SHA256,
            "shards": {name: weights.sources[name] for name in sorted(weights.sources)},
        },
        "reference_contract": {
            "router": "fp32 sigmoid(logits), correction bias for selection only, top-8, renorm, scale 2",
            "polynorm": "per-expert sigmoid(coeff), FP32 moments/poly/multiply, BF16 cast before scale/down",
            "mhc": "BF16 projections with FP32 RMS/control, clamp, sigmoid, 20-step FP32 Sinkhorn, h_post coefficient 1",
        },
        "constants": {
            "router_layer": ROUTER_LAYER,
            "expert": EXPERT,
            "top_k": TOP_K,
            "route_scale": ROUTE_SCALE,
            "hidden_clamp": HIDDEN_CLAMP,
            "polynorm_output_scale": POLYNORM_OUTPUT_SCALE,
            "polynorm_bias_clamp": POLYNORM_BIAS_CLAMP,
            "mhc_layer": MHC_LAYER,
            "mhc_expansion": 4,
            "sinkhorn_iters": SINKHORN_ITERS,
            "h_post_coeff": H_POST_COEFF,
            "eps": EPS,
        },
        "generator": {
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "tf32": False,
        },
        "fixtures": generated,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "fixtures": generated}, indent=2))


if __name__ == "__main__":
    main()
