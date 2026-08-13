#!/usr/bin/env python3
"""Validate Motif 3 topology and project the locked MQ87-88-FIT payload."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path

SOURCE_REPO = "Motif-Technologies/Motif-3"
SOURCE_REVISION = "ccceb1a5fd7b5eb32e47841216b3caf5666c07bc"
EXPECTED_PARAMETERS = 314_841_775_750
EXPECTED_TENSORS = 2_236
EXPECTED_SHARDS = 155

N_LAYER = 53
N_DENSE = 2
N_EXPERT = 384
N_EXPERT_USED = 8
N_SIGNAL_HEAD = 64
N_NOISE_HEAD = 16
N_KV_HEAD = 16
HIDDEN = 4096
ROUTED_FF = 1280
SWA_WINDOW = 128
MAX_CONTEXT = 262_144

QUANT_LAYOUT = {
    "F32": (1, 4),
    "BF16": (1, 2),
    "Q8_0": (32, 34),
    "IQ2_XXS": (256, 66),
    "Q2_K": (256, 84),
}

GLOBAL_SHAPES = {
    "model.embed_tokens.weight": [220_160, 4096],
    "model.norm.weight": [4096],
    "lm_head.weight": [220_160, 4096],
}

ATTENTION_SHAPES = {
    "self_attn.q_norm.weight": [1024],
    "self_attn.kv_norm.weight": [512],
    "self_attn.wq_a.weight": [1024, 4096],
    "self_attn.wq_b.weight": [15_360, 1024],
    "self_attn.wq_b_gate.weight": [8192, 1024],
    "self_attn.wkv_a.weight": [576, 4096],
    "self_attn.wkv_b.weight": [4096, 512],
    "self_attn.lambda_proj.weight": [64, 4096],
    "self_attn.wo.weight": [4096, 8192],
}

DENSE_MLP_SHAPES = {
    "mlp.act_fn.bias": [1],
    "mlp.act_fn.weight": [3],
    "mlp.down_proj.weight": [4096, 12_288],
    "mlp.gate_proj.weight": [12_288, 4096],
    "mlp.up_proj.weight": [12_288, 4096],
}

SPARSE_MOE_SHAPES = {
    "moe.expert_bias": [384],
    "moe.experts.act_fn.bias": [384, 1],
    "moe.experts.act_fn.weight": [384, 3],
    "moe.experts.down_proj": [384, 4096, 1280],
    "moe.experts.gate_up_proj": [384, 2560, 4096],
    "moe.router.gate.weight": [384, 4096],
    "moe.shared_experts.act_fn.bias": [1],
    "moe.shared_experts.act_fn.weight": [3],
    "moe.shared_experts.down_proj.weight": [4096, 1280],
    "moe.shared_experts.gate_proj.weight": [1280, 4096],
    "moe.shared_experts.up_proj.weight": [1280, 4096],
}

MTP_SHAPES = {
    "model.mtp_layers.0.embed_norm.weight": [4096],
    "model.mtp_layers.0.final_layernorm.weight": [4096],
    "model.mtp_layers.0.input_layernorm.weight": [4096],
    "model.mtp_layers.0.input_proj.weight": [4096, 8192],
    "model.mtp_layers.0.mlp.act_fn.bias": [1],
    "model.mtp_layers.0.mlp.act_fn.weight": [3],
    "model.mtp_layers.0.mlp.down_proj.weight": [4096, 12_288],
    "model.mtp_layers.0.mlp.gate_proj.weight": [12_288, 4096],
    "model.mtp_layers.0.mlp.up_proj.weight": [12_288, 4096],
    "model.mtp_layers.0.post_attention_layernorm.weight": [4096],
    "model.mtp_layers.0.self_attn.kv_norm.weight": [512],
    "model.mtp_layers.0.self_attn.lambda_proj.weight": [64, 4096],
    "model.mtp_layers.0.self_attn.q_norm.weight": [1024],
    "model.mtp_layers.0.self_attn.wkv_a.weight": [576, 4096],
    "model.mtp_layers.0.self_attn.wkv_b.weight": [4096, 512],
    "model.mtp_layers.0.self_attn.wo.weight": [4096, 8192],
    "model.mtp_layers.0.self_attn.wq_a.weight": [1024, 4096],
    "model.mtp_layers.0.self_attn.wq_b.weight": [15_360, 1024],
    "model.mtp_layers.0.self_attn.wq_b_gate.weight": [8192, 1024],
}

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--headers", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aligned(value: int, alignment: int = 32) -> int:
    return (value + alignment - 1) // alignment * alignment


def tensor_elements(shape: list[int]) -> int:
    return math.prod(shape)


def storage_bytes(shape: list[int], quant_type: str) -> int:
    block, type_size = QUANT_LAYOUT[quant_type]
    elements = tensor_elements(shape)
    if elements % block:
        raise ValueError(
            f"{quant_type} tensor has {elements} elements, not divisible by block {block}: {shape}"
        )
    return elements // block * type_size


def mhc_shapes(prefix: str) -> dict[str, list[int]]:
    return {
        f"{prefix}.alpha_post": [1],
        f"{prefix}.alpha_pre": [1],
        f"{prefix}.alpha_res": [1],
        f"{prefix}.bias_post": [4],
        f"{prefix}.bias_pre": [4],
        f"{prefix}.bias_res": [4, 4],
        f"{prefix}.proj_post.weight": [4, 16_384],
        f"{prefix}.proj_pre.weight": [4, 16_384],
        f"{prefix}.proj_res.weight": [16, 16_384],
        f"{prefix}.rms_norm.weight": [16_384],
    }


def target_for(name: str) -> tuple[str, str, int | None]:
    if name == "model.embed_tokens.weight":
        return "embedding", "Q8_0", None
    if name == "lm_head.weight":
        return "lm_head", "Q8_0", None
    if name == "model.norm.weight":
        return "norm", "F32", None

    if name.startswith("model.mtp_layers."):
        if ".act_fn." in name:
            return "mtp_polynorm", "F32", None
        if "_norm.weight" in name or "layernorm.weight" in name:
            return "mtp_norm", "F32", None
        return "mtp_projection", "Q8_0", None

    match = LAYER_RE.match(name)
    if not match:
        raise KeyError(f"unclassified tensor: {name}")
    layer = int(match.group(1))
    suffix = match.group(2)

    if suffix in {"input_layernorm.weight", "post_attention_layernorm.weight"}:
        return "norm", "F32", layer
    if suffix.startswith("mhc_"):
        if ".proj_" in suffix:
            return "mhc_small_projection", "BF16", layer
        if ".rms_norm." in suffix:
            return "mhc_norm", "F32", layer
        return "mhc_control", "F32", layer
    if suffix.startswith("self_attn."):
        if suffix in {"self_attn.q_norm.weight", "self_attn.kv_norm.weight"}:
            return "gdla_norm", "F32", layer
        return "gdla_projection", "Q8_0", layer

    if layer < N_DENSE and suffix.startswith("mlp."):
        if ".act_fn." in suffix:
            return "dense_polynorm", "F32", layer
        return "dense_mlp", "Q8_0", layer

    if layer >= N_DENSE and suffix.startswith("moe."):
        if suffix == "moe.expert_bias":
            return "router_control", "F32", layer
        if suffix == "moe.router.gate.weight":
            return "router", "F32", layer
        if suffix.startswith("moe.experts.act_fn."):
            return "expert_polynorm", "F32", layer
        if suffix == "moe.experts.gate_up_proj":
            return "routed_gate_up", "IQ2_XXS", layer
        if suffix == "moe.experts.down_proj":
            return "routed_down", "Q2_K", layer
        if suffix.startswith("moe.shared_experts.act_fn."):
            return "shared_polynorm", "F32", layer
        if suffix.startswith("moe.shared_experts."):
            return "shared_expert", "Q8_0", layer

    raise KeyError(f"unclassified tensor: {name}")


def require_tensor(
    tensors: dict[str, dict], name: str, shape: list[int], errors: list[str]
) -> None:
    tensor = tensors.get(name)
    if tensor is None:
        errors.append(f"missing tensor: {name}")
        return
    if tensor["shape"] != shape:
        errors.append(f"shape mismatch {name}: {tensor['shape']} != {shape}")
    if tensor["dtype"] != "BF16":
        errors.append(f"dtype mismatch {name}: {tensor['dtype']} != BF16")


def expected_tensor_shapes() -> dict[str, list[int]]:
    expected = dict(GLOBAL_SHAPES)
    for layer in range(N_LAYER):
        base = f"model.layers.{layer}."
        expected[base + "input_layernorm.weight"] = [4096]
        expected[base + "post_attention_layernorm.weight"] = [4096]
        for suffix, shape in ATTENTION_SHAPES.items():
            expected[base + suffix] = shape
        for mhc in ("mhc_attn", "mhc_ffn"):
            for suffix, shape in mhc_shapes(mhc).items():
                expected[base + suffix] = shape
        block = DENSE_MLP_SHAPES if layer < N_DENSE else SPARSE_MOE_SHAPES
        for suffix, shape in block.items():
            expected[base + suffix] = shape
    expected.update(MTP_SHAPES)
    return expected


def topology_digest(tensors: dict[str, dict]) -> str:
    rows = [
        [name, tensor["dtype"], tensor["shape"], tensor["shard"]]
        for name, tensor in sorted(tensors.items())
    ]
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def cache_projection() -> dict:
    full_layers = list(range(0, N_LAYER, 4))
    swa_layers = sorted(set(range(N_LAYER)) - set(full_layers))

    # Full-attention history stays as [kv_latent(512), k_pe(64)] in BF16.
    full_latent_bytes = len(full_layers) * MAX_CONTEXT * (512 + 64) * 2

    # Conservative bring-up allowance: each SWA layer may retain an expanded
    # 129-token K/V ring. It is bounded and is not full-history materialization.
    ring_tokens = SWA_WINDOW + 1
    swa_ring_bytes = (
        len(swa_layers)
        * ring_tokens
        * N_KV_HEAD
        * (192 + 128)
        * 2
    )
    identity_bytes = N_LAYER * 4096
    total = full_latent_bytes + swa_ring_bytes + identity_bytes
    return {
        "context_tokens": MAX_CONTEXT,
        "full_attention_layers": full_layers,
        "sliding_window_layers": swa_layers,
        "full_latent_kv_bytes": full_latent_bytes,
        "swa_expanded_ring_upper_bound_bytes": swa_ring_bytes,
        "position_cache_identity_bytes": identity_bytes,
        "total_bytes": total,
        "total_gib": total / 2**30,
        "note": (
            "SWA uses only a bounded 129-token ring. Full-history expanded K/V "
            "is forbidden; full layers retain latent KV plus decoupled RoPE key."
        ),
    }


def project(inventory: list[dict]) -> dict:
    by_type: collections.Counter[str] = collections.Counter()
    by_family: collections.Counter[str] = collections.Counter()
    tensor_counts: collections.Counter[str] = collections.Counter()
    raw = 0
    aligned_total = 0
    for tensor in inventory:
        size = storage_bytes(tensor["shape"], tensor["target_type"])
        by_type[tensor["target_type"]] += size
        by_family[tensor["family"]] += size
        tensor_counts[tensor["target_type"]] += 1
        raw += size
        aligned_total += aligned(size)

    metadata_estimate = 4 * 1024 * 1024 + sum(
        128 + len(tensor["name"].encode()) for tensor in inventory
    )
    artifact_estimate = aligned_total + metadata_estimate
    cache = cache_projection()
    preferred_other = (6 + 3 + 8) * 2**30
    spark_capacity_gib = 121.6
    projected_with_reserve = artifact_estimate + cache["total_bytes"] + preferred_other
    return {
        "variant": "MQ87-88-FIT",
        "source_revision": SOURCE_REVISION,
        "tensor_count": len(inventory),
        "raw_tensor_bytes": raw,
        "aligned_tensor_bytes": aligned_total,
        "metadata_estimate_bytes": metadata_estimate,
        "estimated_artifact_bytes": artifact_estimate,
        "estimated_artifact_gib": artifact_estimate / 2**30,
        "nominal_class_gib": [87, 88],
        "hard_file_size_gate": None,
        "acceptance": "measured single-GB10 resident serving",
        "bytes_by_type": dict(sorted(by_type.items())),
        "bytes_by_family": dict(sorted(by_family.items())),
        "tensor_counts_by_type": dict(sorted(tensor_counts.items())),
        "latent_cache_256k": cache,
        "spark_advisory_projection": {
            "assumed_total_unified_memory_gib": spark_capacity_gib,
            "workspace_gib": 6,
            "server_session_gib": 3,
            "memavailable_reserve_gib": 8,
            "model_cache_workspace_server_reserve_gib": projected_with_reserve / 2**30,
            "projected_headroom_gib": spark_capacity_gib - projected_with_reserve / 2**30,
            "authoritative_gate": "measured physical memory on the target Spark",
        },
        "note": (
            "Exact GGML block sizes and 32-byte tensor alignment are used. "
            "The recipe has no automatic size-reduction ladder."
        ),
    }


def validate_config(config: dict, errors: list[str]) -> None:
    expected = {
        "model_type": "Motif",
        "num_hidden_layers": N_LAYER,
        "n_dense_first_layers": N_DENSE,
        "hidden_size": HIDDEN,
        "intermediate_size": 12_288,
        "moe_intermediate_size": ROUTED_FF,
        "num_experts": N_EXPERT,
        "experts_top_k": N_EXPERT_USED,
        "num_shared_experts": 1,
        "num_attention_heads": 80,
        "num_key_value_heads": N_KV_HEAD,
        "num_noise_heads": N_NOISE_HEAD,
        "head_dim": 192,
        "v_head_dim": 128,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 1024,
        "kv_lora_rank": 512,
        "mhc_enabled": True,
        "mhc_expansion_rate": 4,
        "mhc_sinkhorn_iters": 20,
        "score_func": "sigmoid",
        "route_norm": True,
        "route_scale": 2,
        "sliding_window": SWA_WINDOW,
        "sliding_window_period": 4,
        "max_position_embeddings": MAX_CONTEXT,
        "num_nextn_predict_layers": 1,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            errors.append(f"config mismatch {key}: {config.get(key)!r} != {value!r}")
    if config.get("hidden_act") != "poly_norm":
        errors.append(f"hidden_act is not poly_norm: {config.get('hidden_act')!r}")
    if config.get("attention_cls") != "gdla":
        errors.append(f"attention_cls is not gdla: {config.get('attention_cls')!r}")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    header_manifest = json.loads(args.headers.read_text())
    tensors = header_manifest["tensors"]
    errors: list[str] = []

    if header_manifest.get("repo") != SOURCE_REPO:
        errors.append(f"source repo mismatch: {header_manifest.get('repo')}")
    if header_manifest.get("revision") != SOURCE_REVISION:
        errors.append(f"source revision mismatch: {header_manifest.get('revision')}")
    if header_manifest.get("shard_count") != EXPECTED_SHARDS:
        errors.append(f"shard count mismatch: {header_manifest.get('shard_count')}")
    if len(tensors) != EXPECTED_TENSORS:
        errors.append(f"tensor count mismatch: {len(tensors)} != {EXPECTED_TENSORS}")

    validate_config(config, errors)
    expected = expected_tensor_shapes()
    if len(expected) != EXPECTED_TENSORS:
        errors.append(f"internal expected tensor count: {len(expected)} != {EXPECTED_TENSORS}")
    for name, shape in expected.items():
        require_tensor(tensors, name, shape, errors)
    extras = sorted(set(tensors) - set(expected))
    if extras:
        errors.append(f"unexpected tensors: {len(extras)}; first={extras[:10]}")

    inventory: list[dict] = []
    family_counts: collections.Counter[str] = collections.Counter()
    source_dtype_counts: collections.Counter[str] = collections.Counter()
    parameters = 0
    for name, metadata in sorted(tensors.items()):
        try:
            family, target_type, layer = target_for(name)
        except KeyError as exc:
            errors.append(str(exc))
            continue
        elements = tensor_elements(metadata["shape"])
        parameters += elements
        family_counts[family] += 1
        source_dtype_counts[metadata["dtype"]] += 1
        inventory.append(
            {
                "name": name,
                "shape": metadata["shape"],
                "source_dtype": metadata["dtype"],
                "parameters": elements,
                "shard": metadata["shard"],
                "family": family,
                "layer": layer,
                "target_type": target_type,
            }
        )

    if parameters != EXPECTED_PARAMETERS:
        errors.append(f"parameter count mismatch: {parameters} != {EXPECTED_PARAMETERS}")
    if source_dtype_counts != collections.Counter({"BF16": EXPECTED_TENSORS}):
        errors.append(f"source dtype histogram mismatch: {dict(source_dtype_counts)}")

    projection = project(inventory) if len(inventory) == EXPECTED_TENSORS else None
    architecture = {
        "schema_version": 1,
        "source": {"repo": SOURCE_REPO, "revision": SOURCE_REVISION},
        "config_sha256": sha256(args.config),
        "tensor_header_sha256": sha256(args.headers),
        "topology_sha256": topology_digest(tensors),
        "architecture": "motif3",
        "parameters": parameters,
        "tensor_count": len(tensors),
        "layers": N_LAYER,
        "dense_layers": [0, 1],
        "sparse_layers": list(range(2, N_LAYER)),
        "attention": {
            "kind": "GDLA",
            "full_layers": list(range(0, N_LAYER, 4)),
            "sliding_layers": [i for i in range(N_LAYER) if i % 4 != 0],
            "sliding_window": SWA_WINDOW,
            "query_heads": 80,
            "signal_heads": N_SIGNAL_HEAD,
            "noise_heads": N_NOISE_HEAD,
            "kv_heads": N_KV_HEAD,
            "qk_head_dim": 192,
            "rope_head_dim": 64,
            "value_head_dim": 128,
            "q_lora_rank": 1024,
            "kv_lora_rank": 512,
            "input_dependent_lambda": True,
            "elementwise_output_gate": True,
            "persistent_full_history": "kv_latent + RoPE k_pe only",
        },
        "moe": {
            "routed_experts": N_EXPERT,
            "top_k": N_EXPERT_USED,
            "shared_experts": 1,
            "expert_width": ROUTED_FF,
            "router": "sigmoid + top-k + normalized weights * route_scale(2)",
            "activation": "Expert-Specific PolyNorm",
        },
        "mhc": {
            "streams": 4,
            "sinkhorn_iterations": 20,
            "fp32_projection_accumulation_and_mappings": True,
        },
        "mtp_layers": 1,
        "max_context": MAX_CONTEXT,
        "family_tensor_counts": dict(sorted(family_counts.items())),
        "source_dtype_counts": dict(sorted(source_dtype_counts.items())),
        "validation": {"ok": not errors, "errors": errors},
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "architecture.json": architecture,
        "tensor-inventory.json": {
            "schema_version": 1,
            "source_revision": SOURCE_REVISION,
            "tensor_count": len(inventory),
            "tensors": inventory,
        },
    }
    if projection is not None:
        outputs["projection-mq87-fit.json"] = projection
    for filename, data in outputs.items():
        path = args.out_dir / filename
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        print(f"wrote {path}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    assert projection is not None
    print(f"validated parameters: {parameters:,}")
    print(f"validated tensors: {len(tensors):,}")
    print(f"MQ87-88 projected artifact: {projection['estimated_artifact_gib']:.6f} GiB")
    print(f"256K latent cache projection: {projection['latent_cache_256k']['total_gib']:.6f} GiB")


if __name__ == "__main__":
    main()
