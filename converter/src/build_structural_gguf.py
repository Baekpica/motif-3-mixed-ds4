#!/usr/bin/env python3
"""Write a metadata-only Motif 3 GGUF template for the DS4 quantizer.

The template records final tensor names, shapes, and quant types but no tensor
payloads. The DS4 C quantizer regenerates every payload from the pinned
safetensors. A source map records the exact origin of every GGUF tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

SOURCE_REPO = "Motif-Technologies/Motif-3"
SOURCE_REVISION = "ccceb1a5fd7b5eb32e47841216b3caf5666c07bc"
EXPECTED_SOURCE_TENSORS = 2_236
EXPECTED_GGUF_TENSORS = 2_287
ALIGNMENT = 32

VARIANTS = {
    "mq87": {
        "name": "Motif-3 MQ87-88 FIT",
        "quantization": "MQ87-88-FIT",
    },
    "q8-reference": {
        "name": "Motif-3 Q8_0 Reference",
        "quantization": "Q8_0-REFERENCE",
    },
}

TYPE_LAYOUT = {
    "F32": (1, 4),
    "BF16": (1, 2),
    "Q8_0": (32, 34),
    "IQ2_XXS": (256, 66),
    "Q2_K": (256, 84),
}

ATTENTION_MAP = {
    "q_norm.weight": "attn_q_a_norm.weight",
    "kv_norm.weight": "attn_kv_a_norm.weight",
    "wq_a.weight": "attn_q_a.weight",
    "wq_b.weight": "attn_q_b.weight",
    "wq_b_gate.weight": "attn_q_gate.weight",
    "wkv_a.weight": "attn_kv_a.weight",
    "wkv_b.weight": "attn_kv_b.weight",
    "lambda_proj.weight": "attn_lambda.weight",
    "wo.weight": "attn_output.weight",
}

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-map", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        default="mq87",
        help="payload policy and public metadata for the generated template",
    )
    parser.add_argument(
        "--gguf-py",
        type=Path,
        default=Path(os.environ.get("GGUF_PY_DIR", "/motif3/llama.cpp/gguf-py")),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(shape: list[int], quant_type: str) -> int:
    block, type_size = TYPE_LAYOUT[quant_type]
    elements = math.prod(shape)
    if elements % block:
        raise ValueError(
            f"{quant_type} element count {elements} is not divisible by {block}: {shape}"
        )
    return elements // block * type_size


def aligned(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def map_layer_tensor(layer: int, suffix: str) -> str:
    prefix = f"blk.{layer}."
    if suffix == "input_layernorm.weight":
        return prefix + "attn_norm.weight"
    if suffix == "post_attention_layernorm.weight":
        return prefix + "ffn_norm.weight"
    if suffix.startswith("self_attn."):
        key = suffix.removeprefix("self_attn.")
        return prefix + ATTENTION_MAP[key]
    if suffix.startswith("mhc_attn.") or suffix.startswith("mhc_ffn."):
        return prefix + suffix
    if suffix.startswith("mlp."):
        key = suffix.removeprefix("mlp.")
        dense_map = {
            "gate_proj.weight": "ffn_gate.weight",
            "up_proj.weight": "ffn_up.weight",
            "down_proj.weight": "ffn_down.weight",
            "act_fn.weight": "ffn_polynorm.weight",
            "act_fn.bias": "ffn_polynorm.bias",
        }
        return prefix + dense_map[key]
    if suffix == "moe.router.gate.weight":
        return prefix + "ffn_gate_inp.weight"
    if suffix == "moe.expert_bias":
        return prefix + "exp_probs_b.bias"
    if suffix == "moe.experts.down_proj":
        return prefix + "ffn_down_exps.weight"
    if suffix.startswith("moe.experts.act_fn."):
        field = suffix.removeprefix("moe.experts.act_fn.")
        return prefix + f"ffn_polynorm_exps.{field}"
    if suffix.startswith("moe.shared_experts."):
        field = suffix.removeprefix("moe.shared_experts.")
        shared_map = {
            "gate_proj.weight": "ffn_gate_shexp.weight",
            "up_proj.weight": "ffn_up_shexp.weight",
            "down_proj.weight": "ffn_down_shexp.weight",
            "act_fn.weight": "ffn_polynorm_shexp.weight",
            "act_fn.bias": "ffn_polynorm_shexp.bias",
        }
        return prefix + shared_map[field]
    raise KeyError(f"unmapped layer tensor: {layer}: {suffix}")


def map_regular_tensor(source_name: str) -> str:
    top = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }
    if source_name in top:
        return top[source_name]
    if source_name.startswith("model.mtp_layers.0."):
        suffix = source_name.removeprefix("model.mtp_layers.0.")
        if suffix.startswith("self_attn."):
            key = suffix.removeprefix("self_attn.")
            return "mtp.0." + ATTENTION_MAP[key]
        if suffix.startswith("mlp."):
            key = suffix.removeprefix("mlp.")
            dense_map = {
                "gate_proj.weight": "ffn_gate.weight",
                "up_proj.weight": "ffn_up.weight",
                "down_proj.weight": "ffn_down.weight",
                "act_fn.weight": "ffn_polynorm.weight",
                "act_fn.bias": "ffn_polynorm.bias",
            }
            return "mtp.0." + dense_map[key]
        return "mtp.0." + suffix
    match = LAYER_RE.match(source_name)
    if not match:
        raise KeyError(f"unmapped source tensor: {source_name}")
    return map_layer_tensor(int(match.group(1)), match.group(2))


def variant_target_type(source_type: str, variant: str) -> str:
    if variant == "mq87":
        return source_type
    if variant == "q8-reference":
        # Control/state tensors stay F32 and Motif mHC projections retain
        # checkpoint BF16. Every weight that is quantized in MQ87 is Q8_0 in
        # this bring-up/imatrix reference artifact.
        if source_type in {"F32", "BF16"}:
            return source_type
        return "Q8_0"
    raise ValueError(f"unknown template variant: {variant}")


def build_tensor_map(inventory: dict[str, Any], variant: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tensor in inventory["tensors"]:
        source_name = tensor["name"]
        source_shape = tensor["shape"]
        if source_name.endswith(".moe.experts.gate_up_proj"):
            match = LAYER_RE.match(source_name)
            assert match is not None
            layer = int(match.group(1))
            if source_shape != [384, 2560, 4096]:
                raise ValueError(f"unexpected fused expert shape: {source_name}: {source_shape}")
            for projection, start, stop in (
                ("gate", 0, 1280),
                ("up", 1280, 2560),
            ):
                shape = [384, 1280, 4096]
                target_type = variant_target_type(tensor["target_type"], variant)
                rows.append(
                    {
                        "name": f"blk.{layer}.ffn_{projection}_exps.weight",
                        "shape": shape,
                        "target_type": target_type,
                        "family": tensor["family"],
                        "source_name": source_name,
                        "source_shape": source_shape,
                        "source_slice": {"axis": 1, "start": start, "stop": stop},
                        "source_shard": tensor["shard"],
                        "bytes": tensor_bytes(shape, target_type),
                    }
                )
            continue

        target_type = variant_target_type(tensor["target_type"], variant)
        rows.append(
            {
                "name": map_regular_tensor(source_name),
                "shape": source_shape,
                "target_type": target_type,
                "family": tensor["family"],
                "source_name": source_name,
                "source_shape": source_shape,
                "source_slice": None,
                "source_shard": tensor["shard"],
                "bytes": tensor_bytes(source_shape, target_type),
            }
        )
    names = [row["name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("duplicate GGUF tensor names")
    if len(rows) != EXPECTED_GGUF_TENSORS:
        raise ValueError(f"GGUF tensor count {len(rows)} != {EXPECTED_GGUF_TENSORS}")
    return rows


def add_model_metadata(
    writer: Any, config: dict[str, Any], model_dir: Path, variant: str
) -> None:
    rope = config["rope_scaling"]
    u32 = writer.add_uint32
    f32 = writer.add_float32
    boolean = writer.add_bool
    string = writer.add_string

    variant_meta = VARIANTS[variant]
    string("general.name", variant_meta["name"])
    string("general.type", "model")
    string("general.source.url", f"https://huggingface.co/{SOURCE_REPO}")
    string("general.source.revision", SOURCE_REVISION)
    string("general.quantization", variant_meta["quantization"])
    u32("general.quantization_version", 2)
    boolean("motif3.template_only", True)

    u32("motif3.block_count", config["num_hidden_layers"])
    writer.add_uint64("motif3.context_length", config["max_position_embeddings"])
    u32("motif3.embedding_length", config["hidden_size"])
    u32("motif3.vocab_size", config["vocab_size"])
    u32("motif3.feed_forward_length", config["intermediate_size"])
    u32("motif3.leading_dense_block_count", config["n_dense_first_layers"])
    u32("motif3.expert_count", config["num_experts"])
    u32("motif3.expert_used_count", config["experts_top_k"])
    u32("motif3.expert_feed_forward_length", config["moe_intermediate_size"])
    u32("motif3.expert_shared_count", config["num_shared_experts"])
    u32("motif3.expert_gating_func", 1)
    boolean("motif3.expert_weights_norm", config["route_norm"])
    f32("motif3.expert_weights_scale", config["route_scale"])
    f32("motif3.expert_score_correction", config["load_balance_coeff"])

    u32("motif3.attention.head_count", config["num_attention_heads"])
    u32("motif3.attention.head_count_kv", config["num_key_value_heads"])
    u32("motif3.attention.noise_head_count", config["num_noise_heads"])
    u32("motif3.attention.key_length", config["head_dim"])
    u32("motif3.attention.value_length", config["v_head_dim"])
    u32("motif3.attention.q_lora_rank", config["q_lora_rank"])
    u32("motif3.attention.kv_lora_rank", config["kv_lora_rank"])
    u32("motif3.attention.rope_dimension_count", config["qk_rope_head_dim"])
    u32("motif3.attention.sliding_window", config["sliding_window"])
    u32("motif3.attention.sliding_window_period", config["sliding_window_period"])
    string("motif3.attention.sliding_window_pattern", config["sliding_window_pattern"])
    boolean("motif3.attention.elementwise_output_gate", config["elementwise_attn_output_gate"])
    f32("motif3.attention.layer_norm_rms_epsilon", config["rms_norm_eps"])

    string("motif3.rope.scaling.type", rope["rope_type"])
    f32("motif3.rope.freq_base", rope["rope_theta"])
    f32("motif3.rope.freq_base_swa", config["swa_rope_theta"])
    f32("motif3.rope.scaling.factor", rope["factor"])
    u32("motif3.rope.scaling.original_context_length", rope["original_max_position_embeddings"])
    f32("motif3.rope.scaling.beta_fast", rope["beta_fast"])
    f32("motif3.rope.scaling.beta_slow", rope["beta_slow"])
    f32("motif3.rope.scaling.mscale", rope["mscale"])
    boolean("motif3.rope.scaling.apply_mscale", rope["apply_yarn_scaling"])

    boolean("motif3.mhc.enabled", config["mhc_enabled"])
    u32("motif3.mhc.expansion_rate", config["mhc_expansion_rate"])
    u32("motif3.mhc.sinkhorn_iterations", config["mhc_sinkhorn_iters"])
    boolean("motif3.mhc.identity_init", config["mhc_identity_init"])
    f32("motif3.mhc.h_post_coefficient", 1.0)
    string("motif3.activation", config["hidden_act"])
    f32("motif3.polynorm.output_scale", config["polynorm_output_scale"])
    f32("motif3.polynorm.bias_clamp", config["polynorm_bias_clamp"])
    boolean("motif3.polynorm.sigmoid_weight", True)
    f32("motif3.hidden_clamp", config["hidden_clamp"])
    u32("motif3.mtp.block_count", config["num_nextn_predict_layers"])

    for filename, key in (
        ("config.json", "motif3.source.config_sha256"),
        ("modeling_motif.py", "motif3.source.modeling_sha256"),
        ("tokenizer.json", "motif3.source.tokenizer_sha256"),
        ("chat_template.jinja", "motif3.source.chat_template_sha256"),
        ("generation_config.json", "motif3.source.generation_config_sha256"),
    ):
        string(key, sha256(model_dir / filename))


def add_tokenizer_metadata(writer: Any, model_dir: Path) -> None:
    from gguf import TokenType
    from transformers import AutoTokenizer

    tokenizer_json = json.loads((model_dir / "tokenizer.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )
    vocab_size = tokenizer.vocab_size
    reverse_vocab = {token_id: token for token, token_id in tokenizer.vocab.items()}
    added = {row["id"]: row for row in tokenizer_json["added_tokens"]}
    tokens: list[str] = []
    token_types: list[int] = []
    for token_id in range(vocab_size):
        token = reverse_vocab.get(token_id)
        if token is None:
            tokens.append(f"[PAD{token_id}]")
            token_types.append(TokenType.UNUSED)
            continue
        tokens.append(token)
        if token_id in added:
            token_types.append(
                TokenType.CONTROL if added[token_id]["special"] else TokenType.USER_DEFINED
            )
        else:
            token_types.append(TokenType.NORMAL)

    merges = [" ".join(pair) for pair in tokenizer_json["model"]["merges"]]
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("motif3")
    writer.add_token_list(tokens)
    writer.add_token_types(token_types)
    writer.add_token_merges(merges)
    writer.add_bos_token_id(tokenizer.bos_token_id)
    writer.add_eos_token_id(tokenizer.eos_token_id)
    writer.add_pad_token_id(tokenizer.pad_token_id)
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)
    writer.add_chat_template((model_dir / "chat_template.jinja").read_text())
    writer.add_uint32("tokenizer.ggml.start_of_turn_token_id", 5)
    writer.add_uint32("tokenizer.ggml.end_of_turn_token_id", 6)
    writer.add_uint32("tokenizer.ggml.tool_token_id", 7)
    writer.add_uint32("tokenizer.ggml.think_token_id", 11)
    writer.add_uint32("tokenizer.ggml.end_think_token_id", 12)


def main() -> None:
    args = parse_args()
    if not args.gguf_py.is_dir():
        raise SystemExit(f"gguf-py not found: {args.gguf_py}")
    sys.path.insert(0, str(args.gguf_py))
    from gguf import GGMLQuantizationType, GGUFWriter

    inventory = json.loads(args.inventory.read_text())
    if inventory["tensor_count"] != EXPECTED_SOURCE_TENSORS:
        raise RuntimeError("source inventory tensor count mismatch")
    if inventory["source_revision"] != SOURCE_REVISION:
        raise RuntimeError("source inventory revision mismatch")
    config = json.loads((args.model_dir / "config.json").read_text())
    rows = build_tensor_map(inventory, args.variant)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(args.out, "motif3")
    writer.add_custom_alignment(ALIGNMENT)
    add_model_metadata(writer, config, args.model_dir, args.variant)
    add_tokenizer_metadata(writer, args.model_dir)
    for row in rows:
        raw_type = GGMLQuantizationType[row["target_type"]]
        writer.add_tensor_info(
            row["name"],
            row["shape"],
            np.dtype(np.float32),
            row["bytes"],
            raw_dtype=raw_type,
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    writer.close()

    header_bytes = args.out.stat().st_size
    header_sha256 = sha256(args.out)
    logical_bytes = aligned(header_bytes) + sum(aligned(row["bytes"]) for row in rows)
    os.truncate(args.out, logical_bytes)

    source_map = {
        "schema_version": 1,
        "source_model": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "variant": args.variant,
        "general_name": VARIANTS[args.variant]["name"],
        "general_quantization": VARIANTS[args.variant]["quantization"],
        "source_tensor_count": inventory["tensor_count"],
        "gguf_tensor_count": len(rows),
        "gate_up_physical_transform": (
            "Each fused [384,2560,4096] source tensor is sliced into gate and up "
            "[384,1280,4096] GGUF tensors. All 384 experts and every value remain present."
        ),
        "payload_bytes": sum(row["bytes"] for row in rows),
        "template_header_bytes": header_bytes,
        "template_header_sha256": header_sha256,
        "template_logical_bytes": logical_bytes,
        "template_sparse_zero_payload": True,
        "tensors": rows,
    }
    args.source_map.parent.mkdir(parents=True, exist_ok=True)
    args.source_map.write_text(
        json.dumps(source_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote metadata-only template {args.out}: {len(rows)} tensors, "
        f"{header_bytes / 2**20:.2f} MiB metadata, "
        f"{logical_bytes / 2**30:.6f} GiB sparse logical size"
    )
    print(
        f"planned payload {source_map['payload_bytes'] / 2**30:.6f} GiB; "
        f"source map -> {args.source_map}"
    )


if __name__ == "__main__":
    main()
