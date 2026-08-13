from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_inventory", ROOT / "converter" / "src" / "build_inventory.py"
)
assert SPEC and SPEC.loader
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


def test_locked_tensor_policy() -> None:
    cases = {
        "model.embed_tokens.weight": ("embedding", "Q8_0", None),
        "model.layers.0.mlp.gate_proj.weight": ("dense_mlp", "Q8_0", 0),
        "model.layers.2.moe.router.gate.weight": ("router", "F32", 2),
        "model.layers.2.moe.experts.gate_up_proj": ("routed_gate_up", "IQ2_XXS", 2),
        "model.layers.52.moe.experts.down_proj": ("routed_down", "Q2_K", 52),
        "model.layers.12.mhc_attn.proj_res.weight": ("mhc_small_projection", "BF16", 12),
        "model.layers.12.mhc_attn.alpha_res": ("mhc_control", "F32", 12),
        "model.mtp_layers.0.input_proj.weight": ("mtp_projection", "Q8_0", None),
    }
    for name, expected in cases.items():
        assert inventory.target_for(name) == expected


def test_expected_topology_count() -> None:
    assert len(inventory.expected_tensor_shapes()) == inventory.EXPECTED_TENSORS


def test_256k_latent_cache_is_under_preferred_budget() -> None:
    projection = inventory.cache_projection()
    assert len(projection["full_attention_layers"]) == 14
    assert len(projection["sliding_window_layers"]) == 39
    assert projection["total_gib"] < 5


def test_exact_locked_payload_when_headers_are_present() -> None:
    headers_path = ROOT / "manifests" / "safetensors-headers.json"
    if not headers_path.exists():
        return
    import json

    tensors = json.loads(headers_path.read_text())["tensors"]
    rows = []
    for name, metadata in tensors.items():
        family, target_type, layer = inventory.target_for(name)
        rows.append(
            {
                "name": name,
                "shape": metadata["shape"],
                "family": family,
                "target_type": target_type,
                "layer": layer,
            }
        )
    projection = inventory.project(rows)
    assert projection["raw_tensor_bytes"] == 94_152_882_712
    assert 87 <= projection["estimated_artifact_gib"] <= 88
