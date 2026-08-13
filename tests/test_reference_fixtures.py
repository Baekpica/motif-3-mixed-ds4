from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "official-final"
PINNED_REVISION = "ccceb1a5fd7b5eb32e47841216b3caf5666c07bc"


def _manifest() -> dict:
    return json.loads((FIXTURES / "manifest.json").read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixture_source_and_payload_hashes_are_pinned() -> None:
    manifest = _manifest()
    assert manifest["source_policy"] == "official-final-only"
    assert manifest["source"]["revision"] == PINNED_REVISION
    for name, metadata in manifest["fixtures"].items():
        assert _sha256(FIXTURES / name) == metadata["sha256"]
        runtime = metadata["runtime_fixture"]
        assert _sha256(FIXTURES / runtime["name"]) == runtime["sha256"]


def test_router_uses_unbiased_scores_as_normalized_weights() -> None:
    values = np.load(FIXTURES / "router-layer2.npz")
    scores = 1.0 / (1.0 + np.exp(-values["logits"]))
    selected = values["selected_experts"]
    selected_scores = np.take_along_axis(scores, selected, axis=1)
    expected = selected_scores / selected_scores.sum(axis=1, keepdims=True) * 2.0
    np.testing.assert_allclose(values["route_weights"], expected, rtol=2e-6, atol=2e-7)
    adjusted = scores + values["expert_bias"]
    chosen = np.take_along_axis(adjusted, selected, axis=1)
    threshold = np.partition(adjusted, -8, axis=1)[:, -8]
    assert np.all(chosen >= threshold[:, None])


def test_expert_polynorm_fp32_contract() -> None:
    values = np.load(FIXTURES / "polynorm-layer2-expert173.npz")
    gate = values["gate"].astype(np.float32)
    up = values["up"].astype(np.float32)
    coeff = 1.0 / (1.0 + np.exp(-values["raw_coeff"].astype(np.float32)))
    bias = np.clip(values["raw_bias"].astype(np.float32), -0.5, 0.5)
    g2 = gate * gate
    g3 = g2 * gate
    poly = (
        coeff[0] * g3 / np.sqrt(np.mean(g3 * g3, axis=-1, keepdims=True) + 1e-6)
        + coeff[1] * g2 / np.sqrt(np.mean(g2 * g2, axis=-1, keepdims=True) + 1e-6)
        + coeff[2] * gate / np.sqrt(np.mean(g2, axis=-1, keepdims=True) + 1e-6)
        + bias
    )
    expected = poly * up * 0.5
    np.testing.assert_allclose(values["activated_fp32"], expected, rtol=2e-5, atol=2e-5)


def test_mhc_sigmoid_sinkhorn_and_mix_contract() -> None:
    values = np.load(FIXTURES / "mhc-layer0-attn.npz")
    expected_pre = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                values["alpha_pre"] * values["projected_pre"] + values["bias_pre"],
                -10.0,
                10.0,
            )
        )
    )
    expected_post = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                values["alpha_post"] * values["projected_post"] + values["bias_post"],
                -10.0,
                10.0,
            )
        )
    )
    np.testing.assert_allclose(values["h_pre"], expected_pre, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(values["h_post"], expected_post, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(values["h_res"].sum(axis=-1), 1.0, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(values["h_res"].sum(axis=-2), 1.0, rtol=2e-5, atol=2e-5)

    reduced = (values["hidden"] * values["h_pre"][..., None]).sum(axis=2)
    mixed = np.einsum("btij,btjd->btid", values["h_res"], values["hidden"])
    np.testing.assert_allclose(values["reduced_input"], reduced, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(values["residual_mixed"], mixed, rtol=2e-5, atol=2e-5)


def test_gdla_expanded_fixture_locks_long_rope_and_differential_heads() -> None:
    values = np.load(FIXTURES / "gdla-expanded-layer0.npz")
    assert values["q_full"].shape == (8, 80, 192)
    assert values["k_full"].shape == (8, 16, 192)
    assert values["value"].shape == (8, 16, 128)
    assert values["diff_attention_fp32"].shape == (8, 64, 128)
    np.testing.assert_array_equal(
        values["probe_positions"],
        np.asarray([0, 1, 127, 128, 4095, 65535, 131071, 262143], dtype=np.int32),
    )
    assert np.isfinite(values["q_pe_probe_fp32"]).all()
    assert np.isfinite(values["output_bf16"]).all()
