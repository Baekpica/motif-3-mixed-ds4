#!/usr/bin/env python3
"""Numerically sample a completed Motif-3 GGUF against pinned BF16 rows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader
from gguf.quants import dequantize
from safetensors import safe_open


SAMPLES: dict[str, list[tuple[int, ...]]] = {
    "token_embd.weight": [(0,), (12345,), (220159,)],
    "output.weight": [(0,), (12345,), (220159,)],
    "blk.0.attn_q_a.weight": [(0,), (511,), (1023,)],
    "blk.1.ffn_gate.weight": [(0,), (6143,), (12287,)],
    "blk.2.mhc_attn.proj_pre.weight": [(0,), (1,), (3,)],
    "blk.2.ffn_gate_inp.weight": [(0,), (191,), (383,)],
    "blk.2.ffn_polynorm_exps.weight": [(0,), (173,), (383,)],
    "blk.2.ffn_gate_shexp.weight": [(0,), (639,), (1279,)],
    "blk.2.ffn_gate_exps.weight": [(0, 0), (173, 639), (383, 1279)],
    "blk.2.ffn_up_exps.weight": [(0, 0), (173, 639), (383, 1279)],
    "blk.2.ffn_down_exps.weight": [(0, 0), (173, 2047), (383, 4095)],
    "blk.26.ffn_gate_exps.weight": [(0, 1279), (173, 0), (383, 639)],
    "blk.26.ffn_up_exps.weight": [(0, 1279), (173, 0), (383, 639)],
    "blk.26.ffn_down_exps.weight": [(0, 4095), (173, 0), (383, 2047)],
    "blk.52.ffn_gate_exps.weight": [(0, 639), (173, 1279), (383, 0)],
    "blk.52.ffn_up_exps.weight": [(0, 639), (173, 1279), (383, 0)],
    "blk.52.ffn_down_exps.weight": [(0, 2047), (173, 4095), (383, 0)],
    "mtp.0.attn_q_a.weight": [(0,), (511,), (1023,)],
    "mtp.0.ffn_gate.weight": [(0,), (6143,), (12287,)],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def source_row(source_root: Path, row: dict, prefix: tuple[int, ...]) -> np.ndarray:
    path = source_root / row["source_shard"]
    with safe_open(path, framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(row["source_name"])
        logical = list(prefix)
        source_slice = row.get("source_slice")
        if source_slice is not None:
            axis = int(source_slice["axis"])
            if axis >= len(logical):
                raise ValueError(f"slice axis is not in sampled prefix: {row['name']}")
            logical[axis] += int(source_slice["start"])
        value = tensor_slice[tuple(logical) + (slice(None),)]
        return value.float().numpy()


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict:
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    actual = np.asarray(actual, dtype=np.float32).reshape(-1)
    if reference.shape != actual.shape:
        raise ValueError(f"row shape differs: {reference.shape} != {actual.shape}")
    delta = actual - reference
    ref_rms = float(np.sqrt(np.mean(reference * reference, dtype=np.float64)))
    rmse = float(np.sqrt(np.mean(delta * delta, dtype=np.float64)))
    denominator = float(np.linalg.norm(reference) * np.linalg.norm(actual))
    cosine = float(np.dot(reference, actual) / denominator) if denominator else 1.0
    return {
        "elements": int(reference.size),
        "reference_rms": ref_rms,
        "actual_rms": float(np.sqrt(np.mean(actual * actual, dtype=np.float64))),
        "rmse": rmse,
        "relative_rmse": rmse / ref_rms if ref_rms else 0.0,
        "cosine": cosine,
        "max_abs": float(np.max(np.abs(delta))),
        "finite": bool(np.isfinite(actual).all()),
        "nonzero": bool(np.any(actual != 0.0)),
    }


def main() -> int:
    args = parse_args()
    source_map = json.loads(args.source_map.read_text(encoding="utf-8"))
    rows = {item["name"]: item for item in source_map["tensors"]}
    reader = GGUFReader(args.gguf, "r")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    errors: list[str] = []
    results: list[dict] = []

    for name, prefixes in SAMPLES.items():
        row = rows.get(name)
        tensor = tensors.get(name)
        if row is None or tensor is None:
            errors.append(f"missing sample tensor: {name}")
            continue
        qtype = GGMLQuantizationType(tensor.tensor_type)
        if qtype.name != row["target_type"]:
            errors.append(f"{name}: {qtype.name} != {row['target_type']}")
            continue
        for prefix in prefixes:
            try:
                packed = np.asarray(tensor.data[prefix])
                actual = dequantize(packed, qtype)
                reference = source_row(args.source, row, prefix)
                sample_metrics = metrics(reference, actual)
            except (IndexError, KeyError, OSError, RuntimeError, ValueError) as exc:
                errors.append(f"{name}{prefix}: {exc}")
                continue
            if not sample_metrics["finite"] or not sample_metrics["nonzero"]:
                errors.append(f"{name}{prefix}: non-finite or all-zero dequantized row")
            if qtype in (GGMLQuantizationType.F32, GGMLQuantizationType.BF16):
                if sample_metrics["max_abs"] != 0.0:
                    errors.append(f"{name}{prefix}: protected row is not source-exact")
            elif qtype == GGMLQuantizationType.Q8_0:
                if not math.isfinite(sample_metrics["cosine"]) or sample_metrics["cosine"] < 0.98:
                    errors.append(f"{name}{prefix}: Q8 cosine below sanity floor")
            elif qtype in (GGMLQuantizationType.IQ2_XXS, GGMLQuantizationType.Q2_K):
                if not math.isfinite(sample_metrics["cosine"]) or sample_metrics["cosine"] < 0.25:
                    errors.append(f"{name}{prefix}: low-bit cosine below corruption floor")
            results.append(
                {
                    "tensor": name,
                    "prefix": list(prefix),
                    "type": qtype.name,
                    **sample_metrics,
                }
            )

    report = {
        "status": "ok" if not errors else "error",
        "source_model": source_map["source_model"],
        "source_revision": source_map["source_revision"],
        "gguf": str(args.gguf),
        "sample_count": len(results),
        "samples": results,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    types = sorted({sample["type"] for sample in results})
    summary = {
        "status": report["status"],
        "sample_count": len(results),
        "min_cosine_by_type": {
            qtype: min(
                sample["cosine"] for sample in results if sample["type"] == qtype
            )
            for qtype in types
        },
        "max_relative_rmse_by_type": {
            qtype: max(
                sample["relative_rmse"]
                for sample in results
                if sample["type"] == qtype
            )
            for qtype in types
        },
        "errors": errors,
        "report": str(args.output),
    }
    print(json.dumps(summary, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
