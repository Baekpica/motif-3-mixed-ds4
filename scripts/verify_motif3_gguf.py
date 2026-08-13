#!/usr/bin/env python3
"""Verify an unsharded or split Motif-3 GGUF against its locked source map."""

from __future__ import annotations

import argparse
import collections
import gc
import json
import re
import sys
from pathlib import Path

from gguf import GGMLQuantizationType, GGUFReader


SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$")
EXPERT_RE = re.compile(r"^blk\.(?P<layer>\d+)\.ffn_(?P<proj>gate|up|down)_exps\.weight$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf", type=Path, help="single file, first shard, or shard directory")
    parser.add_argument("--source-map", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def field(reader: GGUFReader, key: str):
    item = reader.fields.get(key)
    return None if item is None else item.contents()


def resolve_shards(path: Path) -> list[Path]:
    candidates = sorted(path.glob("*.gguf")) if path.is_dir() else [path]
    if not candidates or not candidates[0].is_file():
        raise ValueError(f"no GGUF found at {path}")
    match = SHARD_RE.match(candidates[0].name)
    if match is None:
        if len(candidates) != 1:
            raise ValueError("multiple files found but the first is not split-shard named")
        return candidates
    count = int(match.group("count"))
    stem = match.group("stem")
    shards = [
        candidates[0].parent / f"{stem}-{index:05d}-of-{count:05d}.gguf"
        for index in range(1, count + 1)
    ]
    missing = [str(item) for item in shards if not item.is_file()]
    if missing:
        raise ValueError(f"missing {len(missing)} shard(s): {missing[:3]}")
    return shards


def verify(path: Path, source_map_path: Path, variant: str) -> dict:
    source_map = json.loads(source_map_path.read_text(encoding="utf-8"))
    if source_map.get("variant") != variant:
        raise ValueError(
            f"source-map variant {source_map.get('variant')!r} != {variant!r}"
        )
    expected = {row["name"]: row for row in source_map["tensors"]}
    shards = resolve_shards(path)
    errors: list[str] = []
    seen: set[str] = set()
    type_counts: collections.Counter[str] = collections.Counter()
    total_file_bytes = 0
    total_tensor_bytes = 0

    for shard_index, shard in enumerate(shards):
        size = shard.stat().st_size
        total_file_bytes += size
        reader = GGUFReader(shard, "r")
        if shard_index == 0:
            checks = {
                "general.architecture": "motif3",
                "general.source.revision": source_map["source_revision"],
                "motif3.block_count": 53,
                "motif3.context_length": 262_144,
                "motif3.expert_count": 384,
                "motif3.expert_used_count": 8,
                "motif3.mtp.block_count": 1,
            }
            for key, wanted in checks.items():
                actual = field(reader, key)
                if actual != wanted:
                    errors.append(f"{shard.name}: {key}={actual!r}, expected {wanted!r}")
            if "motif3.template_only" in reader.fields:
                errors.append(f"{shard.name}: final artifact retains motif3.template_only")
        if len(shards) > 1:
            if field(reader, "split.no") != shard_index:
                errors.append(f"{shard.name}: split.no mismatch")
            if field(reader, "split.count") != len(shards):
                errors.append(f"{shard.name}: split.count mismatch")
            if field(reader, "split.tensors.count") != len(expected):
                errors.append(f"{shard.name}: split.tensors.count mismatch")

        for tensor in reader.tensors:
            row = expected.get(tensor.name)
            if row is None:
                errors.append(f"unexpected tensor: {tensor.name}")
                continue
            if tensor.name in seen:
                errors.append(f"duplicate tensor: {tensor.name}")
            seen.add(tensor.name)
            actual_type = GGMLQuantizationType(tensor.tensor_type).name
            if actual_type != row["target_type"]:
                errors.append(
                    f"{tensor.name}: type {actual_type} != {row['target_type']}"
                )
            actual_shape = tensor.shape.tolist()
            expected_shape = list(reversed(row["shape"]))
            if actual_shape != expected_shape:
                errors.append(
                    f"{tensor.name}: shape {actual_shape} != {expected_shape}"
                )
            if tensor.n_bytes != row["bytes"]:
                errors.append(
                    f"{tensor.name}: bytes {tensor.n_bytes} != {row['bytes']}"
                )
            if tensor.data_offset + tensor.n_bytes > size:
                errors.append(f"{shard.name}: tensor exceeds bounds: {tensor.name}")
            type_counts[actual_type] += 1
            total_tensor_bytes += tensor.n_bytes
        del reader
        gc.collect()

    missing = sorted(set(expected) - seen)
    if missing:
        errors.append(f"missing {len(missing)} tensor(s): {missing[:8]}")
    if len(seen) != len(expected):
        errors.append(f"tensor count {len(seen)} != {len(expected)}")

    sparse_layers: dict[str, set[int]] = collections.defaultdict(set)
    for name in seen:
        match = EXPERT_RE.match(name)
        if match:
            sparse_layers[match.group("proj")].add(int(match.group("layer")))
    wanted_sparse = set(range(2, 53))
    for projection in ("gate", "up", "down"):
        if sparse_layers[projection] != wanted_sparse:
            errors.append(
                f"routed {projection} layers differ: "
                f"{sorted(sparse_layers[projection] ^ wanted_sparse)}"
            )

    return {
        "status": "ok" if not errors else "error",
        "variant": variant,
        "source_model": source_map["source_model"],
        "source_revision": source_map["source_revision"],
        "files": [str(item) for item in shards],
        "shard_count": len(shards),
        "file_bytes": total_file_bytes,
        "file_gib": total_file_bytes / 2**30,
        "tensor_bytes": total_tensor_bytes,
        "tensor_count": len(seen),
        "type_tensor_counts": dict(sorted(type_counts.items())),
        "errors": errors,
    }


def main() -> int:
    args = parse_args()
    try:
        report = verify(args.gguf, args.source_map, args.variant)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        report = {"status": "error", "variant": args.variant, "errors": [str(exc)]}
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
