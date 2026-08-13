#!/usr/bin/env python3
"""Fetch safetensors JSON headers without downloading tensor payloads."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import requests
from huggingface_hub import get_token, hf_hub_url

MAX_HEADER_BYTES = 256 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=20)
    return parser.parse_args()


def read_range(url: str, start: int, end: int, token: str | None) -> bytes:
    expected = end - start + 1
    headers = {
        "Accept-Encoding": "identity",
        "Range": f"bytes={start}-{end}",
        "User-Agent": "motif3-mixed-ds4-header-inventory/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with requests.get(url, headers=headers, timeout=(30, 180), stream=True) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    length = response.headers.get("Content-Length", "unknown")
                    raise RuntimeError(
                        f"server ignored Range {start}-{end}: "
                        f"status={response.status_code}, content-length={length}"
                    )
                data = response.raw.read(expected)
                if len(data) != expected:
                    raise RuntimeError(
                        f"short range read: expected {expected}, received {len(data)}"
                    )
                return data
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt != 4:
                time.sleep(2**attempt)
    raise RuntimeError(f"range read failed after retries: {url}: {last_error}")


def fetch_one(repo: str, revision: str, shard: str, token: str | None) -> tuple[str, dict]:
    url = hf_hub_url(repo, shard, revision=revision)
    header_size = int.from_bytes(read_range(url, 0, 7, token), "little")
    if not 2 <= header_size <= MAX_HEADER_BYTES:
        raise RuntimeError(f"{shard}: implausible safetensors header size {header_size}")
    raw = read_range(url, 8, 8 + header_size - 1, token)
    header = json.loads(raw)
    if not isinstance(header, dict):
        raise TypeError(f"{shard}: safetensors header is not an object")
    return shard, header


def atomic_json_dump(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


def main() -> None:
    args = parse_args()
    index = json.loads(args.index.read_text())
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    token = get_token()

    tensors: dict[str, dict] = {}
    shard_headers: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_one, args.repo, args.revision, shard, token): shard
            for shard in shards
        }
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            shard, header = future.result()
            count = 0
            for name, metadata in header.items():
                if name == "__metadata__":
                    continue
                if name in tensors:
                    raise RuntimeError(f"duplicate tensor across shards: {name}")
                if weight_map.get(name) != shard:
                    raise RuntimeError(
                        f"index/header mismatch for {name}: {weight_map.get(name)} != {shard}"
                    )
                tensors[name] = {
                    "dtype": metadata["dtype"],
                    "shape": metadata["shape"],
                    "shard": shard,
                    "data_offsets": metadata["data_offsets"],
                }
                count += 1
            shard_headers[shard] = {
                "header_bytes": len(json.dumps(header)),
                "tensor_count": count,
            }
            print(f"[{completed:03d}/{len(shards):03d}] {shard}: {count} tensors", flush=True)

    missing = sorted(set(weight_map) - set(tensors))
    extra = sorted(set(tensors) - set(weight_map))
    if missing or extra:
        raise RuntimeError(
            f"header inventory mismatch: missing={len(missing)} extra={len(extra)}"
        )

    atomic_json_dump(
        {
            "schema_version": 1,
            "generated_utc": datetime.now(UTC).isoformat(),
            "repo": args.repo,
            "revision": args.revision,
            "index": str(args.index),
            "shard_count": len(shards),
            "tensor_count": len(tensors),
            "shards": shard_headers,
            "tensors": tensors,
        },
        args.out,
    )
    print(f"wrote {len(tensors)} tensors from {len(shards)} shards to {args.out}")


if __name__ == "__main__":
    main()
