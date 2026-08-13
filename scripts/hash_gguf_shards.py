#!/usr/bin/env python3
"""Hash a complete split GGUF set in parallel and write sorted SHA256SUMS."""

from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=11)
    return parser.parse_args()


def digest(path: Path) -> tuple[Path, str]:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(chunk)
    return path, value.hexdigest()


def main() -> int:
    args = parse_args()
    shards = sorted(args.directory.glob(args.pattern))
    if len(shards) != args.expected:
        raise SystemExit(f"expected {args.expected} shards, found {len(shards)}")
    with ThreadPoolExecutor(max_workers=min(args.workers, len(shards))) as pool:
        hashes = dict(pool.map(digest, shards))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(f"{hashes[path]}  {path.name}\n" for path in shards),
        encoding="utf-8",
    )
    print(f"files={len(shards)}")
    print(f"bytes={sum(path.stat().st_size for path in shards)}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
