#!/usr/bin/env python3
"""Assemble a checksum-complete H200-to-DGX-Spark handoff directory.

The public mixed GGUF shards are referenced by immutable Hub revision and
hash, not duplicated in the private bucket.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--model-repo", default="Baekpica/Motif-3-Mixed-Quant-GGUF"
    )
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--q8-revision", required=True)
    parser.add_argument(
        "--imatrix",
        type=Path,
        default=Path("/motif3/calibration/Motif-3-Q8_0-imatrix.dat"),
    )
    parser.add_argument(
        "--imatrix-report",
        type=Path,
        default=Path("/motif3/calibration/Motif-3-Q8_0-imatrix.dat.json"),
    )
    parser.add_argument(
        "--imatrix-partials",
        type=Path,
        default=Path("/motif3/calibration/motif3-q8-imatrix"),
    )
    parser.add_argument(
        "--long-context",
        type=Path,
        default=Path("/motif3/handoff-fixtures/long-context"),
    )
    parser.add_argument("--ds4", type=Path, default=Path("/workspace/ds4"))
    parser.add_argument(
        "--ds4-base",
        default="b0309611041655f4e45671cfd9c9886aff161406",
    )
    parser.add_argument(
        "--ds4-revision",
        default="bbce7eecf54703ae315328d4e240531c5a9f1a22",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(source, destination)


def copy_source_tree(source: Path, destination: Path) -> None:
    """Copy reproducible source while excluding interpreter/test caches."""
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )


def git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout


def main() -> int:
    args = parse_args()
    for option, revision in (
        ("--model-revision", args.model_revision),
        ("--q8-revision", args.q8_revision),
        ("--ds4-base", args.ds4_base),
        ("--ds4-revision", args.ds4_revision),
    ):
        if len(revision) != 40 or any(
            char not in "0123456789abcdef" for char in revision
        ):
            raise SystemExit(f"{option} must be a 40-character lowercase SHA")
    if args.out.exists() and (
        not args.out.is_dir() or any(args.out.iterdir())
    ):
        raise SystemExit(f"refusing to overwrite non-empty handoff: {args.out}")

    reproduction_status = git_output(
        ROOT, "status", "--short", "--untracked-files=no"
    )
    if reproduction_status:
        raise SystemExit(
            "refusing to package a reproduction tree with tracked changes:\n"
            + reproduction_status
        )
    ds4_status = git_output(
        args.ds4, "status", "--short", "--untracked-files=no"
    )
    if ds4_status:
        raise SystemExit(
            "refusing to package a ds4 tree with tracked changes:\n" + ds4_status
        )
    ds4_head = git_output(args.ds4, "rev-parse", "HEAD").strip()
    if ds4_head != args.ds4_revision:
        raise SystemExit(
            f"ds4 HEAD mismatch: expected {args.ds4_revision}, found {ds4_head}"
        )
    args.out.mkdir(parents=True, exist_ok=True)

    # Human-facing entry points and reports.
    copy_file(ROOT / "publish/mixed/LICENSE", args.out / "LICENSE")
    copy_file(ROOT / "handoff/README.md", args.out / "README.md")
    copy_tree(ROOT / "reports", args.out / "reports")
    copy_file(
        ROOT / "scripts/pull_spark_handoff.sh",
        args.out / "scripts/pull_spark_handoff.sh",
    )

    # Keep the complete lightweight reproduction implementation in the
    # handoff.  Large generated data and reports live in their dedicated
    # top-level directories below; this tree contains the code needed to
    # rebuild inventories, fixtures, Q8/imatrix state, and quantized outputs.
    reproduction = args.out / "reproduction"
    for relative in (
        ".gitignore",
        "LICENSE",
        "README.md",
        "WORKPLAN.md",
        "pyproject.toml",
        "uv.lock",
    ):
        copy_file(ROOT / relative, reproduction / relative)
    (reproduction / "GIT-HEAD.txt").write_text(
        git_output(ROOT, "rev-parse", "HEAD"), encoding="utf-8"
    )
    (reproduction / "branch.txt").write_text(
        git_output(ROOT, "branch", "--show-current"), encoding="utf-8"
    )
    (reproduction / "origin.txt").write_text(
        git_output(ROOT, "remote", "get-url", "origin"), encoding="utf-8"
    )
    copy_source_tree(ROOT / "converter", reproduction / "converter")
    for directory, patterns in (
        ("calibration", ("*.py",)),
        ("fixtures", ("*.py",)),
        ("scripts", ("*.py", "*.sh")),
        ("tests", ("*.py",)),
    ):
        for pattern in patterns:
            for source in sorted((ROOT / directory).glob(pattern)):
                copy_file(source, reproduction / directory / source.name)

    # Calibration corpus, final imatrix, and rank-local accumulators preserve
    # the expensive H200 state needed for auditing without requantizing.
    copy_file(
        ROOT / "calibration/calibration.composition.json",
        args.out / "calibration/calibration.composition.json",
    )
    copy_file(
        ROOT / "calibration/calibration.txt",
        args.out / "calibration/calibration.txt",
    )
    copy_file(args.imatrix, args.out / f"calibration/{args.imatrix.name}")
    copy_file(
        args.imatrix_report,
        args.out / f"calibration/{args.imatrix_report.name}",
    )
    copy_tree(args.imatrix_partials, args.out / "calibration/partials")

    copy_tree(ROOT / "fixtures/official-final", args.out / "fixtures/official-final")
    copy_tree(args.long_context, args.out / "fixtures/long-context")
    copy_tree(ROOT / "manifests", args.out / "manifests/source")

    # Preserve the exact committed ds4 development state as both a patch from
    # the pinned base and complete copies of every Motif-touched source/test
    # file. The public branch is canonical; this is an offline audit snapshot.
    ds4_dir = args.out / "ds4"
    ds4_dir.mkdir(parents=True, exist_ok=True)
    (ds4_dir / "HEAD.txt").write_text(
        ds4_head + "\n", encoding="utf-8"
    )
    (ds4_dir / "branch.txt").write_text(
        git_output(args.ds4, "branch", "--show-current"), encoding="utf-8"
    )
    (ds4_dir / "base.txt").write_text(args.ds4_base + "\n", encoding="utf-8")
    (ds4_dir / "origin.txt").write_text(
        git_output(args.ds4, "remote", "get-url", "origin"), encoding="utf-8"
    )
    (ds4_dir / "status.txt").write_text(
        git_output(args.ds4, "status", "--short", "--untracked-files=no"),
        encoding="utf-8",
    )
    (ds4_dir / "tracked.patch").write_text(
        git_output(args.ds4, "diff", "--binary", args.ds4_base, "HEAD"),
        encoding="utf-8",
    )
    touched = (
        "Makefile",
        "README.md",
        "ds4.c",
        "ds4.h",
        "ds4_cuda.cu",
        "ds4_gpu.h",
        "ds4_help.c",
        "ds4_server.c",
        "gguf-tools/deepseek4-quantize.c",
        "tests/test_motif3_loader.c",
        "tests/test_motif3_reference.c",
        "tests/test_motif3_tokenizer.c",
        "tests/test_motif3_cuda.cu",
        "tests/test_motif3_resident.c",
        "tests/test_motif3_long.c",
    )
    for relative in touched:
        copy_file(args.ds4 / relative, ds4_dir / "source" / relative)

    shards = sorted(args.model_dir.glob("Motif-3-MQ87-88-FIT-*.gguf"))
    if len(shards) != 11:
        raise SystemExit(f"expected 11 mixed shards, found {len(shards)}")
    model_dir = args.out / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "repo.txt").write_text(args.model_repo + "\n", encoding="utf-8")
    (model_dir / "revision.txt").write_text(
        args.model_revision + "\n", encoding="utf-8"
    )
    (model_dir / "filenames.txt").write_text(
        "".join(f"{path.name}\n" for path in shards), encoding="utf-8"
    )
    (model_dir / "sha256.txt").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in shards),
        encoding="utf-8",
    )
    copy_file(ROOT / "publish/mixed/README.md", model_dir / "model-card.md")
    (model_dir / "q8-reference.txt").write_text(
        f"Baekpica/Motif-3-GGUF@{args.q8_revision}\n",
        encoding="utf-8",
    )

    checksum_path = args.out / "manifests/SHA256SUMS"
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path
        for path in args.out.rglob("*")
        if path.is_file() and path != checksum_path
    )
    checksum_path.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(args.out)}\n" for path in files),
        encoding="utf-8",
    )
    print(f"handoff files: {len(files) + 1}")
    print(f"handoff bytes: {sum(path.stat().st_size for path in args.out.rglob('*') if path.is_file())}")
    print(f"model: {args.model_repo}@{args.model_revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
