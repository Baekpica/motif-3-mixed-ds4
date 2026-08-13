#!/usr/bin/env python3
"""Build Motif 3 imatrix text from the audited Solar mixed-quant composition.

The source buckets, files, revisions, shares, and seed are fixed. Records are
normalized with the checksum-pinned Healing Mix normalizer, then rendered by
the official Motif 3 chat template and counted with the official tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from huggingface_hub import HfApi, get_token, hf_hub_url

SOURCE_MODEL = "Motif-Technologies/Motif-3"
SOURCE_REVISION = "ccceb1a5fd7b5eb32e47841216b3caf5666c07bc"
REFERENCE_DATASET = "Baekpica/Solar-Open2-120B-A15B-REAM-148E-Healing-Mix"
REFERENCE_REVISION = "1931f3a40cc3463217f9c7d25906f80ded029264"
NORMALIZER_SHA256 = "677274f0d6a93326aa3d76e99e0d251d499317fa455dfb7eefed8fbc93ad1b93"
MIN_BUCKET_COVERAGE = 0.98


@dataclass(frozen=True)
class Bucket:
    name: str
    repo: str
    revision: str
    files: tuple[str, ...]
    share: float


BUCKETS = (
    Bucket(
        "if_chat",
        "nvidia/Nemotron-SFT-Instruction-Following-Chat-v2",
        "1a9454ed054b8544503ab8d8c0a519d141a44c5b",
        ("data/reasoning_on.jsonl", "data/reasoning_off.jsonl"),
        0.22,
    ),
    Bucket(
        "cascade1_reasoning",
        "nvidia/Nemotron-Cascade-SFT-Stage-1",
        "e59a356b107737562f064ec252e6adacbc737899",
        ("general.jsonl", "science.jsonl", "math/math_1.jsonl", "code/code_1.jsonl"),
        0.16,
    ),
    Bucket(
        "cascade2_reasoning",
        "nvidia/Nemotron-Cascade-SFT-Stage-2",
        "54cfa23ba28c9d94aeae15142e73bde6f8d14d8b",
        (
            "instruction-following.jsonl",
            "tool_calling.jsonl",
            "general/general_1.jsonl",
            "math/math_1.jsonl",
            "science.jsonl",
            "swe_repair.jsonl",
            "swe_localization.jsonl",
        ),
        0.16,
    ),
    Bucket(
        "ko",
        "nvidia/Nemotron-SFT-Multilingual-v2",
        "971a252224b75414b1b67c55dbe0446d8b6606a0",
        (
            "ultra-v3_math_ko_translated_final.jsonl",
            "ultra-v3_code_ko_translated_final.jsonl",
            "ultra-v3_stem_ko_translated_postedit_final.jsonl",
        ),
        0.16,
    ),
    Bucket(
        "multilingual_other",
        "nvidia/Nemotron-SFT-Multilingual-v2",
        "971a252224b75414b1b67c55dbe0446d8b6606a0",
        tuple(
            f"ultra-v3_{domain}_{language}_{suffix}.jsonl"
            for language in ("ja", "pt", "hi")
            for domain, suffix in (
                ("math", "translated_final"),
                ("code", "translated_final"),
                ("stem", "translated_postedit_final"),
            )
        ),
        0.12,
    ),
    Bucket(
        "finance",
        "nvidia/Nemotron-SpecializedDomains-Finance-v1",
        "5a21b106168facb96ced11b883c2a9b4788ee939",
        ("data/train.jsonl",),
        0.06,
    ),
    Bucket(
        "swe_agentic",
        "nvidia/Nemotron-SFT-SWE-v2",
        "bd151f3f2d89c4804dda0083d912bd9f6a0a9fb7",
        ("data/agentless.jsonl",),
        0.06,
    ),
    Bucket(
        "code_algo",
        "nvidia/Nemotron-SFT-Competitive-Programming-v2",
        "778afc98a9e027e10b3cd78020c120e93e142ef2",
        (
            "data/exercism.jsonl",
            "data/text_to_sql.jsonl",
            "data/competitive_programming_python_00.jsonl",
        ),
        0.06,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--total-tokens", type=int, default=4_000_000)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_normalizer(path: Path) -> Any:
    digest = file_sha256(path)
    if digest != NORMALIZER_SHA256:
        raise RuntimeError(f"reference normalizer hash mismatch: {digest}")
    spec = importlib.util.spec_from_file_location("healing_mix_normalizer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load normalizer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_to_messages(canonical: dict[str, Any]) -> tuple[list[dict], list[dict] | None]:
    messages: list[dict] = []
    if canonical.get("system"):
        messages.append({"role": "system", "content": canonical["system"]})
    for turn in canonical["turns"]:
        role = turn["role"]
        if role == "assistant":
            message: dict[str, Any] = {
                "role": "assistant",
                "content": turn.get("content", ""),
            }
            if turn.get("reasoning"):
                message["reasoning_content"] = turn["reasoning"]
            if turn.get("tool_calls"):
                message["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": call["arguments"],
                        },
                    }
                    for call in turn["tool_calls"]
                ]
            messages.append(message)
        elif role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "content": turn.get("content", ""),
                    "tool_call_id": turn.get("tool_call_id"),
                }
            )
        else:
            messages.append({"role": role, "content": turn.get("content", "")})
    tools = canonical.get("tools")
    if tools:
        tools = [{"type": "function", "function": tool} for tool in tools]
    return messages, tools


def render_record(record: dict[str, Any], normalizer: Any, tokenizer: Any) -> str | None:
    if isinstance(record.get("messages"), list):
        try:
            canonical = normalizer.normalize(record)
            messages, tools = canonical_to_messages(canonical)
            rendered = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=True,
            )
            if len(rendered) >= 150:
                return rendered
        except normalizer.SkipRecord:
            pass
        except (TypeError, ValueError, KeyError, AttributeError):
            pass

    for key in ("text", "content", "problem", "question", "prompt"):
        value = record.get(key)
        if isinstance(value, str) and len(value.strip()) >= 100:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": value.strip()}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
    return None


class Sampler:
    def __init__(self, seed: int, tokenizer: Any, normalizer: Any) -> None:
        token = get_token()
        self.session = requests.Session()
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self.session.headers["Accept-Encoding"] = "identity"
        self.api = HfApi(token=token)
        self.seed = seed
        self.tokenizer = tokenizer
        self.normalizer = normalizer
        self.file_sizes: dict[tuple[str, str, str], int] = {}

    def file_size(self, bucket: Bucket, filename: str) -> int:
        key = (bucket.repo, bucket.revision, filename)
        if key not in self.file_sizes:
            info = self.api.dataset_info(
                bucket.repo, revision=bucket.revision, files_metadata=True
            )
            if info.sha != bucket.revision:
                raise RuntimeError(
                    f"dataset revision mismatch for {bucket.repo}: {info.sha}"
                )
            for sibling in info.siblings:
                self.file_sizes[(bucket.repo, bucket.revision, sibling.rfilename)] = (
                    sibling.size or 0
                )
        return self.file_sizes.get(key, 0)

    def sample_jsonl(
        self, bucket: Bucket, filename: str, want_tokens: int, *, salt: int
    ) -> list[tuple[str, int]]:
        size = self.file_size(bucket, filename)
        if size <= 0 or want_tokens <= 0:
            return []
        url = hf_hub_url(
            bucket.repo,
            filename,
            repo_type="dataset",
            revision=bucket.revision,
        )
        windows = 8
        window_bytes = max(
            1 << 20,
            min(6 << 20, (want_tokens * 12 + windows - 1) // windows),
        )
        rng = random.Random(self.seed + salt)
        starts = [
            min(size - 1, int(size * (index + rng.random() * 0.7) / windows))
            for index in range(windows)
        ]
        documents: list[tuple[str, int]] = []
        tokens = 0
        for start in starts:
            if tokens >= want_tokens:
                break
            end = min(size - 1, start + window_bytes - 1)
            if end <= start:
                continue
            try:
                response = self.session.get(
                    url,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=300,
                    stream=True,
                )
                if response.status_code != 206:
                    response.close()
                    print(
                        f"    rejected unbounded response for {filename}: "
                        f"HTTP {response.status_code}",
                        file=sys.stderr,
                    )
                    continue
                payload = response.content
                response.close()
            except requests.RequestException as exc:
                print(
                    f"    range failure {filename}: {type(exc).__name__}",
                    file=sys.stderr,
                )
                continue

            lines = payload.decode("utf-8", "ignore").splitlines()
            if start > 0 and lines:
                lines = lines[1:]
            if end < size - 1 and lines:
                lines = lines[:-1]
            for line in lines:
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(record, dict):
                    continue
                rendered = render_record(record, self.normalizer, self.tokenizer)
                if not rendered:
                    continue
                rendered = rendered[:32_000]
                count = len(
                    self.tokenizer.encode(rendered, add_special_tokens=False)
                )
                if count == 0:
                    continue
                documents.append((rendered, count))
                tokens += count
                if tokens >= want_tokens:
                    break
        return documents


def main() -> None:
    args = parse_args()
    if args.total_tokens <= 0:
        raise SystemExit("--total-tokens must be positive")
    if abs(sum(bucket.share for bucket in BUCKETS) - 1.0) > 1e-9:
        raise RuntimeError("calibration shares must sum to 1.0")
    if not args.tokenizer.is_dir():
        raise SystemExit(f"official tokenizer directory missing: {args.tokenizer}")
    if not args.normalizer.is_file():
        raise SystemExit(f"reference normalizer missing: {args.normalizer}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.vocab_size != 220_160:
        raise RuntimeError(f"Motif tokenizer vocab mismatch: {tokenizer.vocab_size}")
    normalizer = load_normalizer(args.normalizer)
    sampler = Sampler(args.seed, tokenizer, normalizer)

    corpus: list[str] = []
    buckets_report: list[dict[str, Any]] = []
    for bucket_index, bucket in enumerate(BUCKETS):
        wanted = int(args.total_tokens * bucket.share)
        per_file = max(1, wanted // len(bucket.files))
        sampled: list[tuple[str, int]] = []
        print(
            f"[{bucket.name}] target {wanted / 1e6:.3f}M Motif tokens "
            f"from {bucket.repo}@{bucket.revision[:8]}",
            flush=True,
        )
        for file_index, filename in enumerate(bucket.files):
            sampled.extend(
                sampler.sample_jsonl(
                    bucket,
                    filename,
                    per_file,
                    salt=bucket_index * 101 + file_index,
                )
            )
        documents = [document for document, _ in sampled]
        actual_tokens = sum(count for _, count in sampled)
        actual_chars = sum(len(document) for document in documents)
        corpus.extend(documents)
        report = {
            "bucket": bucket.name,
            "share": bucket.share,
            "repo": bucket.repo,
            "revision": bucket.revision,
            "files": list(bucket.files),
            "target_tokens": wanted,
            "tokens": actual_tokens,
            "chars": actual_chars,
            "documents": len(documents),
            "coverage": actual_tokens / wanted,
            "status": (
                "ok" if actual_tokens >= wanted * MIN_BUCKET_COVERAGE else "shortfall"
            ),
        }
        buckets_report.append(report)
        print(
            f"    {len(documents)} docs, {actual_tokens / 1e6:.3f}M tokens, "
            f"coverage={report['coverage']:.3f}",
            flush=True,
        )

    shortfalls = [row["bucket"] for row in buckets_report if row["status"] != "ok"]
    if shortfalls:
        raise SystemExit(
            "calibration buckets below 98% of target: " + ", ".join(shortfalls)
        )

    random.Random(args.seed).shuffle(corpus)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n\n".join(document.replace("\r", "") for document in corpus) + "\n",
        encoding="utf-8",
    )

    template_path = args.tokenizer / "chat_template.jinja"
    tokenizer_path = args.tokenizer / "tokenizer.json"
    composition = {
        "schema_version": 1,
        "output": str(args.out),
        "output_sha256": file_sha256(args.out),
        "bytes": args.out.stat().st_size,
        "documents": len(corpus),
        "seed": args.seed,
        "requested_tokens": args.total_tokens,
        "actual_tokens": sum(row["tokens"] for row in buckets_report),
        "source_model": {"repo": SOURCE_MODEL, "revision": SOURCE_REVISION},
        "reference_mix": {
            "dataset": REFERENCE_DATASET,
            "revision": REFERENCE_REVISION,
            "normalizer_sha256": NORMALIZER_SHA256,
            "reuse": "bucket shares, source files, source revisions, record normalization",
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "vocab_size": tokenizer.vocab_size,
            "tokenizer_json_sha256": file_sha256(tokenizer_path),
        },
        "rendering": {
            "template": "official Motif-3 chat_template.jinja",
            "template_sha256": file_sha256(template_path),
            "enable_thinking": True,
            "tools": "preserved where present",
        },
        "buckets": buckets_report,
    }
    composition_path = args.out.with_suffix(".composition.json")
    composition_path.write_text(
        json.dumps(composition, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.out}: {len(corpus)} docs, "
        f"{args.out.stat().st_size / 1e6:.2f} MB"
    )
    print(f"composition -> {composition_path}")


if __name__ == "__main__":
    main()
