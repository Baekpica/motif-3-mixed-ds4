#!/usr/bin/env python3
"""Derive an API fixture that reserves decode space inside native context."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--source-text", type=Path, required=True)
    parser.add_argument("--source-answer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reserve", type=int, default=64)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.reserve <= 0:
        raise SystemExit("--reserve must be positive")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, local_files_only=True
    )
    source_text = args.source_text.read_text(encoding="utf-8")
    source_answer = json.loads(args.source_answer.read_text(encoding="utf-8"))
    marker = "\nQUESTION:"
    question_at = source_text.rfind(marker)
    if question_at < 0:
        raise SystemExit("source fixture has no final QUESTION")
    prefix = source_text[:question_at]
    suffix = source_text[question_at:]
    removable = " x" * args.reserve
    if not prefix.endswith(removable):
        raise SystemExit("final filler does not contain the requested reserve")
    text = prefix[: -len(removable)] + suffix
    messages = [
        {"role": "system", "content": source_answer["system_prompt"]},
        {"role": "user", "content": text},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    ids = list(ids)
    expected_prompt = int(source_answer["actual_tokens"]) - args.reserve
    if len(ids) != expected_prompt:
        raise SystemExit(
            f"decode-reserved prompt has {len(ids)} tokens, expected {expected_prompt}"
        )
    retokenized = tokenizer.encode(rendered, add_special_tokens=False)
    decoded = tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if retokenized != ids or decoded != rendered:
        raise SystemExit("decode-reserved rendered prompt is not token-stable")

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"context-{source_answer['target_tokens']}-server"
    text_path = args.out / f"{stem}.txt"
    token_path = args.out / f"{stem}.tokens.npy"
    answer_path = args.out / f"{stem}.answer.json"
    text_path.write_text(text, encoding="utf-8")
    np.save(token_path, np.asarray(ids, dtype=np.int32), allow_pickle=False)
    answer = {
        "source_target_tokens": source_answer["target_tokens"],
        "prompt_tokens": len(ids),
        "admitted_context": source_answer["target_tokens"],
        "decode_reserve": args.reserve,
        "expected_json": source_answer["expected_json"],
        "system_prompt": source_answer["system_prompt"],
        "official_chat_template": True,
        "enable_thinking": False,
        "rendered_chat_retokenizes_exactly": True,
        "decoded_equals_rendered_chat": True,
    }
    answer_path.write_text(
        json.dumps(answer, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "source_fixture": args.source_text.name,
        "source_answer": args.source_answer.name,
        "derivation": "remove final one-token ' x' filler only",
        "text": text_path.name,
        "tokens": token_path.name,
        "answer": answer_path.name,
        "text_sha256": sha256(text_path),
        "tokens_sha256": sha256(token_path),
        "answer_sha256": sha256(answer_path),
        "prompt_tokens": len(ids),
        "admitted_context": source_answer["target_tokens"],
        "decode_reserve": args.reserve,
    }
    (args.out / "server-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
