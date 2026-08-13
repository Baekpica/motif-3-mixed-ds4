#!/usr/bin/env python3
"""Build deterministic Motif-3 long-context admission/retrieval fixtures.

The NumPy token arrays are authoritative.  Text decodes are included for
inspection and API clients, and the manifest records whether decoding then
re-tokenizing reproduces the exact original IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


SOURCE_MODEL = "Motif-Technologies/Motif-3"
SOURCE_REVISION = "ccceb1a5fd7b5eb32e47841216b3caf5666c07bc"
DEFAULT_LENGTHS = (32_768, 65_536, 131_072, 262_144)
SYSTEM_PROMPT = (
    "You are a precise long-context retrieval assistant. "
    "Follow the user output format exactly."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--length", type=int, action="append", dest="lengths")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_one(tokenizer, target: int) -> tuple[list[int], dict]:
    codes = [
        f"MOTIF-{target}-BEGIN-7Q2K",
        f"MOTIF-{target}-MIDDLE-9R4V",
        f"MOTIF-{target}-END-3X8P",
    ]
    header = (
        "Long-context retrieval fixture. Preserve every explicit RECORD code. "
        "At the final QUESTION, return only a JSON array containing the three "
        "codes in document order.\n"
    )
    needles = [
        f"\nRECORD {index + 1}: The exact code is {code}. Remember it verbatim.\n"
        for index, code in enumerate(codes)
    ]
    question = (
        "\nQUESTION: Return only a JSON array of the exact three RECORD codes "
        "in document order, with no explanation.\n"
    )
    # Motif's tokenizer maps every repetition of " x" to exactly one token.
    # The authoritative count includes the official no-think assistant prefix,
    # matching what ds4-server actually evaluates rather than raw user text.
    def render_chat(user_text: str) -> tuple[str, list[int]]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        rendered_chat = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        rendered_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if hasattr(rendered_ids, "keys"):
            rendered_ids = rendered_ids["input_ids"]
        return rendered_chat, list(rendered_ids)

    fixed_text = header + "".join(needles) + question
    _, fixed_ids = render_chat(fixed_text)
    fixed = len(fixed_ids)
    if fixed >= target:
        raise ValueError(f"target {target} is smaller than fixed fixture content")
    remaining = target - fixed
    segment_lengths = [
        remaining * 10 // 100,
        remaining * 40 // 100,
        remaining * 40 // 100,
        0,
    ]
    segment_lengths[3] = remaining - sum(segment_lengths[:3])

    def render() -> tuple[str, list[tuple[int, int]], int]:
        parts = [header]
        char_spans: list[tuple[int, int]] = []
        for index, needle in enumerate(needles):
            parts.append(" x" * segment_lengths[index])
            start = sum(map(len, parts))
            parts.append(needle)
            char_spans.append((start, start + len(needle)))
        parts.append(" x" * segment_lengths[3])
        question_start_char = sum(map(len, parts))
        parts.append(question)
        return "".join(parts), char_spans, question_start_char

    # Boundary merges in the fixed prose can shift the initial estimate by a
    # handful of tokens.  The final padding segment is corrected until the
    # full text—not a concatenated token stream—is exactly target tokens.
    for _ in range(8):
        text, raw_char_spans, raw_question_start_char = render()
        rendered_chat, ids = render_chat(text)
        encoded = tokenizer(
            rendered_chat,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        if list(encoded["input_ids"]) != ids:
            raise RuntimeError("chat-template tokenization is not reproducible")
        delta = target - len(ids)
        if delta == 0:
            break
        segment_lengths[3] += delta
        if segment_lengths[3] < 0:
            raise ValueError("padding correction became negative")
    else:
        raise RuntimeError(f"could not converge exact text length for {target}")

    offsets = encoded["offset_mapping"]
    char_spans: list[tuple[int, int]] = []
    for raw_start, raw_end in raw_char_spans:
        needle = text[raw_start:raw_end]
        rendered_start = rendered_chat.find(needle)
        if rendered_start < 0:
            raise RuntimeError("record missing from rendered chat")
        char_spans.append((rendered_start, rendered_start + len(needle)))
    question_text = text[raw_question_start_char:]
    question_start_char = rendered_chat.find(question_text)
    if question_start_char < 0:
        raise RuntimeError("question missing from rendered chat")

    def token_span(char_start: int, char_end: int) -> tuple[int, int]:
        indices = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > char_start and start < char_end
        ]
        if not indices:
            raise RuntimeError("fixture span has no tokenizer offsets")
        return indices[0], indices[-1] + 1

    positions: list[dict] = []
    for index, (char_start, char_end) in enumerate(char_spans):
        token_start, token_end = token_span(char_start, char_end)
        positions.append(
            {
                "record": index + 1,
                "code": codes[index],
                "token_start": token_start,
                "token_end": token_end,
            }
        )
    question_start, _ = token_span(question_start_char, len(rendered_chat))
    decoded = tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    retokenized = tokenizer.encode(rendered_chat, add_special_tokens=False)
    return ids, {
        "target_tokens": target,
        "actual_tokens": len(ids),
        "records": positions,
        "question_token_start": question_start,
        "expected_json": codes,
        "official_chat_template": True,
        "enable_thinking": False,
        "system_prompt": SYSTEM_PROMPT,
        "rendered_chat_retokenizes_exactly": retokenized == ids,
        "retokenized_tokens": len(retokenized),
        "decoded_equals_rendered_chat": decoded == rendered_chat,
        "rendered_chat_tail": rendered_chat[-256:],
        "text": text,
    }


def main() -> int:
    args = parse_args()
    lengths = tuple(args.lengths or DEFAULT_LENGTHS)
    if any(length <= 0 or length > 262_144 for length in lengths):
        raise SystemExit("lengths must be in 1..262144")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, local_files_only=True
    )
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source_model": SOURCE_MODEL,
        "source_revision": SOURCE_REVISION,
        "fixture_version": 2,
        "token_count_semantics": "official rendered chat including no-think assistant prefix",
        "system_prompt": SYSTEM_PROMPT,
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "fixtures": [],
    }
    for target in lengths:
        ids, info = build_one(tokenizer, target)
        stem = f"context-{target}"
        tokens_path = args.out / f"{stem}.tokens.npy"
        text_path = args.out / f"{stem}.txt"
        answer_path = args.out / f"{stem}.answer.json"
        np.save(tokens_path, np.asarray(ids, dtype=np.int32), allow_pickle=False)
        text_path.write_text(info.pop("text"), encoding="utf-8")
        answer_path.write_text(
            json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest["fixtures"].append(
            {
                "target_tokens": target,
                "tokens": tokens_path.name,
                "text": text_path.name,
                "answer": answer_path.name,
                "tokens_sha256": sha256(tokens_path),
                "text_sha256": sha256(text_path),
                "answer_sha256": sha256(answer_path),
            }
        )
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
