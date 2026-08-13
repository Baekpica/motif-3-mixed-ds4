#!/usr/bin/env python3
"""Freeze official-final Motif-3 tokenizer and chat-template outputs.

The binary fixture is intentionally trivial so the standalone C runtime can
consume it without a JSON dependency.  Raw cases exercise the exact regex +
byte-level BPE pipeline.  Rendered cases exercise special-token recognition
after the pinned official Jinja template has produced a prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


PINNED_REVISION = "ccceb1a5fd7b5eb32e47841216b3caf5666c07bc"
PINNED_TOKENIZER_SHA256 = (
    "956d8a693e0ede5283c803aba3fe1d4b46b202d9faac007f947c6f5b61ce7864"
)
PINNED_TEMPLATE_SHA256 = (
    "998a066fd57a07187040a896e7fc455b2aa21dd490d156e82b63518be0a49297"
)

RAW_CASES = {
    "empty": "",
    "ascii_words": "Hello world and lower words",
    "ascii_case_boundaries": "HTTPServer lowerCamel UPPER CASE XMLParser",
    "contractions": "I'm we're I'LL she'd can't they've",
    "leading_punctuation": "(hello) [WORLD] +mixed -UPPER",
    "numbers": "0 12 123 1234 1234567 １２３４ ١٢٣٤",
    "whitespace": "  hello   world  \tend   ",
    "newlines": "first\r\n\n  second\n\t\nthird",
    "punctuation_slashes": "x != y///\n/\n...\r\n/path/to/file",
    "code": "def HTTPServer(x: int = 123456):\n    return x != 0\n",
    "korean": "안녕하세요 세계. Motif 모델 테스트입니다.",
    "cjk_kana": "日本語テスト 中文分词 테스트",
    "accents_combining": "Éclair déjà vu; Cafe\u0301 naïve Straße",
    "cyrillic_greek": "Привет мир Αλφα ΒΗΤΑ Γάμμα",
    "arabic": "مرحبا بالعالم ١٢٣٤",
    "emoji_symbols": "Hello 👋🏽 world — 2×GPU ≠ 4/GPU\n",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def chat_cases() -> list[dict[str, Any]]:
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
    return [
        {
            "name": "user_thinking",
            "messages": [{"role": "user", "content": "Hello"}],
            "add_generation_prompt": True,
            "enable_thinking": True,
        },
        {
            "name": "system_user_no_thinking",
            "messages": [
                {"role": "system", "content": "You are precise."},
                {"role": "user", "content": "Hello"},
            ],
            "add_generation_prompt": True,
            "enable_thinking": False,
        },
        {
            "name": "multiturn_thinking",
            "messages": [
                {"role": "system", "content": "Answer briefly."},
                {"role": "user", "content": "One plus one?"},
                {
                    "role": "assistant",
                    "reasoning_content": "I should add the values.",
                    "content": "Two.",
                },
                {"role": "user", "content": "Now double it."},
            ],
            "add_generation_prompt": True,
            "enable_thinking": True,
        },
        {
            "name": "tools_and_response",
            "messages": [
                {"role": "system", "content": "Use tools when useful."},
                {"role": "user", "content": "Weather in Seoul?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_42",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Seoul"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_42",
                    "content": {"temperature_c": 27, "condition": "clear"},
                },
                {"role": "user", "content": "Summarize that."},
            ],
            "tools": [tool],
            "add_generation_prompt": True,
            "enable_thinking": False,
        },
        {
            "name": "completed_conversation",
            "messages": [
                {"role": "user", "content": "Say hi."},
                {"role": "assistant", "content": "Hi!"},
            ],
            "add_generation_prompt": False,
            "enable_thinking": False,
        },
    ]


def write_fixture(path: Path, cases: list[dict[str, Any]]) -> None:
    with path.open("wb") as stream:
        stream.write(struct.pack("<8sII", b"DS4TOK1\0", 1, len(cases)))
        for case in cases:
            name = case["name"].encode("utf-8")
            text = case["text"].encode("utf-8")
            ids = case["ids"]
            stream.write(
                struct.pack(
                    "<IIQQ",
                    int(case["kind"]),
                    len(name),
                    len(text),
                    len(ids),
                )
            )
            stream.write(name)
            stream.write(text)
            stream.write(struct.pack(f"<{len(ids)}i", *ids))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer_path = args.snapshot / "tokenizer.json"
    template_path = args.snapshot / "chat_template.jinja"
    if sha256_file(tokenizer_path) != PINNED_TOKENIZER_SHA256:
        raise RuntimeError("official tokenizer.json hash does not match the pin")
    if sha256_file(template_path) != PINNED_TEMPLATE_SHA256:
        raise RuntimeError("official chat_template.jinja hash does not match the pin")

    tokenizer = AutoTokenizer.from_pretrained(
        args.snapshot,
        local_files_only=True,
        trust_remote_code=True,
    )
    cases: list[dict[str, Any]] = []
    manifest_cases: list[dict[str, Any]] = []
    for name, text in RAW_CASES.items():
        ids = tokenizer.encode(text, add_special_tokens=False)
        cases.append({"kind": 0, "name": name, "text": text, "ids": ids})
        manifest_cases.append({"kind": "raw", "name": name, "text": text, "ids": ids})

    for case in chat_cases():
        kwargs = {
            "add_generation_prompt": case["add_generation_prompt"],
            "enable_thinking": case["enable_thinking"],
        }
        if "tools" in case:
            kwargs["tools"] = case["tools"]
        rendered = tokenizer.apply_chat_template(case["messages"], tokenize=False, **kwargs)
        encoded = tokenizer.apply_chat_template(case["messages"], tokenize=True, **kwargs)
        ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
        if ids and isinstance(ids[0], list):
            if len(ids) != 1:
                raise RuntimeError(f"unexpected batched chat result for {case['name']}")
            ids = ids[0]
        ids = [int(token_id) for token_id in ids]
        cases.append({"kind": 1, "name": case["name"], "text": rendered, "ids": ids})
        manifest_cases.append(
            {
                "kind": "rendered_chat",
                "name": case["name"],
                "rendered": rendered,
                "ids": ids,
                "arguments": case,
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    binary = args.output / "tokenizer-chat.ds4tok"
    write_fixture(binary, cases)
    manifest = {
        "source": "Motif-Technologies/Motif-3",
        "revision": PINNED_REVISION,
        "tokenizer_sha256": PINNED_TOKENIZER_SHA256,
        "chat_template_sha256": PINNED_TEMPLATE_SHA256,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "fixture_sha256": sha256_file(binary),
        "cases": manifest_cases,
    }
    (args.output / "tokenizer-chat.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"wrote {len(cases)} official tokenizer/chat cases to {binary}")


if __name__ == "__main__":
    main()
