#!/usr/bin/env python3
"""Send and strictly validate a deterministic Motif-3 OpenAI long gate."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/v1/chat/completions",
    )
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--answer", type=Path, required=True)
    parser.add_argument("--model", default="motif-3")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=21600.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    answer = json.loads(args.answer.read_text(encoding="utf-8"))
    expected = answer["expected_json"]
    expected_prompt = int(answer.get("prompt_tokens", answer["actual_tokens"]))
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": answer["system_prompt"]},
            {"role": "user", "content": args.text.read_text(encoding="utf-8")},
        ],
        "think": False,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        args.endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code}: {detail}") from exc
    result = json.loads(raw)
    choices = result.get("choices") or []
    if len(choices) != 1:
        raise SystemExit(f"expected one choice, received {len(choices)}")
    content = choices[0].get("message", {}).get("content", "")
    try:
        decoded_content = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"assistant content is not JSON: {content!r}") from exc
    usage = result.get("usage") or {}
    checks = {
        "model": result.get("model") == args.model,
        "prompt_tokens": usage.get("prompt_tokens") == expected_prompt,
        "content": decoded_content == expected,
        "finish_reason": choices[0].get("finish_reason") == "stop",
        "completion_nonempty": int(usage.get("completion_tokens", 0)) > 0,
    }
    record = {
        "endpoint": args.endpoint,
        "model": result.get("model"),
        "content": content,
        "finish_reason": choices[0].get("finish_reason"),
        "usage": usage,
        "expected_json": expected,
        "checks": checks,
        "status": "passed" if all(checks.values()) else "failed",
    }
    rendered = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
