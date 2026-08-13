from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "long-context"
PINNED_REVISION = "ccceb1a5fd7b5eb32e47841216b3caf5666c07bc"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_long_context_fixture_hashes_and_lengths_are_pinned() -> None:
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    assert manifest["source_revision"] == PINNED_REVISION
    assert manifest["fixture_version"] == 2
    assert [row["target_tokens"] for row in manifest["fixtures"]] == [
        32768,
        65536,
        131072,
        262144,
    ]

    for row in manifest["fixtures"]:
        for kind in ("tokens", "text", "answer"):
            path = FIXTURES / row[kind]
            assert path.is_file()
            assert _sha256(path) == row[f"{kind}_sha256"]
        tokens = np.load(FIXTURES / row["tokens"], mmap_mode="r")
        assert tokens.dtype == np.dtype("int32")
        assert tokens.shape == (row["target_tokens"],)


def test_long_context_answers_lock_begin_middle_and_end_codes() -> None:
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    for row in manifest["fixtures"]:
        answer = json.loads((FIXTURES / row["answer"]).read_text())
        expected = answer["expected_json"]
        tag = row["target_tokens"]
        assert expected == [
            f"MOTIF-{tag}-BEGIN-7Q2K",
            f"MOTIF-{tag}-MIDDLE-9R4V",
            f"MOTIF-{tag}-END-3X8P",
        ]


def test_256k_server_fixture_reserves_decode_without_trimming_question() -> None:
    manifest = json.loads((FIXTURES / "server-manifest.json").read_text())
    assert manifest["source_fixture"] == "context-262144.txt"
    assert manifest["prompt_tokens"] == 262080
    assert manifest["admitted_context"] == 262144
    assert manifest["decode_reserve"] == 64
    for kind in ("tokens", "text", "answer"):
        path = FIXTURES / manifest[kind]
        assert path.is_file()
        assert _sha256(path) == manifest[f"{kind}_sha256"]

    source = np.load(FIXTURES / "context-262144.tokens.npy")
    server = np.load(FIXTURES / manifest["tokens"])
    question_and_generation_tail = 25
    reserve = manifest["decode_reserve"]
    expected = np.concatenate(
        (
            source[: -question_and_generation_tail - reserve],
            source[-question_and_generation_tail:],
        )
    )
    assert server.dtype == np.dtype("int32")
    assert server.shape == (262080,)
    assert np.array_equal(server, expected)

    answer = json.loads((FIXTURES / manifest["answer"]).read_text())
    assert answer["expected_json"] == [
        "MOTIF-262144-BEGIN-7Q2K",
        "MOTIF-262144-MIDDLE-9R4V",
        "MOTIF-262144-END-3X8P",
    ]
