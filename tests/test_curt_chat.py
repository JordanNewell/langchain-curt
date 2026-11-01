"""Tests for CurtChatLoader.

All fixtures are synthetic. Never load real conversation content in tests.
"""

from __future__ import annotations

import json

import pytest

from langchain_curt.curt_chat import (
    CurtChatLoader,
    curt_ms_to_iso8601,
    is_curt_conversation,
    is_curt_message,
    parse_curt,
)

MAGIC = "curtis-conversation"


def curt_body(
    conv_id="conv_001",
    title="Test Conversation",
    messages=None,
    **top_extra,
):
    """Build a minimal valid .curt v1 body. All content is synthetic."""
    if messages is None:
        messages = [
            {
                "id": "m1",
                "role": "user",
                "content": "Hello",
                "timestamp": 1700000000000,
            },
            {
                "id": "m2",
                "role": "assistant",
                "content": "Hi there",
                "timestamp": 1700000001000,
                "model": "test-model",
            },
        ]
    body = {
        "curt": MAGIC,
        "version": 1,
        "generator": "synthetic-fixture@0.0.1",
        "conversation": {
            "id": conv_id,
            "title": title,
            "createdAt": 1700000000000,
            "updatedAt": 1700000001000,
            "messages": messages,
        },
    }
    body.update(top_extra)
    return body


def write_curt(path, body):
    path.write_text(json.dumps(body, indent="\t"), encoding="utf-8")
    return path


def load_one(loader):
    sessions = list(loader.lazy_load())
    return sessions


class TestHappyPath:
    def test_single_file(self, tmp_path):
        f = write_curt(tmp_path / "a.curt", curt_body())
        loader = CurtChatLoader(f)
        sessions = load_one(loader)
        assert len(sessions) == 1
        msgs = sessions[0]["messages"]
        assert [type(m).__name__ for m in msgs] == ["HumanMessage", "AIMessage"]
        assert msgs[0].content == "Hello"
        assert msgs[1].content == "Hi there"
        assert msgs[1].additional_kwargs["curt"]["model"] == "test-model"
        # Timestamps: epoch ms -> ISO 8601 UTC.
        assert msgs[0].additional_kwargs["curt"]["timestamp"] == (
            "2023-11-14T22:13:20Z"
        )
        assert loader.stats["files_loaded"] == 1
        assert loader.stats["messages_loaded"] == 2
        assert loader.errors == []

    def test_conversation_metadata_collected(self, tmp_path):
        f = write_curt(tmp_path / "a.curt", curt_body())
        loader = CurtChatLoader(f)
        load_one(loader)
        (meta,) = loader.conversation_metadata
        assert meta["id"] == "conv_001"
        assert meta["title"] == "Test Conversation"
        assert meta["created_at"] == "2023-11-14T22:13:20Z"
        assert meta["source"].endswith("a.curt")

    def test_system_role(self, tmp_path):
        msgs = [
            {"id": "m0", "role": "system", "content": "You are terse.", "timestamp": 0},
            {"id": "m1", "role": "user", "content": "Hi", "timestamp": 1},
        ]
        f = write_curt(tmp_path / "a.curt", curt_body(messages=msgs))
        (session,) = load_one(CurtChatLoader(f))
        assert type(session["messages"][0]).__name__ == "SystemMessage"

    def test_timestamp_zero_and_fractional(self):
        assert curt_ms_to_iso8601(0) == "1970-01-01T00:00:00Z"
        # Spec: validators accept any JSON number; never reject fractional.
        assert curt_ms_to_iso8601(1700000000123.5) == "2023-11-14T22:13:20.123500Z"


class TestMultiFile:
    def test_glob(self, tmp_path):
        for i in (2, 1):
            write_curt(tmp_path / f"chat{i}.curt", curt_body(conv_id=f"conv_{i:03d}"))
        loader = CurtChatLoader(str(tmp_path / "*.curt"))
        sessions = load_one(loader)
        assert len(sessions) == 2
        # Sorted for deterministic ordering.
        assert loader.conversation_metadata[0]["id"] == "conv_001"

    def test_path_list(self, tmp_path):
        fa = write_curt(tmp_path / "a.curt", curt_body(conv_id="a"))
        fb = write_curt(tmp_path / "b.curt", curt_body(conv_id="b"))
        sessions = load_one(CurtChatLoader([fa, fb]))
        assert [
            m["messages"][0].additional_kwargs["curt"]["conversation_id"]
            for m in sessions
        ] == ["a", "b"]

    def test_no_matches_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No .curt files matched"):
            load_one(CurtChatLoader(str(tmp_path / "*.curt")))

    def test_missing_literal_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_one(CurtChatLoader(tmp_path / "nope.curt"))


class TestRejections:
    """Whole-file rejection per spec — a corrupted file is never half-loaded."""

    def test_bad_magic_string(self, tmp_path):
        write_curt(tmp_path / "a.curt", curt_body(curt="wrong-magic"))
        with pytest.raises(ValueError, match="not a valid .curt v1 file"):
            load_one(CurtChatLoader(tmp_path / "a.curt"))

    def test_version_as_string(self, tmp_path):
        write_curt(tmp_path / "a.curt", curt_body(version="1"))
        with pytest.raises(ValueError, match="not a valid .curt v1 file"):
            load_one(CurtChatLoader(tmp_path / "a.curt"))

    def test_version_true_rejected(self, tmp_path):
        # Python gotcha the parser must guard: True == 1.
        write_curt(tmp_path / "a.curt", curt_body(version=True))
        assert parse_curt((tmp_path / "a.curt").read_text(encoding="utf-8")) is None

    def test_unknown_role_rejects_whole_file(self, tmp_path):
        # Spec: role must be user/assistant/system/tool; anything else fails
        # message validation, and failure is whole-file — the valid messages
        # in the same file must NOT load.
        msgs = [
            {"id": "m1", "role": "user", "content": "ok", "timestamp": 1},
            {"id": "m2", "role": "moderator", "content": "nope", "timestamp": 2},
        ]
        write_curt(tmp_path / "a.curt", curt_body(messages=msgs))
        with pytest.raises(ValueError, match="not a valid .curt v1 file"):
            load_one(CurtChatLoader(tmp_path / "a.curt"))

    def test_missing_required_message_field(self, tmp_path):
        msgs = [{"id": "m1", "role": "user", "content": "no timestamp"}]
        write_curt(tmp_path / "a.curt", curt_body(messages=msgs))
        with pytest.raises(ValueError):
            load_one(CurtChatLoader(tmp_path / "a.curt"))

    def test_malformed_json(self, tmp_path):
        (tmp_path / "a.curt").write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="not a valid .curt v1 file"):
            load_one(CurtChatLoader(tmp_path / "a.curt"))

    def test_non_object_body(self, tmp_path):
        (tmp_path / "a.curt").write_text('["a"]', encoding="utf-8")
        assert parse_curt('["a"]') is None

    def test_skip_mode_reports(self, tmp_path):
        write_curt(tmp_path / "good.curt", curt_body())
        write_curt(tmp_path / "bad.curt", curt_body(curt="nope"))
        loader = CurtChatLoader(str(tmp_path / "*.curt"), on_error="skip")
        sessions = load_one(loader)
        assert len(sessions) == 1
        assert loader.stats["files_loaded"] == 1
        assert loader.stats["files_failed"] == 1
        assert len(loader.errors) == 1
        assert "bad.curt" in loader.errors[0][0]

    def test_bad_on_error_arg(self):
        with pytest.raises(ValueError, match="on_error"):
            CurtChatLoader("x", on_error="explode")


class TestToolRole:
    def test_tool_messages_skipped_and_counted(self, tmp_path):
        msgs = [
            {"id": "m1", "role": "user", "content": "q", "timestamp": 1},
            {
                "id": "m2",
                "role": "tool",
                "content": '{"result": 42}',
                "timestamp": 2,
            },
            {"id": "m3", "role": "assistant", "content": "a", "timestamp": 3},
        ]
        write_curt(tmp_path / "a.curt", curt_body(messages=msgs))
        loader = CurtChatLoader(tmp_path / "a.curt")
        (session,) = load_one(loader)
        # Documented loss: tool -> skipped (ToolMessage needs tool_call_id).
        assert [type(m).__name__ for m in session["messages"]] == [
            "HumanMessage",
            "AIMessage",
        ]
        assert loader.stats["tool_messages_skipped"] == 1
        assert loader.conversation_metadata[0]["tool_messages_skipped"] == 1


class TestSpecCompliance:
    def test_bom_tolerated(self, tmp_path):
        f = tmp_path / "a.curt"
        f.write_text("﻿" + json.dumps(curt_body()), encoding="utf-8")
        sessions = load_one(CurtChatLoader(f))
        assert len(sessions) == 1

    def test_unknown_fields_preserved(self, tmp_path):
        msgs = [
            {
                "id": "m1",
                "role": "user",
                "content": "hi",
                "timestamp": 1,
                "tokens": 5,
                "images-inline": {"v": 1},
            }
        ]
        body = curt_body(messages=msgs, extensions=["images-inline"])
        body["conversation"]["folder_hint"] = "keep-me"
        f = write_curt(tmp_path / "a.curt", body)
        loader = CurtChatLoader(f)
        (session,) = load_one(loader)
        kwargs = session["messages"][0].additional_kwargs["curt"]
        # Unknown fields are data: preserved verbatim, never a rejection.
        assert kwargs["unknown_fields"] == {
            "tokens": 5,
            "images-inline": {"v": 1},
        }
        meta = loader.conversation_metadata[0]
        assert meta["unknown_fields"] == {"folder_hint": "keep-me"}
        # `generator` must be ignored by readers (spec), never validated on.
        assert meta["generator"] is None

    def test_empty_title_allowed(self, tmp_path):
        f = write_curt(tmp_path / "a.curt", curt_body(title=""))
        (session,) = load_one(CurtChatLoader(f))
        assert (
            session["messages"][0].additional_kwargs["curt"]["conversation_title"] == ""
        )

    def test_empty_messages_accepted(self, tmp_path):
        # Spec: readers may accept empty message arrays.
        f = write_curt(tmp_path / "a.curt", curt_body(messages=[]))
        (session,) = load_one(CurtChatLoader(f))
        assert session["messages"] == []

    def test_predicates(self):
        assert is_curt_message(
            {"id": "m", "role": "user", "content": "c", "timestamp": 0}
        )
        assert not is_curt_message(
            {"id": "m", "role": "user", "content": "c", "timestamp": True}
        )
        assert not is_curt_conversation(
            {"id": "c", "title": "t", "createdAt": 0, "updatedAt": 0}
        )
        assert is_curt_conversation(
            {"id": "c", "title": "t", "createdAt": 0, "updatedAt": 0, "messages": []}
        )
