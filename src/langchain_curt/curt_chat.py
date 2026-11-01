"""LangChain chat loader for `.curt` v1 conversation files.

`.curt` is an open (MIT), single-file conversation format: one timestamped,
attributed conversation per file with unknown fields preserved on round-trip.
See https://github.com/JordanNewell/dot_curt (SPEC.md) — this module is a
reader, not a reimplementation; the reference implementation and SPEC there
are authoritative.

Dependencies: langchain-core only (BaseChatLoader / ChatSession / messages).
The parser below is stdlib-only and importable without langchain installed
(the langchain imports are guarded).

Spec: `.curt` v1 — https://github.com/JordanNewell/dot_curt (SPEC.md,
"Validation rules (version 1)"). Whole-file rejection on any validation
failure; unknown fields are data and must never cause a rejection.
"""

from __future__ import annotations

import glob as _glob
import json
import math
import os
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised only when langchain-core is installed
    from langchain_core.chat_loaders import BaseChatLoader
    from langchain_core.chat_sessions import ChatSession
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
    )

    _LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover - parser-only standalone use
    _LANGCHAIN_AVAILABLE = False

# ---------------------------------------------------------------------------
# .curt v1 parser — stdlib only, no langchain imports.
# Mirrors the reference implementation (dot_curt src/index.ts: parseCurt,
# isCurtConversation, isCurtMessage).
# ---------------------------------------------------------------------------

CURT_MAGIC = "curtis-conversation"
CURT_VERSION = 1

_CORE_MESSAGE_FIELDS = frozenset({"id", "role", "content", "timestamp", "model"})
_CORE_CONVERSATION_FIELDS = frozenset(
    {"id", "title", "createdAt", "updatedAt", "messages"}
)


def _is_str(value: Any) -> bool:
    return isinstance(value, str)


def _is_number(value: Any) -> bool:
    # JSON numbers only: ``True`` is an ``int`` in Python but not a JSON number.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def curt_ms_to_iso8601(ms: Any) -> str | None:
    """Convert epoch milliseconds to an ISO 8601 UTC string.

    Returns None for non-finite values (the spec's bulk convention treats
    those as "no meaningful timestamp"); the raw number is kept in
    ``additional_kwargs`` instead.
    """
    if not _is_number(ms) or not math.isfinite(ms):
        return None
    return (
        datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def is_curt_message(value: Any) -> bool:
    """Spec-valid message predicate (SPEC.md, "Each message in `messages`")."""
    if not isinstance(value, dict):
        return False
    return (
        _is_str(value.get("id"))
        and value.get("role") in ("user", "assistant", "system", "tool")
        and _is_str(value.get("content"))
        and _is_number(value.get("timestamp"))
    )


def is_curt_conversation(value: Any) -> bool:
    """Spec-valid conversation predicate (SPEC.md, "Validation rules")."""
    if not isinstance(value, dict):
        return False
    messages = value.get("messages")
    return (
        _is_str(value.get("id"))
        and _is_str(value.get("title"))
        and _is_number(value.get("createdAt"))
        and _is_number(value.get("updatedAt"))
        and isinstance(messages, list)
        and all(is_curt_message(m) for m in messages)
    )


def parse_curt(raw: str) -> dict | None:
    """Parse a `.curt` file body. Returns the conversation dict or None.

    Whole-file rejection: any validation failure returns None, never a
    best-effort parse. Unknown fields on the top level, the conversation,
    and messages are preserved verbatim (SPEC.md, "Unknown fields"). A
    single leading UTF-8 BOM is ignored. ``version`` must be exactly the
    number 1 — the string ``"1"`` and ``true`` are rejections.
    """
    raw = raw.removeprefix("﻿")
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    # ``type() is int`` so that ``True`` (== 1 in Python) is rejected.
    if data.get("curt") != CURT_MAGIC or type(data.get("version")) is not int:
        return None
    if data["version"] != CURT_VERSION:
        return None
    conversation = data.get("conversation")
    return conversation if is_curt_conversation(conversation) else None


def _extra_fields(obj: dict, core: frozenset) -> dict:
    """Unknown fields are data: preserved verbatim, never interpreted."""
    return {k: v for k, v in obj.items() if k not in core}


# ---------------------------------------------------------------------------
# Loader — requires langchain-core.
# ---------------------------------------------------------------------------

if _LANGCHAIN_AVAILABLE:

    class CurtChatLoader(BaseChatLoader):
        """Load `.curt` v1 conversation files into LangChain chat sessions.

        Spec: https://github.com/JordanNewell/dot_curt (SPEC.md, v1).
        Each file yields one :class:`~langchain_core.chat_sessions.ChatSession`.

        Args:
            paths: A single path or glob pattern (``str`` or ``Path``), or a
                list of them. Glob patterns are resolved and sorted for
                deterministic ordering.
            on_error: ``"raise"`` (default) rejects the whole batch with a
                ``ValueError`` on the first invalid file, matching the spec's
                whole-file-rejection rule. ``"skip"`` matches the spec's bulk
                convention instead: invalid files are reported (in
                :attr:`errors`, never silently dropped) and valid ones load.

        Role mapping: ``user`` -> HumanMessage, ``assistant`` -> AIMessage,
        ``system`` -> SystemMessage. ``tool`` messages are **skipped and
        counted** (see :attr:`stats`): ToolMessage requires a
        ``tool_call_id``, which `.curt` v1 does not carry (message ``id`` is
        a conversation-local identifier, not tool-call linkage), so a lossy
        conversion would misrepresent agent traces.

        Metadata: per-message timestamp (converted from epoch ms to ISO 8601
        UTC), ``model``, and any unknown/extension fields are preserved under
        ``additional_kwargs["curt"]``. Conversation-level metadata (``id``,
        ``title``, ``createdAt``/``updatedAt`` as ISO 8601, ``generator``,
        unknown fields, per-file counts) is collected in
        :attr:`conversation_metadata` because ChatSession carries no
        metadata slot of its own.

        Attributes:
            stats: Running counters (``files_loaded``, ``files_failed``,
                ``messages_loaded``, ``tool_messages_skipped``).
            errors: ``(path, reason)`` tuples for files rejected under
                ``on_error="skip"``.
            conversation_metadata: One metadata dict per loaded file, in
                load order.
        """

        def __init__(
            self,
            paths: str | Path | list[str | Path],
            *,
            on_error: str = "raise",
        ) -> None:
            if on_error not in ("raise", "skip"):
                msg = f"on_error must be 'raise' or 'skip', got {on_error!r}"
                raise ValueError(msg)
            self._patterns: list[str | Path] = (
                list(paths) if isinstance(paths, (list, tuple)) else [paths]
            )
            self.on_error = on_error
            self.stats = {
                "files_loaded": 0,
                "files_failed": 0,
                "messages_loaded": 0,
                "tool_messages_skipped": 0,
            }
            self.errors: list[Any] = []
            self.conversation_metadata: list[dict] = []

        def _resolve_files(self) -> list[Path]:
            files: list[Path] = []
            for pattern in self._patterns:
                matches = sorted(_glob.glob(os.fspath(pattern)))
                if not matches and not any(c in os.fspath(pattern) for c in "*?["):
                    # A literal path that does not exist is an error, even
                    # under on_error="skip": unlike a corrupt file, the user
                    # almost certainly mistyped it.
                    raise FileNotFoundError(os.fspath(pattern))
                files.extend(Path(m) for m in matches)
            if not files:
                msg = f"No .curt files matched: {self._patterns!r}"
                raise ValueError(msg)
            return files

        def _load_file(self, path: Path) -> ChatSession:
            conversation = parse_curt(path.read_text(encoding="utf-8"))
            if conversation is None:
                msg = (
                    f"{path}: not a valid .curt v1 file "
                    "(bad JSON, wrong magic/version, or failed validation; "
                    "see https://github.com/JordanNewell/dot_curt)"
                )
                raise ValueError(msg)

            messages: list[BaseMessage] = []
            tool_skipped = 0
            for m in conversation["messages"]:
                role = m["role"]
                if role == "tool":
                    tool_skipped += 1
                    continue
                curt_kwargs: dict = {
                    "message_id": m["id"],
                    "conversation_id": conversation["id"],
                    "conversation_title": conversation["title"],
                    "timestamp": curt_ms_to_iso8601(m["timestamp"]),
                    "unknown_fields": _extra_fields(m, _CORE_MESSAGE_FIELDS),
                }
                if curt_kwargs["timestamp"] is None:
                    curt_kwargs["timestamp_raw_ms"] = m["timestamp"]
                if "model" in m:
                    curt_kwargs["model"] = m["model"]
                cls = {
                    "user": HumanMessage,
                    "assistant": AIMessage,
                    "system": SystemMessage,
                }[role]
                messages.append(
                    cls(content=m["content"], additional_kwargs={"curt": curt_kwargs})
                )

            created = curt_ms_to_iso8601(conversation["createdAt"])
            updated = curt_ms_to_iso8601(conversation["updatedAt"])
            meta = {
                "source": str(path),
                "id": conversation["id"],
                "title": conversation["title"],
                "created_at": created,
                "updated_at": updated,
                "generator": None,  # readers must ignore `generator` (spec)
                "unknown_fields": _extra_fields(
                    conversation, _CORE_CONVERSATION_FIELDS
                ),
                "tool_messages_skipped": tool_skipped,
            }
            for key in ("created_at", "updated_at"):
                if meta[key] is None:
                    meta[key + "_raw_ms"] = conversation[
                        "createdAt" if key == "created_at" else "updatedAt"
                    ]
            self.conversation_metadata.append(meta)
            self.stats["files_loaded"] += 1
            self.stats["messages_loaded"] += len(messages)
            self.stats["tool_messages_skipped"] += tool_skipped
            return ChatSession(messages=messages)

        def lazy_load(self) -> Iterator[ChatSession]:
            for path in self._resolve_files():
                try:
                    yield self._load_file(path)
                except (ValueError, OSError) as e:
                    if self.on_error == "raise":
                        raise
                    self.stats["files_failed"] += 1
                    self.errors.append((str(path), str(e)))
