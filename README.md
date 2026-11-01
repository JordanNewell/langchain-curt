# langchain-curt

[LangChain](https://github.com/langchain-ai/langchain) chat loader for
[`.curt`](https://github.com/JordanNewell/dot_curt) — the open (MIT),
single-file AI conversation format. One timestamped, attributed conversation
per file; unknown fields are data and survive round-trips.

```python
from langchain_curt import CurtChatLoader

loader = CurtChatLoader("./conversations/*.curt")
sessions = loader.lazy_load()   # one ChatSession per .curt file
```

- **Role mapping:** `user` → `HumanMessage`, `assistant` → `AIMessage`,
  `system` → `SystemMessage`. `tool` messages are skipped and counted
  (`loader.stats["tool_messages_skipped"]`) — `ToolMessage` requires a
  `tool_call_id` that `.curt` v1 does not carry.
- **Metadata:** per-message timestamps (epoch ms → ISO 8601 UTC), model, and
  all unknown/extension fields under `additional_kwargs["curt"]`;
  conversation-level metadata in `loader.conversation_metadata`.
- **Validation:** spec-compliant whole-file rejection (bad magic, wrong
  version, missing fields → rejected, never half-loaded). `on_error="skip"`
  imports valid files and reports invalid ones in `loader.errors`.
- **Dependencies:** `langchain-core` only. The `.curt` parser is stdlib-only.

Converters into `.curt` exist for Claude Code transcripts
([curt-convert](https://github.com/JordanNewell/curt-convert)), Open WebUI
exports ([curt-openwebui](https://github.com/JordanNewell/curt-openwebui)),
and the WildChat-1M research corpus
([curt-wildchat](https://github.com/JordanNewell/curt-wildchat)).

Spec: https://github.com/JordanNewell/dot_curt · npm reference impl:
[`curt-format`](https://www.npmjs.com/package/curt-format)

## Development

```sh
pip install -e . pytest
pytest
```

MIT License.
