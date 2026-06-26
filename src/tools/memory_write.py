"""Memory-write tool — lets the model save a fact to memory on demand.

The pipeline already writes memory automatically (consolidation, breadcrumbs,
write-back) via `MemoryStore.append_memory` / `append_daily`, but until now
nothing exposed those to the LLM as a callable tool — so "remember this" /
"log that" from chat had no path and the model truthfully said it couldn't.

This wires the existing append machinery to a `remember` tool. It is
deliberately scoped to memory files only (MEMORY.md and daily notes) — it
cannot write anywhere else in the workspace. Append-only: it adds to the end
of the file, never overwrites or deletes. The new line is read back into the
stable prompt on the next turn (writing MEMORY.md bumps its mtime, which the
orchestrator's reconnect check already watches).

Set `MEMORY_WRITE_DISABLED=true` to remove the tool entirely.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from src.memory.store import MemoryStore

log = logging.getLogger(__name__)

# Cap a single write so a runaway call can't dump a wall of text into
# MEMORY.md. Facts should be a line or a short paragraph.
MAX_FACT_CHARS = 4000


REMEMBER_TOOL_DOCS = """\
**remember** — Save a fact to your own memory so it persists across conversations.

When to use:
- The user says "remember…", "log this", "note that", "update memory", or
  tells you something durable about themselves, their preferences, or their work.
- You learn something this turn that future-you should know.

How to use:
- `fact`: what to save. Phrase it so it stands on its own later, out of context
  ("Joe published the 'Building A Private Microwave' blog on 2026-06-25"), not
  a bare fragment ("yes, done").
- `scope`: `long_term` (default) appends to MEMORY.md — durable facts and
  preferences. `daily` appends to today's daily note — time-bound observations.

Notes:
- It APPENDS (adds to the end); it can't edit or reorder existing lines, so make
  each fact self-contained.
- It only writes memory files — it cannot touch anything else.
"""


REMEMBER_SCHEMA = {
    "type": "object",
    "properties": {
        "fact": {
            "type": "string",
            "description": (
                "The fact/line to save, phrased to stand on its own later "
                "(out of context)."
            ),
        },
        "scope": {
            "type": "string",
            "enum": ["long_term", "daily"],
            "description": (
                "long_term → MEMORY.md (durable facts/preferences, default). "
                "daily → today's daily note (time-bound observations)."
            ),
        },
    },
    "required": ["fact"],
    "additionalProperties": False,
}


def _error(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": message}], "is_error": True}


def _ok(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": message}]}


async def _handle_remember(args: dict[str, Any], *, config) -> dict[str, Any]:
    """Append a fact to memory. MCP shape; shared by SDK + provider paths."""
    fact = args.get("fact")
    scope = (args.get("scope") or "long_term").strip().lower()

    if not isinstance(fact, str) or not fact.strip():
        return _error("fact must be a non-empty string")
    if scope not in ("long_term", "daily"):
        return _error("scope must be 'long_term' or 'daily'")
    fact = fact.strip()
    if len(fact) > MAX_FACT_CHARS:
        return _error(
            f"fact is too long ({len(fact)} chars; max {MAX_FACT_CHARS}). "
            "Summarize it to the essential line."
        )

    workspace = getattr(config, "workspace_dir", None)
    if not workspace:
        return _error("no workspace configured; cannot write memory")

    try:
        store = MemoryStore(Path(workspace))
        if scope == "daily":
            store.append_daily(fact)
            where = f"today's daily note ({store.daily_path().name})"
        else:
            store.append_memory(fact)
            where = "long-term memory (MEMORY.md)"
    except Exception as e:
        log.warning("remember tool write failed: %s", e)
        return _error(f"failed to write memory: {e}")

    log.info("remember tool wrote to %s (%d chars)", scope, len(fact))
    return _ok(f"Saved to {where}.")


def memory_write_disabled() -> bool:
    return os.environ.get("MEMORY_WRITE_DISABLED", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def build_remember_sdk_tools(config) -> list:
    """SdkMcpTool wrapper. Returns [] if the SDK isn't installed."""
    try:
        from claude_agent_sdk import tool
    except ImportError:
        return []

    @tool(
        name="remember",
        description=(
            "Save a fact to your own memory so it persists across "
            "conversations. scope=long_term → MEMORY.md (default), "
            "scope=daily → today's daily note. Appends only."
        ),
        input_schema=REMEMBER_SCHEMA,
    )
    async def remember_tool(args: dict[str, Any]) -> dict[str, Any]:
        return await _handle_remember(args, config=config)

    return [remember_tool]
