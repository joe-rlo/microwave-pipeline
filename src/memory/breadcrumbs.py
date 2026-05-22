"""Automatic breadcrumbs — discipline-paper § 5.1 says automatic > prompted.

Three breadcrumb triggers, all of which fire AFTER the user-visible
response (never blocking):

1. **pre_compaction** — written by the orchestrator at the top of its
   compaction routine, before any of the heavy summarization runs.
   The discipline-paper data showed 100% capture rate from this hook
   versus degraded compliance from in-prompt nudges. This is the
   single most reliable breadcrumb.

2. **auto_interval** — written every N tool calls in the current
   session. N defaults to 15 (env-tunable via
   `MEMORY_AUTO_BREADCRUMB_INTERVAL`). Per the paper, the goal is to
   catch the runaway-execution pattern (sessions hitting 100+ tool
   calls before the model thinks to checkpoint).

3. **pre_reset** — fires when the user runs `/new` or a channel
   resets the session. Captures state before context is destroyed so
   the next session can find traces of what was happening.

What a breadcrumb contains:
- Timestamp, trigger reason, session key
- Turn count + cumulative tool-call count for this session
- Last N tool names (deque) — gives the next session a hint of what
  the active workflow was
- Active project + active skill (if any)

What a breadcrumb does NOT contain:
- Prompts, messages, or response content. Those live in the
  `turns` table; breadcrumbs are a *separate* surface for "where was
  I" reconstruction without paging through transcripts.
- Token counts. The session engine tracks those for compaction; the
  breadcrumb just snapshots execution state.

Why a separate module from session/engine.py: breadcrumbs are part of
the cognitive memory pipeline (Phase F per the spec), not session
state per se. Keeping them next to `consolidated_facts` and
`fact_edges` (Phase F.2) means the consolidation pipeline can read
breadcrumbs to recover session-execution context for the nightly
Extract → Link → Brief stages.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import apsw

log = logging.getLogger(__name__)


# Default auto-interval — every 15 tool calls. Bigger than the
# discipline paper's "every 10" suggestion so breadcrumb noise stays
# manageable (Memory-Guardian's data showed 10 fired too often during
# tight tool-loop sequences). Tune via env if needed.
DEFAULT_AUTO_INTERVAL = 15


# How many recent tool names to capture in each breadcrumb. Big enough
# to give context, small enough to fit comfortably in JSON.
RECENT_TOOLS_WINDOW = 10


Trigger = str  # "pre_compaction" | "auto_interval" | "pre_reset"


# --- Schema -----------------------------------------------------------------


def init_tables(conn: apsw.Connection) -> None:
    """Create the breadcrumbs table if it doesn't exist.

    Idempotent: safe to call on every startup. The orchestrator wires
    this in alongside `session_engine.connect()`.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS breadcrumbs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fired_at INTEGER NOT NULL,         -- epoch seconds
            trigger TEXT NOT NULL,              -- pre_compaction | auto_interval | pre_reset
            session_key TEXT NOT NULL,
            turn_count INTEGER NOT NULL,
            tool_call_count INTEGER NOT NULL,
            recent_tools TEXT NOT NULL,         -- JSON array of last N tool names
            active_project TEXT,                -- nullable
            active_skill TEXT                   -- nullable
        )
    """)
    # Cheap index for the most common query: "recent breadcrumbs for a session"
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_breadcrumbs_session_fired
        ON breadcrumbs(session_key, fired_at DESC)
    """)


# --- Writer -----------------------------------------------------------------


@dataclass(frozen=True)
class Breadcrumb:
    """Read-only view of one breadcrumb row."""

    id: int
    fired_at: int
    trigger: Trigger
    session_key: str
    turn_count: int
    tool_call_count: int
    recent_tools: list[str]
    active_project: str | None
    active_skill: str | None


def write_breadcrumb(
    conn: apsw.Connection,
    *,
    trigger: Trigger,
    session_key: str,
    turn_count: int,
    tool_call_count: int,
    recent_tools: Iterable[str],
    active_project: str | None = None,
    active_skill: str | None = None,
    now: int | None = None,
) -> int:
    """Insert one breadcrumb row. Returns the new row's id.

    `now` is injectable for tests. Production uses the wall clock.
    Failures are logged but not raised — a breadcrumb write must NOT
    crash the turn it's hooked from.
    """
    import time
    if trigger not in ("pre_compaction", "auto_interval", "pre_reset"):
        raise ValueError(f"Unknown breadcrumb trigger: {trigger!r}")

    ts = now if now is not None else int(time.time())
    recent_json = json.dumps(list(recent_tools))

    try:
        conn.execute(
            """
            INSERT INTO breadcrumbs
                (fired_at, trigger, session_key, turn_count, tool_call_count,
                 recent_tools, active_project, active_skill)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts, trigger, session_key, turn_count, tool_call_count,
                recent_json, active_project, active_skill,
            ),
        )
    except Exception as e:
        log.warning("Breadcrumb write failed (trigger=%s): %s", trigger, e)
        return -1

    return conn.last_insert_rowid()


def recent_breadcrumbs(
    conn: apsw.Connection,
    *,
    limit: int = 20,
    session_key: str | None = None,
) -> list[Breadcrumb]:
    """Read recent breadcrumbs newest-first. Used by CLI + the
    consolidation pipeline (Phase F.2) to recover execution context."""
    sql = "SELECT * FROM breadcrumbs"
    params: tuple = ()
    if session_key:
        sql += " WHERE session_key = ?"
        params = (session_key,)
    sql += " ORDER BY fired_at DESC LIMIT ?"
    params = (*params, limit)

    rows = list(conn.execute(sql, params))
    out: list[Breadcrumb] = []
    for r in rows:
        try:
            tools = json.loads(r["recent_tools"])
        except (json.JSONDecodeError, TypeError):
            tools = []
        out.append(
            Breadcrumb(
                id=r["id"],
                fired_at=r["fired_at"],
                trigger=r["trigger"],
                session_key=r["session_key"],
                turn_count=r["turn_count"],
                tool_call_count=r["tool_call_count"],
                recent_tools=tools,
                active_project=r["active_project"],
                active_skill=r["active_skill"],
            )
        )
    return out


# --- Tool-call counter ------------------------------------------------------


class ToolCallCounter:
    """Counts tool calls and signals when an auto-interval breadcrumb is due.

    Lifetime: one instance per Orchestrator (i.e., one per running bot
    process). The counter persists across turns within a session but
    resets on `new_session()`.

    Usage:
        counter = ToolCallCounter()  # default interval from env
        for chunk in llm.send(...):
            if chunk["type"] == "tool_use":
                if counter.record_tool_call():
                    # threshold hit — write a breadcrumb
                    write_breadcrumb(conn, trigger="auto_interval", ...)
            counter.note_tool_name(chunk["name"])

    `note_tool_name` is separate from `record_tool_call` so callers can
    keep the deque updated even if they don't want to fire the
    threshold check (e.g., during tests).
    """

    def __init__(self, interval: int | None = None, *, window: int = RECENT_TOOLS_WINDOW):
        self.interval = interval if interval is not None else _default_interval_from_env()
        self.window = window
        self.total_calls = 0
        self._since_last_breadcrumb = 0
        self._recent: deque[str] = deque(maxlen=window)

    def record_tool_call(self) -> bool:
        """Increment counter; return True when interval is hit (caller
        should write a breadcrumb)."""
        self.total_calls += 1
        self._since_last_breadcrumb += 1
        if self._since_last_breadcrumb >= self.interval:
            self._since_last_breadcrumb = 0
            return True
        return False

    def note_tool_name(self, name: str) -> None:
        """Track the rolling window of last N tool names."""
        if name:
            self._recent.append(name)

    @property
    def recent_tools(self) -> list[str]:
        return list(self._recent)

    def reset(self) -> None:
        """Called on session reset. Lifetime counter goes to zero, and
        the deque clears so the next session's breadcrumbs don't leak
        last-session's tool sequence."""
        self.total_calls = 0
        self._since_last_breadcrumb = 0
        self._recent.clear()


def _default_interval_from_env() -> int:
    raw = os.environ.get("MEMORY_AUTO_BREADCRUMB_INTERVAL", "")
    if not raw:
        return DEFAULT_AUTO_INTERVAL
    try:
        n = int(raw)
    except ValueError:
        log.warning(
            "MEMORY_AUTO_BREADCRUMB_INTERVAL=%r is not an integer; using default %d",
            raw, DEFAULT_AUTO_INTERVAL,
        )
        return DEFAULT_AUTO_INTERVAL
    if n < 1:
        log.warning(
            "MEMORY_AUTO_BREADCRUMB_INTERVAL=%d is invalid; using default %d",
            n, DEFAULT_AUTO_INTERVAL,
        )
        return DEFAULT_AUTO_INTERVAL
    return n
