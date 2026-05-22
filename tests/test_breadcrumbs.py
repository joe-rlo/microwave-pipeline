"""Tests for the breadcrumbs module (Phase F.1).

Coverage:
- init_tables is idempotent
- write_breadcrumb persists rows with the expected shape
- Invalid triggers rejected at the function boundary
- Failures inside write_breadcrumb are swallowed (return -1) not raised —
  the orchestrator hooks MUST NOT crash turns when a breadcrumb fails
- recent_breadcrumbs returns newest-first; honors limit + session filter
- ToolCallCounter signals threshold correctly, doesn't double-fire,
  resets on new sessions
- Env override for the interval works; bad values fall back to default

We use apsw against a fresh in-memory database per test for isolation.
"""

from __future__ import annotations

import time
from pathlib import Path

import apsw
import pytest

from src.memory.breadcrumbs import (
    DEFAULT_AUTO_INTERVAL,
    RECENT_TOOLS_WINDOW,
    Breadcrumb,
    ToolCallCounter,
    _default_interval_from_env,
    init_tables,
    recent_breadcrumbs,
    write_breadcrumb,
)


@pytest.fixture
def conn() -> apsw.Connection:
    """Fresh in-memory apsw connection with row-by-name access."""
    c = apsw.Connection(":memory:")
    c.row_trace = lambda cursor, row: {
        d[0]: v for d, v in zip(cursor.getdescription(), row)
    }
    init_tables(c)
    return c


# --- Schema -----------------------------------------------------------------


class TestSchema:
    def test_init_tables_idempotent(self, conn):
        # Calling twice must not raise.
        init_tables(conn)
        init_tables(conn)
        rows = list(conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='breadcrumbs'"))
        assert len(rows) == 1


# --- write_breadcrumb -------------------------------------------------------


class TestWriteBreadcrumb:
    def test_basic_write(self, conn):
        row_id = write_breadcrumb(
            conn,
            trigger="pre_compaction",
            session_key="sess-1",
            turn_count=5,
            tool_call_count=12,
            recent_tools=["webfetch", "github_list_repos"],
            active_project="microwave-os",
            active_skill="github",
            now=1_700_000_000,
        )
        assert row_id > 0

        rows = list(conn.execute("SELECT * FROM breadcrumbs"))
        assert len(rows) == 1
        row = rows[0]
        assert row["trigger"] == "pre_compaction"
        assert row["session_key"] == "sess-1"
        assert row["turn_count"] == 5
        assert row["tool_call_count"] == 12
        # recent_tools stored as JSON
        import json
        assert json.loads(row["recent_tools"]) == ["webfetch", "github_list_repos"]
        assert row["active_project"] == "microwave-os"
        assert row["active_skill"] == "github"
        assert row["fired_at"] == 1_700_000_000

    def test_null_active_fields(self, conn):
        # No active project / skill is the common case for fresh sessions.
        write_breadcrumb(
            conn,
            trigger="auto_interval",
            session_key="sess-1",
            turn_count=0,
            tool_call_count=15,
            recent_tools=[],
        )
        row = list(conn.execute("SELECT * FROM breadcrumbs"))[0]
        assert row["active_project"] is None
        assert row["active_skill"] is None

    def test_rejects_unknown_trigger(self, conn):
        with pytest.raises(ValueError, match="Unknown breadcrumb trigger"):
            write_breadcrumb(
                conn, trigger="bogus", session_key="s", turn_count=0,
                tool_call_count=0, recent_tools=[],
            )

    def test_db_error_returns_minus_one_no_raise(self, conn):
        # Drop the table to force a real failure inside the INSERT.
        conn.execute("DROP TABLE breadcrumbs")
        # MUST NOT raise — the orchestrator hooks rely on this.
        row_id = write_breadcrumb(
            conn,
            trigger="pre_reset",
            session_key="s",
            turn_count=0,
            tool_call_count=0,
            recent_tools=[],
        )
        assert row_id == -1

    def test_default_now_uses_walltime(self, conn):
        before = int(time.time())
        write_breadcrumb(
            conn, trigger="pre_reset", session_key="s",
            turn_count=0, tool_call_count=0, recent_tools=[],
        )
        after = int(time.time())
        ts = list(conn.execute("SELECT fired_at FROM breadcrumbs"))[0]["fired_at"]
        assert before <= ts <= after


# --- recent_breadcrumbs -----------------------------------------------------


class TestRecentBreadcrumbs:
    def test_newest_first(self, conn):
        for i, ts in enumerate([1_000, 2_000, 3_000]):
            write_breadcrumb(
                conn, trigger="auto_interval", session_key=f"s{i}",
                turn_count=i, tool_call_count=i, recent_tools=[], now=ts,
            )
        out = recent_breadcrumbs(conn)
        assert [b.fired_at for b in out] == [3_000, 2_000, 1_000]
        # Returned as Breadcrumb dataclasses
        assert all(isinstance(b, Breadcrumb) for b in out)

    def test_limit_respected(self, conn):
        for i in range(5):
            write_breadcrumb(
                conn, trigger="auto_interval", session_key="s",
                turn_count=i, tool_call_count=i, recent_tools=[], now=i,
            )
        out = recent_breadcrumbs(conn, limit=2)
        assert len(out) == 2

    def test_session_filter(self, conn):
        write_breadcrumb(conn, trigger="pre_reset", session_key="a",
                         turn_count=0, tool_call_count=0, recent_tools=[])
        write_breadcrumb(conn, trigger="pre_reset", session_key="b",
                         turn_count=0, tool_call_count=0, recent_tools=[])
        a_only = recent_breadcrumbs(conn, session_key="a")
        assert len(a_only) == 1
        assert a_only[0].session_key == "a"

    def test_malformed_recent_tools_falls_back_to_empty_list(self, conn):
        # Defensive: if JSON in the column got corrupted somehow, reader
        # returns [] rather than raising.
        conn.execute(
            "INSERT INTO breadcrumbs (fired_at, trigger, session_key, "
            "turn_count, tool_call_count, recent_tools) VALUES (?, ?, ?, ?, ?, ?)",
            (1, "pre_reset", "s", 0, 0, "not-json"),
        )
        out = recent_breadcrumbs(conn)
        assert len(out) == 1
        assert out[0].recent_tools == []


# --- ToolCallCounter --------------------------------------------------------


class TestToolCallCounter:
    def test_threshold_fires_on_exact_interval(self):
        c = ToolCallCounter(interval=3)
        assert c.record_tool_call() is False  # 1
        assert c.record_tool_call() is False  # 2
        assert c.record_tool_call() is True   # 3 → fire
        # Counter resets after a fire — next 3 calls fire again.
        assert c.record_tool_call() is False  # 4
        assert c.record_tool_call() is False  # 5
        assert c.record_tool_call() is True   # 6 → fire

    def test_total_calls_tracks_lifetime(self):
        c = ToolCallCounter(interval=2)
        for _ in range(5):
            c.record_tool_call()
        assert c.total_calls == 5

    def test_recent_tools_capped_at_window(self):
        c = ToolCallCounter(interval=999, window=3)
        for name in ["a", "b", "c", "d", "e"]:
            c.note_tool_name(name)
        assert c.recent_tools == ["c", "d", "e"]

    def test_empty_name_ignored(self):
        c = ToolCallCounter()
        c.note_tool_name("")
        c.note_tool_name(None)  # type: ignore[arg-type]
        assert c.recent_tools == []

    def test_reset_clears_state(self):
        c = ToolCallCounter(interval=5)
        for _ in range(3):
            c.record_tool_call()
        c.note_tool_name("github_list_repos")
        c.reset()
        assert c.total_calls == 0
        assert c.recent_tools == []
        # After reset, threshold counter starts over too
        for _ in range(4):
            assert c.record_tool_call() is False
        assert c.record_tool_call() is True

    def test_default_interval_from_constant(self, monkeypatch):
        monkeypatch.delenv("MEMORY_AUTO_BREADCRUMB_INTERVAL", raising=False)
        c = ToolCallCounter()
        assert c.interval == DEFAULT_AUTO_INTERVAL

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MEMORY_AUTO_BREADCRUMB_INTERVAL", "5")
        c = ToolCallCounter()
        assert c.interval == 5

    def test_env_non_integer_falls_back_to_default(self, monkeypatch, caplog):
        monkeypatch.setenv("MEMORY_AUTO_BREADCRUMB_INTERVAL", "abc")
        with caplog.at_level("WARNING"):
            n = _default_interval_from_env()
        assert n == DEFAULT_AUTO_INTERVAL
        assert any("not an integer" in r.message for r in caplog.records)

    def test_env_negative_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MEMORY_AUTO_BREADCRUMB_INTERVAL", "-1")
        n = _default_interval_from_env()
        assert n == DEFAULT_AUTO_INTERVAL
