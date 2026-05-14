"""Tests for pipeline 3.1: cross-session summary.

Covers the three independently-testable pieces:
- `session_summary` module: slug + topic parsing + min-turn gate
- `MemoryStore.save_session_summary` / `load_recent_session_summaries`
- `Orchestrator._build_session_recap` formatting (state-only)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.memory.store import MemoryStore, _parse_session_summary
from src.pipeline.session_summary import (
    MIN_TURNS_FOR_SUMMARY,
    SessionSummaryResult,
    _extract_topic_and_body,
    _slugify,
    generate_session_summary,
)
from src.session.models import Turn


def _turn(role: str, content: str) -> Turn:
    return Turn(session_id="s", channel="repl", user_id="u", role=role, content=content)


class TestSlugify:
    def test_lowercase_kebab(self):
        assert _slugify("Pipeline Improvements") == "pipeline-improvements"

    def test_collapse_punct(self):
        assert _slugify("Signal & Telegram!") == "signal-telegram"

    def test_empty_becomes_general(self):
        """A slugless filename would silently overwrite itself across
        sessions — guarantee a non-empty result."""
        assert _slugify("") == "general"
        assert _slugify("   ") == "general"
        assert _slugify("???") == "general"

    def test_collapses_repeated_separators(self):
        assert _slugify("foo  ---  bar") == "foo-bar"


class TestTopicExtraction:
    def test_trailing_topic_line(self):
        raw = "We worked on X.\n\nTOPIC: pipeline-improvements"
        topic, body = _extract_topic_and_body(raw)
        assert topic == "pipeline-improvements"
        assert "We worked on X." in body
        assert "TOPIC:" not in body

    def test_case_insensitive(self):
        topic, _ = _extract_topic_and_body("body\n\nTopic: novel-chapter-4")
        assert topic == "novel-chapter-4"

    def test_no_topic_line_defaults_to_general(self):
        """Model occasionally drops the TOPIC: footer — fall back to
        'general' rather than crashing the session close hook."""
        topic, body = _extract_topic_and_body("just a summary, no footer")
        assert topic == "general"
        assert "just a summary" in body

    def test_topic_with_spaces_slugged(self):
        topic, _ = _extract_topic_and_body("body\nTOPIC: Signal Formatting")
        assert topic == "signal-formatting"


class TestGenerateSessionSummary:
    @pytest.mark.asyncio
    async def test_skips_when_below_min_turns(self):
        """Two-turn sessions are usually just "hi/hey" — summarizing
        them pollutes retrieval. The min-turn gate must hold."""
        turns = [_turn("user", "hi"), _turn("assistant", "hey")]
        result = await generate_session_summary(turns)
        assert result is None

    @pytest.mark.asyncio
    async def test_min_turn_count_constant_documented(self):
        """The threshold is load-bearing — pin its value so a casual
        bump doesn't silently change behavior."""
        assert MIN_TURNS_FOR_SUMMARY == 3

    @pytest.mark.asyncio
    async def test_parses_topic_and_body_from_response(self):
        """When the LLM returns a clean summary + TOPIC line, the
        result carries both, separated."""
        fake = "Joe worked on pipeline improvements...\n\nTOPIC: pipeline-improvements"
        with patch(
            "src.pipeline.session_summary.SingleTurnClient"
        ) as cli_cls:
            cli_cls.return_value.query = AsyncMock(return_value=fake)
            turns = [_turn("user", f"msg {i}") for i in range(5)]
            result = await generate_session_summary(turns)
        assert isinstance(result, SessionSummaryResult)
        assert result.topic_slug == "pipeline-improvements"
        assert "TOPIC:" not in result.body
        assert "pipeline improvements" in result.body
        assert result.turn_count == 5

    @pytest.mark.asyncio
    async def test_empty_llm_response_returns_none(self):
        """Sonnet sometimes returns empty on transient failures —
        callers expect None so the close hook can skip persistence."""
        with patch(
            "src.pipeline.session_summary.SingleTurnClient"
        ) as cli_cls:
            cli_cls.return_value.query = AsyncMock(return_value="")
            turns = [_turn("user", f"msg {i}") for i in range(5)]
            result = await generate_session_summary(turns)
        assert result is None

    @pytest.mark.asyncio
    async def test_llm_exception_returns_none(self):
        """Generation errors at session close must not propagate —
        `/new` and shutdown both depend on this being non-blocking."""
        with patch(
            "src.pipeline.session_summary.SingleTurnClient"
        ) as cli_cls:
            cli_cls.return_value.query = AsyncMock(side_effect=RuntimeError("boom"))
            turns = [_turn("user", f"msg {i}") for i in range(5)]
            result = await generate_session_summary(turns)
        assert result is None


class TestMemoryStoreSessionSummaries:
    def test_sessions_dir_under_memory(self, tmp_path):
        """sessions/ nests under memory/ so the indexer picks it up
        without needing a second corpus root."""
        store = MemoryStore(tmp_path)
        assert store.sessions_dir == tmp_path / "memory" / "sessions"

    def test_ensure_dirs_creates_sessions(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.ensure_dirs()
        assert store.sessions_dir.is_dir()

    def test_save_writes_frontmatter_and_body(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.ensure_dirs()
        started = datetime(2026, 5, 13, 9, 14)
        ended = datetime(2026, 5, 13, 10, 2)
        path = store.save_session_summary(
            body="We discussed pipeline improvements.",
            started_at=started,
            ended_at=ended,
            topic_slug="pipeline-improvements",
            project="microwaveos",
            turn_count=14,
        )
        text = path.read_text()
        # YAML-shaped header
        assert text.startswith("---\n")
        assert "started: 2026-05-13T09:14:00" in text
        assert "ended: 2026-05-13T10:02:00" in text
        assert "topic: pipeline-improvements" in text
        assert "project: microwaveos" in text
        assert "turns: 14" in text
        # Body preserved verbatim
        assert "We discussed pipeline improvements." in text

    def test_filename_is_timestamp_plus_slug(self, tmp_path):
        """Filename pattern lets `sorted(...)` double as chronological
        order without parsing frontmatter."""
        store = MemoryStore(tmp_path)
        store.ensure_dirs()
        path = store.save_session_summary(
            body="x",
            started_at=datetime(2026, 5, 13, 9, 14),
            ended_at=datetime(2026, 5, 13, 10, 2),
            topic_slug="pipeline-improvements",
        )
        assert path.name == "2026-05-13-0914-pipeline-improvements.md"

    def test_project_null_when_missing(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.ensure_dirs()
        path = store.save_session_summary(
            body="x",
            started_at=datetime(2026, 5, 13, 9, 14),
            ended_at=datetime(2026, 5, 13, 10, 2),
            topic_slug="general",
        )
        assert "project: null" in path.read_text()

    def test_load_recent_returns_newest_first(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.ensure_dirs()
        # Write three out of order — Tuesday, Sunday, Wednesday.
        for hour, slug in [(9, "tuesday"), (15, "sunday"), (11, "wednesday")]:
            store.save_session_summary(
                body=f"{slug} body",
                started_at=datetime(2026, 5, 12, hour) if slug == "tuesday"
                else datetime(2026, 5, 10, hour) if slug == "sunday"
                else datetime(2026, 5, 13, hour),
                ended_at=datetime(2026, 5, 13, hour + 1),
                topic_slug=slug,
            )
        entries = store.load_recent_session_summaries(n=3)
        # Filename-sorted desc → Wednesday (5-13), Tuesday (5-12), Sunday (5-10)
        assert [e["topic"] for e in entries] == ["wednesday", "tuesday", "sunday"]

    def test_load_recent_caps_at_n(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.ensure_dirs()
        for i in range(5):
            store.save_session_summary(
                body=f"summary {i}",
                started_at=datetime(2026, 5, i + 1),
                ended_at=datetime(2026, 5, i + 1, 1),
                topic_slug=f"topic-{i}",
            )
        entries = store.load_recent_session_summaries(n=2)
        assert len(entries) == 2

    def test_load_empty_dir_returns_empty(self, tmp_path):
        """Fresh install — no sessions/ yet — must return [] cleanly,
        not raise. The orchestrator's recap path depends on this."""
        store = MemoryStore(tmp_path)
        # NOTE: deliberately not calling ensure_dirs()
        assert store.load_recent_session_summaries() == []


class TestParseSessionSummary:
    def test_roundtrip(self, tmp_path):
        """Files written by save_session_summary must parse back to
        the same metadata fields. This is the most likely place a
        format drift would silently break retrieval."""
        store = MemoryStore(tmp_path)
        store.ensure_dirs()
        path = store.save_session_summary(
            body="prose here",
            started_at=datetime(2026, 5, 13, 9, 14),
            ended_at=datetime(2026, 5, 13, 10, 2),
            topic_slug="pipeline",
            project="microwaveos",
            turn_count=7,
        )
        parsed = _parse_session_summary(path.read_text(), path)
        assert parsed["topic"] == "pipeline"
        assert parsed["project"] == "microwaveos"
        assert parsed["turns"] == 7
        assert parsed["body"] == "prose here"

    def test_missing_frontmatter_falls_through(self, tmp_path):
        """A hand-edited file without frontmatter shouldn't crash —
        whole content becomes body, metadata fields stay empty."""
        path = tmp_path / "x.md"
        path.write_text("just prose, no header")
        parsed = _parse_session_summary(path.read_text(), path)
        assert parsed["body"] == "just prose, no header"
        assert parsed["topic"] == ""
        assert parsed["turns"] == 0

    def test_null_project_becomes_none(self):
        """Frontmatter `project: null` must surface as Python None so
        downstream `if project:` checks work; an empty string would
        give a wrong-shape '[Project: ]' label in recap output."""
        text = (
            "---\nstarted: 2026-05-13T09:00:00\n"
            "ended: 2026-05-13T10:00:00\ntopic: x\n"
            "project: null\nturns: 5\n---\n\nbody"
        )
        parsed = _parse_session_summary(text, Path("/x"))
        assert parsed["project"] is None
