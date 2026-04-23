"""Tests for the recent-turn retrieval path (Step B).

Covers SessionEngine.search_recent_turns, MemorySearcher._recent_turn_fragments,
and the assembler's "Recent conversation" block.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.memory.search import MemorySearcher
from src.pipeline.assembly import _format_fragments
from src.session.engine import SessionEngine
from src.session.models import MemoryFragment, Turn


def _engine(tmp_path, **kwargs) -> SessionEngine:
    engine = SessionEngine(tmp_path / "test.db", **kwargs)
    engine.connect()
    return engine


def _add(engine, **kw) -> None:
    defaults = dict(
        session_id=kw.pop("session_id", "s1"),
        channel=kw.pop("channel", "telegram"),
        user_id=kw.pop("user_id", "u1"),
        role=kw.pop("role", "user"),
        content=kw.pop("content", ""),
        timestamp=kw.pop("timestamp", datetime.now()),
    )
    engine.add_turn(Turn(**defaults, **kw))


class TestSearchRecentTurns:
    def test_matches_single_word(self, tmp_path):
        eng = _engine(tmp_path)
        _add(eng, content="We discussed the orchestrator pipeline")
        _add(eng, content="Something totally unrelated about cats", role="assistant")
        hits = eng.search_recent_turns("orchestrator")
        assert len(hits) == 1
        assert "orchestrator" in hits[0].content
        eng.close()

    def test_multi_word_ranking(self, tmp_path):
        eng = _engine(tmp_path)
        _add(eng, content="orchestrator only")
        _add(eng, content="orchestrator and pipeline together", role="assistant")
        _add(eng, content="pipeline only", role="user")
        hits = eng.search_recent_turns("orchestrator pipeline")
        # The turn matching both words must rank first
        assert hits[0].content == "orchestrator and pipeline together"
        eng.close()

    def test_respects_time_window(self, tmp_path):
        eng = _engine(tmp_path)
        old = datetime.now() - timedelta(hours=100)
        fresh = datetime.now() - timedelta(hours=1)
        _add(eng, content="orchestrator old", timestamp=old)
        _add(eng, content="orchestrator fresh", timestamp=fresh)
        hits = eng.search_recent_turns("orchestrator", hours=48)
        # The 100h-old turn is outside the 48h window and must be excluded
        contents = [h.content for h in hits]
        assert "orchestrator fresh" in contents
        assert "orchestrator old" not in contents
        eng.close()

    def test_ignores_short_words(self, tmp_path):
        # Words under 3 chars are stripped so stop-words like "is", "a", "to"
        # don't spam-match every turn in the DB.
        eng = _engine(tmp_path)
        _add(eng, content="is a to the of in it")
        hits = eng.search_recent_turns("is a to")
        assert hits == []
        eng.close()

    def test_empty_query_returns_nothing(self, tmp_path):
        eng = _engine(tmp_path)
        _add(eng, content="anything at all")
        assert eng.search_recent_turns("!@#$%") == []
        assert eng.search_recent_turns("") == []
        eng.close()

    def test_channel_and_user_filters(self, tmp_path):
        eng = _engine(tmp_path)
        _add(eng, content="orchestrator from telegram", channel="telegram", user_id="u1")
        _add(eng, content="orchestrator from repl", channel="repl", user_id="u1")
        _add(eng, content="orchestrator other user", channel="telegram", user_id="u2")

        hits = eng.search_recent_turns("orchestrator", channel="telegram", user_id="u1")
        contents = [h.content for h in hits]
        assert "orchestrator from telegram" in contents
        assert "orchestrator from repl" not in contents
        assert "orchestrator other user" not in contents
        eng.close()

    def test_finds_compaction_summary(self, tmp_path):
        # The compaction summary is stored as role='system' with a metadata
        # tag. Recent-turn search must include it so post-compaction context
        # is recoverable.
        eng = _engine(tmp_path)
        _add(
            eng,
            role="system",
            content="Summary: we discussed orchestrator refactor",
            metadata={"type": "compaction_summary"},
        )
        hits = eng.search_recent_turns("orchestrator")
        assert len(hits) == 1
        assert hits[0].role == "system"
        eng.close()


class TestSearcherRecentTurnFragments:
    def test_no_session_engine_returns_empty(self, tmp_path):
        # No engine wired → safe no-op, existing behavior preserved.
        s = MemorySearcher.__new__(MemorySearcher)
        s.session_engine = None
        assert s._recent_turn_fragments("anything") == []

    def test_produces_turn_source_type(self, tmp_path):
        eng = _engine(tmp_path)
        _add(eng, content="orchestrator and pipeline discussion", role="user")
        _add(eng, content="yes we should refactor that", role="assistant")

        s = MemorySearcher.__new__(MemorySearcher)
        s.session_engine = eng
        frags = s._recent_turn_fragments("orchestrator")

        assert len(frags) >= 1
        assert all(f.source_type == "turn" for f in frags)
        # The speaker label must be prepended so the LLM can tell who said what
        assert any(f.content.startswith("User:") for f in frags)
        eng.close()

    def test_summary_turn_labeled_as_summary(self, tmp_path):
        eng = _engine(tmp_path)
        _add(
            eng,
            role="system",
            content="we covered orchestrator changes",
            metadata={"type": "compaction_summary"},
        )
        s = MemorySearcher.__new__(MemorySearcher)
        s.session_engine = eng
        frags = s._recent_turn_fragments("orchestrator")
        assert len(frags) == 1
        assert frags[0].content.startswith("Summary:")
        eng.close()


class TestAssemblyMixedSources:
    def test_fragments_and_turns_rendered_separately(self):
        frags = [
            MemoryFragment(
                id=1, content="Durable fact from MEMORY.md",
                source="/workspace/MEMORY.md",
                timestamp=datetime(2026, 4, 20),
                source_type="fragment",
            ),
            MemoryFragment(
                id=-5, content="User: can you generate this weekly?",
                source="session:abc",
                timestamp=datetime(2026, 4, 22, 15, 36),
                source_type="turn",
            ),
        ]
        out = _format_fragments(frags)
        assert "[Retrieved memory]" in out
        assert "[Recent conversation" in out
        # Durable section shows the source file; turn section doesn't
        assert "MEMORY.md" in out
        assert "User: can you generate this weekly?" in out
        # Sections appear in order: durable first, then turns
        assert out.index("[Retrieved memory]") < out.index("[Recent conversation")

    def test_only_turns_still_renders(self):
        frags = [
            MemoryFragment(
                id=-1, content="Microwave: sure thing",
                source="session:abc",
                timestamp=datetime(2026, 4, 22, 15, 36),
                source_type="turn",
            ),
        ]
        out = _format_fragments(frags)
        assert "[Recent conversation" in out
        assert "[Retrieved memory]" not in out

    def test_only_fragments_still_renders(self):
        # Back-compat: default source_type is "fragment", so old callers
        # that don't set the field get the old behavior.
        frags = [
            MemoryFragment(
                id=1, content="Fact",
                source="MEMORY.md",
                timestamp=datetime(2026, 4, 1),
            ),
        ]
        out = _format_fragments(frags)
        assert "[Retrieved memory]" in out
        assert "[Recent conversation" not in out
