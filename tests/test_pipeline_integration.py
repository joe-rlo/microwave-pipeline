"""Integration tests for the cognitive pipeline.

These tests validate the pipeline wiring without live LLM calls.
"""

import pytest
import tempfile
from pathlib import Path

from src.config import Config
from src.memory.store import MemoryStore
from src.session.engine import SessionEngine
from src.session.models import Turn


class TestMemoryStore:
    def test_identity_roundtrip(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.ensure_dirs()
        store.save_identity("# Test Identity")
        assert store.load_identity() == "# Test Identity"

    def test_memory_append(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.ensure_dirs()
        store.append_memory("Fact 1")
        store.append_memory("Fact 2")
        memory = store.load_memory()
        assert "Fact 1" in memory
        assert "Fact 2" in memory

    def test_daily_notes(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.ensure_dirs()
        store.append_daily("Today's note")
        content = store.load_daily()
        assert "Today's note" in content

    def test_stable_context_assembly(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.ensure_dirs()
        store.save_identity("# Identity")
        store.append_memory("Important fact")
        ctx = store.assemble_stable_context()
        assert "# Identity" in ctx
        assert "Important fact" in ctx


class TestSessionEngine:
    def test_add_and_get_turns(self, tmp_path):
        db_path = tmp_path / "test.db"
        engine = SessionEngine(db_path)
        engine.connect()

        sid = engine.new_session_id()
        engine.add_turn(Turn(
            session_id=sid, channel="test", user_id="u1",
            role="user", content="Hello",
        ))
        engine.add_turn(Turn(
            session_id=sid, channel="test", user_id="u1",
            role="assistant", content="Hi there!",
        ))

        turns = engine.get_turns(sid)
        assert len(turns) == 2
        assert turns[0].role == "user"
        assert turns[1].role == "assistant"
        engine.close()

    def test_token_counting(self, tmp_path):
        db_path = tmp_path / "test.db"
        engine = SessionEngine(db_path)
        engine.connect()

        count = engine.count_tokens("Hello world, this is a test.")
        assert count > 0
        engine.close()

    def test_compaction_detection(self, tmp_path):
        db_path = tmp_path / "test.db"
        engine = SessionEngine(db_path, context_limit=100, compaction_threshold=0.5)
        engine.connect()

        sid = engine.new_session_id()
        # Add enough content to trigger compaction
        for i in range(20):
            engine.add_turn(Turn(
                session_id=sid, channel="test", user_id="u1",
                role="user", content=f"Message {i} " * 20,
            ))

        assert engine.needs_compaction(sid)
        engine.close()


class TestImportanceAwareCompaction:
    """Importance-aware split exempts substantive turns from rollup —
    they stay verbatim past the recent-N boundary."""

    def _engine(self, tmp_path):
        engine = SessionEngine(tmp_path / "test.db")
        engine.connect()
        return engine

    def _add(self, engine, sid, role, content, **meta):
        engine.add_turn(Turn(
            session_id=sid, channel="test", user_id="u1",
            role=role, content=content, metadata=meta,
        ))

    def test_no_metadata_treats_all_old_as_summarizable(self, tmp_path):
        engine = self._engine(tmp_path)
        sid = engine.new_session_id()
        for i in range(10):
            self._add(engine, sid, "user", f"msg {i}")
        to_sum, to_keep, recent = engine.get_turns_for_compaction(sid, keep_recent=4)
        assert len(to_sum) == 6
        assert len(to_keep) == 0
        assert len(recent) == 4

    def test_complex_triage_keeps_verbatim(self, tmp_path):
        engine = self._engine(tmp_path)
        sid = engine.new_session_id()
        # 6 simple old turns, 2 complex old turns, 4 recent
        for i in range(6):
            self._add(engine, sid, "assistant", f"simple {i}", triage_complexity="simple")
        for i in range(2):
            self._add(engine, sid, "assistant", f"complex {i}", triage_complexity="complex")
        for i in range(4):
            self._add(engine, sid, "assistant", f"recent {i}")
        to_sum, to_keep, recent = engine.get_turns_for_compaction(sid, keep_recent=4)
        assert len(to_sum) == 6
        assert len(to_keep) == 2
        assert all("complex" in t.content for t in to_keep)
        assert len(recent) == 4

    def test_high_confidence_keeps_verbatim(self, tmp_path):
        engine = self._engine(tmp_path)
        sid = engine.new_session_id()
        for i in range(5):
            self._add(engine, sid, "assistant", f"low {i}", reflection_confidence=0.5)
        for i in range(3):
            self._add(engine, sid, "assistant", f"high {i}", reflection_confidence=0.9)
        for i in range(4):
            self._add(engine, sid, "assistant", f"recent {i}")
        to_sum, to_keep, recent = engine.get_turns_for_compaction(sid, keep_recent=4)
        assert len(to_sum) == 5
        assert len(to_keep) == 3
        assert all("high" in t.content for t in to_keep)

    def test_threshold_tunable(self, tmp_path):
        engine = self._engine(tmp_path)
        sid = engine.new_session_id()
        for i in range(4):
            self._add(engine, sid, "assistant", f"med {i}", reflection_confidence=0.65)
        for i in range(4):
            self._add(engine, sid, "assistant", "recent")
        # Default threshold 0.7 — 0.65 doesn't qualify, all summarized
        to_sum, to_keep, _ = engine.get_turns_for_compaction(sid, keep_recent=4)
        assert len(to_sum) == 4 and len(to_keep) == 0
        # Lower threshold to 0.6 — now 0.65 qualifies as important
        to_sum2, to_keep2, _ = engine.get_turns_for_compaction(
            sid, keep_recent=4, importance_threshold=0.6,
        )
        assert len(to_sum2) == 0 and len(to_keep2) == 4

    def test_either_signal_alone_suffices(self, tmp_path):
        engine = self._engine(tmp_path)
        sid = engine.new_session_id()
        # Complex but low confidence — kept (complex alone is enough)
        self._add(engine, sid, "assistant", "complex-only",
                  triage_complexity="complex", reflection_confidence=0.3)
        # Simple but high confidence — kept (confidence alone is enough)
        self._add(engine, sid, "assistant", "confident-only",
                  triage_complexity="simple", reflection_confidence=0.9)
        # Simple and low confidence — summarized
        self._add(engine, sid, "assistant", "boring",
                  triage_complexity="simple", reflection_confidence=0.4)
        for i in range(4):
            self._add(engine, sid, "assistant", "recent")
        to_sum, to_keep, _ = engine.get_turns_for_compaction(sid, keep_recent=4)
        assert len(to_sum) == 1
        assert len(to_keep) == 2
        assert {t.content for t in to_keep} == {"complex-only", "confident-only"}

    def test_short_session_returns_empty_old_buckets(self, tmp_path):
        engine = self._engine(tmp_path)
        sid = engine.new_session_id()
        for i in range(3):
            self._add(engine, sid, "user", f"msg {i}")
        to_sum, to_keep, recent = engine.get_turns_for_compaction(sid, keep_recent=6)
        assert to_sum == [] and to_keep == []
        assert len(recent) == 3
