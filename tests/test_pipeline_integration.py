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
