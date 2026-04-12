"""Tests for Stage 3: Assembly."""

import pytest
from datetime import datetime

from src.pipeline.assembly import _format_fragments
from src.session.models import MemoryFragment


class TestFormatFragments:
    def test_empty_list(self):
        assert _format_fragments([]) == ""

    def test_single_fragment(self):
        frags = [
            MemoryFragment(
                id=1, content="Test fact", source="MEMORY.md",
                timestamp=datetime(2026, 4, 11), score=0.9,
            )
        ]
        result = _format_fragments(frags)
        assert "MEMORY.md" in result
        assert "Test fact" in result
        assert "2026-04-11" in result

    def test_multiple_fragments_numbered(self):
        frags = [
            MemoryFragment(id=1, content="Fact A", source="a.md",
                           timestamp=datetime(2026, 1, 1), score=0.9),
            MemoryFragment(id=2, content="Fact B", source="b.md",
                           timestamp=datetime(2026, 1, 2), score=0.8),
        ]
        result = _format_fragments(frags)
        assert "[1." in result
        assert "[2." in result
