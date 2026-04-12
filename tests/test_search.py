"""Tests for Stage 2: Search and memory retrieval."""

import pytest
from datetime import datetime

from src.memory.search import MemorySearcher
from src.session.models import MemoryFragment


class TestTextSimilarity:
    def test_identical(self):
        sim = MemorySearcher._text_similarity("hello world", "hello world")
        assert sim == 1.0

    def test_no_overlap(self):
        sim = MemorySearcher._text_similarity("hello world", "foo bar")
        assert sim == 0.0

    def test_partial_overlap(self):
        sim = MemorySearcher._text_similarity("hello world foo", "hello bar baz")
        assert 0.0 < sim < 1.0

    def test_empty_string(self):
        sim = MemorySearcher._text_similarity("", "hello")
        assert sim == 0.0


class TestMMRSelect:
    def _make_frag(self, id: int, content: str, score: float) -> MemoryFragment:
        return MemoryFragment(
            id=id, content=content, source="test",
            timestamp=datetime.now(), score=score,
        )

    def test_returns_top_k(self):
        candidates = [
            self._make_frag(1, "alpha beta", 0.9),
            self._make_frag(2, "gamma delta", 0.8),
            self._make_frag(3, "epsilon zeta", 0.7),
        ]
        searcher = MemorySearcher.__new__(MemorySearcher)
        selected = searcher._mmr_select(candidates, k=2, lambda_param=0.7)
        assert len(selected) == 2
        assert selected[0].id == 1  # highest score first

    def test_returns_all_when_fewer_than_k(self):
        candidates = [self._make_frag(1, "only one", 0.9)]
        searcher = MemorySearcher.__new__(MemorySearcher)
        selected = searcher._mmr_select(candidates, k=5, lambda_param=0.7)
        assert len(selected) == 1
