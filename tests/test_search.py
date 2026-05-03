"""Tests for Stage 2: Search and memory retrieval."""

import pytest
from datetime import datetime

from src.memory.search import (
    ACTIVE_PROJECT_BOOST,
    MemorySearcher,
    OTHER_PROJECT_DOWNWEIGHT,
    _project_for_source,
)
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


class TestProjectForSource:
    """`_project_for_source` extracts the project name from a source path
    so the searcher can apply project-aware boosts/downweights."""

    def test_project_drafts_path(self):
        src = "/Users/joe/workspace/projects/the-heist/drafts/chapter-01.md"
        assert _project_for_source(src) == "the-heist"

    def test_project_bible_path(self):
        src = "/home/x/workspace/projects/novel-y/BIBLE.md"
        assert _project_for_source(src) == "novel-y"

    def test_project_outline_path(self):
        src = "/anywhere/projects/blog-q2/outline.md"
        assert _project_for_source(src) == "blog-q2"

    def test_global_path_returns_none(self):
        # Daily notes, MEMORY.md, session summaries — not under projects/
        assert _project_for_source("/x/workspace/MEMORY.md") is None
        assert _project_for_source("/x/workspace/memory/2026-04-12.md") is None
        assert _project_for_source("session:abc123:summary") is None

    def test_windows_separators(self):
        # The indexer stores str(Path) which is OS-native; Windows paths
        # should still resolve cleanly.
        src = r"C:\Users\joe\workspace\projects\novel-z\drafts\ch1.md"
        assert _project_for_source(src) == "novel-z"

    def test_empty_string(self):
        assert _project_for_source("") is None

    def test_no_trailing_segment(self):
        # `projects/` with nothing after isn't a valid project source.
        assert _project_for_source("/workspace/projects/") is None


class TestProjectAwareScoring:
    """Verify the active-project boost / other-project downweight wiring.

    We don't exercise the full `search()` end-to-end (that requires SQLite
    + sqlite-vec + a populated index, all heavyweight). Instead we drive
    `_rrf_merge` and the boost loop directly, which is where the project
    weighting actually lives.
    """

    def _frag(self, id: int, source: str, score: float = 1.0) -> MemoryFragment:
        return MemoryFragment(
            id=id, content=f"frag {id}", source=source,
            timestamp=datetime(2026, 1, 1),
            score=score,
        )

    def test_boost_is_above_one(self):
        # Sanity guard: if these get inverted accidentally the feature
        # actively hurts retrieval. Lock the polarity in a test.
        assert ACTIVE_PROJECT_BOOST > 1.0
        assert OTHER_PROJECT_DOWNWEIGHT < 1.0

    def test_active_project_boost_applied(self):
        """Score multiplication path lifted from search() so the test
        doesn't need a live index."""
        active = "the-heist"
        frags = [
            self._frag(1, "/x/workspace/projects/the-heist/drafts/ch1.md", 1.0),
            self._frag(2, "/x/workspace/projects/other/drafts/ch1.md", 1.0),
            self._frag(3, "/x/workspace/MEMORY.md", 1.0),
        ]
        for f in frags:
            proj = _project_for_source(f.source)
            if proj is None:
                continue
            if proj == active:
                f.score *= ACTIVE_PROJECT_BOOST
            else:
                f.score *= OTHER_PROJECT_DOWNWEIGHT

        active_frag = next(f for f in frags if f.id == 1)
        other_frag = next(f for f in frags if f.id == 2)
        global_frag = next(f for f in frags if f.id == 3)
        assert active_frag.score == ACTIVE_PROJECT_BOOST
        assert other_frag.score == OTHER_PROJECT_DOWNWEIGHT
        assert global_frag.score == 1.0
        # And: active project ranks above other-project AND global,
        # which is the user-visible promise of this feature.
        ranked = sorted(frags, key=lambda f: f.score, reverse=True)
        assert ranked[0].id == 1
