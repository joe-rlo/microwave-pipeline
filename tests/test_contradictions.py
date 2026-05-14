"""Pipeline 3.4 Phase A — embedding-similarity contradiction queue.

These tests use hand-crafted embedding vectors so the cosine math is
deterministic and the LLM/embeddings provider never gets called.

Phase A = queue-only. The tests cover:
- `_cosine` math correctness (orthogonal, identical, opposite vectors)
- `find_similar_pairs` filters by threshold and skips exact dupes
- Source-filter scoping (only MEMORY.md fragments by default)
- `render_queue` markdown shape for empty and non-empty cases
- `write_queue` persists to disk and is overwrite-on-each-run
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.memory.contradictions import (
    DEFAULT_CONTRADICTION_THRESHOLD,
    SimilarFragmentPair,
    _cosine,
    find_similar_pairs,
    render_queue,
    write_queue,
)
from src.memory.embeddings import EMBEDDING_DIMENSION
from src.memory.index import MemoryIndex, _serialize_vector


class TestCosine:
    def test_identical_vectors_are_one(self):
        v = [0.6, 0.8]  # already unit-length (3-4-5 triangle)
        assert _cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_are_zero(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert _cosine(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_are_minus_one(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine(a, b) == pytest.approx(-1.0)

    def test_empty_inputs_return_zero(self):
        """Defensive: an unindexed row shouldn't crash the scan; we
        want a 0.0 similarity (which is below any sane threshold) so
        the pair is dropped, not surfaced."""
        assert _cosine([], [1.0]) == 0.0
        assert _cosine([1.0, 2.0], []) == 0.0

    def test_mismatched_length_returns_zero(self):
        """Two embedding versions in the same index would produce
        mismatched dims — better to drop the pair than crash mid-scan."""
        assert _cosine([1.0, 0.0], [0.6, 0.8, 0.0]) == 0.0

    def test_zero_norm_input_returns_zero(self):
        """All-zero vector → undefined cosine → return 0.0 instead of
        crashing on division-by-zero."""
        assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


@pytest.fixture
def index_with_vecs(tmp_path):
    """Build a real MemoryIndex with sqlite-vec but seeded with
    hand-crafted vectors instead of running real embedding calls.

    Yields the index; the test inserts whatever fragments it needs.
    """
    embedder = MagicMock()  # never called — embeddings are hand-inserted
    index = MemoryIndex(tmp_path / "test.db", embedder)
    index.connect()
    if not index._has_vec:
        pytest.skip("sqlite-vec extension not available in this env")
    return index


def _insert(index: MemoryIndex, content: str, source: str, vec: list[float]) -> int:
    """Insert a fragment + its hand-crafted embedding, returning the id.

    Vector is auto-padded to EMBEDDING_DIMENSION so tests can specify
    a tiny vec (just the first few dims) without worrying about the
    1536-dim shape the schema requires.
    """
    padded = list(vec) + [0.0] * (EMBEDDING_DIMENSION - len(vec))
    index.conn.execute(
        "INSERT INTO fragments (content, source, timestamp, embedding_version) "
        "VALUES (?, ?, ?, ?)",
        (content, source, datetime.now().isoformat(), "test"),
    )
    fid = index.conn.last_insert_rowid()
    index.conn.execute(
        "INSERT INTO fragments_fts (rowid, content) VALUES (?, ?)",
        (fid, content),
    )
    index.conn.execute(
        "INSERT INTO fragments_vec (id, embedding) VALUES (?, ?)",
        (fid, _serialize_vector(padded)),
    )
    return fid


class TestFindSimilarPairs:
    def test_returns_empty_when_no_pairs_above_threshold(self, index_with_vecs):
        """Two near-orthogonal vectors should not surface — the queue
        should stay empty rather than padded with noise."""
        _insert(index_with_vecs, "Joe likes coffee.", "/x/MEMORY.md", [1.0, 0.0])
        _insert(index_with_vecs, "Berlin is in Europe.", "/x/MEMORY.md", [0.0, 1.0])
        pairs = find_similar_pairs(index_with_vecs, source_filter="%MEMORY.md")
        assert pairs == []

    def test_flags_high_similarity_pair(self, index_with_vecs):
        """A nearly-identical pair (cos ≈ 0.999) must be flagged."""
        _insert(
            index_with_vecs,
            "Joe's dog is named Biscuit.",
            "/x/MEMORY.md",
            [0.6, 0.8],
        )
        _insert(
            index_with_vecs,
            "Joe's dog is named Max.",
            "/x/MEMORY.md",
            [0.601, 0.799],
        )
        pairs = find_similar_pairs(index_with_vecs, source_filter="%MEMORY.md")
        assert len(pairs) == 1
        assert pairs[0].similarity >= 0.99
        assert "Biscuit" in pairs[0].content_a
        assert "Max" in pairs[0].content_b

    def test_skips_exact_duplicate_text(self, index_with_vecs):
        """Indexing artifacts can produce two rows with identical
        content (e.g. re-index without delete-source). Those are not
        contradictions — they're noise. The queue must skip them."""
        _insert(index_with_vecs, "Joe's dog is Biscuit.", "/x/MEMORY.md", [0.6, 0.8])
        _insert(index_with_vecs, "Joe's dog is Biscuit.", "/x/MEMORY.md", [0.6, 0.8])
        pairs = find_similar_pairs(index_with_vecs, source_filter="%MEMORY.md")
        assert pairs == []

    def test_source_filter_scopes_to_memory_md(self, index_with_vecs):
        """When source_filter is set, fragments from other files
        (daily notes, project files) must not produce cross-source
        pairs — those legitimately repeat themes."""
        _insert(
            index_with_vecs, "Joe's dog is Max.", "/x/MEMORY.md", [0.6, 0.8],
        )
        _insert(
            index_with_vecs, "Talked about Joe's dog Max today.",
            "/x/memory/2026-05-13.md", [0.6, 0.8],
        )
        pairs = find_similar_pairs(index_with_vecs, source_filter="%MEMORY.md")
        # Only one MEMORY.md fragment → no pair possible.
        assert pairs == []

    def test_no_source_filter_includes_all_sources(self, index_with_vecs):
        """Passing source_filter=None should pair across files. The
        flag is opt-in for broader scans, useful for debugging."""
        _insert(index_with_vecs, "Fact A.", "/x/MEMORY.md", [0.6, 0.8])
        _insert(index_with_vecs, "Fact A in other file.", "/x/daily.md", [0.6, 0.8])
        pairs = find_similar_pairs(index_with_vecs, source_filter=None)
        assert len(pairs) == 1

    def test_pairs_sorted_by_similarity_descending(self, index_with_vecs):
        """The most suspicious pair should appear first so the user
        can stop scanning once severity drops off."""
        _insert(index_with_vecs, "A.", "/x/MEMORY.md", [1.0, 0.0])
        _insert(index_with_vecs, "B.", "/x/MEMORY.md", [0.95, 0.31225])  # cos ≈ 0.95
        _insert(index_with_vecs, "C.", "/x/MEMORY.md", [0.85, 0.5267])   # cos ≈ 0.85
        pairs = find_similar_pairs(
            index_with_vecs, source_filter="%MEMORY.md", threshold=0.80,
        )
        # We don't enforce exact length here (depends on inter-pair sims) —
        # just that sorting is descending.
        for i in range(len(pairs) - 1):
            assert pairs[i].similarity >= pairs[i + 1].similarity

    def test_threshold_default_documented(self):
        """The 0.80 default is load-bearing — pin it so a casual bump
        doesn't silently inflate (or shrink) the queue across versions.
        Phase B will tune from queue data; until then, hold this line."""
        assert DEFAULT_CONTRADICTION_THRESHOLD == 0.80

    def test_custom_threshold_filters(self, index_with_vecs):
        """A high custom threshold should drop borderline pairs."""
        _insert(index_with_vecs, "A.", "/x/MEMORY.md", [1.0, 0.0])
        _insert(index_with_vecs, "B.", "/x/MEMORY.md", [0.95, 0.31225])  # cos ≈ 0.95
        pairs_loose = find_similar_pairs(
            index_with_vecs, source_filter="%MEMORY.md", threshold=0.80,
        )
        pairs_strict = find_similar_pairs(
            index_with_vecs, source_filter="%MEMORY.md", threshold=0.99,
        )
        assert len(pairs_loose) >= 1
        assert pairs_strict == []


class TestRenderQueue:
    def test_empty_pairs_show_clean_message(self):
        """An empty queue is a successful state — phrase it as such
        ('looks clean') rather than just printing 'no entries'."""
        out = render_queue([])
        assert "✓" in out
        assert "clean" in out.lower()
        assert "# Contradiction queue" in out

    def test_pair_renders_both_contents_as_blockquotes(self):
        pair = SimilarFragmentPair(
            id_a=5,
            id_b=8,
            content_a="Joe's dog is named Biscuit.",
            content_b="Joe's dog is named Max.",
            source_a="/x/MEMORY.md",
            source_b="/x/MEMORY.md",
            similarity=0.92,
        )
        out = render_queue([pair])
        assert "## 1. Similarity 0.920" in out
        assert "> Joe's dog is named Biscuit." in out
        assert "> Joe's dog is named Max." in out
        assert "#5" in out and "#8" in out

    def test_queue_explains_manual_resolution(self):
        """First-time users running `memory scan` need to know the
        queue is just a flag — resolution is a manual MEMORY.md edit."""
        pair = SimilarFragmentPair(
            id_a=1, id_b=2,
            content_a="A.", content_b="B.",
            source_a="/x/MEMORY.md", source_b="/x/MEMORY.md",
            similarity=0.85,
        )
        out = render_queue([pair])
        assert "manual" in out.lower()
        assert "MEMORY.md" in out

    def test_pair_count_in_header(self):
        """User scanning the file should see the count without
        having to count headings themselves."""
        pairs = [
            SimilarFragmentPair(
                id_a=i, id_b=i + 10,
                content_a=f"A{i}", content_b=f"B{i}",
                source_a="/x/MEMORY.md", source_b="/x/MEMORY.md",
                similarity=0.82,
            )
            for i in range(3)
        ]
        out = render_queue(pairs)
        assert "3 pair" in out


class TestWriteQueue:
    def test_writes_to_disk(self, tmp_path):
        queue_path = tmp_path / "contradictions.md"
        out = write_queue([], queue_path)
        assert out == queue_path
        assert queue_path.exists()
        assert "Contradiction queue" in queue_path.read_text()

    def test_overwrites_on_each_run(self, tmp_path):
        """Last-week's queue shouldn't accumulate — each scan starts
        from a clean slate so resolved entries drop out."""
        queue_path = tmp_path / "contradictions.md"
        pair = SimilarFragmentPair(
            id_a=1, id_b=2,
            content_a="OLD ENTRY", content_b="OLD ENTRY 2",
            source_a="/x/MEMORY.md", source_b="/x/MEMORY.md",
            similarity=0.9,
        )
        write_queue([pair], queue_path)
        assert "OLD ENTRY" in queue_path.read_text()
        write_queue([], queue_path)
        assert "OLD ENTRY" not in queue_path.read_text()

    def test_creates_parent_dirs(self, tmp_path):
        """Fresh install — `workspace/memory/` may not exist yet —
        write_queue should mkdir rather than fail."""
        queue_path = tmp_path / "fresh" / "memory" / "contradictions.md"
        write_queue([], queue_path)
        assert queue_path.exists()
