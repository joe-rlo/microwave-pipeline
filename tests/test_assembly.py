"""Tests for Stage 3: Assembly."""

import pytest
from datetime import date, datetime

from src.health.retrieval.base import Evidence
from src.pipeline.assembly import _format_evidence, _format_fragments
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


def _ev(**kw) -> Evidence:
    return Evidence(
        source=kw.get("source", "pubmed"),
        title=kw.get("title", "T"),
        snippet=kw.get("snippet", "S"),
        url=kw.get("url", "https://example.test/x"),
        published=kw.get("published"),
        quality_score=kw.get("quality_score", 0.5),
    )


class TestFormatEvidence:
    """The [Evidence context] block is the LLM's contract surface for
    citations — the health-qa skill says 'cite by number from this
    block.' Getting the shape right (numbered, source-tagged, URL on
    its own line) is what makes the citation extraction robust."""

    def test_empty_returns_empty(self):
        assert _format_evidence([]) == ""

    def test_single_evidence(self):
        out = _format_evidence([_ev(
            source="pubmed",
            title="Effect of metformin on cardiovascular outcomes",
            snippet="Randomized trial of 5,000 patients showed...",
            url="https://pubmed.ncbi.nlm.nih.gov/12345/",
            published=date(2024, 3, 15),
        )])
        # Header explains the citation contract
        assert "Evidence context" in out
        assert "cite" in out.lower()
        # Citation is numbered and source-tagged
        assert "[1]" in out
        assert "PubMed" in out
        # Title is quoted
        assert '"Effect of metformin' in out
        # URL is on its own line
        assert "https://pubmed.ncbi.nlm.nih.gov/12345/" in out
        # Published date in ISO format
        assert "2024-03-15" in out
        # Excerpt prefix appears
        assert "Excerpt:" in out
        assert "5,000 patients" in out

    def test_multiple_evidence_numbered(self):
        out = _format_evidence([
            _ev(source="pubmed", url="https://x/1"),
            _ev(source="medlineplus", url="https://x/2"),
            _ev(source="openfda", url="https://x/3"),
        ])
        assert "[1]" in out
        assert "[2]" in out
        assert "[3]" in out

    def test_source_display_names(self):
        out = _format_evidence([
            _ev(source="pubmed", url="https://x/1"),
            _ev(source="openfda", url="https://x/2"),
            _ev(source="medlineplus", url="https://x/3"),
            _ev(source="cdc", url="https://x/4"),
            _ev(source="clinicaltrials", url="https://x/5"),
        ])
        assert "PubMed" in out
        assert "openFDA" in out
        assert "MedlinePlus" in out
        assert "CDC" in out
        assert "ClinicalTrials.gov" in out

    def test_undated_evidence_omits_date_label(self):
        out = _format_evidence([_ev(
            source="medlineplus",
            url="https://x/1",
            published=None,
            snippet="A summary.",
        )])
        assert "Published" not in out
        assert "Excerpt: A summary." in out

    def test_no_snippet_no_excerpt_label(self):
        out = _format_evidence([_ev(
            source="medlineplus", url="https://x/1", snippet="",
            published=date(2024, 1, 1),
        )])
        assert "Excerpt:" not in out
        assert "2024-01-01" in out

    def test_unknown_title_handled(self):
        out = _format_evidence([_ev(title="", url="https://x/1")])
        assert "(untitled)" in out
