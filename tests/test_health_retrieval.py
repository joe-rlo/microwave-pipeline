"""Retrieval-layer tests — fan-out, dedup, ranking, and per-source
failure isolation.

We use synthetic EvidenceSource subclasses here rather than mocking
out network calls; the orchestrator's contract is "every source is an
async function returning a list" and that's what we exercise.
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta

import pytest

from src.health.retrieval.base import Evidence, EvidenceSource
from src.health.retrieval.orchestrator import (
    RetrievalOrchestrator,
    _dedupe_by_url,
    _rank,
    _recency_score,
    _topic_match,
)


def _ev(**kw) -> Evidence:
    """Test helper — build an Evidence with sensible defaults."""
    return Evidence(
        source=kw.get("source", "pubmed"),
        title=kw.get("title", "T"),
        snippet=kw.get("snippet", "S"),
        url=kw.get("url", "https://example.test/x"),
        published=kw.get("published"),
        quality_score=kw.get("quality_score", 0.5),
        raw=kw.get("raw", {}),
    )


class _FakeSource(EvidenceSource):
    """Returns canned results, optionally raises, optionally sleeps."""

    def __init__(
        self,
        name: str,
        results: list[Evidence] | Exception | None = None,
        delay: float = 0.0,
        timeout: float = 5.0,
        authority: float = 0.5,
    ):
        self.name = name
        self.authority_weight = authority
        self.timeout_seconds = timeout
        self._results = results if results is not None else []
        self._delay = delay
        self.calls: list[tuple[str, str | None]] = []

    async def search(self, query: str, topic: str | None = None) -> list[Evidence]:
        self.calls.append((query, topic))
        if self._delay:
            await asyncio.sleep(self._delay)
        if isinstance(self._results, Exception):
            raise self._results
        # Stamp source name on each result for test convenience
        return [
            Evidence(
                source=self.name,
                title=r.title,
                snippet=r.snippet,
                url=r.url,
                published=r.published,
                quality_score=r.quality_score,
                raw=r.raw,
            )
            for r in self._results
        ]


class TestDedupe:
    def test_keeps_first_occurrence(self):
        a = _ev(url="https://x.test/a", title="first")
        b = _ev(url="https://x.test/a", title="second")
        c = _ev(url="https://x.test/c", title="third")
        out = _dedupe_by_url([a, b, c])
        assert len(out) == 2
        assert out[0].title == "first"
        assert out[1].title == "third"

    def test_case_insensitive_url(self):
        a = _ev(url="https://X.test/A")
        b = _ev(url="https://x.test/a")
        out = _dedupe_by_url([a, b])
        assert len(out) == 1

    def test_drops_empty_url_entries(self):
        out = _dedupe_by_url([_ev(url=""), _ev(url="https://x.test/a")])
        assert len(out) == 1
        assert out[0].url == "https://x.test/a"


class TestRecencyScore:
    def test_undated_neutral(self):
        assert _recency_score(None, date(2026, 5, 3)) == 0.5

    def test_today_full(self):
        d = date(2026, 5, 3)
        assert _recency_score(d, d) == 1.0

    def test_old_decays(self):
        today = date(2026, 5, 3)
        old = today - timedelta(days=5 * 365)  # one half-life
        assert _recency_score(old, today) == pytest.approx(0.5, abs=0.01)
        ancient = today - timedelta(days=10 * 365)  # two half-lives
        assert _recency_score(ancient, today) == pytest.approx(0.25, abs=0.01)


class TestTopicMatch:
    def test_no_topic_no_bonus(self):
        ev = _ev(title="x", snippet="y")
        assert _topic_match(ev, "") == 0.0

    def test_match_in_title(self):
        ev = _ev(title="Diabetes management", snippet="...")
        assert _topic_match(ev, "diabetes") == 1.0

    def test_match_in_snippet(self):
        ev = _ev(title="x", snippet="An overview of metformin")
        assert _topic_match(ev, "metformin") == 1.0

    def test_no_match(self):
        ev = _ev(title="cancer", snippet="treatment")
        assert _topic_match(ev, "diabetes") == 0.0


class TestRanking:
    def test_authority_dominates(self):
        """Higher-authority source ranks above lower-authority even when
        the lower one has a slightly more recent date — authority is
        the load-bearing signal."""
        recent = date(2026, 1, 1)
        older = date(2024, 1, 1)
        high = _ev(source="pubmed", url="https://a", published=older)
        low = _ev(source="cdc", url="https://b", published=recent)
        sources = [
            _FakeSource("pubmed", authority=1.0),
            _FakeSource("cdc", authority=0.3),
        ]
        ranked = _rank([low, high], query="q", topic=None, sources=sources)
        assert ranked[0].source == "pubmed"

    def test_recency_breaks_ties(self):
        """Same source -> recency wins."""
        sources = [_FakeSource("pubmed", authority=1.0)]
        old = _ev(source="pubmed", url="https://a", published=date(2020, 1, 1))
        new = _ev(source="pubmed", url="https://b", published=date(2026, 1, 1))
        ranked = _rank([old, new], query="q", topic=None, sources=sources)
        assert ranked[0].url == "https://b"

    def test_topic_match_bonus(self):
        sources = [_FakeSource("pubmed", authority=0.5)]
        no_topic = _ev(source="pubmed", url="https://a", title="general health")
        on_topic = _ev(source="pubmed", url="https://b", title="diabetes care")
        ranked = _rank(
            [no_topic, on_topic], query="q", topic="diabetes", sources=sources,
        )
        assert ranked[0].title == "diabetes care"


class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_no_sources_returns_empty(self):
        orch = RetrievalOrchestrator(sources=[])
        assert await orch.search("anything") == []

    @pytest.mark.asyncio
    async def test_fans_out_to_all_sources(self):
        a = _FakeSource("pubmed", results=[_ev(url="https://x.test/1", title="A")])
        b = _FakeSource("medlineplus", results=[_ev(url="https://x.test/2", title="B")])
        orch = RetrievalOrchestrator(sources=[a, b])
        results = await orch.search("metformin", topic="diabetes")
        # Both sources called with the same args
        assert a.calls == [("metformin", "diabetes")]
        assert b.calls == [("metformin", "diabetes")]
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_one_source_failure_does_not_kill_retrieval(self):
        """The whole point of the per-source try/except — a flaky API
        doesn't take down the turn."""
        good = _FakeSource("pubmed", results=[_ev(url="https://x.test/1")])
        bad = _FakeSource("openfda", results=RuntimeError("API down"))
        orch = RetrievalOrchestrator(sources=[good, bad])
        results = await orch.search("anything")
        assert len(results) == 1
        assert results[0].source == "pubmed"

    @pytest.mark.asyncio
    async def test_timeout_caps_slow_source(self):
        slow = _FakeSource(
            "cdc",
            results=[_ev(url="https://x.test/1")],
            delay=1.0,
            timeout=0.05,  # forces timeout
        )
        fast = _FakeSource(
            "pubmed", results=[_ev(url="https://x.test/2")],
        )
        orch = RetrievalOrchestrator(sources=[slow, fast])
        results = await orch.search("anything")
        # slow timed out; fast came through
        assert len(results) == 1
        assert results[0].source == "pubmed"

    @pytest.mark.asyncio
    async def test_truncates_to_max_results(self):
        many = _FakeSource(
            "pubmed",
            results=[_ev(url=f"https://x.test/{i}") for i in range(20)],
        )
        orch = RetrievalOrchestrator(sources=[many])
        results = await orch.search("q", max_results=5)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_dedupes_across_sources(self):
        """Two sources returning the same URL — dedup keeps only one."""
        a = _FakeSource("pubmed", results=[_ev(url="https://shared.test/x", title="A")])
        b = _FakeSource("medlineplus", results=[_ev(url="https://shared.test/x", title="B")])
        orch = RetrievalOrchestrator(sources=[a, b])
        results = await orch.search("anything")
        assert len(results) == 1
        # Higher-authority source wins (pubmed registered first)
        assert results[0].title == "A"
