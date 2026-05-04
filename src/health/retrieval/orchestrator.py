"""Parallel fan-out across registered evidence sources.

Single entry point: `RetrievalOrchestrator.search(query, topic)`. The
orchestrator:

1. Calls every registered source's `search()` concurrently, with a
   per-source timeout so one slow source can't stall the whole turn.
2. Catches per-source failures (network, parse, timeout) and logs
   them — one bad source returns 0 results, doesn't break retrieval.
3. Deduplicates results by URL across sources (PubMed and MedlinePlus
   sometimes link to the same NIH page).
4. Ranks by a weighted score combining source authority, recency, and
   query/topic match, then truncates to `max_results`.

The ranking is deliberately simple. Authority dominates because the
LLM should weight a NEJM abstract over a Wikipedia-equivalent;
recency is a tiebreaker, and topic match is a small bonus. Phase 4
revisits this with real-usage tuning.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timezone

from src.health.retrieval.base import Evidence, EvidenceSource

log = logging.getLogger(__name__)


# Hardcoded recency half-life (days) for the ranking score's decay term.
# 5 years aligns with how fast clinical evidence ages out — older
# studies aren't useless but should rank below recent ones.
_RECENCY_HALF_LIFE_DAYS = 5 * 365


class RetrievalOrchestrator:
    """Fans out a query to every registered EvidenceSource in parallel."""

    def __init__(self, sources: list[EvidenceSource]):
        self.sources = list(sources)

    async def search(
        self,
        query: str,
        topic: str | None = None,
        max_results: int = 8,
    ) -> list[Evidence]:
        """Run every source concurrently, dedupe, rank, truncate.

        Returns an empty list rather than raising when no source has
        results. The caller (assembly) treats no-evidence as
        "honestly tell the user we couldn't find authoritative
        coverage" rather than as an error condition.
        """
        if not self.sources:
            log.debug("RetrievalOrchestrator: no sources registered")
            return []

        # Wrap each source's search in its own timeout so one slow
        # source doesn't anchor the whole gather.
        async def _one(src: EvidenceSource) -> list[Evidence]:
            try:
                return await asyncio.wait_for(
                    src.search(query, topic),
                    timeout=src.timeout_seconds,
                )
            except asyncio.TimeoutError:
                log.warning(f"Retrieval source {src.name!r} timed out")
                return []
            except Exception as e:
                # One bad source must NOT break retrieval. Log + continue.
                log.warning(f"Retrieval source {src.name!r} failed: {e}")
                return []

        per_source = await asyncio.gather(*(_one(s) for s in self.sources))
        all_results: list[Evidence] = []
        for results in per_source:
            all_results.extend(results)

        if not all_results:
            return []

        deduped = _dedupe_by_url(all_results)
        ranked = _rank(deduped, query=query, topic=topic, sources=self.sources)
        return ranked[:max_results]


def _dedupe_by_url(results: list[Evidence]) -> list[Evidence]:
    """Keep the first occurrence of each URL.

    Multiple sources can link to the same canonical page (NLM /
    PubMed crosslinks). When that happens, prefer the entry that
    arrived first — sources are gather'd in registration order, and
    callers register higher-authority sources first.
    """
    seen: set[str] = set()
    out: list[Evidence] = []
    for ev in results:
        key = ev.url.strip().lower() if ev.url else ""
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(ev)
    return out


def _rank(
    results: list[Evidence],
    *,
    query: str,
    topic: str | None,
    sources: list[EvidenceSource],
) -> list[Evidence]:
    """Sort results by a weighted score; highest first.

    Score = 0.5 * authority + 0.3 * recency + 0.1 * quality + 0.1 * topic_match
    (clamped to 0..1 components, not enforced as a probability).

    Tweakable; the relative weights matter more than the absolutes.
    Authority dominates because most user queries get better answers
    from one strong NEJM abstract than from five weaker references.
    """
    name_to_authority = {s.name: float(s.authority_weight) for s in sources}
    today = date.today()
    norm_topic = (topic or "").strip().lower()

    def score(ev: Evidence) -> float:
        authority = name_to_authority.get(ev.source, 0.5)
        recency = _recency_score(ev.published, today)
        quality = max(0.0, min(1.0, ev.quality_score))
        topic_match = _topic_match(ev, norm_topic)
        return (
            0.5 * authority
            + 0.3 * recency
            + 0.1 * quality
            + 0.1 * topic_match
        )

    return sorted(results, key=score, reverse=True)


def _recency_score(published: date | None, today: date) -> float:
    """Map a publication date to a 0..1 freshness score.

    Exponential decay with `_RECENCY_HALF_LIFE_DAYS` half-life.
    Undated entries get 0.5 — neither penalized nor rewarded — so
    consumer-facing pages without explicit dates (MedlinePlus) don't
    auto-lose to slightly newer journal articles.
    """
    if published is None:
        return 0.5
    days = max(0, (today - published).days)
    return 0.5 ** (days / _RECENCY_HALF_LIFE_DAYS)


def _topic_match(ev: Evidence, norm_topic: str) -> float:
    """Tiny bonus when the source returned content matching the topic.

    Substring match in title/snippet — cheap, no NLP. The ranking
    formula caps this at a 10% contribution so wrong topic guesses
    by triage don't dominate the rank.
    """
    if not norm_topic:
        return 0.0
    haystack = (ev.title + " " + ev.snippet).lower()
    return 1.0 if norm_topic in haystack else 0.0


def now_utc() -> datetime:
    """Test-friendly UTC clock (importable to monkeypatch)."""
    return datetime.now(tz=timezone.utc)
