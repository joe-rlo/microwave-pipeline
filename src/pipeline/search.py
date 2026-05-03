"""Stage 2: Search — triage-driven hybrid retrieval.

Takes triage parameters and runs memory search with those specific settings.
"""

from __future__ import annotations

import logging

from src.memory.search import MemorySearcher
from src.session.models import SearchResult, TriageResult

log = logging.getLogger(__name__)


async def search(
    query: str,
    triage_result: TriageResult,
    searcher: MemorySearcher,
) -> SearchResult:
    """Run memory search shaped by triage parameters.

    If triage says no memory needed, returns empty result immediately.
    """
    if not triage_result.needs_memory:
        log.info("Triage says no memory needed, skipping search")
        return SearchResult(fragments=[], strategy_used="skipped", search_time_ms=0)

    result = await searcher.search(query, triage_result)
    log.info(
        f"Search: {len(result.fragments)} fragments in {result.search_time_ms}ms "
        f"({result.strategy_used})"
    )
    return result
