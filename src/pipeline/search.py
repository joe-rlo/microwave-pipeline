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
    active_project: str | None = None,
    include_phi: bool = True,
) -> SearchResult:
    """Run memory search shaped by triage parameters.

    If triage says no memory needed, returns empty result immediately.

    `active_project` is forwarded to the searcher so retrieval can
    weight the active project's fragments higher than other projects'.

    `include_phi` is forwarded to gate recent-turn recall across the BAA
    boundary — pass False when the current turn is not on the BAA path.
    """
    if not triage_result.needs_memory:
        log.info("Triage says no memory needed, skipping search")
        return SearchResult(fragments=[], strategy_used="skipped", search_time_ms=0)

    result = await searcher.search(
        query, triage_result, active_project=active_project, include_phi=include_phi,
    )
    log.info(
        f"Search: {len(result.fragments)} fragments in {result.search_time_ms}ms "
        f"({result.strategy_used})"
    )
    return result
