"""Stage 3: Assembly — system prompt construction and context management.

Builds stable context (system prompt) and dynamic context (per-turn memory).
Identifies fragments for write-back promotion to MEMORY.md.

Reconnect detection is handled by the orchestrator via file mtime, not here.
"""

from __future__ import annotations

import logging
from datetime import datetime

from src.memory.index import MemoryIndex
from src.memory.store import MemoryStore
from src.session.models import AssemblyResult, MemoryFragment, SearchResult

log = logging.getLogger(__name__)

PROMOTION_THRESHOLD = 3  # retrievals across different sessions


def assemble(
    search_result: SearchResult,
    memory_store: MemoryStore,
    memory_index: MemoryIndex,
    channel: str | None = None,
) -> AssemblyResult:
    """Assemble stable and dynamic context for this turn."""
    # Build stable context (for reconnect if needed)
    stable_prompt = memory_store.assemble_stable_context(channel=channel)

    # Build dynamic context: datetime + retrieved fragments
    # Datetime goes here (not in stable prompt) so it doesn't trigger reconnects
    dynamic_parts = []
    dynamic_parts.append(f"[Current datetime: {datetime.now().isoformat(timespec='minutes')}]")

    fragments_text = _format_fragments(search_result.fragments)
    if fragments_text:
        dynamic_parts.append(fragments_text)

    memory_context = "\n\n".join(dynamic_parts)

    # Token budget estimate (rough: 4 chars per token)
    token_budget_used = len(stable_prompt) // 4 + len(memory_context) // 4

    # Check for promotion candidates
    promote_candidates = []
    try:
        candidates = memory_index.get_promotion_candidates(min_retrievals=PROMOTION_THRESHOLD)
        for c in candidates:
            promote_candidates.append(
                MemoryFragment(
                    id=c["id"],
                    content=c["content"],
                    source=c["source"],
                    timestamp=None,
                    retrieval_count=c["retrieval_count"],
                )
            )
    except Exception as e:
        log.debug(f"Promotion check failed: {e}")

    if promote_candidates:
        log.info(f"Found {len(promote_candidates)} fragments eligible for promotion")

    return AssemblyResult(
        stable_prompt=stable_prompt,
        memory_context=memory_context,
        token_budget_used=token_budget_used,
        promote_candidates=promote_candidates,
    )


def _format_fragments(fragments: list[MemoryFragment]) -> str:
    """Format retrieved fragments for prepending to user message."""
    if not fragments:
        return ""

    lines = []
    for i, frag in enumerate(fragments, 1):
        source_label = frag.source.split("/")[-1] if "/" in frag.source else frag.source
        ts_label = frag.timestamp.strftime("%Y-%m-%d") if frag.timestamp else "unknown"
        lines.append(f"[{i}. {source_label} ({ts_label})]")
        lines.append(frag.content.strip())
        lines.append("")

    return "\n".join(lines)


def promote_fragments(
    candidates: list[MemoryFragment],
    memory_store: MemoryStore,
) -> bool:
    """Promote high-retrieval fragments to MEMORY.md. Returns True if anything changed."""
    if not candidates:
        return False

    promoted = []
    existing_memory = memory_store.load_memory()

    for frag in candidates:
        if frag.content.strip() in existing_memory:
            continue
        promoted.append(frag.content.strip())

    if promoted:
        for fact in promoted:
            memory_store.append_memory(fact)
        log.info(f"Promoted {len(promoted)} fragments to MEMORY.md")
        return True

    return False
