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
    output_dir: str = "",
) -> AssemblyResult:
    """Assemble stable and dynamic context for this turn."""
    # Build stable context (for reconnect if needed)
    stable_prompt = memory_store.assemble_stable_context(channel=channel)

    # Build dynamic context: datetime + capabilities + retrieved fragments
    # Datetime goes here (not in stable prompt) so it doesn't trigger reconnects
    dynamic_parts = []
    dynamic_parts.append(f"[Current datetime: {datetime.now().isoformat(timespec='minutes')}]")

    # File output instructions — channel-aware
    if channel == "telegram":
        dynamic_parts.append(
            '[File output — Telegram]\n'
            'Tables, charts, flowcharts, diagrams, timelines, and comparisons '
            'CANNOT be rendered properly in Telegram. Instead, output them as '
            'complete HTML documents inside ```html code fences. The system will '
            'automatically extract them and send as file attachments.\n\n'
            'HTML requirements:\n'
            '- Self-contained: inline CSS, no external stylesheets\n'
            '- Include <!DOCTYPE html> and full <html> structure\n'
            '- Descriptive <title> tag (used for the filename)\n'
            '- Dark mode: use prefers-color-scheme media query\n'
            '- For charts: use Chart.js via CDN\n'
            '- For diagrams: use a ```mermaid code fence (auto-wrapped in HTML)\n\n'
            'CRITICAL mobile layout rules (files open on phones):\n'
            '- viewport meta tag: <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            '- Tables: use width:100%, word-wrap:break-word, and font-size ~14px\n'
            '- Tables: do NOT use fixed-width columns. Use percentage or auto widths.\n'
            '- If a table has 5+ columns, use a card/list layout instead (each row becomes a stacked card)\n'
            '- Body padding: 12px-16px, never 0\n'
            '- Max-width on content containers: 100vw, with overflow-x:auto as fallback\n'
            '- Test assumption: screen is 375px wide\n\n'
            'Write a brief summary OUTSIDE the code fence explaining what the file contains. '
            'The code fence content becomes the file; the surrounding text becomes the message.'
        )
    else:
        dynamic_parts.append(
            '[File output]\n'
            'When creating complete files (code, HTML, configs, documents), '
            'use code fences with language hints. Complete HTML documents '
            '(with <!DOCTYPE html>) will be auto-extracted as downloadable files.'
        )

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
