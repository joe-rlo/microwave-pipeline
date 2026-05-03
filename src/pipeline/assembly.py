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
from src.skills import Skill

log = logging.getLogger(__name__)

PROMOTION_THRESHOLD = 3  # retrievals across different sessions


_LENGTH_HINTS = {
    "simple": (
        "[Response length] The user's message is simple — match its brevity. "
        "Aim for under ~50 words. One short paragraph or a single line is "
        "often right. No preamble, no signoff. Don't pad."
    ),
    "complex": (
        "[Response length] The user is asking for depth — take whatever space "
        "the answer needs. Develop the argument, walk through the steps, "
        "show the reasoning. Don't artificially compress."
    ),
    # 'moderate' adds no hint — that's the default response calibration.
}


def assemble(
    search_result: SearchResult,
    memory_store: MemoryStore,
    memory_index: MemoryIndex,
    channel: str | None = None,
    output_dir: str = "",
    active_skill: Skill | None = None,
    complexity: str = "moderate",
    bible_path=None,
    tool_catalog: str = "",
) -> AssemblyResult:
    """Assemble stable and dynamic context for this turn."""
    # Build stable context (for reconnect if needed). If a project is
    # active, its BIBLE.md joins the stable prompt — that way per-project
    # canon is in front of the LLM for the whole session, not retrieved
    # piecemeal each turn.
    stable_prompt = memory_store.assemble_stable_context(
        channel=channel, bible_path=bible_path
    )

    # Build dynamic context: datetime + capabilities + retrieved fragments
    # Datetime goes here (not in stable prompt) so it doesn't trigger reconnects
    dynamic_parts = []
    dynamic_parts.append(f"[Current datetime: {datetime.now().isoformat(timespec='minutes')}]")

    # Tool catalog block — only present when at least one tool is wired
    # in. The Agent SDK already advertises each tool's name + schema to
    # the model, but those descriptions are terse. The catalog gives the
    # model strategic guidance ("use this when X, ask before calling if
    # Y, don't fabricate the result"). Goes near the top of the dynamic
    # context so it's not buried under retrieval noise.
    if tool_catalog:
        dynamic_parts.append(f"[Tools available]\n{tool_catalog.strip()}")

    # Active skill block goes BEFORE channel file-output instructions so
    # channel rules appear later in the prompt (higher recency priority).
    # The explicit note below reinforces that precedence.
    if active_skill is not None:
        dynamic_parts.append(
            f"[Active skill: {active_skill.name}]\n"
            f"{active_skill.body.strip()}\n\n"
            "Note: these skill instructions are additive to IDENTITY.md. "
            "If anything above conflicts with the channel formatting rules "
            "that follow (message length, markdown syntax, attachment "
            "behavior), the channel rules win."
        )

    # File output instructions — channel-aware
    if channel == "signal":
        dynamic_parts.append(
            '[File output — Signal]\n'
            'Signal renders plain text with only inline formatting '
            '(`**bold**`, `*italic*`, `~strike~`, `` `code` ``). No headers, '
            'no link markup, no tables, no HTML in the message body.\n\n'
            'For tabular data: output a markdown table normally — the channel '
            'will auto-convert it to a card layout (`**Header:** value` blocks) '
            'that reads well on a phone. Do not pre-flatten tables yourself.\n\n'
            'For charts, diagrams, flowcharts, or any visual output: produce a '
            'complete self-contained HTML document inside a ```html code fence. '
            'The pipeline extracts it and sends as a file attachment. Keep the '
            'surrounding text brief — that becomes the message body.\n\n'
            'HTML requirements (same as Telegram):\n'
            '- Self-contained: inline CSS, no external stylesheets\n'
            '- <!DOCTYPE html> + descriptive <title>\n'
            '- Mobile viewport meta tag, 12–16px body padding\n'
            '- Dark mode via prefers-color-scheme'
        )
    elif channel == "telegram":
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

    # Length hint goes LAST so it's the most-recent rule the LLM reads —
    # closest to the user message, hardest to forget while writing.
    length_hint = _LENGTH_HINTS.get(complexity)
    if length_hint:
        dynamic_parts.append(length_hint)

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
    """Format retrieved context for prepending to user message.

    Two sources get rendered as distinct blocks so the LLM can weight them
    appropriately:
    - "Retrieved memory" — durable fragments (MEMORY.md, identity, notes).
      These are curated, committed facts.
    - "Recent conversation" — live turns from the last ~48h. These are raw,
      unvetted, and may contain questions or half-formed thoughts.
    """
    if not fragments:
        return ""

    durable = [f for f in fragments if f.source_type == "fragment"]
    turns = [f for f in fragments if f.source_type == "turn"]

    blocks: list[str] = []
    if durable:
        lines = ["[Retrieved memory]"]
        for i, frag in enumerate(durable, 1):
            source_label = frag.source.split("/")[-1] if "/" in frag.source else frag.source
            ts_label = frag.timestamp.strftime("%Y-%m-%d") if frag.timestamp else "unknown"
            lines.append(f"[{i}. {source_label} ({ts_label})]")
            lines.append(frag.content.strip())
            lines.append("")
        blocks.append("\n".join(lines).rstrip())

    if turns:
        lines = [
            "[Recent conversation — raw turns from the last ~48h, possibly "
            "incomplete or recanted. Treat as context, not as committed facts.]"
        ]
        for frag in turns:
            ts_label = frag.timestamp.strftime("%m-%d %H:%M") if frag.timestamp else "?"
            lines.append(f"[{ts_label}] {frag.content.strip()}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


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
