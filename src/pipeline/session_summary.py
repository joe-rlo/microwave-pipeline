"""Session-close summarization.

When a session ends (`/new`, app shutdown, timeout), generate a
~200-word Sonnet summary of what was worked on, what was decided, and
what's still hanging. Persist to `workspace/memory/sessions/` as a
markdown file with YAML frontmatter; the indexer picks it up so future
sessions can retrieve it.

The point isn't a perfect minutes-document — it's enough connective
tissue that the next session doesn't start cold. Acceptance: a fresh
`/new` followed by "what were we doing yesterday?" should surface the
relevant summary.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from src.llm.client import SingleTurnClient
from src.session.models import Turn

log = logging.getLogger(__name__)


# Don't summarize tiny sessions — pure social pings or a single
# bot-meta question generate noisy "we said hi" entries that pollute
# retrieval. Threshold is intentionally low (3 substantive turns)
# because real working sessions usually clear it easily.
MIN_TURNS_FOR_SUMMARY = 3


_SUMMARY_PROMPT = """\
You are summarizing a finished conversation between Joe and his personal
AI assistant. The summary will be retrieved verbatim in future sessions
to remind the assistant what was being worked on.

Write a coherent ~200-word summary in plain prose (not bullets). Cover:
- What Joe was working on (the actual subject, not "we chatted")
- Key decisions, plans, or facts established
- Anything left hanging — open questions, things to revisit

Then on a final line, output a topic slug:
  TOPIC: <short-kebab-case-slug>

The slug should name the project or subject area in 1-3 words
(e.g., `pipeline-improvements`, `signal-formatting`, `novel-chapter-4`).
Use `general` if the session truly had no single subject.

Do not include any other formatting (no headers, no JSON, no preamble).
Start the prose immediately and end with the TOPIC: line.\
"""


@dataclass
class SessionSummaryResult:
    body: str           # ~200-word prose
    topic_slug: str     # filename-safe slug, e.g. "pipeline-improvements"
    turn_count: int     # how many turns fed the summary


def _slugify(text: str) -> str:
    """Filename-safe slug, conservative — drops anything that's not
    [a-z0-9-]. Empty input becomes 'general' so callers never end up
    with a slugless filename."""
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "general"


def _format_turns_for_summary(turns: Iterable[Turn]) -> str:
    """Compact transcript: role + content, truncated per turn.

    System turns (compaction summaries) are kept — they're the most
    information-dense rows we have for any older portion of the
    conversation. Skip empty turns defensively."""
    parts = []
    for t in turns:
        content = (t.content or "").strip()
        if not content:
            continue
        # Cap individual turns so a single rambling message can't crowd
        # out everything else from the Sonnet context window.
        if len(content) > 1500:
            content = content[:1500].rstrip() + "…"
        parts.append(f"{t.role}: {content}")
    return "\n\n".join(parts)


def _extract_topic_and_body(raw: str) -> tuple[str, str]:
    """Pull the trailing `TOPIC:` line off the summary and slugify it.

    Tolerant of variations — `Topic:`, missing line break, or no topic
    at all. Falls back to 'general' rather than crashing the close.
    """
    text = raw.strip()
    match = re.search(r"^TOPIC:\s*(.+?)\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    if match:
        topic = _slugify(match.group(1))
        body = (text[:match.start()] + text[match.end():]).strip()
        return topic, body
    return "general", text


async def generate_session_summary(
    turns: list[Turn],
    model: str = "sonnet",
    auth_mode: str = "max",
    api_key: str = "",
    cli_path: str = "",
    workspace_dir: str = "",
) -> SessionSummaryResult | None:
    """Produce a session summary from `turns`, or None if too short.

    Returns None when:
    - turns has fewer than MIN_TURNS_FOR_SUMMARY user+assistant entries
      (the LLM call would just summarize noise)
    - The Sonnet call fails / returns empty (caller should log and skip)

    The summary is not persisted here — the caller (orchestrator close
    hook) decides where it lands so storage and generation can be
    tested independently.
    """
    substantive = [t for t in turns if t.role in ("user", "assistant", "system")]
    if len(substantive) < MIN_TURNS_FOR_SUMMARY:
        log.info(
            f"Session has only {len(substantive)} substantive turn(s); "
            f"skipping summary (min {MIN_TURNS_FOR_SUMMARY})"
        )
        return None

    transcript = _format_turns_for_summary(substantive)
    if not transcript:
        return None

    client = SingleTurnClient(
        model=model, auth_mode=auth_mode, api_key=api_key, cli_path=cli_path,
        workspace_dir=workspace_dir,
    )
    try:
        raw = await client.query(_SUMMARY_PROMPT, transcript[:16000])
    except Exception as e:
        log.warning(f"Session summary generation failed: {e}")
        return None

    if not raw or not raw.strip():
        log.warning("Session summary returned empty")
        return None

    topic, body = _extract_topic_and_body(raw)
    return SessionSummaryResult(
        body=body,
        topic_slug=topic,
        turn_count=len(substantive),
    )
