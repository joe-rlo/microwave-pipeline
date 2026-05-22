"""Brief stage — graph + breadcrumbs → BRIEFING.md.

The "pre-game routine" surface. Reads the current consolidated graph
plus recent breadcrumbs, asks Sonnet to write a tight executive-style
briefing the next session reads at startup.

What's in the brief (per spec §12.2):
- Active projects + status
- Recent decisions (last 7 days)
- Open commitments + deadlines
- Pending contradictions ("you said X on Tuesday and Y on Friday —
  worth resolving?")
- Personality reminders (preferences)

Token cap: BRIEFING.md targets ~2K tokens so it stays cached after the
first turn of the session (Anthropic prompt cache TTL is 5 minutes
sliding; the brief gets re-cached on every turn that lands within
that window). The prompt tells the model "tight executive summary,
not a transcript."

Output: writes to workspace/BRIEFING.md, returns the path. The
stable-context load order (Phase F.3) reads it into the system prompt
on the next reconnect.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Awaitable, Callable

import apsw

from src.memory.consolidation.schema import ExtractedFact

log = logging.getLogger(__name__)


# Lookback windows for the prompt inputs. Decisions and commitments
# stay relevant longer; breadcrumbs are about recent execution state
# so 24h is plenty.
DECISIONS_LOOKBACK_DAYS = 7
BREADCRUMBS_LIMIT = 10


# Suggested brief length. The model is asked to hit this; it may go
# over for high-activity weeks, which is fine — the consumer caps at
# its own budget anyway.
TARGET_BRIEF_CHARS = 6_000


BRIEF_SYSTEM_PROMPT = """\
You are writing the morning briefing for a personal AI assistant.
Your reader IS the assistant: it starts each new session with your
briefing in its context window so it knows where things stand.

Voice: tight executive summary. No filler. No preamble. Start with
the most important thing. ~2000 tokens max.

Required sections (use these exact headings — markdown level 2):

## Active projects
What's in flight, status, current focus. One bullet per project.

## Recent decisions
Decisions made in the last 7 days. Lead with the decision, not the
discussion. Include the date in parentheses where helpful.

## Open commitments
Anything dated, anything promised. Include deadlines.

## Pending review
Contradictions or proposals waiting for the user. If empty, omit the
section.

## How the user wants to work
Standing preferences — voice, format, anti-patterns. Short bullets.
If you have only one or two, that's fine.

Anti-patterns:
- Don't restate facts that don't change behavior in the next session.
- Don't pad with generic platitudes.
- Don't include explanations of how you got the facts.
- Don't mention this briefing system in the briefing itself.
"""


LLMCall = Callable[[str, str], Awaitable[str]]


async def run_brief(
    *,
    conn: apsw.Connection,
    llm_call: LLMCall,
    output_path: Path,
    lookback_days: int = DECISIONS_LOOKBACK_DAYS,
    breadcrumbs_limit: int = BREADCRUMBS_LIMIT,
    now: int | None = None,
) -> Path | None:
    """Generate BRIEFING.md from the current graph state.

    Returns the path written, or None if there was nothing to brief on
    (fresh install with no facts) — caller skips writing in that case.
    """
    now_ts = now if now is not None else int(time.time())
    cutoff = now_ts - lookback_days * 86400

    projects = _load_facts_of_type(conn, "project_state", cutoff)
    decisions = _load_facts_of_type(conn, "decision", cutoff)
    commitments = _load_facts_of_type(conn, "commitment", cutoff)
    preferences = _load_facts_of_type(conn, "preference", cutoff=None)
    contradictions = _load_pending_contradictions(conn)
    breadcrumbs = _load_recent_breadcrumbs(conn, limit=breadcrumbs_limit)

    if not any([projects, decisions, commitments, preferences, contradictions]):
        log.info("Brief: no facts to summarize; skipping write")
        return None

    user_message = _compose_user_message(
        projects=projects,
        decisions=decisions,
        commitments=commitments,
        preferences=preferences,
        contradictions=contradictions,
        breadcrumbs=breadcrumbs,
    )

    raw = ""
    try:
        raw = await llm_call(BRIEF_SYSTEM_PROMPT, user_message)
    except Exception as e:
        log.warning("Brief LLM call failed: %s", e)
        return None

    brief_text = raw.strip()
    if not brief_text:
        log.warning("Brief: empty response; skipping write")
        return None

    # Write atomically — generate to a tempfile alongside, then rename.
    # A failed brief mid-write must not leave a half-baked file the
    # next session reads as truth.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(brief_text + "\n", encoding="utf-8")
    tmp.replace(output_path)
    log.info("Brief: wrote %d chars to %s", len(brief_text), output_path)
    return output_path


# --- Inputs -----------------------------------------------------------------


def _load_facts_of_type(
    conn: apsw.Connection, fact_type: str, cutoff: int | None,
) -> list[ExtractedFact]:
    """Load active (non-superseded) facts of one type. `cutoff` filters
    by extraction time when set; None means "all time" (used for
    preferences, which don't expire)."""
    sql = """
        SELECT id, extracted_at, fact_type, content, confidence,
               source_note, source_excerpt, superseded_by
        FROM consolidated_facts
        WHERE fact_type = ? AND superseded_by IS NULL
    """
    params: tuple = (fact_type,)
    if cutoff is not None:
        sql += " AND extracted_at >= ?"
        params = (fact_type, cutoff)
    sql += " ORDER BY extracted_at DESC"

    rows = list(conn.execute(sql, params))
    return [
        ExtractedFact(
            id=r["id"], extracted_at=int(r["extracted_at"]),
            fact_type=r["fact_type"], content=r["content"],
            confidence=float(r["confidence"]),
            source_note=r["source_note"], source_excerpt=r["source_excerpt"],
            superseded_by=r["superseded_by"],
        )
        for r in rows
    ]


def _load_pending_contradictions(conn: apsw.Connection) -> list[dict]:
    rows = list(conn.execute(
        """
        SELECT pc.id, pc.detected_at, pc.fact_a_id, pc.fact_b_id, pc.explanation,
               fa.content AS a_content, fb.content AS b_content
        FROM pending_contradictions pc
        JOIN consolidated_facts fa ON fa.id = pc.fact_a_id
        JOIN consolidated_facts fb ON fb.id = pc.fact_b_id
        WHERE pc.status = 'pending'
        ORDER BY pc.detected_at DESC
        """
    ))
    return [dict(r) for r in rows]


def _load_recent_breadcrumbs(conn: apsw.Connection, *, limit: int) -> list[dict]:
    try:
        rows = list(conn.execute(
            "SELECT trigger, session_key, active_project, recent_tools, fired_at "
            "FROM breadcrumbs ORDER BY fired_at DESC LIMIT ?",
            (limit,),
        ))
    except Exception:
        # Breadcrumbs table may not exist in test setups that init only
        # consolidation tables; treat as empty rather than blowing up.
        return []
    return [dict(r) for r in rows]


def _compose_user_message(
    *,
    projects: list[ExtractedFact],
    decisions: list[ExtractedFact],
    commitments: list[ExtractedFact],
    preferences: list[ExtractedFact],
    contradictions: list[dict],
    breadcrumbs: list[dict],
) -> str:
    parts: list[str] = []

    if projects:
        parts.append("[Project state facts]")
        for f in projects:
            parts.append(f"- {f.content}")
        parts.append("")

    if decisions:
        parts.append("[Decisions (last 7 days)]")
        for f in decisions:
            parts.append(f"- {f.content}")
        parts.append("")

    if commitments:
        parts.append("[Open commitments]")
        for f in commitments:
            parts.append(f"- {f.content}")
        parts.append("")

    if preferences:
        parts.append("[User preferences]")
        for f in preferences:
            parts.append(f"- {f.content}")
        parts.append("")

    if contradictions:
        parts.append("[Pending contradictions to surface for review]")
        for c in contradictions:
            parts.append(
                f"- Conflict: \"{c['a_content']}\" vs \"{c['b_content']}\". "
                f"Why: {c['explanation']}"
            )
        parts.append("")

    if breadcrumbs:
        parts.append("[Recent execution breadcrumbs (context only — don't quote)]")
        for b in breadcrumbs:
            parts.append(
                f"- {b.get('trigger')} on {b.get('session_key')} "
                f"project={b.get('active_project') or '-'}"
            )
        parts.append("")

    parts.append(
        f"Write the briefing now. Target around {TARGET_BRIEF_CHARS} characters."
    )
    return "\n".join(parts)
