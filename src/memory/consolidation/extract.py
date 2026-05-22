"""Extract stage — daily notes + recent turns → structured facts.

Per the discipline-paper / memory-pipeline drafts:
- Reads the last N hours of inputs (default 24h)
- Inputs: daily-note files modified in the window, recent turns from
  the session engine, recent breadcrumbs (execution context)
- LLM call: Haiku via the selector under stage `consolidation_extract`
- Output: structured `ExtractedFact` rows, persisted to
  `consolidated_facts`. Returns the inserted rows so the Link stage
  can consume them directly without a re-read.

Design choices worth flagging:

- **Pure-ish.** This module reads from the filesystem and database
  but doesn't reach for global state. The input gathering is split
  into small helpers so tests can mock each piece.
- **No-op on empty.** If the lookback window has no content, returns
  []. Doesn't write anything, doesn't error.
- **Confidence floor.** Facts below MIN_CONFIDENCE (0.5) are dropped.
  The extractor's prompt asks for a calibrated score; we trust it
  enough to filter the low-quality long tail.
- **Stable IDs.** Each fact gets a content-hash-derived ID so re-runs
  on the same input don't create duplicates. Re-extraction is
  idempotent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Awaitable, Callable, Iterable

import apsw

from src.memory.consolidation.schema import (
    FACT_TYPES,
    ExtractedFact,
    FactType,
)

log = logging.getLogger(__name__)


# Facts below this confidence get dropped. The extractor returns floats
# in [0,1]; the prompt tells it to use 0.4+ for anything worth keeping.
# 0.5 is the practical floor — anything lower is noise that pollutes
# the graph without adding signal.
MIN_CONFIDENCE = 0.5


# Schema hint passed to the JSON-retry helper. Mirrors the schema in
# microwave-health-profile-spec.md but for general (non-PHI) facts.
EXTRACT_SCHEMA_HINT = """{
  "facts": [
    {
      "fact_type": "decision" | "preference" | "commitment" | "learning" | "person" | "project_state",
      "content": "<one-sentence fact>",
      "confidence": <float 0.0-1.0>,
      "source_excerpt": "<the sentence(s) this came from>"
    }
  ]
}"""


EXTRACT_SYSTEM_PROMPT = """\
You are an extractor for a personal AI assistant's memory consolidation
pipeline. You read recent conversation turns and daily notes and identify
new, factual information worth preserving in structured form.

What to extract:
- decision: a clear choice made between alternatives
  ("we decided to use Bedrock for BAA")
- preference: a stated user preference about how to work
  ("Joe wants terse responses, no trailing summaries")
- commitment: a date, deadline, or "I will do X by Y"
  ("ship Phase B by 2026-06-15")
- learning: a non-obvious fact or correction the agent surfaced
  ("Bedrock streams events differently from OpenAI SSE")
- person: a named individual + relationship/context
  ("Sarah from Acme Corp, met 2026-05-10, interested in API access")
- project_state: status of an ongoing project or piece of work
  ("microwave-os: Phase A complete, Phase B in progress")

What NOT to extract:
- Hypotheticals or speculation
- Generic platitudes or filler
- Anything from a writing project's fictional context
- Anything you can't trace to a specific source sentence
- Conversational glue ("ok", "thanks", "got it")

Confidence calibration:
- 1.0: stated directly and unambiguously
- 0.7-0.9: stated with minor ambiguity (no date / no scope)
- 0.5-0.6: strongly implied but not explicit
- Below 0.5: do not extract

Return ONLY valid JSON matching this schema:
""" + EXTRACT_SCHEMA_HINT


# A `LLMCall` is (system, user) -> response_text, matching the shape
# that `src.llm.selector.get_stage_callable()` returns.
LLMCall = Callable[[str, str], Awaitable[str]]


async def run_extract(
    *,
    conn: apsw.Connection,
    llm_call: LLMCall,
    lookback_hours: int = 24,
    daily_notes_dir: Path | None = None,
    recent_turns: Iterable[dict] = (),
    recent_breadcrumbs: Iterable[dict] = (),
    now: int | None = None,
) -> list[ExtractedFact]:
    """Run the Extract stage end-to-end.

    Returns the facts that were inserted into `consolidated_facts`.
    Returns [] when there's no content in the lookback window or the
    LLM call returns nothing extractable.
    """
    now_ts = now if now is not None else int(time.time())
    cutoff = now_ts - lookback_hours * 3600

    notes = _read_recent_notes(daily_notes_dir, cutoff) if daily_notes_dir else []
    turn_text = _format_turns(recent_turns)
    breadcrumb_text = _format_breadcrumbs(recent_breadcrumbs)

    user_message = _compose_user_message(notes, turn_text, breadcrumb_text)
    if not user_message.strip():
        log.info("Extract: nothing in lookback window; skipping LLM call")
        return []

    raw = ""
    try:
        raw = await llm_call(EXTRACT_SYSTEM_PROMPT, user_message)
    except Exception as e:
        log.warning("Extract LLM call failed: %s", e)
        return []

    data = _parse_extract_response(raw)
    if data is None:
        log.warning("Extract: malformed JSON; dropping turn")
        return []

    facts: list[ExtractedFact] = []
    for item in data.get("facts", []):
        fact = _validate_and_build_fact(item, now_ts=now_ts)
        if fact is None:
            continue
        # Idempotency: skip if already present
        existing = list(conn.execute(
            "SELECT 1 FROM consolidated_facts WHERE id = ?", (fact.id,),
        ))
        if existing:
            log.debug("Extract: fact %s already present, skipping", fact.id)
            continue
        try:
            conn.execute(
                """
                INSERT INTO consolidated_facts
                    (id, extracted_at, fact_type, content, confidence,
                     source_note, source_excerpt, superseded_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    fact.id, fact.extracted_at, fact.fact_type, fact.content,
                    fact.confidence, fact.source_note, fact.source_excerpt,
                ),
            )
        except Exception as e:
            log.warning("Extract: insert failed for fact %s: %s", fact.id, e)
            continue
        facts.append(fact)

    log.info("Extract: %d new facts persisted", len(facts))
    return facts


# --- Input gathering --------------------------------------------------------


def _read_recent_notes(dir_path: Path, cutoff: int) -> list[tuple[str, str]]:
    """Return (path, content) for every daily note modified after `cutoff`.

    Skips empty files and respects mtime — older notes are excluded
    even if they live in the directory.
    """
    if not dir_path.exists():
        return []
    out: list[tuple[str, str]] = []
    for path in sorted(dir_path.glob("*.md")):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime < cutoff:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if not text.strip():
            continue
        out.append((str(path), text))
    return out


def _format_turns(turns: Iterable[dict]) -> str:
    """Format conversation turns as `role: content` lines."""
    lines: list[str] = []
    for t in turns:
        role = t.get("role", "?")
        content = (t.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _format_breadcrumbs(breadcrumbs: Iterable[dict]) -> str:
    """Format breadcrumb rows as a short header."""
    lines: list[str] = []
    for b in breadcrumbs:
        ts = b.get("fired_at", 0)
        trig = b.get("trigger", "?")
        tools = b.get("recent_tools", [])
        proj = b.get("active_project") or "-"
        skill = b.get("active_skill") or "-"
        lines.append(
            f"[{ts}] trigger={trig} project={proj} skill={skill} "
            f"recent_tools={tools}"
        )
    return "\n".join(lines)


def _compose_user_message(
    notes: list[tuple[str, str]],
    turn_text: str,
    breadcrumb_text: str,
) -> str:
    parts: list[str] = []
    if turn_text:
        parts.append(f"[Recent conversation turns]\n{turn_text}")
    if breadcrumb_text:
        parts.append(f"[Recent execution breadcrumbs]\n{breadcrumb_text}")
    for path, text in notes:
        parts.append(f"[Daily note: {path}]\n{text}")
    return "\n\n".join(parts)


# --- LLM response parsing ---------------------------------------------------


def _parse_extract_response(raw: str) -> dict | None:
    """Parse the LLM's JSON response. Tolerates code-fence wrapping."""
    from src.pipeline.json_utils import extract_json
    return extract_json(raw)


def _validate_and_build_fact(item: dict, *, now_ts: int) -> ExtractedFact | None:
    """Validate one fact dict from the LLM and build an ExtractedFact.

    Returns None for any malformed / under-confident / unsupported entry
    so the caller can skip it without branching. Logs why at debug level
    so prompt-tuning has signal.
    """
    if not isinstance(item, dict):
        return None

    fact_type = item.get("fact_type")
    if fact_type not in FACT_TYPES:
        log.debug("Extract: dropping fact with bad type %r", fact_type)
        return None

    content = (item.get("content") or "").strip()
    if not content:
        log.debug("Extract: dropping fact with empty content")
        return None

    try:
        confidence = float(item.get("confidence", 0.0))
    except (TypeError, ValueError):
        log.debug("Extract: dropping fact with non-numeric confidence")
        return None
    if confidence < MIN_CONFIDENCE:
        log.debug(
            "Extract: dropping low-confidence fact (%.2f < %.2f)",
            confidence, MIN_CONFIDENCE,
        )
        return None

    source_excerpt = (item.get("source_excerpt") or "").strip()
    source_note = (item.get("source_note") or "").strip()

    return ExtractedFact(
        id=_fact_id(fact_type, content),
        extracted_at=now_ts,
        fact_type=fact_type,  # type: ignore[arg-type] — narrowed above
        content=content,
        confidence=confidence,
        source_note=source_note,
        source_excerpt=source_excerpt,
        superseded_by=None,
    )


def _fact_id(fact_type: FactType, content: str) -> str:
    """Stable hash-derived ID — re-running on the same content is idempotent.

    Includes fact_type so two different fact types covering the same
    content don't collide ("a was here" as a decision vs. as a learning).
    """
    h = hashlib.sha256()
    h.update(fact_type.encode("utf-8"))
    h.update(b":")
    h.update(content.encode("utf-8"))
    return "fact_" + h.hexdigest()[:16]
