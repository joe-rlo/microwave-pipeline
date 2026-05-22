"""Pipeline orchestration — ties Extract → Link → Brief together.

One entry point: `run_consolidation()`. The caller (cron job, CLI
command, scheduler) supplies the config and the workspace paths;
this module:

1. Resolves the per-stage LLM callables via the selector
2. Reads inputs for Extract (daily notes + recent turns + recent
   breadcrumbs)
3. Runs Extract, passes the new facts into Link, then runs Brief on
   the resulting graph
4. Returns a structured result with counts so the CLI / scheduler can
   surface them

Why split this from the stage modules: each stage is independently
useful (manual re-run, debugging), and putting the orchestration
elsewhere keeps each stage file focused on one thing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import apsw

from src.memory.consolidation.brief import run_brief
from src.memory.consolidation.extract import run_extract
from src.memory.consolidation.link import run_link
from src.memory.consolidation.schema import init_tables as init_consolidation_tables

log = logging.getLogger(__name__)


# Stage names. Mirror src/llm/selector.py's expected keys so per-stage
# env overrides like `LLM_STAGE_CONSOLIDATION_EXTRACT=near:anthropic/claude-haiku-4-5`
# take effect without further plumbing.
STAGE_EXTRACT = "consolidation_extract"
STAGE_LINK = "consolidation_link"
STAGE_BRIEF = "consolidation_brief"


# Default fallback models per stage (overridden by env overrides). These
# match the §5 stage-defaults table in the spec.
DEFAULT_EXTRACT_MODEL = "claude-haiku-4-5"
DEFAULT_LINK_MODEL = "claude-sonnet-4-6"
DEFAULT_BRIEF_MODEL = "claude-sonnet-4-6"


@dataclass(frozen=True)
class ConsolidationResult:
    """Counts surfaced to the CLI / scheduler / audit log."""

    new_facts: int
    edges: int
    contradictions: int
    briefing_path: str | None
    duration_ms: int


async def run_consolidation(
    *,
    conn: apsw.Connection,
    config: Any,
    daily_notes_dir: Path | None = None,
    breadcrumbs_limit: int = 20,
    recent_turns_limit: int = 50,
    briefing_path: Path | None = None,
    lookback_hours: int = 24,
    now: int | None = None,
) -> ConsolidationResult:
    """Run Extract → Link → Brief.

    Tables are init'd at the top so this can run against a fresh DB
    (e.g., from `python3 src/main.py memory consolidate` on first
    install). Idempotent: re-running on the same input doesn't
    duplicate facts (see `extract._fact_id()`).
    """
    start = time.time()
    init_consolidation_tables(conn)

    extract_call = _stage_callable(config, STAGE_EXTRACT, DEFAULT_EXTRACT_MODEL)
    link_call = _stage_callable(config, STAGE_LINK, DEFAULT_LINK_MODEL)
    brief_call = _stage_callable(config, STAGE_BRIEF, DEFAULT_BRIEF_MODEL)

    # --- Extract ---
    recent_turns = _load_recent_turns(conn, limit=recent_turns_limit)
    recent_breadcrumbs = _load_recent_breadcrumbs(conn, limit=breadcrumbs_limit)

    new_facts = await run_extract(
        conn=conn,
        llm_call=extract_call,
        lookback_hours=lookback_hours,
        daily_notes_dir=daily_notes_dir,
        recent_turns=recent_turns,
        recent_breadcrumbs=recent_breadcrumbs,
        now=now,
    )

    # --- Link ---
    edges, contradictions = await run_link(
        conn=conn,
        new_facts=new_facts,
        llm_call=link_call,
        now=now,
    )

    # --- Brief ---
    brief_out: Path | None = None
    if briefing_path is not None:
        brief_out = await run_brief(
            conn=conn,
            llm_call=brief_call,
            output_path=briefing_path,
            now=now,
        )

    duration_ms = int((time.time() - start) * 1000)
    return ConsolidationResult(
        new_facts=len(new_facts),
        edges=len(edges),
        contradictions=len(contradictions),
        briefing_path=str(brief_out) if brief_out else None,
        duration_ms=duration_ms,
    )


# --- Helpers ---------------------------------------------------------------


def _stage_callable(config: Any, stage: str, fallback_model: str):
    """Build a (system, user) -> str callable for a consolidation stage.

    Mirrors how triage/reflection call `get_stage_callable` — same env
    overrides apply (`LLM_STAGE_CONSOLIDATION_EXTRACT`, etc.).
    """
    from src.llm.selector import get_stage_callable

    return get_stage_callable(
        stage,
        fallback_model=fallback_model,
        auth_mode=getattr(config, "auth_mode", "max"),
        api_key=getattr(config, "anthropic_api_key", ""),
        cli_path=getattr(config, "cli_path", ""),
        workspace_dir=str(getattr(config, "workspace_dir", "")),
    )


def _load_recent_turns(conn: apsw.Connection, *, limit: int) -> list[dict]:
    """Pull the most-recent N turns across all sessions.

    The Extract stage filters by recency in its own logic (lookback
    window). Pulling 50 turns is cheap and gives the model enough
    context to extract from conversational turns.

    Column note: the real `turns` schema uses `timestamp` (DATETIME),
    not `created_at`. Earlier rev assumed `created_at` and silently
    failed in production until smoke-test caught it.
    """
    try:
        rows = list(conn.execute(
            """
            SELECT role, content, timestamp
            FROM turns
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        ))
    except Exception as e:
        log.debug("Consolidation: couldn't read turns (%s); continuing", e)
        return []
    return [dict(r) for r in rows]


def _load_recent_breadcrumbs(conn: apsw.Connection, *, limit: int) -> list[dict]:
    """Pull the most-recent breadcrumb rows for Extract context."""
    try:
        rows = list(conn.execute(
            """
            SELECT trigger, session_key, recent_tools, active_project,
                   active_skill, fired_at
            FROM breadcrumbs
            ORDER BY fired_at DESC
            LIMIT ?
            """,
            (limit,),
        ))
    except Exception:
        return []
    out: list[dict] = []
    for r in rows:
        d = dict(r)
        # recent_tools comes back as a JSON string; decode for the
        # Extract input formatter which expects a list.
        import json
        try:
            d["recent_tools"] = json.loads(d.get("recent_tools") or "[]")
        except json.JSONDecodeError:
            d["recent_tools"] = []
        out.append(d)
    return out
