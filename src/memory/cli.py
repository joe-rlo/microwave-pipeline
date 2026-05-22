"""`microwaveos memory ...` subcommand.

Mirrors the scheduler/skills/projects CLI shape. Currently only one
action — `health` — but kept namespaced so future memory-curation
helpers (search, export, etc.) have a natural home.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from src.config import load_config


def memory_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="microwaveos memory",
        description="Memory curation, consolidation, and inspection helpers.",
    )
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser(
        "health",
        help="LLM-based contradiction detector (Haiku pass over MEMORY.md).",
    )
    scan_parser = sub.add_parser(
        "scan",
        help=(
            "Embedding-similarity contradiction scan; writes the queue to "
            "workspace/memory/contradictions.md. No auto-resolution."
        ),
    )
    scan_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Cosine similarity threshold; defaults to 0.80 (Phase A spec). "
            "Lower = more false positives but more flagged pairs."
        ),
    )

    # --- Consolidation pipeline (Phase F) ---
    consolidate_parser = sub.add_parser(
        "consolidate",
        help=(
            "Run the Extract → Link → Brief consolidation pipeline now. "
            "Normally a nightly cron job; this is the manual trigger."
        ),
    )
    consolidate_parser.add_argument(
        "--lookback-hours", type=int, default=24,
        help="Window of daily notes / turns / breadcrumbs to read (default 24).",
    )
    consolidate_parser.add_argument(
        "--no-brief", action="store_true",
        help="Skip the BRIEFING.md write (Extract+Link only).",
    )

    facts_parser = sub.add_parser(
        "facts",
        help="List consolidated facts from the graph.",
    )
    facts_parser.add_argument(
        "--type",
        choices=("decision", "preference", "commitment", "learning",
                 "person", "project_state"),
        default=None,
        help="Filter by fact type.",
    )
    facts_parser.add_argument(
        "--limit", type=int, default=20,
        help="Maximum number to show (default 20, newest first).",
    )
    facts_parser.add_argument(
        "--include-superseded", action="store_true",
        help="Include facts that have been superseded by newer ones.",
    )

    contra_parser = sub.add_parser(
        "contradictions",
        help="List pending contradictions awaiting your resolution.",
    )
    contra_parser.add_argument(
        "--limit", type=int, default=20,
        help="Maximum number to show.",
    )

    resolve_parser = sub.add_parser(
        "resolve",
        help="Resolve a pending contradiction.",
    )
    resolve_parser.add_argument(
        "id", type=int, help="Contradiction id (from `memory contradictions`).",
    )
    resolve_parser.add_argument(
        "--keep",
        choices=("a", "b", "both", "dismiss"),
        required=True,
        help="Which side to keep (or `both` to keep them as siblings, "
             "or `dismiss` to drop the flag without changing facts).",
    )

    sub.add_parser(
        "briefing",
        help="Print the current BRIEFING.md.",
    )

    bc_parser = sub.add_parser(
        "breadcrumbs",
        help="Show recent automatic breadcrumbs (pre-compaction, auto-interval, pre-reset).",
    )
    bc_parser.add_argument(
        "--limit", type=int, default=20,
        help="Maximum number to show (newest first).",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.action == "health":
        return asyncio.run(_health())
    if args.action == "scan":
        return asyncio.run(_scan(threshold=args.threshold))
    if args.action == "consolidate":
        return asyncio.run(_consolidate(
            lookback_hours=args.lookback_hours,
            no_brief=args.no_brief,
        ))
    if args.action == "facts":
        return _facts(
            fact_type=args.type, limit=args.limit,
            include_superseded=args.include_superseded,
        )
    if args.action == "contradictions":
        return _contradictions(limit=args.limit)
    if args.action == "resolve":
        return _resolve(args.id, keep=args.keep)
    if args.action == "briefing":
        return _briefing()
    if args.action == "breadcrumbs":
        return _breadcrumbs(limit=args.limit)

    return 1


async def _health() -> int:
    """Run the contradiction detector and print results.

    Surfaces flagged contradictions without modifying MEMORY.md —
    resolution is always a manual edit. Returns exit code 0 even when
    contradictions are found, since "found contradictions" is a
    successful run, not a process error.
    """
    from src.memory.health import detect_contradictions, format_for_cli
    from src.memory.store import MemoryStore

    config = load_config()
    config.ensure_dirs()
    store = MemoryStore(config.workspace_dir)
    store.ensure_dirs()
    memory_text = store.load_memory()

    if not memory_text.strip():
        print("✓ MEMORY.md is empty — nothing to check.")
        return 0

    contradictions = await detect_contradictions(
        memory_text,
        model=config.model_triage,  # Haiku-tier — same speed/cost as triage
        auth_mode=config.auth_mode,
        api_key=config.anthropic_api_key,
        cli_path=config.cli_path,
        workspace_dir=str(config.workspace_dir),
    )
    print(format_for_cli(contradictions), file=sys.stdout)
    return 0


async def _scan(threshold: float | None) -> int:
    """Run the embedding-similarity scan and persist the queue.

    Pure-vector pass — no LLM call. Cheap enough that the spec
    suggests running it on a weekly cron once Phase A has shipped.
    """
    from src.memory.contradictions import (
        DEFAULT_CONTRADICTION_THRESHOLD,
        find_similar_pairs,
        write_queue,
    )
    from src.memory.embeddings import EmbeddingClient
    from src.memory.index import MemoryIndex
    from src.memory.store import MemoryStore

    config = load_config()
    config.ensure_dirs()
    store = MemoryStore(config.workspace_dir)
    store.ensure_dirs()

    embedder = EmbeddingClient(api_key=config.openai_api_key)
    index = MemoryIndex(config.db_path, embedder)
    index.connect()

    cutoff = threshold if threshold is not None else DEFAULT_CONTRADICTION_THRESHOLD

    # Scope to MEMORY.md specifically — daily notes and project files
    # legitimately repeat themes ("worked on chapter 4") in ways that
    # aren't contradictions. Phase A targets the curated memory file.
    source_filter = f"%{store.memory_path.name}"

    pairs = find_similar_pairs(
        index,
        source_filter=source_filter,
        threshold=cutoff,
    )
    queue_path = store.daily_dir / "contradictions.md"
    write_queue(pairs, queue_path)

    if not pairs:
        print(
            f"✓ No fragment pairs at similarity ≥ {cutoff:.2f}. "
            f"Queue written to {queue_path} (empty)."
        )
    else:
        print(
            f"⚠ {len(pairs)} pair(s) flagged at similarity ≥ {cutoff:.2f}.\n"
            f"Queue written to {queue_path}.\n"
            f"Review and resolve via direct MEMORY.md edits."
        )
    index.close()
    return 0


# --- Phase F: consolidation CLI handlers ---


async def _consolidate(*, lookback_hours: int, no_brief: bool) -> int:
    """Run Extract → Link → Brief once and print a summary.

    The same entrypoint the scheduler calls — running it by hand here
    is the manual-trigger path (debugging, first-run priming).
    """
    from src.memory.breadcrumbs import init_tables as init_bc_tables
    from src.memory.consolidation import run_consolidation
    from src.session.engine import SessionEngine

    config = load_config()
    config.ensure_dirs()

    engine = SessionEngine(config.db_path)
    engine.connect()
    init_bc_tables(engine.conn)

    briefing_path = None if no_brief else config.workspace_dir / "BRIEFING.md"
    daily_dir = config.workspace_dir / "memory"

    result = await run_consolidation(
        conn=engine.conn,
        config=config,
        daily_notes_dir=daily_dir if daily_dir.exists() else None,
        briefing_path=briefing_path,
        lookback_hours=lookback_hours,
    )

    print(
        f"✓ Consolidation complete in {result.duration_ms} ms\n"
        f"  new facts:        {result.new_facts}\n"
        f"  edges added:      {result.edges}\n"
        f"  contradictions:   {result.contradictions}\n"
        f"  briefing written: {result.briefing_path or '(skipped)'}"
    )
    if result.contradictions:
        print(
            f"\n⚠ {result.contradictions} contradiction(s) flagged. "
            f"Run `microwaveos memory contradictions` to review."
        )
    return 0


def _facts(*, fact_type: str | None, limit: int, include_superseded: bool) -> int:
    """List facts from `consolidated_facts`, newest first."""
    from datetime import datetime
    from src.session.engine import SessionEngine

    config = load_config()
    engine = SessionEngine(config.db_path)
    engine.connect()
    conn = engine.conn

    sql = """
        SELECT id, extracted_at, fact_type, content, confidence,
               source_note, superseded_by
        FROM consolidated_facts
        WHERE 1=1
    """
    params: list = []
    if fact_type:
        sql += " AND fact_type = ?"
        params.append(fact_type)
    if not include_superseded:
        sql += " AND superseded_by IS NULL"
    sql += " ORDER BY extracted_at DESC LIMIT ?"
    params.append(limit)

    try:
        rows = list(conn.execute(sql, tuple(params)))
    except Exception as e:
        print(f"Could not read consolidated_facts: {e}")
        print("Hint: run `microwaveos memory consolidate` first to populate.")
        return 1

    if not rows:
        msg = f"No facts of type {fact_type!r}." if fact_type else "No facts in the graph yet."
        print(msg + " Run `memory consolidate` to populate.")
        return 0

    for r in rows:
        when = datetime.fromtimestamp(r["extracted_at"]).strftime("%Y-%m-%d")
        flag = " [superseded]" if r["superseded_by"] else ""
        print(
            f"[{when}] {r['fact_type']:<14} ({r['confidence']:.2f}){flag}"
            f"\n  {r['content']}"
        )
    return 0


def _contradictions(*, limit: int) -> int:
    """List pending contradictions for review."""
    from datetime import datetime
    from src.session.engine import SessionEngine

    config = load_config()
    engine = SessionEngine(config.db_path)
    engine.connect()
    conn = engine.conn

    try:
        rows = list(conn.execute(
            """
            SELECT pc.id, pc.detected_at, pc.explanation,
                   fa.content AS a_content, fb.content AS b_content
            FROM pending_contradictions pc
            JOIN consolidated_facts fa ON fa.id = pc.fact_a_id
            JOIN consolidated_facts fb ON fb.id = pc.fact_b_id
            WHERE pc.status = 'pending'
            ORDER BY pc.detected_at DESC
            LIMIT ?
            """,
            (limit,),
        ))
    except Exception as e:
        print(f"Could not read pending_contradictions: {e}")
        return 1

    if not rows:
        print("✓ No pending contradictions.")
        return 0

    for r in rows:
        when = datetime.fromtimestamp(r["detected_at"]).strftime("%Y-%m-%d")
        print(
            f"[#{r['id']}] {when}\n"
            f"  A: {r['a_content']}\n"
            f"  B: {r['b_content']}\n"
            f"  Why: {r['explanation']}\n"
            f"  Resolve: microwaveos memory resolve {r['id']} --keep a|b|both|dismiss"
        )
    return 0


def _resolve(contradiction_id: int, *, keep: str) -> int:
    """Resolve a pending contradiction.

    `--keep a`: status='accepted_a', fact B marked superseded by A.
    `--keep b`: status='accepted_b', fact A marked superseded by B.
    `--keep both`: status='both_kept', no fact changes.
    `--keep dismiss`: status='dismissed', no fact changes.
    """
    from src.session.engine import SessionEngine

    config = load_config()
    engine = SessionEngine(config.db_path)
    engine.connect()
    conn = engine.conn

    rows = list(conn.execute(
        "SELECT id, fact_a_id, fact_b_id, status FROM pending_contradictions WHERE id = ?",
        (contradiction_id,),
    ))
    if not rows:
        print(f"No contradiction #{contradiction_id}.")
        return 1
    row = rows[0]
    if row["status"] != "pending":
        print(f"Contradiction #{contradiction_id} already resolved as {row['status']!r}.")
        return 1

    new_status_map = {
        "a": "accepted_a",
        "b": "accepted_b",
        "both": "both_kept",
        "dismiss": "dismissed",
    }
    new_status = new_status_map[keep]

    try:
        conn.execute(
            "UPDATE pending_contradictions SET status = ? WHERE id = ?",
            (new_status, contradiction_id),
        )
        if keep == "a":
            conn.execute(
                "UPDATE consolidated_facts SET superseded_by = ? WHERE id = ?",
                (row["fact_a_id"], row["fact_b_id"]),
            )
        elif keep == "b":
            conn.execute(
                "UPDATE consolidated_facts SET superseded_by = ? WHERE id = ?",
                (row["fact_b_id"], row["fact_a_id"]),
            )
    except Exception as e:
        print(f"Could not resolve contradiction: {e}")
        return 1

    print(f"✓ Contradiction #{contradiction_id} resolved as {new_status!r}.")
    return 0


def _briefing() -> int:
    """Print the current BRIEFING.md contents."""
    config = load_config()
    path = config.workspace_dir / "BRIEFING.md"
    if not path.exists():
        print(
            "No BRIEFING.md yet. Run `microwaveos memory consolidate` to "
            "generate one from the current graph."
        )
        return 0
    print(path.read_text(encoding="utf-8"))
    return 0


def _breadcrumbs(*, limit: int) -> int:
    """Show recent breadcrumbs newest-first."""
    from datetime import datetime
    from src.memory.breadcrumbs import (
        init_tables as init_bc_tables,
        recent_breadcrumbs,
    )
    from src.session.engine import SessionEngine

    config = load_config()
    engine = SessionEngine(config.db_path)
    engine.connect()
    init_bc_tables(engine.conn)
    rows = recent_breadcrumbs(engine.conn, limit=limit)

    if not rows:
        print("No breadcrumbs yet — they fire automatically as the bot runs.")
        return 0

    for b in rows:
        when = datetime.fromtimestamp(b.fired_at).strftime("%Y-%m-%d %H:%M:%S")
        proj = b.active_project or "-"
        skill = b.active_skill or "-"
        tools = ", ".join(b.recent_tools) if b.recent_tools else "(none)"
        print(
            f"[{when}] {b.trigger}\n"
            f"  session={b.session_key} turns={b.turn_count} tool_calls={b.tool_call_count}\n"
            f"  project={proj} skill={skill}\n"
            f"  recent_tools={tools}"
        )
    return 0
