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
        description="Memory curation — auditing and inspection helpers for MEMORY.md.",
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
