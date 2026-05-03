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
        help="Scan MEMORY.md for likely contradictions (no auto-resolution).",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.action == "health":
        return asyncio.run(_health())

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
