"""Unified ingest — import data from any source into MicrowaveOS memory.

Takes the normalized output from source-specific importers and writes it
into MicrowaveOS's memory index (SQLite + embeddings) and workspace files.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from src.config import load_config
from src.memory.embeddings import EmbeddingClient
from src.memory.index import MemoryIndex
from src.memory.store import MemoryStore

log = logging.getLogger(__name__)


def ingest_sessions(
    sessions: list[dict],
    source_name: str,
    memory_index: MemoryIndex,
) -> int:
    """Ingest conversation sessions into the memory index.

    Each session's turns are chunked, embedded, and stored for vector + FTS search.
    Returns total fragments indexed.
    """
    total = 0
    for session in sessions:
        # Build a single text from the conversation turns
        lines = []
        for turn in session.get("turns", []):
            role = turn["role"]
            content = turn["content"]
            lines.append(f"{role}: {content}")

        text = "\n\n".join(lines)
        if not text.strip():
            continue

        # Parse timestamp
        timestamp = None
        started = session.get("started_at") or session.get("timestamp")
        if started:
            try:
                if isinstance(started, str):
                    timestamp = datetime.fromisoformat(started.replace("Z", "+00:00"))
                else:
                    timestamp = datetime.fromtimestamp(started / 1000)
            except (ValueError, TypeError, OSError):
                pass

        session_id = session.get("session_id", "unknown")
        source = f"{source_name}/session/{session_id}"
        ids = memory_index.index_text(text, source=source, timestamp=timestamp)
        total += len(ids)

    log.info(f"Indexed {total} fragments from {len(sessions)} {source_name} sessions")
    return total


def ingest_memories(
    memories: list[dict],
    source_name: str,
    memory_store: MemoryStore,
    memory_index: MemoryIndex,
    merge_to_memory_md: bool = True,
) -> int:
    """Ingest memory/knowledge files into MicrowaveOS.

    - Indexes all content for vector + FTS search
    - Optionally appends key content to MEMORY.md
    Returns total fragments indexed.
    """
    total = 0
    existing_memory = memory_store.load_memory()

    for mem in memories:
        content = mem.get("content", "")
        if not content.strip():
            continue

        # Determine source label
        path = mem.get("path", mem.get("group", mem.get("source", "unknown")))
        source = f"{source_name}/memory/{path}"

        # Index for search
        ids = memory_index.index_text(content, source=source)
        total += len(ids)

        # Optionally merge to MEMORY.md
        if merge_to_memory_md:
            # For MEMORY.md files, append directly (deduplicated)
            if path in ("MEMORY.md", "") or mem.get("memory_md"):
                for line in content.strip().splitlines():
                    line = line.strip()
                    if line and line not in existing_memory and not line.startswith("#"):
                        memory_store.append_memory(line)
                        existing_memory += "\n" + line

    log.info(f"Indexed {total} fragments from {source_name} memories")
    return total


def ingest_daily_notes(
    daily_notes: list[dict],
    source_name: str,
    memory_store: MemoryStore,
    memory_index: MemoryIndex,
) -> int:
    """Ingest daily notes into MicrowaveOS workspace + index."""
    total = 0
    for note in daily_notes:
        date_str = note.get("date", "")
        content = note.get("content", "")
        if not content.strip() or not date_str:
            continue

        # Write to daily notes file
        from datetime import date
        try:
            day = date.fromisoformat(date_str)
            header = f"[Imported from {source_name}]\n"
            memory_store.append_daily(header + content, day=day)
        except ValueError:
            pass

        # Index for search
        source = f"{source_name}/daily/{date_str}"
        ids = memory_index.index_text(content, source=source)
        total += len(ids)

    log.info(f"Indexed {total} fragments from {len(daily_notes)} {source_name} daily notes")
    return total
