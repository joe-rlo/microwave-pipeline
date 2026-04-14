"""Import conversation history and memory from Hermes Agent.

Data source:
- Database: ~/.hermes/state.db (SQLite with FTS5)
- Memories: ~/.hermes/memories/
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)


def find_hermes_dir() -> Path | None:
    """Find the Hermes data directory."""
    # Check HERMES_HOME env var first
    import os
    hermes_home = os.getenv("HERMES_HOME")
    if hermes_home:
        path = Path(hermes_home)
        if path.exists():
            return path

    default = Path.home() / ".hermes"
    if default.exists():
        return default
    return None


def import_sessions(hermes_dir: Path) -> list[dict]:
    """Import conversation sessions from Hermes.

    Returns list of sessions, each with:
    - session_id, title, started_at, model, turns: [{role, content}]
    """
    db_path = hermes_dir / "state.db"
    if not db_path.exists():
        log.warning(f"Hermes database not found at {db_path}")
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    sessions = []
    try:
        session_rows = conn.execute(
            "SELECT id, title, model, started_at, ended_at, message_count "
            "FROM sessions ORDER BY started_at ASC"
        ).fetchall()

        for srow in session_rows:
            msg_rows = conn.execute(
                "SELECT role, content, tool_calls, tool_name, token_count "
                "FROM messages WHERE session_id = ? ORDER BY rowid ASC",
                (srow["id"],),
            ).fetchall()

            turns = []
            for mrow in msg_rows:
                role = mrow["role"]
                content = mrow["content"] or ""

                # Skip tool calls/results — only import user/assistant text
                if role not in ("user", "assistant"):
                    continue
                if not content.strip():
                    continue

                turns.append({
                    "role": role,
                    "content": content.strip(),
                })

            if turns:
                sessions.append({
                    "session_id": srow["id"],
                    "title": srow["title"] or "",
                    "started_at": srow["started_at"],
                    "model": srow["model"] or "",
                    "turns": turns,
                })

    finally:
        conn.close()

    log.info(f"Imported {len(sessions)} sessions from Hermes")
    return sessions


def import_memories(hermes_dir: Path) -> list[dict]:
    """Import memory files from Hermes.

    Returns list of [{path, content}].
    """
    memories_dir = hermes_dir / "memories"
    if not memories_dir.exists():
        return []

    memories = []
    for md_file in sorted(memories_dir.rglob("*.md")):
        content = md_file.read_text()
        if content.strip():
            rel_path = md_file.relative_to(memories_dir)
            memories.append({
                "path": str(rel_path),
                "content": content,
            })

    log.info(f"Imported {len(memories)} memory files from Hermes")
    return memories
