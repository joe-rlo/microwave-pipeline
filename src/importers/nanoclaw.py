"""Import conversation history and memory from NanoClaw.

Data sources:
- Messages: store/messages.db (SQLite)
- Per-group memory: <group_folder>/CLAUDE.md
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)


def find_nanoclaw_dir(search_path: Path | None = None) -> Path | None:
    """Find the NanoClaw installation directory.

    NanoClaw stores data relative to its install dir, so we look for
    the store/messages.db file.
    """
    candidates = []
    if search_path:
        candidates.append(search_path)

    # Common locations
    candidates.extend([
        Path.home() / "nanoclaw",
        Path.home() / ".nanoclaw",
        Path.home() / "Development" / "nanoclaw",
        Path.home() / "projects" / "nanoclaw",
    ])

    for path in candidates:
        db = path / "store" / "messages.db"
        if db.exists():
            return path
    return None


def import_sessions(nanoclaw_dir: Path) -> list[dict]:
    """Import conversations from NanoClaw.

    Returns list of conversations grouped by chat, each with:
    - chat_jid, turns: [{role, sender, content, timestamp}]
    """
    db_path = nanoclaw_dir / "store" / "messages.db"
    if not db_path.exists():
        log.warning(f"NanoClaw database not found at {db_path}")
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    conversations = []
    try:
        # Get distinct chats
        chats = conn.execute(
            "SELECT DISTINCT chat_jid FROM messages ORDER BY chat_jid"
        ).fetchall()

        for chat in chats:
            chat_jid = chat["chat_jid"]
            msg_rows = conn.execute(
                "SELECT content, sender, sender_name, timestamp, "
                "is_from_me, is_bot_message "
                "FROM messages WHERE chat_jid = ? ORDER BY timestamp ASC",
                (chat_jid,),
            ).fetchall()

            turns = []
            for mrow in msg_rows:
                content = mrow["content"] or ""
                if not content.strip():
                    continue

                # Map NanoClaw roles to user/assistant
                if mrow["is_bot_message"]:
                    role = "assistant"
                else:
                    role = "user"

                turns.append({
                    "role": role,
                    "sender": mrow["sender_name"] or mrow["sender"] or "",
                    "content": content.strip(),
                    "timestamp": mrow["timestamp"],
                })

            if turns:
                conversations.append({
                    "chat_jid": chat_jid,
                    "turns": turns,
                })

    finally:
        conn.close()

    log.info(f"Imported {len(conversations)} conversations from NanoClaw")
    return conversations


def import_memories(nanoclaw_dir: Path) -> list[dict]:
    """Import per-group CLAUDE.md memory files.

    Returns list of [{group, content}].
    """
    memories = []
    for claude_md in sorted(nanoclaw_dir.rglob("CLAUDE.md")):
        content = claude_md.read_text()
        if content.strip():
            group = claude_md.parent.name
            memories.append({
                "group": group,
                "content": content,
            })

    # Also check for conversation FTS content
    db_path = nanoclaw_dir / "store" / "messages.db"
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT group_folder, filename, content FROM conversations "
                "ORDER BY group_folder, date"
            ).fetchall()
            for row in rows:
                memories.append({
                    "group": row["group_folder"],
                    "source": row["filename"],
                    "content": row["content"],
                })
            conn.close()
        except Exception as e:
            log.debug(f"Could not read NanoClaw conversations table: {e}")

    log.info(f"Imported {len(memories)} memory entries from NanoClaw")
    return memories
