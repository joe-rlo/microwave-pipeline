"""Session engine: conversation history, context window management, compaction."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

import apsw
import tiktoken

from src.db import connect as db_connect
from src.session.models import Turn

log = logging.getLogger(__name__)


class SessionEngine:
    def __init__(self, db_path: Path, context_limit: int = 200_000, compaction_threshold: float = 0.8):
        self.db_path = db_path
        self.context_limit = context_limit
        self.compaction_threshold = compaction_threshold
        self.conn: apsw.Connection | None = None
        self._encoder = tiktoken.get_encoding("cl100k_base")

    def connect(self) -> None:
        self.conn = db_connect(self.db_path)
        self._init_tables()

    def _init_tables(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                token_count INTEGER,
                metadata JSON
            )
        """)
        # apsw doesn't support executescript, so run indexes separately
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_turns_channel_user ON turns(channel, user_id)"
        )

    def get_last_session_id(self, channel: str | None = None) -> str | None:
        """Get the most recent session ID, optionally filtered by channel."""
        if channel:
            for row in self.conn.execute(
                "SELECT session_id FROM turns WHERE channel = ? "
                "ORDER BY timestamp DESC LIMIT 1",
                (channel,),
            ):
                return row["session_id"]
        else:
            for row in self.conn.execute(
                "SELECT session_id FROM turns ORDER BY timestamp DESC LIMIT 1",
            ):
                return row["session_id"]
        return None

    def count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def new_session_id(self) -> str:
        return str(uuid.uuid4())[:8]

    def add_turn(self, turn: Turn) -> int:
        """Store a conversation turn. Returns the turn ID."""
        token_count = self.count_tokens(turn.content)
        self.conn.execute(
            "INSERT INTO turns (session_id, channel, user_id, role, content, timestamp, "
            "token_count, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                turn.session_id,
                turn.channel,
                turn.user_id,
                turn.role,
                turn.content,
                turn.timestamp.isoformat(),
                token_count,
                json.dumps(turn.metadata),
            ),
        )
        return self.conn.last_insert_rowid()

    def get_turns(self, session_id: str, limit: int | None = None) -> list[Turn]:
        """Get conversation turns for a session, ordered chronologically."""
        query = "SELECT * FROM turns WHERE session_id = ? ORDER BY timestamp ASC"
        params: list = [session_id]
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        rows = list(self.conn.execute(query, params))
        return [self._row_to_turn(row) for row in rows]

    def get_recent_turns(self, channel: str, user_id: str, limit: int = 10) -> list[Turn]:
        """Get recent turns for a channel/user pair, across sessions."""
        rows = list(self.conn.execute(
            "SELECT * FROM turns WHERE channel = ? AND user_id = ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (channel, user_id, limit),
        ))
        return [self._row_to_turn(row) for row in reversed(rows)]

    def search_recent_turns(
        self,
        query: str,
        hours: float = 48.0,
        limit: int = 5,
        channel: str | None = None,
        user_id: str | None = None,
        exclude_session: str | None = None,
    ) -> list[Turn]:
        """Find recent user/assistant turns matching a query.

        Volume is low (hundreds of turns/day) so a plain LIKE scan is fine —
        no FTS needed. Each query word contributes; turns that match more
        words rank higher, ties broken by recency.
        Returns up to `limit` turns, most relevant first.
        """
        import re

        words = [w.lower() for w in re.findall(r"[a-zA-Z0-9]{3,}", query)]
        if not words:
            return []

        # Build SQL and params together, in order, so placeholder positions
        # match their values. Order: time, optional channel, optional user,
        # optional session exclusion, word LIKEs, limit.
        filters: list[str] = ["role IN ('user', 'assistant', 'system')"]
        params: list = []

        filters.append("timestamp >= datetime('now', ?)")
        params.append(f"-{int(hours * 3600)} seconds")

        if channel:
            filters.append("channel = ?")
            params.append(channel)
        if user_id:
            filters.append("user_id = ?")
            params.append(user_id)
        if exclude_session:
            filters.append("session_id != ?")
            params.append(exclude_session)

        clauses = ["LOWER(content) LIKE ?" for _ in words]
        params.extend(f"%{w}%" for w in words)

        sql = (
            "SELECT * FROM turns WHERE "
            + " AND ".join(filters)
            + " AND (" + " OR ".join(clauses) + ") "
            "ORDER BY timestamp DESC LIMIT ?"
        )
        params.append(limit * 4)  # over-fetch so we can score and trim

        rows = list(self.conn.execute(sql, params))
        if not rows:
            return []

        # Score: word hits + small recency bonus
        now = datetime.now()
        scored: list[tuple[float, Turn]] = []
        for row in rows:
            turn = self._row_to_turn(row)
            content_l = turn.content.lower()
            hits = sum(1 for w in words if w in content_l)
            if hits == 0:
                continue
            age_hours = (now - turn.timestamp).total_seconds() / 3600
            recency = max(0.0, 1.0 - (age_hours / hours))
            scored.append((hits + 0.3 * recency, turn))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [t for _, t in scored[:limit]]

    def session_token_count(self, session_id: str) -> int:
        """Total tokens used in a session."""
        for row in self.conn.execute(
            "SELECT COALESCE(SUM(token_count), 0) AS total FROM turns WHERE session_id = ?",
            (session_id,),
        ):
            return row["total"]
        return 0

    def needs_compaction(self, session_id: str) -> bool:
        """Check if session is approaching context window limit."""
        used = self.session_token_count(session_id)
        return used >= self.context_limit * self.compaction_threshold

    def get_turns_for_compaction(self, session_id: str, keep_recent: int = 6) -> tuple[list[Turn], list[Turn]]:
        """Split turns into old (to compact) and recent (to keep)."""
        all_turns = self.get_turns(session_id)
        if len(all_turns) <= keep_recent:
            return [], all_turns
        split = len(all_turns) - keep_recent
        return all_turns[:split], all_turns[split:]

    def replace_with_summary(self, session_id: str, old_turn_ids: list[int], summary: str) -> None:
        """Replace old turns with a compaction summary."""
        if not old_turn_ids:
            return

        placeholders = ",".join("?" for _ in old_turn_ids)
        self.conn.execute(
            f"DELETE FROM turns WHERE id IN ({placeholders})",
            old_turn_ids,
        )

        self.conn.execute(
            "INSERT INTO turns (session_id, channel, user_id, role, content, token_count, metadata) "
            "VALUES (?, 'system', 'system', 'system', ?, ?, ?)",
            (
                session_id,
                summary,
                self.count_tokens(summary),
                json.dumps({"type": "compaction_summary"}),
            ),
        )
        log.info(f"Compacted {len(old_turn_ids)} turns into summary for session {session_id}")

    def _row_to_turn(self, row) -> Turn:
        return Turn(
            id=row["id"],
            session_id=row["session_id"],
            channel=row["channel"],
            user_id=row["user_id"],
            role=row["role"],
            content=row["content"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            token_count=row["token_count"] or 0,
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def get_recent_stats(self, channel: str, user_id: str, limit: int = 10) -> dict:
        """Aggregate pipeline stats from recent assistant turns."""
        rows = list(self.conn.execute(
            "SELECT metadata FROM turns WHERE channel = ? AND user_id = ? AND role = 'assistant' "
            "ORDER BY timestamp DESC LIMIT ?",
            (channel, user_id, limit),
        ))

        intents: list[str] = []
        fragments: list[int] = []
        confidences: list[float] = []

        for row in rows:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            if meta.get("triage_intent"):
                intents.append(meta["triage_intent"])
            if meta.get("search_fragments") is not None:
                fragments.append(int(meta["search_fragments"]))
            if meta.get("reflection_confidence") is not None:
                confidences.append(float(meta["reflection_confidence"]))

        intent_counts: dict[str, int] = {}
        for intent in intents:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

        return {
            "turns_sampled": len(rows),
            "intent_distribution": intent_counts,
            "avg_search_fragments": sum(fragments) / len(fragments) if fragments else 0.0,
            "avg_reflection_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        }

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None
