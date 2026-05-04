"""Health audit log — non-PHI metadata about health-routed turns.

Phase 1 ships only `health_audit` (general path + decline_phi). The
`phi_audit` table from the spec is Phase 2 work — it requires the
per-user encryption key plumbing, which doesn't land until BAA wiring.

What's deliberately NOT in this table: prompt text, response text,
user IDs in cleartext. The spec is explicit: "If something goes wrong,
you reconstruct what happened from the route and source list, not from
a prompt transcript." Audit captures route, sources, model, timing —
enough to debug routing decisions and tune retrieval, never enough to
reconstruct the conversation.

Retention is configurable via `HEALTH_AUDIT_RETENTION_DAYS` (default
2555 days = 7 years to align with HIPAA defaults). Phase 1 doesn't
auto-prune; Phase 2 will.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.db import connect as db_connect

log = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS health_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,           -- unix epoch seconds (UTC)
    route TEXT NOT NULL,                  -- "general" | "phi" | "decline_phi"
    triage_phi_class TEXT,                -- "general" | "personal" | "unknown"
    triage_health_topic TEXT,             -- e.g. "diabetes", or NULL
    sources_queried TEXT,                 -- JSON array of source names
    sources_returned TEXT,                -- JSON array of {name, count}
    llm_provider TEXT,                    -- "anthropic" | "bedrock" | NULL on decline
    llm_model TEXT,
    latency_ms INTEGER,
    token_count_input INTEGER,
    token_count_output INTEGER
    -- intentionally no prompt or response body (see module docstring)
);
"""

_INSERT = """
INSERT INTO health_audit (
    timestamp, route, triage_phi_class, triage_health_topic,
    sources_queried, sources_returned, llm_provider, llm_model,
    latency_ms, token_count_input, token_count_output
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


@dataclass
class HealthAuditRow:
    """Per-turn audit record. Built by the orchestrator at the end of
    a health-routed turn and handed to `HealthAuditWriter.write()`."""

    route: str
    triage_phi_class: str | None = None
    triage_health_topic: str | None = None
    sources_queried: list[str] = field(default_factory=list)
    sources_returned: list[dict] = field(default_factory=list)
    llm_provider: str | None = None
    llm_model: str | None = None
    latency_ms: int | None = None
    token_count_input: int | None = None
    token_count_output: int | None = None


class HealthAuditWriter:
    """Append-only writer for the `health_audit` table.

    Owns its own apsw connection so audit failures can be isolated from
    the rest of the session-engine's traffic. Connections are cheap on
    SQLite — one per writer is fine.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = None

    def connect(self) -> None:
        self.conn = db_connect(self.db_path)
        self.conn.execute(_SCHEMA)

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def write(self, row: HealthAuditRow, *, ts: int | None = None) -> None:
        """Insert one audit row.

        Failures are logged but never propagate — audit is a
        side-channel; an audit DB hiccup must not break the user's
        turn. The downside (missed audit row on a transient SQLite
        error) is acceptable for personal-use scope.
        """
        if self.conn is None:
            log.warning("HealthAuditWriter not connected; dropping audit row")
            return
        try:
            self.conn.execute(
                _INSERT,
                (
                    int(ts if ts is not None else time.time()),
                    row.route,
                    row.triage_phi_class,
                    row.triage_health_topic,
                    json.dumps(row.sources_queried) if row.sources_queried else None,
                    json.dumps(row.sources_returned) if row.sources_returned else None,
                    row.llm_provider,
                    row.llm_model,
                    row.latency_ms,
                    row.token_count_input,
                    row.token_count_output,
                ),
            )
        except Exception as e:
            log.warning(f"Audit write failed (dropping row): {e}")

    def list_recent(self, limit: int = 50) -> list[dict]:
        """Read the most recent audit rows, newest first.

        Used by `health audit list` (CLI) and `health status` to show
        the user what's been routed. Returns dicts (apsw's Row type),
        which the caller can render.
        """
        if self.conn is None:
            return []
        rows = list(self.conn.execute(
            "SELECT * FROM health_audit ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ))
        return [dict(r) for r in rows]
