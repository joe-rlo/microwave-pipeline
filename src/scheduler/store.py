"""Persistence for scheduled jobs.

Stores jobs in the existing memory DB so the scheduler shares a single
source of truth with the session engine and memory index. Keeps
`cron_expr` verbatim and computes `next_fire` on the fly via croniter —
no denormalized next-run column to fall out of sync.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import apsw

from src.db import connect as db_connect

log = logging.getLogger(__name__)


@dataclass
class ScheduledJob:
    id: int | None = None
    name: str = ""
    cron_expr: str = ""
    mode: str = "llm"  # "llm" | "direct" | "script"
    prompt_or_text: str = ""
    target_channel: str = ""
    recipient_id: str = ""
    enabled: bool = True
    timezone: str = "America/New_York"
    # For LLM mode: if set, wrap output as HTML card-view with one card per
    # item, split on this separator. Default "---" matches the convention in
    # the seed Substack prompt.
    card_split: str = "---"
    # Only meaningful for LLM mode. If False, deliver as plain text.
    card_view: bool = True
    # If set, scheduler loads this skill's body as the system prompt for
    # the LLM call and runs any pre-fetch script before the call. The job's
    # `prompt_or_text` is used as the kickoff user message ("the trigger").
    skill_name: str = ""
    created_at: datetime | None = None
    last_run_at: datetime | None = None
    last_error: str | None = None


class SchedulerStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn: apsw.Connection | None = None

    def connect(self) -> None:
        self.conn = db_connect(self.db_path)
        self._init_tables()

    # Canonical table DDL — keep in sync with the migration below so both
    # fresh installs and upgraded installs converge on the same schema.
    # Includes every column that existed as of the last migration; the
    # ALTER-based backfills below are only for DBs created *before* that
    # column was added to this canonical DDL.
    _CREATE_SQL = """
        CREATE TABLE IF NOT EXISTS scheduled_jobs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT UNIQUE NOT NULL,
            cron_expr       TEXT NOT NULL,
            mode            TEXT NOT NULL CHECK (mode IN ('llm', 'direct', 'script')),
            prompt_or_text  TEXT NOT NULL,
            target_channel  TEXT NOT NULL,
            recipient_id    TEXT NOT NULL,
            enabled         INTEGER NOT NULL DEFAULT 1,
            timezone        TEXT NOT NULL DEFAULT 'America/New_York',
            card_split      TEXT NOT NULL DEFAULT '---',
            card_view       INTEGER NOT NULL DEFAULT 1,
            skill_name      TEXT NOT NULL DEFAULT '',
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            last_run_at     TEXT,
            last_error      TEXT
        )
    """

    def _init_tables(self) -> None:
        # Older DBs have a CHECK constraint of IN ('llm', 'direct') that
        # blocks inserting `mode='script'`. SQLite can't ALTER a CHECK in
        # place, so rebuild the table if we detect the old signature.
        self._migrate_relax_mode_check()

        self.conn.execute(self._CREATE_SQL)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scheduled_jobs_enabled ON scheduled_jobs(enabled)"
        )
        # Backfill new columns if the table predates them
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(scheduled_jobs)")}
        if "card_split" not in cols:
            self.conn.execute(
                "ALTER TABLE scheduled_jobs ADD COLUMN card_split TEXT NOT NULL DEFAULT '---'"
            )
        if "card_view" not in cols:
            self.conn.execute(
                "ALTER TABLE scheduled_jobs ADD COLUMN card_view INTEGER NOT NULL DEFAULT 1"
            )
        if "skill_name" not in cols:
            self.conn.execute(
                "ALTER TABLE scheduled_jobs ADD COLUMN skill_name TEXT NOT NULL DEFAULT ''"
            )

    def _migrate_relax_mode_check(self) -> None:
        """Rebuild scheduled_jobs if its CHECK constraint predates 'script' mode."""
        rows = list(self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='scheduled_jobs'"
        ))
        if not rows:
            return  # fresh install — CREATE below will do the right thing
        ddl = rows[0]["sql"] or ""
        # Detect the pre-script signature. Use a loose match so we don't miss
        # single/double quote variations or whitespace differences.
        if "'script'" in ddl:
            return  # already migrated
        if "'llm'" not in ddl or "'direct'" not in ddl:
            return  # unrecognized — leave alone, safer than blindly rewriting

        log.info("Migrating scheduled_jobs: relaxing mode CHECK to include 'script'")
        existing_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(scheduled_jobs)")}
        self.conn.execute("BEGIN")
        try:
            self.conn.execute(self._CREATE_SQL.replace("scheduled_jobs", "scheduled_jobs_new"))
            # Intersect old + new column sets so the migration survives drift
            # in either direction (missing columns backfill via DEFAULTs).
            new_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(scheduled_jobs_new)")}
            shared = [c for c in existing_cols if c in new_cols]
            col_list = ", ".join(shared)
            self.conn.execute(
                f"INSERT INTO scheduled_jobs_new ({col_list}) "
                f"SELECT {col_list} FROM scheduled_jobs"
            )
            self.conn.execute("DROP TABLE scheduled_jobs")
            self.conn.execute("ALTER TABLE scheduled_jobs_new RENAME TO scheduled_jobs")
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def add(self, job: ScheduledJob) -> int:
        self.conn.execute(
            "INSERT INTO scheduled_jobs "
            "(name, cron_expr, mode, prompt_or_text, target_channel, recipient_id, "
            " enabled, timezone, card_split, card_view, skill_name) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job.name, job.cron_expr, job.mode, job.prompt_or_text,
                job.target_channel, job.recipient_id,
                int(job.enabled), job.timezone,
                job.card_split, int(job.card_view), job.skill_name,
            ),
        )
        return self.conn.last_insert_rowid()

    def remove(self, name_or_id: str | int) -> bool:
        """Delete a job by name or integer id. Returns True if anything was deleted."""
        if isinstance(name_or_id, int) or str(name_or_id).isdigit():
            self.conn.execute(
                "DELETE FROM scheduled_jobs WHERE id = ?", (int(name_or_id),)
            )
        else:
            self.conn.execute(
                "DELETE FROM scheduled_jobs WHERE name = ?", (name_or_id,)
            )
        return self.conn.changes() > 0

    def set_enabled(self, name_or_id: str | int, enabled: bool) -> bool:
        if isinstance(name_or_id, int) or str(name_or_id).isdigit():
            self.conn.execute(
                "UPDATE scheduled_jobs SET enabled = ? WHERE id = ?",
                (int(enabled), int(name_or_id)),
            )
        else:
            self.conn.execute(
                "UPDATE scheduled_jobs SET enabled = ? WHERE name = ?",
                (int(enabled), name_or_id),
            )
        return self.conn.changes() > 0

    def list_all(self) -> list[ScheduledJob]:
        rows = list(self.conn.execute(
            "SELECT * FROM scheduled_jobs ORDER BY name"
        ))
        return [self._row_to_job(r) for r in rows]

    def list_enabled(self) -> list[ScheduledJob]:
        rows = list(self.conn.execute(
            "SELECT * FROM scheduled_jobs WHERE enabled = 1 ORDER BY name"
        ))
        return [self._row_to_job(r) for r in rows]

    def get_by_name(self, name: str) -> ScheduledJob | None:
        for r in self.conn.execute(
            "SELECT * FROM scheduled_jobs WHERE name = ?", (name,)
        ):
            return self._row_to_job(r)
        return None

    def mark_ran(self, job_id: int, error: str | None) -> None:
        self.conn.execute(
            "UPDATE scheduled_jobs SET last_run_at = datetime('now'), last_error = ? "
            "WHERE id = ?",
            (error, job_id),
        )

    def mark_baseline(self, job_id: int, when: datetime) -> None:
        """Fast-forward last_run_at without recording a fire.

        Used for stale jobs: when the daemon comes back online after missed
        fires, we don't want to catch up the whole backlog — we just set a
        baseline so the next real fire happens at the next scheduled time.
        """
        self.conn.execute(
            "UPDATE scheduled_jobs SET last_run_at = ? WHERE id = ?",
            (when.isoformat(), job_id),
        )

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    # --- helpers ---

    def _row_to_job(self, row) -> ScheduledJob:
        def _parse(ts: str | None) -> datetime | None:
            if not ts:
                return None
            try:
                return datetime.fromisoformat(ts)
            except ValueError:
                # SQLite's `datetime('now')` uses space separator; isoformat
                # requires T. Normalize.
                return datetime.fromisoformat(ts.replace(" ", "T"))

        return ScheduledJob(
            id=row["id"],
            name=row["name"],
            cron_expr=row["cron_expr"],
            mode=row["mode"],
            prompt_or_text=row["prompt_or_text"],
            target_channel=row["target_channel"],
            recipient_id=row["recipient_id"],
            enabled=bool(row["enabled"]),
            timezone=row["timezone"],
            card_split=row["card_split"] or "---",
            card_view=bool(row["card_view"]),
            skill_name=row["skill_name"] or "",
            created_at=_parse(row["created_at"]),
            last_run_at=_parse(row["last_run_at"]),
            last_error=row["last_error"],
        )
