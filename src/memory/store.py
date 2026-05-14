"""Read/write markdown workspace files (IDENTITY.md, MEMORY.md, daily notes)."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)


def _parse_session_summary(text: str, path: Path) -> dict:
    """Lenient frontmatter parser for session summary files.

    Not a real YAML parser — frontmatter is a flat string→string map by
    convention, so a line-by-line split is enough and avoids pulling in
    PyYAML for this one feature. If the file has no frontmatter, the
    whole content becomes `body` and metadata fields stay empty.
    """
    meta: dict[str, str] = {}
    body = text
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            header = text[4:end]
            body = text[end + 5:].strip()
            for line in header.splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    meta[k.strip()] = v.strip()
    return {
        "path": path,
        "started": meta.get("started", ""),
        "ended": meta.get("ended", ""),
        "topic": meta.get("topic", ""),
        "project": meta.get("project", "") if meta.get("project") not in (None, "null", "") else None,
        "turns": int(meta["turns"]) if meta.get("turns", "").isdigit() else 0,
        "body": body,
    }


class MemoryStore:
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self.identity_path = workspace_dir / "IDENTITY.md"
        self.memory_path = workspace_dir / "MEMORY.md"
        self.daily_dir = workspace_dir / "memory"

    @property
    def channels_dir(self) -> Path:
        return self.workspace_dir / "channels"

    @property
    def sessions_dir(self) -> Path:
        """Cross-session summaries — one markdown file per closed session.

        Lives under `daily_dir` so the indexer picks it up alongside daily
        notes without needing a second corpus root. Filenames carry a
        timestamp + topic slug so listing-by-name doubles as listing-by-time.
        """
        return self.daily_dir / "sessions"

    def ensure_dirs(self) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self.channels_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    # --- Identity ---

    def load_identity(self) -> str:
        if self.identity_path.exists():
            return self.identity_path.read_text()
        return ""

    def save_identity(self, content: str) -> None:
        self.identity_path.write_text(content)

    # --- Long-term memory ---

    def load_memory(self) -> str:
        if self.memory_path.exists():
            return self.memory_path.read_text()
        return ""

    def save_memory(self, content: str) -> None:
        self.memory_path.write_text(content)

    def append_memory(self, fact: str) -> None:
        existing = self.load_memory()
        separator = "\n\n" if existing.strip() else ""
        self.memory_path.write_text(existing + separator + fact.strip() + "\n")

    # --- Daily notes ---

    def daily_path(self, day: date | None = None) -> Path:
        day = day or date.today()
        return self.daily_dir / f"{day.isoformat()}.md"

    def load_daily(self, day: date | None = None) -> str:
        path = self.daily_path(day)
        if path.exists():
            return path.read_text()
        return ""

    def append_daily(self, content: str, day: date | None = None) -> None:
        path = self.daily_path(day)
        existing = path.read_text() if path.exists() else ""
        separator = "\n\n" if existing.strip() else ""
        path.write_text(existing + separator + content.strip() + "\n")

    def load_recent_daily(self, days: int = 2) -> str:
        """Load today's and yesterday's daily notes."""
        parts = []
        today = date.today()
        for i in range(days):
            day = today - timedelta(days=i)
            content = self.load_daily(day)
            if content:
                label = "Today" if i == 0 else "Yesterday" if i == 1 else day.isoformat()
                parts.append(f"[Daily notes — {label} ({day.isoformat()})]\n{content}")
        return "\n\n".join(parts)

    # --- Session summaries ---

    def session_summary_path(
        self,
        started_at: datetime,
        topic_slug: str,
    ) -> Path:
        """Build the canonical path for a session summary file.

        Format: `<YYYY-MM-DD-HHMM>-<slug>.md`. Timestamp comes first so
        `sorted(sessions_dir.iterdir())` yields chronological order
        without parsing frontmatter.
        """
        stamp = started_at.strftime("%Y-%m-%d-%H%M")
        return self.sessions_dir / f"{stamp}-{topic_slug}.md"

    def save_session_summary(
        self,
        body: str,
        started_at: datetime,
        ended_at: datetime,
        topic_slug: str,
        project: str | None = None,
        turn_count: int = 0,
    ) -> Path:
        """Write a session summary with YAML frontmatter; return its path.

        Frontmatter fields are flat strings so a glance at the file
        (without a YAML parser) is still readable. `project` is optional
        — kept null when no project was active.
        """
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        path = self.session_summary_path(started_at, topic_slug)
        project_line = f'project: {project}\n' if project else 'project: null\n'
        frontmatter = (
            "---\n"
            f"started: {started_at.isoformat(timespec='seconds')}\n"
            f"ended: {ended_at.isoformat(timespec='seconds')}\n"
            f"topic: {topic_slug}\n"
            f"{project_line}"
            f"turns: {turn_count}\n"
            "---\n\n"
        )
        path.write_text(frontmatter + body.strip() + "\n", encoding="utf-8")
        return path

    def load_recent_session_summaries(self, n: int = 3) -> list[dict]:
        """Return the N most recent session summaries, newest first.

        Each entry: `{path, started, ended, topic, project, turns, body}`.
        Frontmatter is parsed leniently — missing fields default to ""
        rather than raising, so a hand-edited or malformed file doesn't
        break session start.

        Sorted by filename (which is `<timestamp>-<slug>.md`), which is
        cheaper than reading frontmatter from every file just to sort.
        Falls back to mtime if filenames don't conform.
        """
        if not self.sessions_dir.exists():
            return []
        files = sorted(
            self.sessions_dir.glob("*.md"),
            key=lambda p: p.name,
            reverse=True,
        )[:n]
        entries: list[dict] = []
        for path in files:
            try:
                text = path.read_text(encoding="utf-8")
            except Exception as e:
                log.warning(f"Could not read session summary {path.name}: {e}")
                continue
            entries.append(_parse_session_summary(text, path))
        return entries

    # --- Channel-specific rules ---

    def channel_config_path(self, channel: str) -> Path:
        return self.channels_dir / f"{channel}.md"

    def load_channel_config(self, channel: str) -> str:
        """Load channel-specific rules (e.g. telegram.md, repl.md)."""
        path = self.channel_config_path(channel)
        if path.exists():
            return path.read_text()
        return ""

    # --- Stable context assembly ---

    def assemble_stable_context(
        self,
        channel: str | None = None,
        bible_path=None,
    ) -> str:
        """Build the stable system prompt from identity + channel rules + memory + bible.

        When `bible_path` is provided and points to an existing file, the
        active project's BIBLE.md is appended as a labeled section. That
        way per-project canon (characters, world, established facts) is
        in the system prompt for the whole session — the LLM never
        contradicts it without retrieval cost.

        NOTE: Daily notes are NOT included here — they're indexed and
        retrieved per-turn via the search pipeline instead. The old
        blanket-concat approach (today + yesterday into every prompt)
        bloated the stable prefix with content that's usually
        irrelevant to the current turn, weakening prompt-cache hits.
        Today's notes still surface when their content actually matches
        the query.

        NOTE: Current datetime is NOT included here — it changes every turn
        and would cause unnecessary reconnects. It goes in dynamic context instead.
        """
        sections = []

        identity = self.load_identity()
        if identity:
            sections.append(identity)

        # Channel-specific rules
        if channel:
            channel_rules = self.load_channel_config(channel)
            if channel_rules:
                sections.append(channel_rules)

        memory = self.load_memory()
        if memory:
            sections.append(f"[Long-term memory]\n{memory}")

        # Daily notes intentionally NOT concatenated here — see
        # docstring. They're indexed via _index_workspace and surface
        # through the search pipeline per turn when relevant. Keeps
        # the stable prefix smaller, prompt cache hits stronger.

        if bible_path is not None and bible_path.is_file():
            try:
                bible = bible_path.read_text(encoding="utf-8").strip()
            except Exception:
                bible = ""
            if bible:
                sections.append(
                    f"[Project bible — {bible_path.parent.name}]\n{bible}"
                )

        return "\n\n---\n\n".join(sections)

    def stable_context_mtime(
        self, channel: str | None = None, bible_path=None
    ) -> float:
        """Return the latest mtime across files that make up stable context.

        Used to detect real changes (write-back, project switch) without
        comparing prompt strings that include volatile data.

        Includes the active project's BIBLE.md when given — that's how
        `/bible add` updates propagate without an explicit reconnect call.

        NOTE: Daily notes are NOT tracked here anymore — they no longer
        contribute to the stable prompt (see `assemble_stable_context`),
        so a daily-note write shouldn't trigger an LLM reconnect.
        """
        mtimes = []
        for path in [self.identity_path, self.memory_path]:
            if path.exists():
                mtimes.append(path.stat().st_mtime)
        # Channel config
        if channel:
            ch_path = self.channel_config_path(channel)
            if ch_path.exists():
                mtimes.append(ch_path.stat().st_mtime)
        # Project bible
        if bible_path is not None and bible_path.is_file():
            mtimes.append(bible_path.stat().st_mtime)
        return max(mtimes) if mtimes else 0.0
