"""Read/write markdown workspace files (IDENTITY.md, MEMORY.md, daily notes)."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)


class MemoryStore:
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self.identity_path = workspace_dir / "IDENTITY.md"
        self.memory_path = workspace_dir / "MEMORY.md"
        self.daily_dir = workspace_dir / "memory"

    @property
    def channels_dir(self) -> Path:
        return self.workspace_dir / "channels"

    def ensure_dirs(self) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self.channels_dir.mkdir(parents=True, exist_ok=True)

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

    def assemble_stable_context(self, channel: str | None = None) -> str:
        """Build the stable system prompt from identity + memory + daily notes + channel rules.

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

        daily = self.load_recent_daily()
        if daily:
            sections.append(daily)

        return "\n\n---\n\n".join(sections)

    def stable_context_mtime(self, channel: str | None = None) -> float:
        """Return the latest mtime across files that make up stable context.

        Used to detect real changes (write-back, day rollover) without
        comparing prompt strings that include volatile data.
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
        # Check today's and yesterday's daily notes
        today = date.today()
        for i in range(2):
            day_path = self.daily_path(today - timedelta(days=i))
            if day_path.exists():
                mtimes.append(day_path.stat().st_mtime)
        return max(mtimes) if mtimes else 0.0
