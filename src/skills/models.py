"""Data model for a skill."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Skill:
    name: str
    description: str  # one-line summary — used for /skills list and future auto-match
    body: str  # the markdown body below the frontmatter — goes into the prompt
    triggers: list[str] = field(default_factory=list)  # keyword hints for auto-match (v2)
    directory: Path | None = None  # where the skill lives on disk
    # True when a fetch.py exists alongside SKILL.md. Scheduler-only for v1
    # (interactive chat doesn't invoke fetch scripts — see the spec).
    has_fetch: bool = False

    @property
    def fetch_path(self) -> Path | None:
        if not self.directory:
            return None
        p = self.directory / "fetch.py"
        return p if p.is_file() else None
