"""Data model for a writing project."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Recognized project types. Each maps to a default skill and scaffolding.
PROJECT_TYPES = ("blog", "novel", "screenplay")


@dataclass
class Project:
    name: str
    type: str  # blog | novel | screenplay
    skill: str = ""  # which skill auto-activates when project is active
    status: str = "drafting"  # drafting | revising | paused | done | archived
    description: str = ""
    target_words: int = 0
    voice_notes: str = ""  # body of PROJECT.md after frontmatter
    created_at: datetime | None = None
    directory: Path | None = None

    # Convenience: derived paths and file-presence flags. Populated by the
    # loader so callers don't have to recompute repeatedly.
    bible_path: Path | None = None
    outline_path: Path | None = None
    drafts_dir: Path | None = None
    notes_dir: Path | None = None

    @property
    def has_bible(self) -> bool:
        return bool(self.bible_path and self.bible_path.is_file())

    @property
    def has_outline(self) -> bool:
        return bool(self.outline_path and self.outline_path.is_file())

    def list_drafts(self) -> list[Path]:
        if not self.drafts_dir or not self.drafts_dir.is_dir():
            return []
        return sorted(p for p in self.drafts_dir.iterdir() if p.is_file())

    def word_count(self) -> int:
        """Sum of word counts across all draft files. Cheap whitespace
        split — close enough for status displays, not for billing."""
        total = 0
        for p in self.list_drafts():
            try:
                total += len(p.read_text(encoding="utf-8").split())
            except Exception:
                continue
        return total
