"""Disk discovery and lifecycle for writing projects.

Projects live at `<projects_dir>/<name>/PROJECT.md`. Each one owns its
own subdirectory (drafts, notes, optional bible/outline) — the loader's
only job is to find them, parse the frontmatter, and surface the
canonical paths so callers don't recompute them everywhere.

Scaffolding (the `new` action) is type-aware: a novel gets a chapters
draft layout + BIBLE template, a screenplay gets a FOUNTAIN-friendly
single-file layout, a blog gets a single-draft layout.
"""

from __future__ import annotations

import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

from src.frontmatter import split_frontmatter
from src.projects.models import PROJECT_TYPES, Project

log = logging.getLogger(__name__)


class ProjectNotFound(KeyError):
    """Raised when the caller asks for a project that isn't on disk."""


# Default skill to auto-activate when the user enters a project of each
# type and PROJECT.md doesn't override it.
_DEFAULT_SKILLS = {
    "blog": "blog-writing",
    "novel": "novel-writing",
    "screenplay": "screenplay-writing",
}


class ProjectLoader:
    def __init__(self, projects_dir: Path):
        self.projects_dir = projects_dir

    # --- discovery ---

    def list_names(self, include_archived: bool = False) -> list[str]:
        if not self.projects_dir.is_dir():
            return []
        names: list[str] = []
        for child in self.projects_dir.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            if (child / "PROJECT.md").is_file():
                names.append(child.name)
        names.sort()
        if include_archived:
            archive_dir = self.projects_dir / ".archived"
            if archive_dir.is_dir():
                for child in archive_dir.iterdir():
                    if child.is_dir() and (child / "PROJECT.md").is_file():
                        names.append(f".archived/{child.name}")
        return names

    def list_all(self) -> list[Project]:
        out: list[Project] = []
        for name in self.list_names():
            try:
                out.append(self.load(name))
            except Exception as e:
                log.warning(f"Failed to load project {name!r}: {e}")
        return out

    def load(self, name: str) -> Project:
        path = self.projects_dir / name / "PROJECT.md"
        if not path.is_file():
            raise ProjectNotFound(name)

        text = path.read_text(encoding="utf-8")
        meta, body = split_frontmatter(text)

        proj_dir = path.parent
        project_type = str(meta.get("type", "")).strip() or _infer_type(proj_dir)
        if project_type not in PROJECT_TYPES:
            log.warning(
                f"Project {name!r} declares unknown type {project_type!r}; "
                f"defaulting to 'blog'"
            )
            project_type = "blog"

        skill = str(meta.get("skill", "")).strip() or _DEFAULT_SKILLS.get(project_type, "")
        target_words = _parse_int(meta.get("target_words"))
        created_at = _parse_date(meta.get("created"))

        return Project(
            name=name,
            type=project_type,
            skill=skill,
            status=str(meta.get("status", "drafting")).strip() or "drafting",
            description=str(meta.get("description", "")).strip(),
            target_words=target_words,
            voice_notes=body.strip(),
            created_at=created_at,
            directory=proj_dir,
            bible_path=proj_dir / "BIBLE.md",
            outline_path=proj_dir / "outline.md",
            drafts_dir=proj_dir / "drafts",
            notes_dir=proj_dir / "notes",
        )

    # --- mutation ---

    def scaffold(self, name: str, project_type: str, description: str = "") -> Path:
        """Create the directory layout for a new project of the given type.

        Returns the path to the created PROJECT.md.
        """
        if not _valid_name(name):
            raise ValueError(
                f"Invalid project name {name!r}: lowercase letters, digits, "
                f"hyphens only"
            )
        if project_type not in PROJECT_TYPES:
            raise ValueError(
                f"Unknown project type {project_type!r}; must be one of "
                f"{PROJECT_TYPES}"
            )
        target = self.projects_dir / name
        if target.exists():
            raise FileExistsError(f"Project {name!r} already exists at {target}")

        target.mkdir(parents=True)
        (target / "drafts").mkdir()
        (target / "notes").mkdir()

        # Type-specific scaffolding
        if project_type in ("novel", "screenplay"):
            (target / "BIBLE.md").write_text(
                _bible_template(name, project_type), encoding="utf-8"
            )
            (target / "outline.md").write_text(
                _outline_template(project_type), encoding="utf-8"
            )
        elif project_type == "blog":
            (target / "outline.md").write_text(
                _outline_template("blog"), encoding="utf-8"
            )
            (target / "drafts" / "draft.md").write_text("", encoding="utf-8")

        if project_type == "screenplay":
            (target / "drafts" / "screenplay.fountain").write_text("", encoding="utf-8")

        path = target / "PROJECT.md"
        path.write_text(
            _project_template(name, project_type, description), encoding="utf-8"
        )
        return path

    def archive(self, name: str) -> bool:
        """Move a project under .archived/. Returns True if moved."""
        target = self.projects_dir / name
        if not target.is_dir():
            return False
        archive_dir = self.projects_dir / ".archived"
        archive_dir.mkdir(exist_ok=True)
        dest = archive_dir / name
        if dest.exists():
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            dest = archive_dir / f"{name}-{stamp}"
        shutil.move(str(target), str(dest))
        return True

    def remove(self, name: str) -> bool:
        target = self.projects_dir / name
        if not target.is_dir():
            return False
        shutil.rmtree(target)
        return True


# --- helpers ---


def _valid_name(name: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", name))


def _parse_int(value) -> int:
    if value is None:
        return 0
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return 0


def _parse_date(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).strip())
    except ValueError:
        return None


def _infer_type(directory: Path) -> str:
    """Best-effort type guess when PROJECT.md frontmatter omits `type`.

    Looks for telltale files: a `.fountain` file → screenplay, a
    `BIBLE.md` → novel, otherwise blog.
    """
    drafts = directory / "drafts"
    if drafts.is_dir():
        if any(p.suffix == ".fountain" for p in drafts.iterdir()):
            return "screenplay"
    if (directory / "BIBLE.md").is_file():
        return "novel"
    return "blog"


# --- templates ---


def _project_template(name: str, project_type: str, description: str) -> str:
    skill = _DEFAULT_SKILLS.get(project_type, "")
    desc = description or f"What this {project_type} is about."
    return f"""---
name: {name}
type: {project_type}
skill: {skill}
status: drafting
description: >
  {desc}
created: {datetime.now().date().isoformat()}
---

# {name}

## Voice notes
Project-specific voice or tone notes. The skill ({skill or 'none'}) handles
the broad rules; use this section for what's unique to *this* assignment.

## Project-specific rules
- Add anything here that should override skill defaults for this project.
"""


def _bible_template(name: str, project_type: str) -> str:
    if project_type == "screenplay":
        return f"""# Bible — {name}

## Logline
One sentence: who wants what, what stops them.

## Characters
### NAME (role)
- Trait 1
- Trait 2
- Voice: how they speak — sentence length, vocabulary, tics

## World / setting
- Time period, location, tone

## Established facts
(Auto- and manually-added canonical facts. The bot suggests additions
when it drafts new scenes; you approve them with `/bible add`.)
"""
    return f"""# Bible — {name}

## Premise
One paragraph: what this is about.

## Characters
### NAME (role in the story)
- Backstory hook
- Voice: how they speak, recurring phrasing
- Arc: where they start, where they end

## World / setting
- Time period
- Location(s)
- Rules of the world (especially if speculative)

## Established facts
(Auto- and manually-added canonical facts. The bot suggests additions
when it drafts new chapters; you approve them with `/bible add`.)
"""


def _outline_template(project_type: str) -> str:
    if project_type == "blog":
        return """# Outline

## Working title
TBD

## Hook / lede
What's the opening line? What tension does it create?

## Main argument
- Point 1
- Point 2
- Point 3

## Supporting evidence / examples
- Where do they sit in the structure?

## Close
What does the reader take with them?
"""
    if project_type == "novel":
        return """# Outline

## Premise
One paragraph.

## Structure
| Chapter | Scene | Status | Notes |
|---------|-------|--------|-------|
| 1       |       | todo   |       |
| 2       |       | todo   |       |

## Major beats
- Inciting incident:
- Midpoint:
- Climax:
- Resolution:
"""
    if project_type == "screenplay":
        return """# Outline

## Logline
One sentence.

## Acts
| Act | Beat | Scene | Status | Notes |
|-----|------|-------|--------|-------|
| I   | Setup |      | todo   |       |
| I   | Inciting incident |  | todo   |       |
| II  | Rising action |  | todo   |       |
| II  | Midpoint |       | todo   |       |
| III | Climax |         | todo   |       |
| III | Resolution |     | todo   |       |
"""
    return "# Outline\n\nTBD\n"
