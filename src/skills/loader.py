"""Disk discovery for skills.

Each skill lives at `<skills_dir>/<name>/SKILL.md`. The file starts with
a small YAML-ish frontmatter block (name, description, optional triggers),
followed by the skill body. Adjacent files in the directory (like
`fetch.py` or reference material) are the skill's own resources.

Frontmatter parsing is shared with projects via `src.frontmatter`.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from src.frontmatter import as_list, split_frontmatter
from src.skills.models import Skill

log = logging.getLogger(__name__)


class SkillNotFound(KeyError):
    """Raised when the caller asks for a skill that isn't on disk."""


class SkillLoader:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir

    # --- discovery ---

    def list_names(self) -> list[str]:
        """Return skill names sorted alphabetically. Skips dotfiles."""
        if not self.skills_dir.is_dir():
            return []
        names: list[str] = []
        for child in self.skills_dir.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            if (child / "SKILL.md").is_file():
                names.append(child.name)
        return sorted(names)

    def list_all(self) -> list[Skill]:
        """Load every skill from disk. Bad skills are skipped with a log
        warning — listing should never crash the caller."""
        skills: list[Skill] = []
        for name in self.list_names():
            try:
                skills.append(self.load(name))
            except Exception as e:
                log.warning(f"Failed to load skill {name!r}: {e}")
        return skills

    def load(self, name: str) -> Skill:
        """Load a single skill by name. Raises SkillNotFound if missing."""
        path = self.skills_dir / name / "SKILL.md"
        if not path.is_file():
            raise SkillNotFound(name)

        text = path.read_text(encoding="utf-8")
        meta, body = split_frontmatter(text)

        # Name in the frontmatter must match the directory name — keeps
        # CLI lookups honest. We could auto-correct, but silently rewriting
        # file state on read is a footgun later.
        fm_name = meta.get("name", name)
        if fm_name != name:
            log.warning(
                f"Skill directory {name!r} declares frontmatter name={fm_name!r}; "
                f"using directory name for consistency"
            )

        skill_dir = path.parent
        return Skill(
            name=name,
            description=str(meta.get("description", "")).strip(),
            body=body.strip(),
            triggers=as_list(meta.get("triggers")),
            directory=skill_dir,
            has_fetch=(skill_dir / "fetch.py").is_file(),
        )

    # --- mutation (used by the `skills new` / `skills remove` CLI) ---

    def scaffold(self, name: str, description: str = "") -> Path:
        """Create an empty skill directory with a template SKILL.md."""
        if not _valid_name(name):
            raise ValueError(
                f"Invalid skill name {name!r}: lowercase letters, digits, hyphens only"
            )
        target = self.skills_dir / name
        if target.exists():
            raise FileExistsError(f"Skill {name!r} already exists at {target}")
        target.mkdir(parents=True)
        path = target / "SKILL.md"
        path.write_text(_scaffold_template(name, description), encoding="utf-8")
        return path

    def remove(self, name: str) -> bool:
        """Delete a skill directory. Returns True if anything was removed."""
        target = self.skills_dir / name
        if not target.is_dir():
            return False
        import shutil
        shutil.rmtree(target)
        return True


# --- helpers ---


def _valid_name(name: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", name))


def _scaffold_template(name: str, description: str) -> str:
    desc = description or f"What {name} does. Update before use."
    return f"""---
name: {name}
description: >
  {desc}
triggers:
  - {name}
---

# {name}

Write the skill body here. This goes into the pipeline's dynamic context
whenever the skill is active. You can include:

- Voice/tone rules
- Topic guardrails or anti-patterns
- Output format requirements
- Domain-specific vocabulary
- Example outputs that work (keep them short — they count against tokens)

Keep instructions additive to IDENTITY.md. If anything here conflicts
with channel formatting rules (message length, markdown syntax), the
channel rules win.
"""
