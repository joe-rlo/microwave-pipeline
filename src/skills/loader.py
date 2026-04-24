"""Disk discovery for skills.

Each skill lives at `<skills_dir>/<name>/SKILL.md`. The file starts with
a small YAML-ish frontmatter block (name, description, optional triggers),
followed by the skill body. Adjacent files in the directory (like
`fetch.py` or reference material) are the skill's own resources.

We don't pull in a YAML dep — the frontmatter here is a tiny, predictable
subset (scalars and short lists) that a 30-line parser covers cleanly.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from src.skills.models import Skill

log = logging.getLogger(__name__)


class SkillNotFound(KeyError):
    """Raised when the caller asks for a skill that isn't on disk."""


_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


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
        meta, body = _split_frontmatter(text)

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
            triggers=_as_list(meta.get("triggers")),
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


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Pull the frontmatter dict off the front of a SKILL.md. Returns
    ({}, full_text) when there's no frontmatter."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw = m.group(1)
    body = text[m.end():]
    return _parse_frontmatter(raw), body


def _parse_frontmatter(raw: str) -> dict:
    """Minimal YAML-ish parser covering what skill frontmatter uses:

    - `key: scalar` — string
    - `key:` followed by `  - item` lines — list of strings
    - `key: >` / `key: |` — folded/literal multiline scalar (we treat both
      the same: join lines with spaces for `>`, keep newlines for `|`).

    Deliberately doesn't support nested maps or anchors — skill
    frontmatter shouldn't need them, and the smaller the grammar, the
    fewer surprise bugs.
    """
    out: dict = {}
    lines = raw.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        if ":" not in line:
            i += 1
            continue

        key, rest = line.split(":", 1)
        key = key.strip()
        value = rest.strip()

        # Folded or literal multiline scalar
        if value in (">", "|"):
            fold = value == ">"
            chunks: list[str] = []
            i += 1
            while i < len(lines) and (lines[i].startswith("  ") or lines[i].startswith("\t") or not lines[i].strip()):
                chunks.append(lines[i].strip())
                i += 1
            joined = " ".join(c for c in chunks if c) if fold else "\n".join(chunks).strip()
            out[key] = joined
            continue

        # List — indented `- item` lines follow
        if value == "":
            items: list[str] = []
            i += 1
            while i < len(lines) and (lines[i].startswith("  -") or lines[i].startswith("- ")):
                item = lines[i].lstrip()
                if item.startswith("- "):
                    items.append(item[2:].strip().strip('"').strip("'"))
                i += 1
            out[key] = items
            continue

        # Scalar
        out[key] = _strip_quotes(value)
        i += 1
    return out


def _strip_quotes(s: str) -> str:
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    return s


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        # single-string case: split on comma as a convenience
        return [s.strip() for s in value.split(",") if s.strip()]
    return []


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
