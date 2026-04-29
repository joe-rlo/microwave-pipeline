"""Minimal YAML-ish frontmatter parser.

Shared by skills (`SKILL.md`) and projects (`PROJECT.md`). Deliberately
small — supports just what those configs need:

- `key: scalar` — string value (auto strips quotes)
- `key:` followed by `  - item` lines — list of strings
- `key: >` / `key: |` — folded/literal multiline scalar

Doesn't support nested maps, anchors, or any advanced YAML. We pulled
in this hand-rolled parser instead of a YAML dep so MicrowaveOS doesn't
gain a transitive PyYAML installation just for two metadata keys.
"""

from __future__ import annotations

import re

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Pull the frontmatter dict off the front of a markdown file.

    Returns ({}, full_text) when there's no frontmatter header.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    return _parse_frontmatter(m.group(1)), text[m.end():]


def _parse_frontmatter(raw: str) -> dict:
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
            while i < len(lines) and (
                lines[i].startswith("  ") or lines[i].startswith("\t") or not lines[i].strip()
            ):
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

        out[key] = _strip_quotes(value)
        i += 1
    return out


def _strip_quotes(s: str) -> str:
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    return s


def as_list(value) -> list[str]:
    """Coerce a frontmatter value to a list of strings.

    Used by callers because some fields (triggers, tags) are nicer to
    accept either as a YAML list or a comma-separated string.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        return [s.strip() for s in value.split(",") if s.strip()]
    return []
