"""Minimal YAML-ish frontmatter parser.

Shared by skills (`SKILL.md`) and projects (`PROJECT.md`). Deliberately
small — supports just what those configs need:

- `key: scalar` — string value (auto strips quotes)
- `key:` followed by `  - item` lines — list of strings
- `key:` followed by `  subkey: value` lines — one-level nested map
- `key: >` / `key: |` — folded/literal multiline scalar

Nested maps are limited to one level deep — that's enough for the
skill `pipeline:` block (pipeline 2.3) without growing into a full
YAML implementation. Deeper nesting silently flattens, which is fine
because no current consumer needs it.

Doesn't support anchors, multi-doc separators, or any advanced YAML.
We pulled in this hand-rolled parser instead of a YAML dep so
MicrowaveOS doesn't gain a transitive PyYAML installation just for a
handful of metadata keys.
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

        # `key:` with no value — could be a list (`  - item`) or a
        # one-level nested map (`  subkey: value`). Peek at the next
        # non-blank line to decide; default to empty list when the
        # block is empty (keeps the old behavior for skills that
        # declare `triggers:` with nothing under it).
        if value == "":
            # Find next non-blank line.
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            peek = lines[j] if j < len(lines) else ""
            if peek.lstrip().startswith("- "):
                items: list[str] = []
                i += 1
                while i < len(lines) and (lines[i].startswith("  -") or lines[i].startswith("- ")):
                    item = lines[i].lstrip()
                    if item.startswith("- "):
                        items.append(item[2:].strip().strip('"').strip("'"))
                    i += 1
                out[key] = items
                continue
            # Nested map. Consume indented `subkey: value` lines.
            # We deliberately accept only string-shaped values here —
            # the pipeline block (the only current consumer) takes
            # values like "off", "high", "4000" and parses them
            # downstream. Keeping all sub-values as strings keeps the
            # parser simple and the call sites explicit about coercion.
            if peek.startswith(("  ", "\t")) and ":" in peek:
                submap: dict[str, str] = {}
                i += 1
                while i < len(lines):
                    sub = lines[i]
                    if not sub.strip():
                        i += 1
                        continue
                    if not (sub.startswith("  ") or sub.startswith("\t")):
                        break
                    if ":" not in sub:
                        i += 1
                        continue
                    sk, _, sv = sub.strip().partition(":")
                    submap[sk.strip()] = _strip_quotes(sv.strip())
                    i += 1
                out[key] = submap
                continue
            # Empty block — preserve the old empty-list default so
            # callers that did `triggers:` with nothing under it
            # don't suddenly start getting an empty dict.
            out[key] = []
            i += 1
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
