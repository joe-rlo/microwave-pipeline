"""Markdown → Signal-rendered text.

Signal renders a small set of inline styles natively when the message text
contains the right syntax: `**bold**`, `*italic*` / `_italic_`,
`~strikethrough~`, and `` `monospace` ``. There is no header/link/code-block
markup, and no HTML.

This module normalizes LLM output into that subset and, like the Telegram
formatter, expands markdown tables into inline "card" blocks so tabular
data stays readable on a phone screen.
"""

from __future__ import annotations

import re

# A markdown table: header row, separator row, zero or more data rows.
_TABLE_RE = re.compile(
    r"(?m)"
    r"(^\|[^\n]*\|[ \t]*\n"
    r"\|[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)+\|[ \t]*\n"
    r"(?:\|[^\n]*\|[ \t]*\n?)*)"
)

_CODE_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n?(.*?)```", re.DOTALL)
_HEADER_RE = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*#*\s*$")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_HR_RE = re.compile(r"(?m)^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")


def _parse_table_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def _table_to_cards_text(md: str) -> str:
    """Render a markdown table as a sequence of card-layout blocks.

    Each row becomes `**Header:** value` lines, separated by a blank line
    between cards. Identical shape to the Telegram formatter, but uses
    Signal's `**bold**` markdown instead of `<b>` tags.
    """
    lines = [l for l in md.strip().split("\n") if l.strip()]
    if len(lines) < 3:
        return md.strip()

    header = _parse_table_row(lines[0])
    rows = [_parse_table_row(l) for l in lines[2:]]

    cards: list[str] = []
    for row in rows:
        card_lines: list[str] = []
        for i, cell in enumerate(row):
            if not cell:
                continue
            label = header[i] if i < len(header) else f"col{i+1}"
            card_lines.append(f"**{label}:** {cell}")
        if card_lines:
            cards.append("\n".join(card_lines))

    if not cards:
        return md.strip()
    return "\n\n".join(cards)


def markdown_to_signal_text(text: str) -> str:
    """Normalize LLM markdown into Signal-rendered plain text.

    - Tables → card-layout blocks (`**Header:** value`).
    - Headers → bold on their own line (Signal has no header markup).
    - `[text](url)` → `text (url)` so the URL auto-linkifies.
    - Fenced code blocks → stripped of backticks, content kept as plain text
      (Signal does not render multi-line code blocks; single-line `` `code` ``
      is preserved).
    - Any stray HTML tags (e.g., if a Telegram-formatted string leaks through)
      are removed.
    - `**bold**`, `*italic*`, `_italic_`, `~strike~`, `` `code` `` pass through
      untouched — Signal clients render them on receipt.
    """
    # 1. Tables first — their cells may contain inline formatting that should
    # still pass through, so expand before doing other transforms.
    text = _TABLE_RE.sub(lambda m: _table_to_cards_text(m.group(1)), text)

    # 2. Fenced code blocks: keep the inner content, drop the fences.
    # Signal can't render multi-line code blocks; stripping keeps the text
    # legible instead of showing raw ``` markers.
    text = _CODE_FENCE_RE.sub(lambda m: m.group(1).rstrip("\n"), text)

    # 3. Horizontal rules → blank line.
    text = _HR_RE.sub("", text)

    # 4. Headers → bold on their own line.
    text = _HEADER_RE.sub(lambda m: f"**{m.group(2)}**", text)

    # 5. Inline links: Signal has no link markup — render as "text (url)" so
    # the URL part auto-linkifies on the client.
    text = _LINK_RE.sub(lambda m: f"{m.group(1)} ({m.group(2)})", text)

    # 6. Strip any stray HTML tags (defensive — e.g., if something in the
    # pipeline accidentally emits HTML meant for Telegram).
    text = _HTML_TAG_RE.sub("", text)

    # 7. Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text
