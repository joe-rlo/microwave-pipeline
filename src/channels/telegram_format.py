"""Markdown → Telegram-HTML conversion and table extraction.

Telegram's legacy Markdown parser is fragile (one unbalanced `*` breaks the
whole message) and doesn't understand `**bold**`. We convert LLM output to
Telegram's HTML subset instead: b, i, u, s, a, code, pre, blockquote.

Markdown tables render poorly in Telegram, so we extract them and send them
as standalone .html file attachments.
"""

from __future__ import annotations

import html
import re

# A markdown table: header row, separator row, zero or more data rows.
_TABLE_RE = re.compile(
    r"(?m)"
    r"(^\|[^\n]*\|[ \t]*\n"
    r"\|[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)+\|[ \t]*\n"
    r"(?:\|[^\n]*\|[ \t]*\n?)*)"
)

_CODE_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\n?(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_BOLD_RE = re.compile(r"\*\*([^*\n][^*\n]*?)\*\*")
_ITAL_AST_RE = re.compile(r"(?<![*\w])\*([^*\n]+?)\*(?!\*)")
_ITAL_UND_RE = re.compile(r"(?<![_\w])_([^_\n]+?)_(?!_)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_HEADER_RE = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*#*\s*$")
_HR_RE = re.compile(r"(?m)^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")


def _parse_table_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def _table_to_cards_html(md: str) -> str:
    """Render a markdown table as a sequence of card-layout blocks (HTML).

    Each row becomes a block of `<b>Header:</b> value` lines. Cards are
    separated by blank lines. Returns HTML ready to splice into the
    output (already escaped where needed).
    """
    lines = [l for l in md.strip().split("\n") if l.strip()]
    if len(lines) < 3:
        # Not a full table (need header + separator + ≥1 data row);
        # fall back to preformatted so it at least stays legible.
        return f"<pre>{html.escape(md)}</pre>"

    header = _parse_table_row(lines[0])
    rows = [_parse_table_row(l) for l in lines[2:]]  # skip separator

    cards: list[str] = []
    for row in rows:
        card_lines: list[str] = []
        for i, cell in enumerate(row):
            if not cell:
                continue
            label = header[i] if i < len(header) else f"col{i+1}"
            label_html = _inline_md_to_html(label) if label else f"col{i+1}"
            cell_html = _inline_md_to_html(cell)
            card_lines.append(f"<b>{label_html}:</b> {cell_html}")
        if card_lines:
            cards.append("\n".join(card_lines))

    if not cards:
        return f"<pre>{html.escape(md)}</pre>"
    return "\n" + "\n\n".join(cards) + "\n"


def _inline_md_to_html(text: str) -> str:
    """Inline markdown → HTML, for use inside table cells.

    Escapes HTML first, then applies bold/italic/code/link.
    """
    # Stash inline code first
    stashed: list[str] = []

    def _stash(s: str) -> str:
        stashed.append(s)
        return f"\x00PH{len(stashed)-1}\x00"

    text = _INLINE_CODE_RE.sub(lambda m: _stash(f"<code>{html.escape(m.group(1))}</code>"), text)
    text = html.escape(text, quote=False)
    text = _BOLD_RE.sub(r"<strong>\1</strong>", text)
    text = _ITAL_AST_RE.sub(r"<em>\1</em>", text)
    text = _ITAL_UND_RE.sub(r"<em>\1</em>", text)
    text = _LINK_RE.sub(r'<a href="\2">\1</a>', text)
    for i, s in enumerate(stashed):
        text = text.replace(f"\x00PH{i}\x00", s)
    return text


def markdown_to_telegram_html(text: str) -> str:
    """Convert LLM-style markdown to Telegram's HTML subset.

    Supported output tags: b, i, code, pre, a, blockquote.
    Headers become bold lines. Tables become inline card-layout blocks
    (one `<b>Header:</b> value` line per column, blank line between rows).
    """
    # 1. Stash fenced and inline code so their contents aren't mangled.
    stashed: list[str] = []

    def _stash(s: str) -> str:
        stashed.append(s)
        return f"\x00PH{len(stashed)-1}\x00"

    # Tables first, before code stashing, so a ``` inside a table cell
    # doesn't get stashed prematurely. Tables are rendered as card blocks.
    text = _TABLE_RE.sub(lambda m: _stash(_table_to_cards_html(m.group(1))), text)

    def _handle_fence(m: re.Match) -> str:
        lang = m.group(1).strip()
        code = m.group(2).rstrip("\n")
        escaped = html.escape(code)
        if lang:
            return _stash(
                f'<pre><code class="language-{html.escape(lang)}">{escaped}</code></pre>'
            )
        return _stash(f"<pre>{escaped}</pre>")

    text = _CODE_FENCE_RE.sub(_handle_fence, text)
    text = _INLINE_CODE_RE.sub(
        lambda m: _stash(f"<code>{html.escape(m.group(1))}</code>"), text
    )

    # 2. Escape HTML special chars in the remaining prose.
    text = html.escape(text, quote=False)

    # 3. Horizontal rules → blank line (Telegram has no <hr>).
    text = _HR_RE.sub("", text)

    # 4. Headers → bold on their own line.
    text = _HEADER_RE.sub(lambda m: f"<b>{m.group(2)}</b>", text)

    # 5. Inline formatting. Bold before italic so `**x**` wins over `*x*`.
    text = _BOLD_RE.sub(r"<b>\1</b>", text)
    text = _ITAL_AST_RE.sub(r"<i>\1</i>", text)
    text = _ITAL_UND_RE.sub(r"<i>\1</i>", text)
    text = _LINK_RE.sub(r'<a href="\2">\1</a>', text)

    # 6. Restore stashed code.
    for i, s in enumerate(stashed):
        text = text.replace(f"\x00PH{i}\x00", s)

    # 7. Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text
