"""Render scheduler LLM output as an HTML card-view attachment.

Each card has a header, body (preserving whitespace), a word-count meta
row, and a per-card Copy button. A "Copy all" button appears at the top
when there are ≥ 2 cards. Self-contained HTML — no external assets.

Per `feedback_signal_card_view.md`: Signal's mobile client has painful
text-selection ergonomics. Copy buttons sidestep the UI entirely.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass


@dataclass
class Card:
    label: str
    body: str

    @property
    def word_count(self) -> int:
        return len(re.findall(r"\S+", self.body))


def split_into_cards(content: str, separator: str = "---") -> list[Card]:
    """Split raw LLM output into cards on a separator line.

    A separator line is a line whose only non-whitespace content is the
    separator string. `---` embedded in prose won't split.
    """
    pattern = re.compile(rf"(?m)^\s*{re.escape(separator)}\s*$")
    chunks = [c.strip() for c in pattern.split(content) if c.strip()]

    cards: list[Card] = []
    for i, chunk in enumerate(chunks, 1):
        label, body = _extract_label(chunk, fallback=f"Item {i}")
        cards.append(Card(label=label, body=body))
    return cards


_LABEL_RE = re.compile(
    r"^(?:\*\*|__)?"                 # optional bold markers
    r"(Note\s+\d+|Item\s+\d+|Part\s+\d+|\d+[\.\)])"  # "Note 1", "1.", "1)"
    r"[^\n]{0,80}"                   # short continuation (e.g. "— hot take")
    r"(?:\*\*|__)?\s*$",
    re.IGNORECASE,
)


def _extract_label(chunk: str, fallback: str) -> tuple[str, str]:
    """Pull a short leading label line off a chunk, if one exists.

    The Substack prompt tells the LLM to prefix each note like
    `Note 1 — hot take`; we harvest that as the card header so the card
    title is informative without re-scanning the body.
    """
    lines = chunk.split("\n", 1)
    first = lines[0].strip()
    if len(first) <= 80 and _LABEL_RE.match(first):
        body = lines[1].strip() if len(lines) > 1 else ""
        # Strip bold markers from the displayed label.
        label = re.sub(r"^\*\*|\*\*$|^__|__$", "", first).strip()
        return label, body or first
    return fallback, chunk


def render_card_view(
    cards: list[Card],
    title: str = "Notes",
) -> str:
    """Render the cards as a full HTML document."""
    if not cards:
        return _empty_doc(title)

    copy_all_button = ""
    if len(cards) >= 2:
        copy_all_button = (
            '<div class="toolbar">'
            '<button class="btn btn-copy-all" data-action="copy-all" type="button">'
            'Copy all</button></div>'
        )

    cards_html = "\n".join(_render_card(c, i) for i, c in enumerate(cards))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{
    --bg: #fafafa; --fg: #1a1a1a; --muted: #6b6b6b; --accent: #c2410c;
    --card-bg: #ffffff; --border: #e5e5e5;
    --btn-bg: #f3f3f3; --btn-fg: #222; --btn-hover: #e5e5e5;
    --ok-bg: #dcfce7; --ok-fg: #166534;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #0f0f0f; --fg: #ececec; --muted: #9a9a9a; --accent: #fb923c;
      --card-bg: #181818; --border: #2a2a2a;
      --btn-bg: #262626; --btn-fg: #ececec; --btn-hover: #333;
      --ok-bg: #14532d; --ok-fg: #bbf7d0;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--bg); color: var(--fg); margin: 0;
    font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    padding: 14px; max-width: 820px; margin: 0 auto;
  }}
  h1 {{ font-size: 20px; margin: 4px 0 16px; color: var(--accent); }}
  .toolbar {{ margin: 0 0 14px; }}
  .card {{
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px; margin-bottom: 14px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  }}
  .card-label {{
    font-size: 13px; font-weight: 600; color: var(--accent);
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;
  }}
  .card-body {{
    white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word;
    font-size: 15.5px; line-height: 1.55;
  }}
  .meta {{
    display: flex; align-items: center; justify-content: space-between;
    margin-top: 12px; padding-top: 10px; border-top: 1px dashed var(--border);
    gap: 10px;
  }}
  .word-count {{ font-size: 12px; color: var(--muted); }}
  .btn {{
    appearance: none; border: 1px solid var(--border); background: var(--btn-bg);
    color: var(--btn-fg); font-size: 14px; font-weight: 500;
    padding: 10px 14px; border-radius: 8px; cursor: pointer;
    min-height: 40px; min-width: 80px;
  }}
  .btn:active {{ background: var(--btn-hover); }}
  .btn-copy-all {{ width: 100%; min-height: 44px; font-size: 15px; }}
  .btn.copied {{
    background: var(--ok-bg); color: var(--ok-fg); border-color: transparent;
  }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
{copy_all_button}
{cards_html}
<script>
(function() {{
  async function copyText(text, button) {{
    try {{
      if (navigator.clipboard && navigator.clipboard.writeText) {{
        await navigator.clipboard.writeText(text);
      }} else {{
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed'; ta.style.opacity = '0';
        document.body.appendChild(ta); ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
      }}
      flash(button, 'Copied');
    }} catch (e) {{
      flash(button, 'Failed');
    }}
  }}
  function flash(btn, text) {{
    if (!btn) return;
    const original = btn.textContent;
    btn.textContent = text === 'Copied' ? '✓ Copied' : text;
    btn.classList.add('copied');
    setTimeout(() => {{
      btn.textContent = original;
      btn.classList.remove('copied');
    }}, 1800);
  }}
  document.addEventListener('click', (ev) => {{
    const btn = ev.target.closest('button[data-action]');
    if (!btn) return;
    if (btn.dataset.action === 'copy-card') {{
      const card = btn.closest('.card');
      const body = card && card.querySelector('.card-body');
      if (body) copyText(body.textContent, btn);
    }} else if (btn.dataset.action === 'copy-all') {{
      const parts = Array.from(document.querySelectorAll('.card-body'))
        .map(el => el.textContent.trim());
      copyText(parts.join('\\n\\n---\\n\\n'), btn);
    }}
  }});
}})();
</script>
</body>
</html>
"""


def _render_card(card: Card, idx: int) -> str:
    return (
        '<div class="card">'
        f'<div class="card-label">{html.escape(card.label)}</div>'
        f'<div class="card-body">{html.escape(card.body)}</div>'
        '<div class="meta">'
        f'<span class="word-count">{card.word_count} words</span>'
        f'<button class="btn" data-action="copy-card" type="button">Copy</button>'
        '</div>'
        '</div>'
    )


def _empty_doc(title: str) -> str:
    return (
        f'<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}'
        '</title></head><body><p>(empty)</p></body></html>'
    )


def plain_text_fallback(cards: list[Card], sep: str = "\n\n---\n\n") -> str:
    """Plain-text version of the cards — goes in the Signal message body so
    desktop users and anything that can't render HTML still have the content."""
    return sep.join(f"{c.label}\n{c.body}" for c in cards)
