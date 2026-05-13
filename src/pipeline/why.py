"""`/why` chat command — surface the retrieval that fed the last turn.

Read-only inspection tool. Doesn't trigger a new pipeline run; just
formats the SearchResult the orchestrator cached during the previous
turn's Stage 2 (search). Useful for two cases:

1. The bot said something surprising — show me what it was reading.
2. Tuning retrieval — see whether a query is pulling the right
   fragments without trial-and-erroring through real conversation.

Syntax:
    /why            top 5 fragments, source + 100-char snippet
    /why -v         same plus retrieval scores (REPL convention)
    /why scores     same as -v, mobile-friendly verb (Signal / Telegram)

Default hides scores because score values without prior calibration
on what 0.82 means in your corpus are noise — they answer a tuning
question, not "why did you say that?" The verbose form is gated for
when you're actively tuning.
"""

from __future__ import annotations

from pathlib import Path

from src.session.models import MemoryFragment

# Cap on how many fragments to render. Matches the spec's "top 5" call.
# Search may have returned more; we trim here for legibility.
_MAX_FRAGMENTS = 5

# Snippet length cap per fragment. Long enough to recognize the
# content, short enough that a 5-fragment output fits a phone screen.
_SNIPPET_CHARS = 100


async def handle_why_command(text: str, orchestrator) -> str | None:
    """Return a reply for `/why` commands, else None.

    Async to match the existing handler contract (skill / project /
    bible), even though no I/O is involved.
    """
    stripped = text.strip()
    lower = stripped.lower()

    if lower == "/why":
        verbose = False
    elif lower in ("/why -v", "/why scores", "/why verbose"):
        verbose = True
    else:
        return None

    last = getattr(orchestrator, "_last_search_result", None)
    if last is None:
        return (
            "No retrieval cached yet — send a substantive message first, "
            "then /why will show what it pulled."
        )

    fragments = list(last.fragments or [])
    if not fragments:
        return (
            "Last turn ran search but retrieval came back empty. "
            f"Strategy: {last.strategy_used}"
        )

    workspace_dir = _get_workspace_dir(orchestrator)
    lines = ["why: last turn retrieved"]
    for frag in fragments[:_MAX_FRAGMENTS]:
        lines.append(_format_fragment(frag, workspace_dir, verbose))
    return "\n".join(lines)


def _format_fragment(
    frag: MemoryFragment,
    workspace_dir: Path | None,
    verbose: bool,
) -> str:
    """One fragment → one or two lines of output.

    Default form (no verbose):
        workspace/MEMORY.md
          "Joe's primary project is..."

    Verbose form:
        [0.82] workspace/MEMORY.md
          "Joe's primary project is..."
    """
    source = _abbreviate_source(frag.source, workspace_dir)
    snippet = _snippet(frag.content)
    if verbose:
        header = f"  [{frag.score:.2f}] {source}"
    else:
        header = f"  {source}"
    return f"{header}\n      {snippet!r}"


def _snippet(content: str) -> str:
    """Single-line snippet of fragment content, capped at _SNIPPET_CHARS."""
    text = (content or "").strip().replace("\n", " ").replace("\r", " ")
    # Collapse runs of whitespace introduced by the newline replace
    text = " ".join(text.split())
    if len(text) <= _SNIPPET_CHARS:
        return text
    return text[: _SNIPPET_CHARS - 1].rstrip() + "…"


def _abbreviate_source(source: str, workspace_dir: Path | None) -> str:
    """Trim absolute paths to something readable.

    If the source is inside the workspace dir, strip that prefix
    (`/Users/joe/.microwaveos/workspace/MEMORY.md` → `workspace/MEMORY.md`).
    Sources from session summaries / API origins get returned
    unchanged — they aren't filesystem paths.
    """
    if not source:
        return "(unknown source)"
    # Non-filesystem sources (e.g. "session:abc:summary") — pass through
    if not source.startswith("/") and ":" in source:
        return source
    try:
        path = Path(source)
    except (TypeError, ValueError):
        return source
    if workspace_dir is not None:
        try:
            rel = path.resolve().relative_to(workspace_dir.resolve())
            # Show "workspace/<rel>" so it's clear which root, matching
            # the convention used throughout IDENTITY / skills / etc.
            return f"workspace/{rel}"
        except ValueError:
            pass  # not under workspace; fall through to last-2-components
    # Generic: show the last 2 path components so logs / system files
    # are at least recognizable
    parts = path.parts
    if len(parts) >= 2:
        return f".../{parts[-2]}/{parts[-1]}"
    return source


def _get_workspace_dir(orchestrator) -> Path | None:
    """Defensive accessor — older orchestrator instances or test
    doubles may not have a config attached."""
    config = getattr(orchestrator, "config", None)
    if config is None:
        return None
    return getattr(config, "workspace_dir", None)
