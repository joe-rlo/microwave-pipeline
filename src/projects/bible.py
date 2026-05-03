"""BIBLE.md update flow for novel/screenplay projects.

The LLM surfaces "Possibly new for BIBLE" suggestions at the end of every
draft (this is in the novel-writing and screenplay-writing skill bodies).
The user commits any of them with `/bible add <name> [description]`.

Bible writes are user-approved on purpose — auto-writing risks polluting
the project's canonical state with mid-draft inventions the user hasn't
endorsed.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Section heading where new entries land. We use a fixed string so the
# parser can find it reliably; users can rearrange the file freely as
# long as this header stays present.
_FACTS_HEADING = "## Established facts"


async def handle_bible_command(text: str, orchestrator) -> str | None:
    """Return a reply for `/bible*` commands, else None.

    Recognized:
    - `/bible add <name> [description]`   append a new entry
    - `/bible show`                        print the current bible
    - `/bible`                             same as show
    """
    stripped = text.strip()
    lower = stripped.lower()

    if lower in ("/bible", "/bible show"):
        return _show_bible(orchestrator)

    if not lower.startswith("/bible add "):
        if lower.startswith("/bible"):
            return (
                "Usage: `/bible add <name> [description]` or `/bible show`. "
                "BIBLE updates only work when a project is active."
            )
        return None

    # `/bible add <name> [description...]`
    rest = stripped[len("/bible add "):].strip()
    if not rest:
        return "Usage: `/bible add <name> [description]`"

    # First whitespace-separated token is the name; everything after is
    # the description. Names with spaces should be wrapped in quotes —
    # detect that case so "Detective Walsh" works.
    name, description = _split_name_and_description(rest)

    project = orchestrator.get_active_project()
    if project is None:
        return "No active project. Use `/project <name>` to activate one first."
    if project.bible_path is None:
        return f"Project {project.name!r} has no BIBLE.md. Try `/bible show` first."

    try:
        added_to = _append_entry(project.bible_path, name, description)
    except Exception as e:
        log.exception("bible append failed")
        return f"Could not update BIBLE.md: {e}"

    return (
        f"✓ added **{name}** to BIBLE.md"
        + (f"\n\n> {description}" if description else "")
        + f"\n\n_({added_to})_"
    )


def _show_bible(orchestrator) -> str:
    project = orchestrator.get_active_project()
    if project is None:
        return "No active project. Use `/project <name>` to activate one first."
    if not project.has_bible:
        return f"Project {project.name!r} has no BIBLE.md yet."
    text = project.bible_path.read_text(encoding="utf-8")
    if len(text) > 3500:
        text = text[:3500] + "\n\n…(truncated, see " + str(project.bible_path) + ")"
    return text


def _split_name_and_description(s: str) -> tuple[str, str]:
    """Pull a name + description out of a `/bible add ...` argument string.

    Quoted names ("Detective Walsh") preserve internal spaces. Otherwise
    the first whitespace-delimited token is the name.
    """
    s = s.strip()
    if s.startswith('"') or s.startswith("'"):
        quote = s[0]
        end = s.find(quote, 1)
        if end > 0:
            return s[1:end].strip(), s[end + 1:].strip()
    parts = s.split(None, 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1].strip()


def _append_entry(bible_path: Path, name: str, description: str) -> str:
    """Append `### name\\n\\ndescription` under the established-facts heading.

    Creates the heading if it doesn't exist yet. Returns a short
    description of where the entry landed.
    """
    text = bible_path.read_text(encoding="utf-8") if bible_path.is_file() else ""

    entry = f"\n### {name}\n"
    if description:
        entry += f"\n{description}\n"

    if _FACTS_HEADING in text:
        # Insert right after the heading (and any blank line following it).
        # We insert at the bottom of the file if the heading is the last
        # section — appending is safer than trying to find a section break.
        new_text = text.rstrip() + "\n" + entry
    else:
        # No facts section yet — create one.
        new_text = text.rstrip() + f"\n\n{_FACTS_HEADING}\n" + entry

    bible_path.write_text(new_text, encoding="utf-8")
    return str(bible_path)
