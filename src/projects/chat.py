"""Shared `/project ...` chat-command parser.

Mirrors `src.skills.chat.handle_skill_command`. Channels call this
before routing a message to the pipeline; if it's a project command,
the function mutates orchestrator state and returns a reply.
"""

from __future__ import annotations

from src.projects.loader import ProjectNotFound


def handle_project_command(text: str, orchestrator) -> str | None:
    """Return a reply string for `/project*` commands, else None.

    Recognized:
    - `/project <name>`              activate
    - `/project off|none|clear`      deactivate
    - `/project`                     show current
    - `/project status`              detailed status (drafts, words, bible)
    - `/projects`                    list all
    """
    stripped = text.strip()
    lower = stripped.lower()

    if lower == "/projects":
        return _list_projects(orchestrator)

    if lower in ("/project", "/project status"):
        return _current_project(orchestrator, detailed=(lower.endswith("status")))

    if not lower.startswith("/project "):
        return None

    arg = stripped[len("/project "):].strip()
    if arg.lower() == "status":
        return _current_project(orchestrator, detailed=True)
    if arg.lower() in ("off", "none", "clear"):
        orchestrator.clear_active_project()
        return "Active project cleared."

    try:
        project = orchestrator.set_active_project(arg)
    except ProjectNotFound:
        return f"No project named '{arg}'. Try `/projects` to list available."
    except Exception as e:
        return f"Could not activate project: {e}"

    skill_note = f" (auto-activates skill: {project.skill})" if project.skill else ""
    return (
        f"Active project: {project.name} [{project.type}, {project.status}]"
        f"{skill_note}\n\n"
        f"{project.description or '(no description)'}"
    )


def _current_project(orchestrator, detailed: bool) -> str:
    p = orchestrator.get_active_project()
    if p is None:
        return (
            "No active project. Use `/project <name>` to activate one, "
            "or `/projects` to list."
        )
    lines = [f"Active project: {p.name} [{p.type}, {p.status}]"]
    if p.skill:
        lines.append(f"Skill: {p.skill}")
    if detailed:
        words = p.word_count()
        if words or p.target_words:
            target = f" / {p.target_words:,}" if p.target_words else ""
            lines.append(f"Words: {words:,}{target}")
        drafts = p.list_drafts()
        if drafts:
            lines.append(f"Drafts: {len(drafts)} files")
            # Show the most recent (by mtime) for orientation
            recent = max(drafts, key=lambda d: d.stat().st_mtime)
            lines.append(f"Most recent draft: {recent.name}")
        if p.has_bible:
            lines.append("BIBLE.md: present")
        if p.has_outline:
            lines.append("Outline: present")
    if p.description:
        lines.append("")
        lines.append(p.description)
    return "\n".join(lines)


def _list_projects(orchestrator) -> str:
    projects = orchestrator.list_projects()
    if not projects:
        return "No projects yet. Create one with `microwaveos projects new <name> --type blog|novel|screenplay`."
    active = orchestrator.get_active_project()
    active_name = active.name if active else None
    lines = ["Available projects:"]
    for p in projects:
        marker = "→ " if p.name == active_name else "  "
        words = p.word_count()
        words_str = f", {words:,} words" if words else ""
        desc = p.description or "(no description)"
        if len(desc) > 60:
            desc = desc[:57] + "…"
        lines.append(
            f"{marker}{p.name} [{p.type}, {p.status}{words_str}] — {desc}"
        )
    lines.append("")
    lines.append("Use `/project <name>` to activate, `/project off` to deactivate.")
    return "\n".join(lines)
