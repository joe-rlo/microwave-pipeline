"""Shared `/skill ...` chat-command parser.

Channels (REPL, Telegram, Signal) call `handle_skill_command(text, orch)`
before routing a message to the pipeline. If the text is a skill command,
the function mutates orchestrator state and returns a human-readable
reply for the channel to deliver. Otherwise it returns None and the
channel sends the message through the pipeline as usual.
"""

from __future__ import annotations

from src.skills.loader import SkillNotFound


async def handle_skill_command(text: str, orchestrator) -> str | None:
    """Return a reply string if `text` is a `/skill` / `/skills` command,
    else None.

    Recognized:
    - `/skill <name>`           activate
    - `/skill off` / `none`     deactivate
    - `/skill`                  show current
    - `/skills`                 list all
    """
    stripped = text.strip()
    lower = stripped.lower()

    if lower == "/skills":
        return _list_skills(orchestrator)

    if lower == "/skill" or lower == "/skill ":
        return _current_skill(orchestrator)

    if not lower.startswith("/skill "):
        return None

    arg = stripped[len("/skill "):].strip()
    if arg.lower() in ("off", "none", "clear"):
        orchestrator.clear_active_skill()
        return "Active skill cleared."

    try:
        skill = orchestrator.set_active_skill(arg)
    except SkillNotFound:
        return f"No skill named '{arg}'. Try `/skills` to list available."
    except Exception as e:
        return f"Could not activate skill: {e}"

    desc = skill.description or "(no description)"
    return f"Active skill: {skill.name}\n\n{desc}"


def _current_skill(orchestrator) -> str:
    skill = orchestrator.get_active_skill()
    if skill is None:
        return "No active skill. Use `/skill <name>` to activate one, or `/skills` to list."
    return f"Active skill: {skill.name}\n\n{skill.description or '(no description)'}"


def _list_skills(orchestrator) -> str:
    skills = orchestrator.list_skills()
    if not skills:
        return (
            "No skills found. Create one with `microwaveos skills new <name>`."
        )
    lines = ["Available skills:"]
    active = orchestrator.get_active_skill()
    active_name = active.name if active else None
    for s in skills:
        marker = "→ " if s.name == active_name else "  "
        desc = s.description or "(no description)"
        if len(desc) > 80:
            desc = desc[:77] + "…"
        lines.append(f"{marker}{s.name} — {desc}")
    lines.append("")
    lines.append("Use `/skill <name>` to activate, `/skill off` to deactivate.")
    return "\n".join(lines)
