"""One-off: repoint the morning-briefing job at the new skill.

Run once, then delete. Updates skill_name + replaces the legacy
inline prompt with a short trigger message (the real instructions
live in skills/morning-briefing/SKILL.md now).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path.home() / ".microwaveos" / "data" / "memory.db"
TRIGGER = (
    "Compose today's morning briefing using the live weather and news "
    "data in [Pre-fetch context]. Four cards, --- separated, in the "
    "order specified by the skill body."
)


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "UPDATE scheduled_jobs SET skill_name = ?, prompt_or_text = ? "
        "WHERE name = ?",
        ("morning-briefing", TRIGGER, "morning-briefing"),
    )
    conn.commit()
    for r in conn.execute(
        "SELECT name, mode, skill_name, length(prompt_or_text) AS plen, "
        "target_channel, recipient_id FROM scheduled_jobs WHERE name = ?",
        ("morning-briefing",),
    ):
        print(dict(r))


if __name__ == "__main__":
    main()
