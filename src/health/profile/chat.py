"""Shared `/profile ...` chat-command parser (Phase G.1.d).

Channels (REPL, Telegram, Signal) call `handle_profile_command(text, orch)`
before routing a message to the pipeline. If the text matches a
profile command, the function returns a reply string for the channel
to deliver. Otherwise returns None and the channel sends the message
through the pipeline as usual.

Mirrors `src/skills/chat.py` and `src/projects/chat.py` exactly so the
channel-side dispatch list stays uniform.

Commands recognized this phase:

  /profile                      Show summary (counts per section + last_updated)
  /profile show <section>       Detailed view of one section
  /profile audit [N]            Recent change log (default 10 entries)
  /profile clear "clear my profile"
                                Nuclear option. Requires the phrase in
                                the same message — no multi-turn dance.
                                Bot tells the user how to type it if
                                they just type `/profile clear`.
  /profile export               Write decrypted JSON to
                                workspace/output/profile-YYYY-MM-DD.json
                                and return the path

Deferred:

  /profile edit <section>       Needs interactive editor flow (a
                                multi-turn dialog that's a real
                                design exercise — DM if you want it)
  /profile setup                Multi-turn wizard for first-time data
                                entry (age range, current meds, etc.).
                                Needs orchestrator-side conversation
                                state tracking that doesn't exist yet.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


# Sections accepted by `/profile show <section>`. Must match
# HealthProfile attribute names.
SECTIONS = (
    "demographics",
    "conditions",
    "medications",
    "allergies",
    "family_history",
    "lifestyle",
    "labs",
    "concerns",
)


_CLEAR_PHRASE = "clear my profile"


async def handle_profile_command(text: str, orchestrator) -> str | None:
    """Return a reply string for `/profile ...` commands, else None.

    Errors are caught and surfaced as user-readable messages — a
    profile command must not crash the bot turn.
    """
    stripped = text.strip()
    lower = stripped.lower()

    if not lower.startswith("/profile"):
        return None

    # Strip leading "/profile" and any whitespace
    arg = stripped[len("/profile"):].strip()

    try:
        if not arg:
            return _cmd_show_summary(orchestrator)

        if arg.lower().startswith("show"):
            section = arg[len("show"):].strip().lower()
            if not section:
                return _cmd_show_summary(orchestrator)
            if section not in SECTIONS:
                return (
                    f"Unknown section {section!r}. Options: "
                    f"{', '.join(SECTIONS)}."
                )
            return _cmd_show_section(orchestrator, section)

        if arg.lower().startswith("audit"):
            tail = arg[len("audit"):].strip()
            try:
                limit = int(tail) if tail else 10
            except ValueError:
                return "Audit usage: `/profile audit [N]` (N is an integer)."
            limit = max(1, min(limit, 100))
            return _cmd_audit(orchestrator, limit)

        if arg.lower().startswith("clear"):
            tail = arg[len("clear"):].strip().strip('"').strip("'")
            if tail != _CLEAR_PHRASE:
                return (
                    "Profile clear is destructive and not reversible.\n"
                    f'To confirm, send: `/profile clear "{_CLEAR_PHRASE}"`\n'
                    "(Tip: the CLI `microwaveos profile clear` has an "
                    "interactive prompt if you'd rather use it.)"
                )
            return _cmd_clear(orchestrator)

        if arg.lower() == "export":
            return _cmd_export(orchestrator)

        return (
            f"Unknown profile command: `/profile {arg}`.\n"
            "Try `/profile`, `/profile show <section>`, "
            "`/profile audit`, `/profile export`, or "
            f'`/profile clear \"{_CLEAR_PHRASE}\"`.'
        )
    except Exception as e:
        log.warning("Profile command %r raised: %s", arg, e)
        return f"Profile command failed: {e}"


# --- Command implementations --------------------------------------------


def _cmd_show_summary(orchestrator) -> str:
    from src.health.profile.store import load_profile

    config = orchestrator.config
    loaded = load_profile(
        orchestrator.session_engine.conn,
        key_source=config.health.phi_encryption_key_source,
    )
    p = loaded.profile
    if loaded.version == 0 and p.is_empty:
        return (
            "No profile data yet. Health turns will accumulate proposals "
            "that you can review with `/profile show pending`."
        )

    last = p.last_updated.isoformat(timespec='seconds')
    pending = sum(1 for u in p.pending_updates if u.status == "pending")

    return (
        f"**Profile** (v{loaded.version}, updated {last})\n"
        f"  • demographics: {_demo_count(p.demographics)}/6 fields\n"
        f"  • conditions:   {len(p.conditions)}\n"
        f"  • medications:  {len(p.medications)}\n"
        f"  • allergies:    {len(p.allergies)}\n"
        f"  • family:       {len(p.family_history)}\n"
        f"  • lifestyle:    {_lifestyle_count(p.lifestyle)}/5 fields\n"
        f"  • labs:         {len(p.labs)}\n"
        f"  • concerns:     {len(p.concerns)}\n"
        f"  • pending:      {pending}"
    )


def _cmd_show_section(orchestrator, section: str) -> str:
    from src.health.profile.store import load_profile

    config = orchestrator.config
    loaded = load_profile(
        orchestrator.session_engine.conn,
        key_source=config.health.phi_encryption_key_source,
    )
    p = loaded.profile

    if section == "demographics":
        d = p.demographics
        lines = ["**Demographics**"]
        for label, fld in [
            ("age range",    d.age_range),
            ("sex (birth)",  d.sex_assigned_at_birth),
            ("gender id",    d.gender_identity),
            ("height range", d.height_range),
            ("weight range", d.weight_range),
            ("pregnancy",    d.pregnancy_status),
        ]:
            if fld is not None:
                lines.append(f"  • {label}: {fld.value}")
        if len(lines) == 1:
            lines.append("  (no fields set)")
        return "\n".join(lines)

    if section == "lifestyle":
        lf = p.lifestyle
        lines = ["**Lifestyle**"]
        for label, fld in [
            ("smoking",  lf.smoking),
            ("alcohol",  lf.alcohol),
            ("exercise", lf.exercise_frequency),
            ("sleep",    lf.sleep_hours_typical),
            ("diet",     lf.diet_pattern),
        ]:
            if fld is not None:
                lines.append(f"  • {label}: {fld.value}")
        if len(lines) == 1:
            lines.append("  (no fields set)")
        return "\n".join(lines)

    items = getattr(p, section, None)
    if not isinstance(items, list):
        return f"Unknown section: {section}"
    if not items:
        return f"**{section.title()}**\n  (none)"

    lines = [f"**{section.title()}** ({len(items)})"]
    for entry in items:
        headline = (
            getattr(entry, "name", None)
            or getattr(entry, "substance", None)
            or getattr(entry, "test_name", None)
            or getattr(entry, "relation", None)
            or getattr(entry, "text", None)
            or "?"
        )
        bits = []
        for attr in ("status", "severity", "dose", "date"):
            val = getattr(entry, attr, None)
            if val:
                bits.append(f"{attr}={val}")
        suffix = (" — " + ", ".join(bits)) if bits else ""
        lines.append(f"  • {headline}{suffix}")
    return "\n".join(lines)


def _cmd_audit(orchestrator, limit: int) -> str:
    from src.health.profile.store import list_change_log

    rows = list_change_log(orchestrator.session_engine.conn, limit=limit)
    if not rows:
        return "No profile changes recorded yet."

    lines = ["**Profile audit** (newest first)"]
    for r in rows:
        ts = datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d %H:%M")
        lines.append(f"  [{ts}] {r['operation']:<8} section={r['section']}")
    return "\n".join(lines)


def _cmd_clear(orchestrator) -> str:
    from src.health.profile.store import _user_key_id, load_profile

    config = orchestrator.config
    conn = orchestrator.session_engine.conn
    loaded = load_profile(
        conn, key_source=config.health.phi_encryption_key_source,
    )
    if loaded.version == 0 and loaded.profile.is_empty:
        return "No profile data to clear."

    user_id = loaded.profile.user_id
    conn.execute(
        "DELETE FROM health_profiles WHERE user_id = ?", (user_id,),
    )
    conn.execute(
        "DELETE FROM profile_change_log WHERE user_key_id = ?",
        (_user_key_id(user_id),),
    )
    return "✓ Profile cleared. All entries and change log removed."


def _cmd_export(orchestrator) -> str:
    from src.health.profile.store import load_profile

    config = orchestrator.config
    loaded = load_profile(
        orchestrator.session_engine.conn,
        key_source=config.health.phi_encryption_key_source,
    )
    if loaded.version == 0 and loaded.profile.is_empty:
        return "No profile data to export."

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"profile-{datetime.now().strftime('%Y-%m-%d')}.json"

    # Atomic write
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(loaded.profile.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(out_path)
    return (
        f"✓ Wrote profile to `{out_path}`\n"
        "⚠ Contents are decrypted PHI — don't commit, don't share, "
        "delete when done."
    )


# --- Helpers --------------------------------------------------------------


def _demo_count(d) -> int:
    return sum(
        1 for f in [
            d.age_range, d.sex_assigned_at_birth, d.gender_identity,
            d.height_range, d.weight_range, d.pregnancy_status,
        ] if f is not None
    )


def _lifestyle_count(lf) -> int:
    return sum(
        1 for f in [
            lf.smoking, lf.alcohol, lf.exercise_frequency,
            lf.sleep_hours_typical, lf.diet_pattern,
        ] if f is not None
    )
