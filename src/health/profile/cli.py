"""`microwaveos profile ...` CLI surface (Phase G.1.c).

Commands shipped in this phase:

  profile show                     summary (counts per section, last_updated)
  profile show <section>           detailed view of one section
  profile audit                    encrypted change log (op + section + ts)
  profile clear                    delete the entire profile (typed confirm)
  profile export <path>            write decrypted profile to a JSON file

Deferred to Phase G.1.d (REPL integration):
  profile edit <section>           opens $EDITOR — needs an interactive
                                   confirmation flow that fits the REPL
                                   shape better than a one-shot CLI
  profile delete <section> <item>  needs section-specific item addressing
  profile undo                     needs the soft-delete buffer to be
                                   populated by `profile delete` first
  profile setup                    interactive — REPL is the right home

All commands use `phi_encryption_key_source` from HealthConfig (default
"keychain"). Tests pin it to "env" via the autouse fixture.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.config import load_config
from src.health.profile import (
    DEFAULT_USER_ID,
    HealthProfile,
    init_tables,
    list_change_log,
    load_profile,
)
from src.session.engine import SessionEngine


log = logging.getLogger(__name__)


# Section names accepted by `profile show <section>`. Must match
# attribute names on HealthProfile.
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


def profile_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="microwaveos profile",
        description=(
            "Health profile — view, audit, export, or clear the "
            "encrypted profile. See the spec at "
            "microwave-health-profile-spec.md for design intent."
        ),
    )
    sub = parser.add_subparsers(dest="action", required=True)

    p_show = sub.add_parser(
        "show",
        help="Show the profile summary, or one section in detail.",
    )
    p_show.add_argument(
        "section",
        nargs="?",
        default=None,
        choices=SECTIONS,
        help=f"Optional section name. One of: {', '.join(SECTIONS)}.",
    )

    p_audit = sub.add_parser(
        "audit",
        help="Show recent profile change log entries (newest first).",
    )
    p_audit.add_argument(
        "--limit", type=int, default=20,
        help="Maximum number of entries (default 20).",
    )

    p_clear = sub.add_parser(
        "clear",
        help="Delete the entire profile. Requires typed confirmation.",
    )
    p_clear.add_argument(
        "--yes-really",
        action="store_true",
        help="Skip the interactive 'yes-i-really-mean-it' prompt "
             "(use only for scripted deployments).",
    )

    p_export = sub.add_parser(
        "export",
        help="Write the decrypted profile as JSON to a file.",
    )
    p_export.add_argument(
        "path",
        type=Path,
        help="Destination file path. Parent must exist.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.action == "show":
        return _cmd_show(args.section)
    if args.action == "audit":
        return _cmd_audit(args.limit)
    if args.action == "clear":
        return _cmd_clear(yes_really=args.yes_really)
    if args.action == "export":
        return _cmd_export(args.path)
    return 1


# --- Command handlers ------------------------------------------------------


def _cmd_show(section: str | None) -> int:
    config = load_config()
    engine = _connect(config)
    loaded = load_profile(
        engine.conn, key_source=config.health.phi_encryption_key_source,
    )
    p = loaded.profile

    if section:
        return _render_section(p, section)

    # Summary view
    if loaded.version == 0 and p.is_empty:
        print(
            "No profile data yet. Health turns will accumulate "
            "structured facts once Phase G.2 (extractor) is wired."
        )
        return 0

    print(f"Profile version: {loaded.version}")
    print(f"Last updated:    {p.last_updated.isoformat(timespec='seconds')}")
    print()
    print(f"  demographics:  {_demographics_summary(p.demographics)}")
    print(f"  conditions:    {len(p.conditions)}")
    print(f"  medications:   {len(p.medications)}")
    print(f"  allergies:     {len(p.allergies)}")
    print(f"  family:        {len(p.family_history)}")
    print(f"  lifestyle:     {_lifestyle_summary(p.lifestyle)}")
    print(f"  labs:          {len(p.labs)}")
    print(f"  concerns:      {len(p.concerns)} "
          f"({sum(1 for c in p.concerns if c.status == 'active')} active)")
    if p.pending_updates:
        print(f"  pending:       {len(p.pending_updates)} awaiting review "
              f"(run `microwaveos memory contradictions` for now; "
              f"the profile review CLI lands in Phase G.2)")
    return 0


def _cmd_audit(limit: int) -> int:
    config = load_config()
    engine = _connect(config)
    rows = list_change_log(engine.conn, limit=limit)
    if not rows:
        print("No profile changes recorded yet.")
        return 0

    for r in rows:
        ts = datetime.fromtimestamp(r["timestamp"]).isoformat(timespec="seconds")
        print(f"[{ts}] {r['operation']:<8} section={r['section']}")
    return 0


def _cmd_clear(*, yes_really: bool) -> int:
    config = load_config()
    engine = _connect(config)

    # Show what's about to be lost
    loaded = load_profile(
        engine.conn, key_source=config.health.phi_encryption_key_source,
    )
    if loaded.version == 0 and loaded.profile.is_empty:
        print("No profile data to clear.")
        return 0

    print(
        "About to delete the entire health profile and its change log. "
        "This is NOT reversible — once cleared, the encrypted blob is "
        "removed from the database. Any exports you made are unaffected."
    )
    print()
    print(f"  current version: {loaded.version}")
    print(f"  last updated:    {loaded.profile.last_updated.isoformat(timespec='seconds')}")
    print()

    if not yes_really:
        try:
            answer = input("Type 'clear my profile' to confirm: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 1
        if answer != "clear my profile":
            print("Aborted (phrase didn't match).")
            return 1

    # Delete the row + all change log entries for this user.
    user_id = loaded.profile.user_id
    engine.conn.execute(
        "DELETE FROM health_profiles WHERE user_id = ?", (user_id,),
    )
    from src.health.profile.store import _user_key_id
    engine.conn.execute(
        "DELETE FROM profile_change_log WHERE user_key_id = ?",
        (_user_key_id(user_id),),
    )
    print("✓ Profile cleared.")
    return 0


def _cmd_export(path: Path) -> int:
    config = load_config()
    engine = _connect(config)
    loaded = load_profile(
        engine.conn, key_source=config.health.phi_encryption_key_source,
    )

    if loaded.version == 0 and loaded.profile.is_empty:
        print("No profile data to export.")
        return 0

    path = Path(path)
    if not path.parent.exists():
        print(f"Parent directory does not exist: {path.parent}")
        return 1

    # Write atomically — generate to .tmp then rename so a write
    # interrupted mid-flight doesn't leave a half-baked file.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(loaded.profile.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)
    print(
        f"✓ Wrote profile to {path}\n"
        f"  WARNING: contents are decrypted plaintext. Treat this file "
        f"as PHI — don't commit it, don't email it, delete when done."
    )
    return 0


# --- Render helpers --------------------------------------------------------


def _render_section(p: HealthProfile, section: str) -> int:
    """Detailed view of one section."""
    if section == "demographics":
        d = p.demographics
        print("Demographics:")
        for label, field in [
            ("age range",    d.age_range),
            ("sex (birth)",  d.sex_assigned_at_birth),
            ("gender id",    d.gender_identity),
            ("height range", d.height_range),
            ("weight range", d.weight_range),
            ("pregnancy",    d.pregnancy_status),
        ]:
            if field is not None:
                print(f"  {label:<14} {field.value} {_provenance_note(field)}")
        return 0

    if section == "lifestyle":
        lf = p.lifestyle
        print("Lifestyle:")
        for label, field in [
            ("smoking",   lf.smoking),
            ("alcohol",   lf.alcohol),
            ("exercise",  lf.exercise_frequency),
            ("sleep",     lf.sleep_hours_typical),
            ("diet",      lf.diet_pattern),
        ]:
            if field is not None:
                print(f"  {label:<10} {field.value} {_provenance_note(field)}")
        return 0

    # List-shaped sections
    items = getattr(p, section, None)
    if items is None or not isinstance(items, list):
        print(f"Unknown section: {section}")
        return 1
    if not items:
        print(f"No entries in {section}.")
        return 0

    print(f"{section.title()} ({len(items)}):")
    for entry in items:
        # Each section has its own shape — try to render the "headline"
        # field plus status/severity if present, then provenance.
        headline = (
            getattr(entry, "name", None)
            or getattr(entry, "substance", None)
            or getattr(entry, "test_name", None)
            or getattr(entry, "relation", None)
            or getattr(entry, "text", None)
            or "?"
        )
        status_bits = []
        for attr in ("status", "severity", "dose", "date"):
            val = getattr(entry, attr, None)
            if val:
                status_bits.append(f"{attr}={val}")
        status = (" [" + ", ".join(status_bits) + "]") if status_bits else ""
        print(f"  • {headline}{status} {_provenance_note(entry.field_meta)}")
    return 0


def _provenance_note(field) -> str:
    """Compact provenance suffix."""
    if field is None:
        return ""
    src = field.source.replace("_", " ")
    return f"\n      (source: {src}, confirmed={field.confirmed}, "\
           f"added {field.added_at.isoformat(timespec='seconds')})"


def _demographics_summary(d) -> str:
    set_fields = sum(
        1 for f in [
            d.age_range, d.sex_assigned_at_birth, d.gender_identity,
            d.height_range, d.weight_range, d.pregnancy_status,
        ] if f is not None
    )
    return f"{set_fields}/6 fields set"


def _lifestyle_summary(lf) -> str:
    set_fields = sum(
        1 for f in [
            lf.smoking, lf.alcohol, lf.exercise_frequency,
            lf.sleep_hours_typical, lf.diet_pattern,
        ] if f is not None
    )
    return f"{set_fields}/5 fields set"


# --- DB connection wiring --------------------------------------------------


def _connect(config) -> SessionEngine:
    """Connect to the shared DB and ensure profile tables exist.

    Profile tables init alongside the existing turns/breadcrumbs/
    consolidation schema. Idempotent — safe to run on every CLI
    invocation.
    """
    engine = SessionEngine(config.db_path)
    engine.connect()
    init_tables(engine.conn)
    return engine
