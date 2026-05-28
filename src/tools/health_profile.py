"""LLM-facing tools for reading the user's health profile.

Three read-only tools, mirroring the safe subset of the `/profile`
slash command (intercepted in `src/health/profile/chat.py`):

- `health_profile_summary` — section counts + last updated
- `health_profile_show`    — detailed view of one section
- `health_profile_audit`   — recent change log (op + section + ts)

What's deliberately NOT here:
- `clear` / destructive ops — only via `/profile clear` (requires the
  literal "clear my profile" phrase, no LLM shortcut).
- `export` — writes decrypted PHI to disk; only via `/profile export`
  so the human is unambiguously in the loop.
- `edit` / `set` — extraction + confirmation is the canonical write
  path; these tools deliberately stay read-only.

Routing note: any turn that calls one of these tools is, by definition,
about the user's health. The orchestrator's triage stage classifies
those as PHI and routes to the BAA Bedrock path before the tool runs —
the decrypted profile never reaches NEAR. The tool handler itself
doesn't need to enforce that; failing to set HEALTH_MODULE_ENABLED is
what gates registration.

PHI handling inside the handler:
- Decryption uses the existing per-user Keychain-derived key.
- Returned JSON is plaintext profile data going straight into the LLM
  prompt. The session that called the tool is already PHI-routed.
- No fields are scrubbed or redacted on the way out; the user asked
  to see their own profile and consent is implicit in the question.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from src.db import connect as db_connect

log = logging.getLogger(__name__)


# What the LLM sees in its system context.
HEALTH_PROFILE_TOOL_DOCS = """\
**health_profile_summary** — Read a high-level snapshot of the user's stored health profile: section counts (conditions, medications, allergies, family history, labs, concerns) plus when it was last updated.

When to use:
- "What's in my health profile?", "give me my profile summary", "what do you have on file for me?".
- As a first step before `health_profile_show` when the user hasn't named a section.

**health_profile_show** — Get the detailed contents of one profile section.

When to use:
- "What meds am I on?" → section=medications
- "What conditions do I have logged?" → section=conditions
- "What allergies?" → section=allergies
- Plus: family_history, lifestyle, demographics, labs, concerns

How to use:
- `section`: one of `demographics | lifestyle | conditions | medications | allergies | family_history | labs | concerns`

**health_profile_audit** — Recent change-log entries for the profile (operation + section + timestamp). Useful for "what's been added recently?" or "did the discontinue go through?".

How to use:
- `limit`: optional (default 10, max 100).
"""


# --- JSON schemas ---------------------------------------------------------


_SECTIONS = (
    "demographics", "lifestyle", "conditions", "medications",
    "allergies", "family_history", "labs", "concerns",
)


HEALTH_PROFILE_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}

HEALTH_PROFILE_SHOW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "section": {
            "type": "string",
            "enum": list(_SECTIONS),
            "description": "Which section of the profile to read",
        },
    },
    "required": ["section"],
    "additionalProperties": False,
}

HEALTH_PROFILE_AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
            "description": "Max log entries to return (default 10)",
        },
    },
    "additionalProperties": False,
}


# --- Helpers --------------------------------------------------------------


def _key_source(config) -> str:
    """Resolve the encryption key source from config, defaulting safely."""
    h = getattr(config, "health", None)
    src = getattr(h, "phi_encryption_key_source", None) if h else None
    return src or "keychain"


def _open_conn(config):
    """Open a fresh DB connection and run the profile init DDL.

    The orchestrator's session_engine.conn is what the slash-command
    path uses, but tool handlers don't have a handle to that. Opening
    a new connection is fine — apsw + the WAL setup in db_connect()
    handle concurrent readers cleanly.

    The init call is idempotent (CREATE TABLE IF NOT EXISTS) and
    cheap, so we don't bother caching here.
    """
    from src.health.profile.store import init_tables

    conn = db_connect(config.db_path)
    init_tables(conn)
    return conn


def _field_value(fld):
    """Demographics/lifestyle fields are wrapped in a typed container
    with `.value`; unwrap for display. Returns None when the field
    itself is None."""
    return getattr(fld, "value", None) if fld is not None else None


def _demo_dict(d) -> dict[str, Any]:
    return {
        "age_range": _field_value(d.age_range),
        "sex_assigned_at_birth": _field_value(d.sex_assigned_at_birth),
        "gender_identity": _field_value(d.gender_identity),
        "height_range": _field_value(d.height_range),
        "weight_range": _field_value(d.weight_range),
        "pregnancy_status": _field_value(d.pregnancy_status),
    }


def _lifestyle_dict(lf) -> dict[str, Any]:
    return {
        "smoking": _field_value(lf.smoking),
        "alcohol": _field_value(lf.alcohol),
        "exercise_frequency": _field_value(lf.exercise_frequency),
        "sleep_hours_typical": _field_value(lf.sleep_hours_typical),
        "diet_pattern": _field_value(lf.diet_pattern),
    }


def _list_entry_dict(entry) -> dict[str, Any]:
    """Generic flattener for the list sections (conditions/meds/etc.).

    Pulls the headline-naming field (whatever variant the section uses)
    plus any optional structured attributes that exist on the entry.
    Skips fields whose value is None so the LLM gets a tight payload.
    """
    out: dict[str, Any] = {}
    for attr in (
        "name", "substance", "test_name", "relation", "text",
        "status", "severity", "dose", "frequency", "date",
        "reason", "notes",
    ):
        val = getattr(entry, attr, None)
        if val is not None and val != "":
            out[attr] = val
    return out


def _section_payload(profile, section: str) -> dict[str, Any] | list[dict]:
    if section == "demographics":
        return _demo_dict(profile.demographics)
    if section == "lifestyle":
        return _lifestyle_dict(profile.lifestyle)
    items = getattr(profile, section, None)
    if not isinstance(items, list):
        raise RuntimeError(f"Unknown section {section!r}")
    return [_list_entry_dict(e) for e in items]


# --- Handlers -------------------------------------------------------------


async def _handle_summary(args: dict[str, Any], *, config) -> str:
    from src.health.profile.store import load_profile

    conn = _open_conn(config)
    try:
        loaded = load_profile(conn, key_source=_key_source(config))
    finally:
        conn.close()

    p = loaded.profile
    if loaded.version == 0 and p.is_empty:
        return json.dumps({
            "status": "empty",
            "note": (
                "No profile data yet. Health turns will accumulate "
                "proposals that the user can review with `/profile show pending`."
            ),
        })

    pending = sum(1 for u in p.pending_updates if u.status == "pending")
    return json.dumps({
        "version": loaded.version,
        "last_updated": p.last_updated.isoformat(timespec="seconds"),
        "section_counts": {
            "demographics_fields_set": sum(
                1 for v in _demo_dict(p.demographics).values() if v is not None
            ),
            "lifestyle_fields_set": sum(
                1 for v in _lifestyle_dict(p.lifestyle).values() if v is not None
            ),
            "conditions": len(p.conditions),
            "medications": len(p.medications),
            "allergies": len(p.allergies),
            "family_history": len(p.family_history),
            "labs": len(p.labs),
            "concerns": len(p.concerns),
        },
        "pending_proposals": pending,
    }, indent=2)


async def _handle_show(args: dict[str, Any], *, config) -> str:
    from src.health.profile.store import load_profile

    section = (args.get("section") or "").strip().lower()
    if section not in _SECTIONS:
        raise RuntimeError(
            f"`section` must be one of {list(_SECTIONS)}; got {section!r}"
        )

    conn = _open_conn(config)
    try:
        loaded = load_profile(conn, key_source=_key_source(config))
    finally:
        conn.close()

    payload = _section_payload(loaded.profile, section)

    # Container-style sections (demographics/lifestyle) return dicts;
    # list sections return lists. Surface a `count` for list sections so
    # the LLM doesn't have to count items itself.
    if isinstance(payload, list):
        return json.dumps({
            "section": section,
            "count": len(payload),
            "entries": payload,
        }, indent=2)
    return json.dumps({"section": section, "fields": payload}, indent=2)


async def _handle_audit(args: dict[str, Any], *, config) -> str:
    from src.health.profile.store import list_change_log

    raw_limit = args.get("limit", 10)
    try:
        limit = max(1, min(100, int(raw_limit)))
    except (TypeError, ValueError):
        raise RuntimeError(f"`limit` must be an integer; got {raw_limit!r}")

    conn = _open_conn(config)
    try:
        rows = list_change_log(conn, limit=limit)
    finally:
        conn.close()

    return json.dumps({
        "count": len(rows),
        "entries": [
            {
                "timestamp": datetime.fromtimestamp(r["timestamp"]).isoformat(
                    timespec="seconds"
                ),
                "operation": r["operation"],
                "section": r["section"],
            }
            for r in rows
        ],
    }, indent=2)


# --- Gating ---------------------------------------------------------------


def health_module_available(config) -> bool:
    """Predicate for tool registration: is the health module on?

    Mirrors how blink/instacart gate registration. If the user hasn't
    enabled HEALTH_MODULE_ENABLED we don't advertise the tools — the
    bot just sees them as unavailable rather than discovering at
    decryption time that the key source isn't set up.
    """
    h = getattr(config, "health", None)
    return bool(h and getattr(h, "enabled", False))
