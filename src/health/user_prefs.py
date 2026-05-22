"""User-controlled health privacy preferences (Phase E).

A small SQLite-backed surface for "how do you want health turns to be
processed?" — surfaced via `microwaveos health prefs` and consumed by
`src.health.router.route()`.

Modes:

- `standard` (default): general-health turns route to the standard
  main pipeline (NEAR Anonymised Claude in current setups). Anonymised
  proxying strips PII metadata but the upstream provider still sees
  the (de-identified) prompt. Personal/PHI turns still get BAA when
  configured.

- `private_tee`: general-health turns route to NEAR's Private TEE
  open-weight models (GPT OSS 120B for complex, Qwen3.5 122B for
  simple). The prompt content stays inside hardware-attested
  isolation; NEAR cannot read it. Quality vs. Anonymised Claude is
  unknown (Open Question 14.3) — flip it on per-user.

Why a separate table from MEMORY.md or a config flag: prefs are
mutable at runtime and per-user-id, not global. Today there's only
one user (`self`) but the schema scoping forward is cheap.

Why not in HealthConfig: HealthConfig is loaded from env at startup
and frozen for the process lifetime. Prefs need to flip from the
CLI without a restart.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Literal

import apsw

log = logging.getLogger(__name__)


PrivacyMode = Literal["standard", "private_tee"]
PRIVACY_MODES = ("standard", "private_tee")


# Single-user MVP — every callsite uses "self". The column exists so
# the Health Profile spec's multi-user shape lands without a migration.
DEFAULT_USER_ID = "self"


@dataclass(frozen=True)
class UserHealthPref:
    user_id: str
    privacy_mode: PrivacyMode
    consent_anonymised_general: bool
    last_updated: int

    @property
    def is_default(self) -> bool:
        """True for prefs that haven't been explicitly set."""
        return (
            self.privacy_mode == "standard"
            and self.consent_anonymised_general is False
        )


def init_tables(conn: apsw.Connection) -> None:
    """Create the user_health_prefs table. Idempotent."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_health_prefs (
            user_id TEXT PRIMARY KEY,
            privacy_mode TEXT NOT NULL DEFAULT 'standard',
            consent_anonymised_general INTEGER NOT NULL DEFAULT 0,
            last_updated INTEGER NOT NULL
        )
    """)


def load_pref(
    conn: apsw.Connection, user_id: str = DEFAULT_USER_ID,
) -> UserHealthPref:
    """Load one user's prefs. Returns a default-shaped pref when the row
    doesn't exist — no need to pre-seed the table."""
    try:
        rows = list(conn.execute(
            "SELECT user_id, privacy_mode, consent_anonymised_general, "
            "last_updated FROM user_health_prefs WHERE user_id = ?",
            (user_id,),
        ))
    except Exception as e:
        # Schema not present (table not init'd yet) — common in narrow
        # tests. Return default.
        log.debug("Could not read user_health_prefs: %s", e)
        return _default_pref(user_id)

    if not rows:
        return _default_pref(user_id)

    r = rows[0]
    mode = r["privacy_mode"]
    if mode not in PRIVACY_MODES:
        log.warning(
            "Unknown privacy_mode %r in user_health_prefs; falling back to standard",
            mode,
        )
        mode = "standard"
    return UserHealthPref(
        user_id=r["user_id"],
        privacy_mode=mode,  # type: ignore[arg-type]
        consent_anonymised_general=bool(r["consent_anonymised_general"]),
        last_updated=int(r["last_updated"]),
    )


def save_pref(
    conn: apsw.Connection,
    *,
    user_id: str = DEFAULT_USER_ID,
    privacy_mode: PrivacyMode | None = None,
    consent_anonymised_general: bool | None = None,
    now: int | None = None,
) -> UserHealthPref:
    """Upsert one user's prefs. Only the fields supplied are updated;
    unspecified fields retain their current value."""
    if privacy_mode is not None and privacy_mode not in PRIVACY_MODES:
        raise ValueError(
            f"Unknown privacy_mode {privacy_mode!r}; expected one of {PRIVACY_MODES}"
        )

    ts = now if now is not None else int(time.time())
    current = load_pref(conn, user_id)

    new_mode = privacy_mode if privacy_mode is not None else current.privacy_mode
    new_consent = (
        consent_anonymised_general
        if consent_anonymised_general is not None
        else current.consent_anonymised_general
    )

    conn.execute(
        """
        INSERT INTO user_health_prefs
            (user_id, privacy_mode, consent_anonymised_general, last_updated)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            privacy_mode = excluded.privacy_mode,
            consent_anonymised_general = excluded.consent_anonymised_general,
            last_updated = excluded.last_updated
        """,
        (user_id, new_mode, 1 if new_consent else 0, ts),
    )

    return UserHealthPref(
        user_id=user_id,
        privacy_mode=new_mode,
        consent_anonymised_general=new_consent,
        last_updated=ts,
    )


def _default_pref(user_id: str) -> UserHealthPref:
    return UserHealthPref(
        user_id=user_id,
        privacy_mode="standard",
        consent_anonymised_general=False,
        last_updated=0,
    )
