"""Encrypted storage for the Health Profile (Phase G.1.b).

One row per user in `health_profiles`. The full profile is serialized
to JSON via Pydantic, encrypted with the per-user derived key
(see `src.health.profile.crypto`), and stored as a BLOB. Every save
appends an encrypted diff entry to `profile_change_log` so the user
can later run `profile audit` and see "this section changed at this
time, by this operation".

Optimistic concurrency:

The Extract → propose loop (Phase G.2) runs in the background. The
user can also be editing the profile via the CLI at the same time. We
use `profile_version` (monotonic int) on the row + on the in-memory
`HealthProfile` to detect lost-update races: if the version on disk
moved between load and save, the save is rejected with
`StaleProfileError` and the caller must reload + reapply.

What this module deliberately doesn't do:

- No CLI surface. That's Phase G.1.c.
- No automatic extraction. That's Phase G.2.
- No slice selector. That's Phase G.3 and integrates with the
  assembly stage.
- No multi-user identity story. Single-user MVP with `user_id="self"`
  baked into the default. The schema's `user_id PRIMARY KEY` is
  forward compat.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


def _utc_now() -> datetime:
    """Naive UTC `now()` — datetime.utcnow() is deprecated in 3.12+.

    Returns a naive datetime (no tzinfo) for Pydantic v2 compatibility
    with the existing model field types (which don't specify timezone).
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)

import apsw

from src.health.profile.crypto import (
    KeySource,
    decrypt_for_user,
    encrypt_for_user,
)
from src.health.profile.models import HealthProfile

log = logging.getLogger(__name__)


DEFAULT_USER_ID = "self"


class StaleProfileError(RuntimeError):
    """Raised when a save would overwrite a newer version of the profile.

    Optimistic concurrency control — the in-memory profile carries the
    version it was loaded at; if the on-disk version has moved past
    that, the save is rejected. Callers should reload, reapply their
    change, and retry.
    """


@dataclass(frozen=True)
class LoadedProfile:
    """What `load_profile` returns. Carries the version so a subsequent
    save can verify nothing changed in between."""

    profile: HealthProfile
    version: int        # monotonic; 0 for fresh-install default
    last_accessed: int  # epoch seconds; 0 for never-saved-yet


# --- Schema -------------------------------------------------------------


def init_tables(conn: apsw.Connection) -> None:
    """Create the profile tables. Idempotent."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS health_profiles (
            user_id TEXT PRIMARY KEY,
            user_key_id TEXT NOT NULL,
            encrypted_profile BLOB NOT NULL,
            profile_version INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            last_updated INTEGER NOT NULL,
            last_accessed INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS profile_change_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            user_key_id TEXT NOT NULL,
            encrypted_change BLOB NOT NULL,
            operation TEXT NOT NULL,
            section TEXT NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_profile_change_log_user_ts "
        "ON profile_change_log(user_key_id, timestamp DESC)"
    )


# --- Load / save ---------------------------------------------------------


def load_profile(
    conn: apsw.Connection,
    user_id: str = DEFAULT_USER_ID,
    *,
    key_source: KeySource = "keychain",
) -> LoadedProfile:
    """Load the user's profile (decrypted) + the version it was at.

    Returns a fresh empty profile with `version=0` when no row exists —
    callers don't need to special-case first-run.
    """
    rows = list(conn.execute(
        "SELECT encrypted_profile, profile_version, last_accessed "
        "FROM health_profiles WHERE user_id = ?",
        (user_id,),
    ))
    if not rows:
        return LoadedProfile(
            profile=HealthProfile.empty(user_id),
            version=0,
            last_accessed=0,
        )

    row = rows[0]
    blob = bytes(row["encrypted_profile"])
    try:
        plaintext = decrypt_for_user(blob, user_id, source=key_source)
    except Exception as e:
        # Don't include the blob or key in the message — never leak
        # crypto material on error.
        log.error("Profile decryption failed for user=%s: %s", user_id, e)
        raise

    profile = HealthProfile.model_validate_json(plaintext.decode("utf-8"))
    return LoadedProfile(
        profile=profile,
        version=int(row["profile_version"]),
        last_accessed=int(row["last_accessed"] or 0),
    )


def save_profile(
    conn: apsw.Connection,
    profile: HealthProfile,
    expected_version: int,
    *,
    operation: str = "modify",
    section: str = "profile",
    key_source: KeySource = "keychain",
    now: Optional[int] = None,
) -> int:
    """Persist `profile`. Returns the new version number.

    `expected_version` is the version the caller loaded. If the on-disk
    version has moved past that (someone else saved in between), this
    raises StaleProfileError and the caller should reload + retry.

    `operation` + `section` are stored verbatim in the change log so the
    audit CLI can render readable history.
    """
    import time
    ts = now if now is not None else int(time.time())

    user_id = profile.user_id
    user_key_id = _user_key_id(user_id)

    # Update last_updated on the in-memory object so the persisted form
    # is consistent with the row's last_updated column.
    profile = profile.model_copy(update={"last_updated": _utc_now()})
    plaintext = profile.model_dump_json().encode("utf-8")
    blob = encrypt_for_user(plaintext, user_id, source=key_source)

    # Check current state for optimistic-concurrency comparison.
    rows = list(conn.execute(
        "SELECT profile_version FROM health_profiles WHERE user_id = ?",
        (user_id,),
    ))
    if rows:
        current = int(rows[0]["profile_version"])
        if current != expected_version:
            raise StaleProfileError(
                f"Profile version drift: expected {expected_version}, found "
                f"{current}. Reload and retry."
            )
        new_version = current + 1
        conn.execute(
            """
            UPDATE health_profiles
               SET encrypted_profile = ?,
                   profile_version = ?,
                   last_updated = ?,
                   last_accessed = ?
             WHERE user_id = ?
            """,
            (blob, new_version, ts, ts, user_id),
        )
    else:
        # Fresh insert. expected_version must be 0 to make sense.
        if expected_version != 0:
            raise StaleProfileError(
                f"No existing profile but expected_version={expected_version}; "
                "expected 0 for a fresh save."
            )
        new_version = 1
        conn.execute(
            """
            INSERT INTO health_profiles
                (user_id, user_key_id, encrypted_profile, profile_version,
                 created_at, last_updated, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, user_key_id, blob, new_version, ts, ts, ts),
        )

    # Append change-log entry. Encrypt the (operation, section, version)
    # tuple so a leaked log doesn't reveal what changed; we deliberately
    # don't store the diff content yet (Phase G.1.c can extend with a
    # diff once the CLI surface drives real edits).
    change_payload = json.dumps({
        "version_from": expected_version,
        "version_to": new_version,
        "ts": ts,
    }).encode("utf-8")
    encrypted_change = encrypt_for_user(change_payload, user_id, source=key_source)
    conn.execute(
        """
        INSERT INTO profile_change_log
            (timestamp, user_key_id, encrypted_change, operation, section)
        VALUES (?, ?, ?, ?, ?)
        """,
        (ts, user_key_id, encrypted_change, operation, section),
    )

    return new_version


def list_change_log(
    conn: apsw.Connection,
    user_id: str = DEFAULT_USER_ID,
    *,
    limit: int = 50,
) -> list[dict]:
    """Read the change log for a user, newest first.

    Returns metadata only — the encrypted_change blob is not decrypted
    here (callers that want it can use `decrypt_change_entry`). For
    the typical `profile audit` use case, op + section + timestamp is
    what the user wants to see.
    """
    user_key_id = _user_key_id(user_id)
    rows = list(conn.execute(
        """
        SELECT id, timestamp, operation, section
        FROM profile_change_log
        WHERE user_key_id = ?
        ORDER BY timestamp DESC, id DESC
        LIMIT ?
        """,
        (user_key_id, limit),
    ))
    return [dict(r) for r in rows]


# --- Helpers ------------------------------------------------------------


def _user_key_id(user_id: str) -> str:
    """Stable, non-secret identifier for the per-user key.

    Stored in plain text in the `user_key_id` column so a future
    re-derivation knows which derivation scheme to use. NOT the key
    itself — just an identifier of which key to look up. Today this
    is the user_id verbatim; future schemes might prefix with a
    version (`v1:self`, `v2:self`).
    """
    return f"v1:{user_id}"
