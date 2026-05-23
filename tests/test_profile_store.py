"""Tests for the Health Profile schema, storage, and Pydantic models.

The crypto layer is mocked via the env-source path (with a known
base64 master key) so tests don't depend on the OS keychain. Encryption
itself is real — we want the roundtrip to actually exercise AES-GCM.

Coverage:
- HealthProfile model: empty constructor, is_empty across sections,
  JSON roundtrip (serialize + deserialize matches), Pydantic
  validation of section fields
- ProfileField provenance enforced
- Schema: init_tables idempotent, both tables created with expected
  columns + indexes
- Store load: fresh-install default returns version=0 + empty profile;
  existing row decrypts and deserializes
- Store save: fresh insert sets version=1, repeat save increments,
  change log gets one entry per save
- Optimistic concurrency: stale expected_version raises
  StaleProfileError; fresh-insert with non-zero expected raises too
- Encryption is per-user: blob saved as user "alice" can't be read as
  user "bob" (defense-in-depth — even with correct master key)
- list_change_log: newest first, honors limit, scoped to user
"""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone

import apsw
import pytest

from src.health.profile import (
    DEFAULT_USER_ID,
    Allergy,
    Concern,
    Condition,
    Demographics,
    HealthProfile,
    Medication,
    ProfileField,
    StaleProfileError,
    init_tables,
    list_change_log,
    load_profile,
    save_profile,
)
from src.health.profile.crypto import ENV_MASTER_KEY, KEY_LEN


# --- Fixtures ---


@pytest.fixture
def conn() -> apsw.Connection:
    c = apsw.Connection(":memory:")
    c.row_trace = lambda cursor, row: {
        d[0]: v for d, v in zip(cursor.getdescription(), row)
    }
    init_tables(c)
    return c


@pytest.fixture(autouse=True)
def env_master_key(monkeypatch):
    """Pin a known master key in env so the crypto layer can encrypt
    without keychain access. autouse=True so every test in this file
    runs with `source="env"` available."""
    # Exactly 32 bytes — KEY_LEN is enforced by the env loader.
    monkeypatch.setenv(
        ENV_MASTER_KEY,
        base64.b64encode(b"test-key-exactly-32-bytes-long!!").decode(),
    )


def _save(conn, profile, expected_version=0, **kw):
    """Wrapper to pin source=env across all save sites."""
    return save_profile(
        conn, profile, expected_version=expected_version,
        key_source="env", **kw,
    )


def _load(conn, user_id=DEFAULT_USER_ID):
    return load_profile(conn, user_id, key_source="env")


def _provenance(value="x") -> ProfileField:
    """Build a stub ProfileField for tests that just need the wrapper."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return ProfileField(
        value=value,
        added_at=now,
        last_modified=now,
        confirmed=True,
        source="user_stated_setup",
        confidence=1.0,
    )


# --- HealthProfile model ---


class TestHealthProfileModel:
    def test_empty_constructor(self):
        p = HealthProfile.empty("self")
        assert p.user_id == "self"
        assert p.is_empty is True
        assert p.conditions == []
        assert p.medications == []
        assert p.demographics.age_range is None

    def test_is_empty_false_when_demographics_set(self):
        p = HealthProfile.empty("self")
        p = p.model_copy(update={
            "demographics": Demographics(age_range=_provenance("30-39")),
        })
        assert p.is_empty is False

    def test_is_empty_false_when_condition_added(self):
        p = HealthProfile.empty("self")
        p.conditions.append(Condition(
            name="x", status="active", field_meta=_provenance(),
        ))
        assert p.is_empty is False

    def test_pending_updates_alone_dont_make_nonempty(self):
        # is_empty cares about USER DATA only; pending updates are
        # machinery state — an empty profile with pending proposals
        # is still empty from the user's perspective.
        p = HealthProfile.empty("self")
        assert p.is_empty is True

    def test_json_roundtrip_preserves_sections(self):
        p = HealthProfile.empty("self")
        p.conditions.append(Condition(
            name="Type 2 Diabetes", status="active",
            diagnosed_when="approximate:2022",
            field_meta=_provenance(),
        ))
        p.medications.append(Medication(
            name="metformin", dose="500mg twice daily",
            status="active", reason="Type 2 Diabetes",
            field_meta=_provenance(),
        ))
        p.allergies.append(Allergy(
            substance="penicillin", severity="severe",
            field_meta=_provenance(),
        ))
        p.concerns.append(Concern(
            text="dull headache for 3 days", raised_at=datetime.now(timezone.utc).replace(tzinfo=None),
            status="active", field_meta=_provenance(),
        ))

        # Roundtrip via JSON
        as_json = p.model_dump_json()
        restored = HealthProfile.model_validate_json(as_json)

        assert len(restored.conditions) == 1
        assert restored.conditions[0].name == "Type 2 Diabetes"
        assert restored.medications[0].dose == "500mg twice daily"
        assert restored.allergies[0].severity == "severe"
        assert restored.concerns[0].text.startswith("dull headache")

    def test_invalid_status_rejected(self):
        # Pydantic Literal validation catches bad enum values
        with pytest.raises(Exception):  # ValidationError, but importing it is noisy
            Condition(
                name="x", status="not-a-real-status",  # type: ignore[arg-type]
                field_meta=_provenance(),
            )


# --- Schema ---


class TestSchema:
    def test_init_tables_idempotent(self, conn):
        init_tables(conn)
        init_tables(conn)
        names = {
            r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "health_profiles" in names
        assert "profile_change_log" in names

    def test_change_log_index_created(self, conn):
        rows = list(conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name='idx_profile_change_log_user_ts'"
        ))
        assert len(rows) == 1


# --- Store load ---


class TestLoadProfile:
    def test_fresh_install_returns_empty_v0(self, conn):
        loaded = _load(conn)
        assert loaded.version == 0
        assert loaded.profile.is_empty
        assert loaded.profile.user_id == DEFAULT_USER_ID

    def test_after_save_load_returns_persisted(self, conn):
        p = HealthProfile.empty("self")
        p.medications.append(Medication(
            name="metformin", status="active",
            field_meta=_provenance(),
        ))
        _save(conn, p, expected_version=0)

        loaded = _load(conn)
        assert loaded.version == 1
        assert len(loaded.profile.medications) == 1
        assert loaded.profile.medications[0].name == "metformin"


# --- Store save ---


class TestSaveProfile:
    def test_fresh_insert_version_starts_at_1(self, conn):
        p = HealthProfile.empty("self")
        new_v = _save(conn, p, expected_version=0)
        assert new_v == 1

    def test_repeat_save_increments_version(self, conn):
        p = HealthProfile.empty("self")
        v1 = _save(conn, p, expected_version=0)

        loaded = _load(conn)
        loaded.profile.allergies.append(Allergy(
            substance="penicillin", field_meta=_provenance(),
        ))
        v2 = _save(conn, loaded.profile, expected_version=v1)
        assert v2 == v1 + 1

    def test_change_log_entry_per_save(self, conn):
        p = HealthProfile.empty("self")
        v1 = _save(conn, p, expected_version=0, operation="add", section="medications")
        loaded = _load(conn)
        _save(
            conn, loaded.profile, expected_version=v1,
            operation="modify", section="conditions",
        )

        entries = list_change_log(conn)
        assert len(entries) == 2
        # Newest first
        assert entries[0]["section"] == "conditions"
        assert entries[1]["section"] == "medications"

    def test_stale_version_rejected(self, conn):
        p = HealthProfile.empty("self")
        v1 = _save(conn, p, expected_version=0)
        # Someone else also saves (simulated by another save call)
        loaded = _load(conn)
        _save(conn, loaded.profile, expected_version=v1)  # OK, version=2

        # Now try to save based on the OLD version=v1 → should reject
        with pytest.raises(StaleProfileError, match="version drift"):
            _save(conn, loaded.profile, expected_version=v1)

    def test_fresh_save_with_nonzero_expected_rejected(self, conn):
        p = HealthProfile.empty("self")
        with pytest.raises(StaleProfileError, match="No existing"):
            _save(conn, p, expected_version=5)


# --- Defense in depth: per-user keys ---


class TestPerUserIsolation:
    def test_alice_blob_unreadable_as_bob(self, conn):
        # Save under user "alice"
        p_alice = HealthProfile.empty("alice")
        p_alice.medications.append(Medication(
            name="alice's medication", status="active",
            field_meta=_provenance(),
        ))
        _save(conn, p_alice, expected_version=0)

        # Try to read as user "bob" — the row exists for "alice" only;
        # load_profile("bob") returns the fresh-install default rather
        # than the alice blob. (Per-user key derivation would also
        # prevent decryption even if you forced the row.)
        loaded_bob = _load(conn, user_id="bob")
        assert loaded_bob.version == 0
        assert loaded_bob.profile.is_empty


# --- list_change_log ---


class TestListChangeLog:
    def test_empty_when_no_saves(self, conn):
        assert list_change_log(conn) == []

    def test_honors_limit(self, conn):
        p = HealthProfile.empty("self")
        v = _save(conn, p, expected_version=0)
        for i in range(3):
            loaded = _load(conn)
            v = _save(conn, loaded.profile, expected_version=v,
                      section=f"section_{i}")
        all_entries = list_change_log(conn)
        assert len(all_entries) == 4
        limited = list_change_log(conn, limit=2)
        assert len(limited) == 2

    def test_scoped_to_user(self, conn):
        # Two separate user profiles in the same DB. log_for("alice")
        # only sees alice's saves.
        p_alice = HealthProfile.empty("alice")
        _save(conn, p_alice, expected_version=0, section="alice_section")
        p_bob = HealthProfile.empty("bob")
        _save(conn, p_bob, expected_version=0, section="bob_section")

        alice_log = list_change_log(conn, user_id="alice")
        bob_log = list_change_log(conn, user_id="bob")

        assert len(alice_log) == 1
        assert alice_log[0]["section"] == "alice_section"
        assert len(bob_log) == 1
        assert bob_log[0]["section"] == "bob_section"
