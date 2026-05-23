"""Tests for the `microwaveos profile` CLI surface (Phase G.1.c).

Each command runs end-to-end against an in-memory DB and env-source
crypto. We patch `load_config` and `_connect` so the CLI uses our
test connection rather than reading the user's actual ~/.microwaveos.

Coverage:
- profile show (empty) — explains nothing yet, exit 0
- profile show (populated) — counts per section + summary fields
- profile show <section> — detail view per section type
- profile show <bad section> — argparse rejects (choices=)
- profile audit (empty) — "no changes" message, exit 0
- profile audit (with entries) — newest-first, op + section visible
- profile clear (empty) — "nothing to clear", exit 0
- profile clear (with data + --yes-really) — deletes profile + log
- profile clear (with data + correct typed input) — succeeds
- profile clear (with data + wrong input) — aborts, profile intact
- profile export (empty) — message, no file written
- profile export (populated) — JSON file written, decrypted content
- profile export (bad parent) — error, exit 1
"""

from __future__ import annotations

import base64
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import apsw
import pytest

from src.health.profile import (
    Allergy,
    Concern,
    Condition,
    Demographics,
    HealthProfile,
    Medication,
    ProfileField,
    init_tables,
    save_profile,
)
from src.health.profile.cli import profile_cli
from src.health.profile.crypto import ENV_MASTER_KEY


# --- Fixtures ---


@pytest.fixture(autouse=True)
def env_master_key(monkeypatch):
    monkeypatch.setenv(
        ENV_MASTER_KEY,
        base64.b64encode(b"test-key-exactly-32-bytes-long!!").decode(),
    )


@pytest.fixture
def conn() -> apsw.Connection:
    c = apsw.Connection(":memory:")
    c.row_trace = lambda cursor, row: {
        d[0]: v for d, v in zip(cursor.getdescription(), row)
    }
    init_tables(c)
    return c


@pytest.fixture
def patched_cli(monkeypatch, conn):
    """Make the CLI use our in-memory conn + env-source crypto."""
    # Build a fake config + a fake _connect that returns the in-memory
    # session-engine-like object the CLI expects.
    class _FakeEngine:
        def __init__(self, c): self.conn = c

    class _FakeHealthConfig:
        phi_encryption_key_source = "env"

    class _FakeConfig:
        health = _FakeHealthConfig()
        # db_path / etc. unused because we patch _connect

    monkeypatch.setattr(
        "src.health.profile.cli.load_config", lambda: _FakeConfig()
    )
    monkeypatch.setattr(
        "src.health.profile.cli._connect",
        lambda config: _FakeEngine(conn),
    )
    return conn


def _provenance(value="x"):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return ProfileField(
        value=value, added_at=now, last_modified=now,
        confirmed=True, source="user_stated_setup", confidence=1.0,
    )


def _seed_profile(conn) -> int:
    """Insert a small populated profile so multiple tests share setup."""
    p = HealthProfile.empty("self")
    p.demographics = Demographics(age_range=_provenance("30-39"))
    p.conditions.append(Condition(
        name="Type 2 Diabetes", status="active",
        diagnosed_when="approximate:2022", field_meta=_provenance(),
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
        text="dull headache for three days",
        raised_at=datetime.now(timezone.utc).replace(tzinfo=None),
        status="active", field_meta=_provenance(),
    ))
    return save_profile(conn, p, expected_version=0, key_source="env")


# --- profile show ---


class TestShow:
    def test_show_empty(self, patched_cli, capsys):
        rc = profile_cli(["show"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "No profile data" in out

    def test_show_populated_summary(self, patched_cli, capsys):
        _seed_profile(patched_cli)
        rc = profile_cli(["show"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Profile version" in out
        assert "conditions:    1" in out
        assert "medications:   1" in out
        assert "allergies:     1" in out
        assert "demographics:" in out
        assert "1/6 fields set" in out  # one demo field

    def test_show_section_demographics(self, patched_cli, capsys):
        _seed_profile(patched_cli)
        rc = profile_cli(["show", "demographics"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "age range" in out
        assert "30-39" in out
        assert "user stated setup" in out  # provenance shown

    def test_show_section_conditions(self, patched_cli, capsys):
        _seed_profile(patched_cli)
        rc = profile_cli(["show", "conditions"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Type 2 Diabetes" in out
        assert "status=active" in out

    def test_show_section_medications(self, patched_cli, capsys):
        _seed_profile(patched_cli)
        rc = profile_cli(["show", "medications"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "metformin" in out
        assert "500mg" in out

    def test_show_empty_section(self, patched_cli, capsys):
        # Seed with only demographics; conditions empty.
        p = HealthProfile.empty("self")
        p.demographics = Demographics(age_range=_provenance("30-39"))
        save_profile(patched_cli, p, expected_version=0, key_source="env")

        rc = profile_cli(["show", "conditions"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "No entries in conditions" in out

    def test_bad_section_rejected_by_argparse(self, patched_cli, capsys):
        with pytest.raises(SystemExit):
            profile_cli(["show", "not-a-section"])


# --- profile audit ---


class TestAudit:
    def test_audit_empty(self, patched_cli, capsys):
        rc = profile_cli(["audit"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "No profile changes" in out

    def test_audit_after_saves(self, patched_cli, capsys):
        _seed_profile(patched_cli)
        rc = profile_cli(["audit"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "section=profile" in out  # default section= used by _seed
        assert "modify" in out  # default operation


# --- profile clear ---


class TestClear:
    def test_clear_empty_no_op(self, patched_cli, capsys):
        rc = profile_cli(["clear", "--yes-really"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "No profile data to clear" in out

    def test_clear_with_yes_really_deletes(self, patched_cli, capsys):
        _seed_profile(patched_cli)
        rc = profile_cli(["clear", "--yes-really"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Profile cleared" in out

        # Confirm the row + change log are gone
        rows = list(patched_cli.execute("SELECT COUNT(*) AS n FROM health_profiles"))
        assert rows[0]["n"] == 0
        rows = list(patched_cli.execute(
            "SELECT COUNT(*) AS n FROM profile_change_log"
        ))
        assert rows[0]["n"] == 0

    def test_clear_with_correct_typed_input(self, patched_cli, capsys, monkeypatch):
        _seed_profile(patched_cli)
        monkeypatch.setattr(
            "builtins.input", lambda *a, **kw: "clear my profile"
        )
        rc = profile_cli(["clear"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Profile cleared" in out

    def test_clear_with_wrong_input_aborts(self, patched_cli, capsys, monkeypatch):
        _seed_profile(patched_cli)
        monkeypatch.setattr(
            "builtins.input", lambda *a, **kw: "yes"
        )
        rc = profile_cli(["clear"])
        out = capsys.readouterr().out
        assert rc == 1
        assert "Aborted" in out
        # Profile must still exist
        rows = list(patched_cli.execute(
            "SELECT COUNT(*) AS n FROM health_profiles"
        ))
        assert rows[0]["n"] == 1

    def test_clear_eof_aborts(self, patched_cli, capsys, monkeypatch):
        _seed_profile(patched_cli)

        def raise_eof(*a, **kw):
            raise EOFError()
        monkeypatch.setattr("builtins.input", raise_eof)

        rc = profile_cli(["clear"])
        assert rc == 1


# --- profile export ---


class TestExport:
    def test_export_empty(self, patched_cli, capsys, tmp_path):
        out_path = tmp_path / "profile.json"
        rc = profile_cli(["export", str(out_path)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "No profile data" in out
        assert not out_path.exists()

    def test_export_populated_writes_json(self, patched_cli, capsys, tmp_path):
        _seed_profile(patched_cli)
        out_path = tmp_path / "profile.json"
        rc = profile_cli(["export", str(out_path)])
        out = capsys.readouterr().out
        assert rc == 0
        assert out_path.exists()

        data = json.loads(out_path.read_text())
        assert data["user_id"] == "self"
        assert len(data["conditions"]) == 1
        assert data["conditions"][0]["name"] == "Type 2 Diabetes"
        # And the user-facing warning was printed
        assert "WARNING" in out
        assert "PHI" in out

    def test_export_bad_parent_dir(self, patched_cli, capsys, tmp_path):
        _seed_profile(patched_cli)
        out_path = tmp_path / "does" / "not" / "exist" / "profile.json"
        rc = profile_cli(["export", str(out_path)])
        out = capsys.readouterr().out
        assert rc == 1
        assert "Parent directory" in out

    def test_export_atomic_no_tmp_left_behind(
        self, patched_cli, tmp_path,
    ):
        _seed_profile(patched_cli)
        out_path = tmp_path / "profile.json"
        profile_cli(["export", str(out_path)])
        assert out_path.exists()
        assert not (tmp_path / "profile.json.tmp").exists()
