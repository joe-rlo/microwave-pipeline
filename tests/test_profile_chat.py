"""Tests for the /profile chat handler (Phase G.1.d).

Each test builds an in-memory orchestrator stub with a real apsw conn
+ env-source crypto. We exercise handle_profile_command() end-to-end
against actual stored profiles.

Coverage:
- Pass-through: non-/profile messages return None
- /profile (no args): empty + populated summary
- /profile show <section>: each section + unknown section error
- /profile audit: empty + populated + N parsing
- /profile clear: needs phrase, succeeds with phrase, no-op when empty
- /profile export: writes file, returns path; no-op when empty
- /profile unknown: helpful error
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import apsw
import pytest

from src.health.profile import (
    Allergy,
    Condition,
    Demographics,
    HealthProfile,
    Medication,
    ProfileField,
    init_tables,
    save_profile,
)
from src.health.profile.chat import handle_profile_command
from src.health.profile.crypto import ENV_MASTER_KEY


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
def orch(conn, tmp_path):
    """Build a minimal orchestrator stub that the chat handler needs."""
    class _SE:
        def __init__(self, c): self.conn = c

    class _Health:
        phi_encryption_key_source = "env"

    class _Config:
        health = _Health()
        output_dir = tmp_path / "output"

    return SimpleNamespace(
        config=_Config(),
        session_engine=_SE(conn),
    )


def _provenance(value="x"):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return ProfileField(
        value=value, added_at=now, last_modified=now,
        confirmed=True, source="user_stated_setup", confidence=1.0,
    )


def _seed_basic(conn) -> int:
    p = HealthProfile.empty("self")
    p.demographics = Demographics(age_range=_provenance("30-39"))
    p.conditions.append(Condition(
        name="T2D", status="active", field_meta=_provenance(),
    ))
    p.medications.append(Medication(
        name="metformin", dose="500mg", status="active",
        field_meta=_provenance(),
    ))
    p.allergies.append(Allergy(
        substance="penicillin", field_meta=_provenance(),
    ))
    return save_profile(conn, p, expected_version=0, key_source="env")


# --- Pass-through ---


@pytest.mark.asyncio
class TestPassThrough:
    async def test_non_profile_returns_none(self, orch):
        assert await handle_profile_command("hello", orch) is None

    async def test_profileless_prefix_returns_none(self, orch):
        assert await handle_profile_command("profile", orch) is None

    async def test_profile_in_middle_returns_none(self, orch):
        # "/profile" must be at start of message
        assert await handle_profile_command(
            "look at /profile show", orch
        ) is None


# --- /profile (summary) ---


@pytest.mark.asyncio
class TestSummary:
    async def test_empty_profile_summary(self, orch):
        reply = await handle_profile_command("/profile", orch)
        assert reply is not None
        assert "No profile data" in reply

    async def test_populated_summary(self, orch):
        _seed_basic(orch.session_engine.conn)
        reply = await handle_profile_command("/profile", orch)
        assert reply is not None
        assert "conditions:" in reply
        assert "medications:" in reply
        assert "1/6 fields" in reply  # one demographic field set

    async def test_summary_with_whitespace(self, orch):
        # /profile with trailing space
        reply = await handle_profile_command("/profile  ", orch)
        assert reply is not None


# --- /profile show <section> ---


@pytest.mark.asyncio
class TestShowSection:
    async def test_show_demographics(self, orch):
        _seed_basic(orch.session_engine.conn)
        reply = await handle_profile_command("/profile show demographics", orch)
        assert reply is not None
        assert "Demographics" in reply
        assert "30-39" in reply

    async def test_show_medications(self, orch):
        _seed_basic(orch.session_engine.conn)
        reply = await handle_profile_command("/profile show medications", orch)
        assert "metformin" in reply
        assert "500mg" in reply

    async def test_show_empty_section(self, orch):
        _seed_basic(orch.session_engine.conn)
        reply = await handle_profile_command("/profile show concerns", orch)
        assert "(none)" in reply.lower()

    async def test_show_unknown_section(self, orch):
        reply = await handle_profile_command("/profile show garbage", orch)
        assert "Unknown section" in reply

    async def test_show_no_args_falls_back_to_summary(self, orch):
        # "/profile show" alone — same as "/profile"
        reply = await handle_profile_command("/profile show", orch)
        # Empty profile → summary message
        assert "No profile data" in reply


# --- /profile audit ---


@pytest.mark.asyncio
class TestAudit:
    async def test_audit_empty(self, orch):
        reply = await handle_profile_command("/profile audit", orch)
        assert "No profile changes" in reply

    async def test_audit_with_entries(self, orch):
        _seed_basic(orch.session_engine.conn)
        reply = await handle_profile_command("/profile audit", orch)
        assert "Profile audit" in reply
        assert "modify" in reply  # default operation in save_profile

    async def test_audit_with_int_arg(self, orch):
        _seed_basic(orch.session_engine.conn)
        reply = await handle_profile_command("/profile audit 5", orch)
        assert reply is not None

    async def test_audit_bad_int_returns_usage(self, orch):
        reply = await handle_profile_command("/profile audit abc", orch)
        assert "Audit usage" in reply


# --- /profile clear ---


@pytest.mark.asyncio
class TestClear:
    async def test_clear_without_phrase_shows_help(self, orch):
        _seed_basic(orch.session_engine.conn)
        reply = await handle_profile_command("/profile clear", orch)
        assert "clear my profile" in reply
        # Confirm nothing was actually cleared
        rows = list(orch.session_engine.conn.execute(
            "SELECT COUNT(*) AS n FROM health_profiles"
        ))
        assert rows[0]["n"] == 1

    async def test_clear_with_wrong_phrase_shows_help(self, orch):
        _seed_basic(orch.session_engine.conn)
        reply = await handle_profile_command(
            '/profile clear "wrong phrase"', orch,
        )
        assert "clear my profile" in reply
        rows = list(orch.session_engine.conn.execute(
            "SELECT COUNT(*) AS n FROM health_profiles"
        ))
        assert rows[0]["n"] == 1

    async def test_clear_with_correct_phrase(self, orch):
        _seed_basic(orch.session_engine.conn)
        reply = await handle_profile_command(
            '/profile clear "clear my profile"', orch,
        )
        assert "Profile cleared" in reply
        rows = list(orch.session_engine.conn.execute(
            "SELECT COUNT(*) AS n FROM health_profiles"
        ))
        assert rows[0]["n"] == 0

    async def test_clear_with_single_quotes(self, orch):
        # Be tolerant of single quotes around the phrase
        _seed_basic(orch.session_engine.conn)
        reply = await handle_profile_command(
            "/profile clear 'clear my profile'", orch,
        )
        assert "Profile cleared" in reply

    async def test_clear_when_empty(self, orch):
        reply = await handle_profile_command(
            '/profile clear "clear my profile"', orch,
        )
        assert "No profile data to clear" in reply


# --- /profile export ---


@pytest.mark.asyncio
class TestExport:
    async def test_export_empty(self, orch):
        reply = await handle_profile_command("/profile export", orch)
        assert "No profile data" in reply

    async def test_export_writes_file(self, orch, tmp_path):
        _seed_basic(orch.session_engine.conn)
        reply = await handle_profile_command("/profile export", orch)
        assert "Wrote profile" in reply
        assert "PHI" in reply  # warning surface

        # Find the file — should be in output_dir with today's date
        out_dir = orch.config.output_dir
        files = list(out_dir.glob("profile-*.json"))
        assert len(files) == 1
        import json
        data = json.loads(files[0].read_text())
        assert data["medications"][0]["name"] == "metformin"


# --- /profile unknown ---


@pytest.mark.asyncio
class TestUnknown:
    async def test_unknown_subcommand(self, orch):
        reply = await handle_profile_command("/profile unicorn", orch)
        assert "Unknown profile command" in reply
        # Should mention valid commands
        assert "show" in reply
        assert "audit" in reply


# --- Error tolerance ---


@pytest.mark.asyncio
class TestErrorTolerance:
    async def test_load_failure_returns_friendly_message(self, monkeypatch, orch):
        """If profile load raises, the handler must surface an error
        message — not crash the channel."""
        from src.health.profile import store as store_mod

        def boom(*a, **kw):
            raise RuntimeError("simulated crypto failure")

        monkeypatch.setattr(store_mod, "load_profile", boom)
        reply = await handle_profile_command("/profile", orch)
        assert reply is not None
        assert "failed" in reply.lower()
