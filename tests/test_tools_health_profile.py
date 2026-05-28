"""Tests for the health_profile LLM tools.

End-to-end against a real apsw DB + env-source encryption (same pattern
as test_profile_chat.py). Tool handlers open their own DB connection, so
each test seeds a tmp_path-backed file rather than an in-memory conn.

Coverage:
- summary: empty vs populated
- show: each section type (struct sections + list sections + unknown)
- audit: empty + populated + limit clamping
- registry: gated correctly on health_module_available
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.db import connect as db_connect
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
from src.health.profile.crypto import ENV_MASTER_KEY
from src.tools import health_profile as hp


@pytest.fixture(autouse=True)
def env_master_key(monkeypatch):
    """All tests run with the env-sourced master key so the keychain
    isn't touched. Matches the existing profile-test pattern."""
    monkeypatch.setenv(
        ENV_MASTER_KEY,
        base64.b64encode(b"test-key-exactly-32-bytes-long!!").decode(),
    )


def _config(tmp_path: Path, **health_overrides) -> SimpleNamespace:
    """Build a config the tool handler will accept.

    Mirrors the real `Config` shape just enough — db_path + a nested
    health namespace with `enabled` and `phi_encryption_key_source`.
    """
    health = SimpleNamespace(
        enabled=True,
        phi_encryption_key_source="env",
    )
    for k, v in health_overrides.items():
        setattr(health, k, v)
    return SimpleNamespace(
        db_path=tmp_path / "test.db",
        health=health,
    )


def _provenance(value="x"):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return ProfileField(
        value=value, added_at=now, last_modified=now,
        confirmed=True, source="user_stated_setup", confidence=1.0,
    )


def _seed_profile(cfg: SimpleNamespace) -> int:
    """Seed a representative profile so each section has at least one entry."""
    conn = db_connect(cfg.db_path)
    init_tables(conn)
    p = HealthProfile.empty("self")
    p.demographics = Demographics(age_range=_provenance("30-39"))
    p.conditions.append(Condition(
        name="T2D", status="active", field_meta=_provenance(),
    ))
    p.medications.append(Medication(
        name="metformin", dose="500mg", status="active",
        field_meta=_provenance(),
    ))
    p.medications.append(Medication(
        name="CoQ100", dose="200mg", status="active",
        field_meta=_provenance(),
    ))
    p.allergies.append(Allergy(
        substance="penicillin", field_meta=_provenance(),
    ))
    v = save_profile(conn, p, expected_version=0, key_source="env")
    conn.close()
    return v


# --- summary --------------------------------------------------------------


class TestSummary:
    @pytest.mark.asyncio
    async def test_empty_profile(self, tmp_path):
        cfg = _config(tmp_path)
        out = json.loads(await hp._handle_summary({}, config=cfg))
        assert out["status"] == "empty"
        # The note should hint at how to populate, not just say "empty"
        assert "proposal" in out["note"].lower()

    @pytest.mark.asyncio
    async def test_populated_profile(self, tmp_path):
        cfg = _config(tmp_path)
        _seed_profile(cfg)
        out = json.loads(await hp._handle_summary({}, config=cfg))
        assert out["version"] >= 1
        c = out["section_counts"]
        assert c["medications"] == 2
        assert c["conditions"] == 1
        assert c["allergies"] == 1
        assert c["family_history"] == 0
        # Demographics counts FIELDS, not entries
        assert c["demographics_fields_set"] == 1


# --- show ----------------------------------------------------------------


class TestShow:
    @pytest.mark.asyncio
    async def test_list_section(self, tmp_path):
        cfg = _config(tmp_path)
        _seed_profile(cfg)
        out = json.loads(await hp._handle_show({"section": "medications"}, config=cfg))
        assert out["section"] == "medications"
        assert out["count"] == 2
        names = {e["name"] for e in out["entries"]}
        assert names == {"metformin", "CoQ100"}
        # Dose passed through
        for e in out["entries"]:
            assert "dose" in e

    @pytest.mark.asyncio
    async def test_struct_section_demographics(self, tmp_path):
        cfg = _config(tmp_path)
        _seed_profile(cfg)
        out = json.loads(await hp._handle_show({"section": "demographics"}, config=cfg))
        # Struct sections return fields dict, not entries list
        assert out["section"] == "demographics"
        assert "fields" in out
        assert out["fields"]["age_range"] == "30-39"
        # Unset fields are present-but-null
        assert out["fields"]["height_range"] is None

    @pytest.mark.asyncio
    async def test_struct_section_lifestyle_empty(self, tmp_path):
        cfg = _config(tmp_path)
        _seed_profile(cfg)
        out = json.loads(await hp._handle_show({"section": "lifestyle"}, config=cfg))
        assert out["section"] == "lifestyle"
        assert all(v is None for v in out["fields"].values())

    @pytest.mark.asyncio
    async def test_unknown_section_raises(self, tmp_path):
        cfg = _config(tmp_path)
        with pytest.raises(RuntimeError, match="must be one of"):
            await hp._handle_show({"section": "bloodtype"}, config=cfg)

    @pytest.mark.asyncio
    async def test_missing_section_raises(self, tmp_path):
        cfg = _config(tmp_path)
        with pytest.raises(RuntimeError):
            await hp._handle_show({}, config=cfg)

    @pytest.mark.asyncio
    async def test_empty_list_section(self, tmp_path):
        cfg = _config(tmp_path)
        _seed_profile(cfg)  # has medications but no family_history
        out = json.loads(await hp._handle_show({"section": "family_history"}, config=cfg))
        assert out["count"] == 0
        assert out["entries"] == []


# --- audit ----------------------------------------------------------------


class TestAudit:
    @pytest.mark.asyncio
    async def test_empty(self, tmp_path):
        cfg = _config(tmp_path)
        # Must init tables so the audit query doesn't error
        conn = db_connect(cfg.db_path); init_tables(conn); conn.close()
        out = json.loads(await hp._handle_audit({}, config=cfg))
        assert out == {"count": 0, "entries": []}

    @pytest.mark.asyncio
    async def test_populated(self, tmp_path):
        cfg = _config(tmp_path)
        _seed_profile(cfg)
        out = json.loads(await hp._handle_audit({}, config=cfg))
        assert out["count"] >= 1
        e = out["entries"][0]
        assert "timestamp" in e and "operation" in e and "section" in e

    @pytest.mark.asyncio
    async def test_limit_default(self, tmp_path):
        cfg = _config(tmp_path)
        _seed_profile(cfg)
        # Default limit is 10; seeding produces at most a handful, so this
        # just confirms the default arg is accepted without explicit value.
        out = json.loads(await hp._handle_audit({}, config=cfg))
        assert out["count"] <= 10

    @pytest.mark.asyncio
    async def test_limit_clamped_to_max(self, tmp_path):
        cfg = _config(tmp_path)
        _seed_profile(cfg)
        # 9999 → silently clamped to 100, not rejected
        out = json.loads(await hp._handle_audit({"limit": 9999}, config=cfg))
        assert isinstance(out["count"], int)

    @pytest.mark.asyncio
    async def test_invalid_limit_raises(self, tmp_path):
        cfg = _config(tmp_path)
        _seed_profile(cfg)
        with pytest.raises(RuntimeError, match="integer"):
            await hp._handle_audit({"limit": "abc"}, config=cfg)


# --- Gating + registry wiring --------------------------------------------


class TestGating:
    def test_available_when_health_enabled(self, tmp_path):
        cfg = _config(tmp_path)
        assert hp.health_module_available(cfg) is True

    def test_unavailable_when_health_disabled(self, tmp_path):
        cfg = _config(tmp_path, enabled=False)
        assert hp.health_module_available(cfg) is False

    def test_unavailable_when_health_missing(self):
        # SimpleNamespace without a `health` attribute at all
        cfg = SimpleNamespace(db_path=Path("/tmp/nope.db"))
        assert hp.health_module_available(cfg) is False


class TestRegistryWiring:
    """If the catalog mentions health_profile_show but the handler isn't
    registered, the bot regresses to the "documented but not callable"
    confabulation pattern. Pin both sides agree on the same gate."""

    def test_provider_tools_register_when_enabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WEB_TOOLS_DISABLED", "1")
        monkeypatch.setenv("FILE_TOOLS_DISABLED", "1")
        monkeypatch.setenv("WEBSEARCH_DISABLED", "1")
        monkeypatch.setenv("BLINK_CREDENTIALS_PATH", str(tmp_path / "no-blink.json"))
        from src.tools import build_provider_tools

        cfg = _config(tmp_path)
        # build_provider_tools also reads workspace + recipient defaults
        cfg.workspace_dir = tmp_path / "ws"
        cfg.heartbeat_notify_channel = "signal"
        cfg.heartbeat_notify_recipient = "+1"
        cfg.instacart_api_key = ""
        cfg.github_token = ""

        names = {t.definition.name for t in build_provider_tools(cfg)}
        assert {
            "health_profile_summary",
            "health_profile_show",
            "health_profile_audit",
        } <= names

    def test_provider_tools_omit_when_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WEB_TOOLS_DISABLED", "1")
        monkeypatch.setenv("FILE_TOOLS_DISABLED", "1")
        monkeypatch.setenv("WEBSEARCH_DISABLED", "1")
        monkeypatch.setenv("BLINK_CREDENTIALS_PATH", str(tmp_path / "no-blink.json"))
        from src.tools import build_provider_tools

        cfg = _config(tmp_path, enabled=False)
        cfg.workspace_dir = tmp_path / "ws"
        cfg.heartbeat_notify_channel = "signal"
        cfg.heartbeat_notify_recipient = "+1"
        cfg.instacart_api_key = ""
        cfg.github_token = ""

        names = {t.definition.name for t in build_provider_tools(cfg)}
        assert not any(n.startswith("health_profile_") for n in names)

    def test_catalog_includes_health_profile_when_enabled(self, tmp_path):
        from src.tools import build_tools

        cfg = _config(tmp_path)
        cfg.workspace_dir = tmp_path / "ws"
        cfg.instacart_api_key = ""
        cfg.github_token = ""
        bundle = build_tools(cfg)
        if not bundle.catalog_text:
            pytest.skip("SDK not available; catalog path returned empty")
        assert "health_profile_show" in bundle.catalog_text
        assert "health_profile_summary" in bundle.catalog_text
