"""Tests for Phase E — private TEE general-health route + user prefs.

Coverage:
- UserHealthPref: init_tables idempotent, load returns default when row
  missing, save upserts, unknown privacy_mode rejected at boundary,
  load tolerates missing table (schema-not-init yet) without raising
- Router: when user_pref.privacy_mode=='private_tee' AND phi_class==
  'general', returns general_private_tee path; standard / None pref
  preserves existing 'general' path behavior; private_tee doesn't
  affect non-general phi_classes
- HealthRoute.use_private_tee field defaults to False; mutually
  exclusive with use_baa_llm in all router outputs
- Factory: build_private_tee_llm returns None when NEAR_API_KEY missing;
  picks GPT OSS 120B for complex, Qwen3.5 for simple/moderate; tools
  list is empty (no tool wiring on the PHI/TEE path)
- CLI: `health prefs show` prints sensible output when table empty;
  `health prefs set --privacy-mode private_tee` persists; warns when
  NEAR_API_KEY is unset
"""

from __future__ import annotations

import time
from pathlib import Path

import apsw
import pytest

from src.health.config import HealthConfig
from src.health.router import HealthRoute, route
from src.health.user_prefs import (
    DEFAULT_USER_ID,
    UserHealthPref,
    init_tables,
    load_pref,
    save_pref,
)
from src.session.models import TriageResult


# --- fixtures ---


@pytest.fixture
def conn() -> apsw.Connection:
    c = apsw.Connection(":memory:")
    c.row_trace = lambda cursor, row: {
        d[0]: v for d, v in zip(cursor.getdescription(), row)
    }
    init_tables(c)
    return c


def _triage(phi_class: str) -> TriageResult:
    return TriageResult(
        intent="question",
        complexity="moderate",
        search_params={},
        needs_memory=True,
        matched_skill=None,
        phi_class=phi_class,
        health_topic=None,
    )


def _baa_config() -> HealthConfig:
    return HealthConfig(
        enabled=True,
        baa_provider="bedrock",
        baa_model_main="us.anthropic.claude-sonnet-4-x",
    )


# --- UserHealthPref store ---


class TestUserHealthPrefs:
    def test_init_tables_idempotent(self, conn):
        init_tables(conn)
        init_tables(conn)
        # Only one row in sqlite_master for our table.
        rows = list(conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='user_health_prefs'"
        ))
        assert len(rows) == 1

    def test_load_returns_default_when_missing(self, conn):
        pref = load_pref(conn)
        assert pref.user_id == DEFAULT_USER_ID
        assert pref.privacy_mode == "standard"
        assert pref.consent_anonymised_general is False
        assert pref.is_default is True

    def test_save_then_load_roundtrip(self, conn):
        save_pref(conn, privacy_mode="private_tee", now=1_700_000_000)
        pref = load_pref(conn)
        assert pref.privacy_mode == "private_tee"
        assert pref.last_updated == 1_700_000_000

    def test_save_partial_field_preserves_others(self, conn):
        save_pref(conn, privacy_mode="private_tee",
                  consent_anonymised_general=True, now=1)
        # Update only privacy_mode; consent should stick
        save_pref(conn, privacy_mode="standard", now=2)
        pref = load_pref(conn)
        assert pref.privacy_mode == "standard"
        assert pref.consent_anonymised_general is True
        assert pref.last_updated == 2

    def test_unknown_privacy_mode_rejected(self, conn):
        with pytest.raises(ValueError, match="Unknown privacy_mode"):
            save_pref(conn, privacy_mode="bogus")  # type: ignore[arg-type]

    def test_load_handles_missing_table_gracefully(self):
        # Connection with no user_health_prefs table — should NOT raise.
        c = apsw.Connection(":memory:")
        c.row_trace = lambda cursor, row: {
            d[0]: v for d, v in zip(cursor.getdescription(), row)
        }
        pref = load_pref(c)
        assert pref.privacy_mode == "standard"
        assert pref.is_default is True

    def test_unknown_mode_in_db_falls_back_to_standard(self, conn, caplog):
        # Bypass save_pref to inject a corrupt row
        conn.execute(
            "INSERT INTO user_health_prefs "
            "(user_id, privacy_mode, consent_anonymised_general, last_updated) "
            "VALUES (?, ?, ?, ?)",
            ("self", "corrupt_value", 0, 1),
        )
        with caplog.at_level("WARNING"):
            pref = load_pref(conn)
        assert pref.privacy_mode == "standard"
        assert any("Unknown privacy_mode" in r.message for r in caplog.records)


# --- Router ---


class TestRouterPrivateTee:
    def test_general_with_private_tee_pref_routes_to_tee(self):
        cfg = HealthConfig(enabled=True)
        pref = UserHealthPref(
            user_id="self", privacy_mode="private_tee",
            consent_anonymised_general=False, last_updated=0,
        )
        r = route(_triage("general"), cfg, pref)
        assert r.path == "general_private_tee"
        assert r.use_private_tee is True
        assert r.use_baa_llm is False
        assert r.enable_retrieval is True
        assert r.require_disclaimer is True

    def test_general_with_standard_pref_routes_to_general(self):
        cfg = HealthConfig(enabled=True)
        pref = UserHealthPref(
            user_id="self", privacy_mode="standard",
            consent_anonymised_general=False, last_updated=0,
        )
        r = route(_triage("general"), cfg, pref)
        assert r.path == "general"
        assert r.use_private_tee is False

    def test_general_without_pref_is_unchanged(self):
        cfg = HealthConfig(enabled=True)
        # No user_pref → behaves like pre-Phase-E
        r = route(_triage("general"), cfg)
        assert r.path == "general"
        assert r.use_private_tee is False

    def test_personal_ignores_private_tee_pref(self):
        # The TEE pref only affects general-health turns. Personal/PHI
        # turns still BAA-or-decline regardless of TEE preference.
        cfg = _baa_config()
        pref = UserHealthPref(
            user_id="self", privacy_mode="private_tee",
            consent_anonymised_general=False, last_updated=0,
        )
        r = route(_triage("personal"), cfg, pref)
        assert r.path == "phi"
        assert r.use_baa_llm is True
        assert r.use_private_tee is False

    def test_disabled_module_with_pref_still_skips(self):
        cfg = HealthConfig(enabled=False)
        pref = UserHealthPref(
            user_id="self", privacy_mode="private_tee",
            consent_anonymised_general=False, last_updated=0,
        )
        r = route(_triage("general"), cfg, pref)
        assert r.path == "skip"

    def test_phi_class_none_with_pref_still_skips(self):
        cfg = HealthConfig(enabled=True)
        pref = UserHealthPref(
            user_id="self", privacy_mode="private_tee",
            consent_anonymised_general=False, last_updated=0,
        )
        r = route(_triage("none"), cfg, pref)
        assert r.path == "skip"

    def test_all_routes_have_use_private_tee_field(self):
        # Pin down: every HealthRoute the router can produce has the
        # use_private_tee field set explicitly. Mutually exclusive
        # with use_baa_llm.
        cfg = HealthConfig(enabled=True)
        baa_cfg = _baa_config()
        tee_pref = UserHealthPref(
            user_id="self", privacy_mode="private_tee",
            consent_anonymised_general=False, last_updated=0,
        )
        for triage, c, pref in [
            (_triage("none"), cfg, None),
            (_triage("general"), cfg, None),
            (_triage("general"), cfg, tee_pref),
            (_triage("personal"), baa_cfg, None),
            (_triage("personal"), cfg, None),  # no baa → decline
        ]:
            r = route(triage, c, pref)
            assert isinstance(r.use_private_tee, bool)
            # Mutually exclusive
            assert not (r.use_baa_llm and r.use_private_tee), (
                f"both flags on for path={r.path}"
            )


# --- Factory ---


class TestPrivateTeeFactory:
    def test_returns_none_without_near_api_key(self, monkeypatch, caplog):
        monkeypatch.delenv("NEAR_API_KEY", raising=False)
        from src.llm.factory import build_private_tee_llm
        class _C: pass
        with caplog.at_level("WARNING"):
            result = build_private_tee_llm(_C())
        assert result is None
        assert any("NEAR_API_KEY" in r.message for r in caplog.records)

    def test_returns_llmsession_with_qwen_for_moderate(self, monkeypatch):
        monkeypatch.setenv("NEAR_API_KEY", "k")
        from src.llm.factory import build_private_tee_llm
        from src.llm.session import LLMSession
        class _C: pass
        llm = build_private_tee_llm(_C(), complexity="moderate")
        assert isinstance(llm, LLMSession)
        assert "Qwen3.5" in llm.model
        # No tools on the TEE path
        assert llm._tools == []
        assert llm._tool_handlers == {}

    def test_returns_llmsession_with_gpt_oss_for_complex(self, monkeypatch):
        monkeypatch.setenv("NEAR_API_KEY", "k")
        from src.llm.factory import build_private_tee_llm
        from src.llm.session import LLMSession
        class _C: pass
        llm = build_private_tee_llm(_C(), complexity="complex")
        assert isinstance(llm, LLMSession)
        assert "gpt-oss" in llm.model.lower()

    def test_simple_complexity_uses_qwen_too(self, monkeypatch):
        monkeypatch.setenv("NEAR_API_KEY", "k")
        from src.llm.factory import build_private_tee_llm
        class _C: pass
        llm = build_private_tee_llm(_C(), complexity="simple")
        assert "Qwen3.5" in llm.model


# --- CLI prefs ---


class TestHealthPrefsCli:
    def test_show_with_no_prefs_set(self, monkeypatch, tmp_path, capsys):
        from src.health.cli import health_cli
        monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "ws"))
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
        rc = health_cli(["prefs", "show"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "standard" in captured.out
        assert "never (using defaults)" in captured.out

    def test_set_persists_and_show_reflects(
        self, monkeypatch, tmp_path, capsys
    ):
        from src.health.cli import health_cli
        monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "ws"))
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("NEAR_API_KEY", "k")  # silence the warn path

        rc = health_cli(["prefs", "set", "--privacy-mode", "private_tee"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "private_tee" in captured.out

        # Re-run show: should reflect the change
        rc = health_cli(["prefs", "show"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "private_tee" in captured.out
        assert "never" not in captured.out  # last_updated is now set

    def test_set_warns_when_near_key_missing(
        self, monkeypatch, tmp_path, capsys
    ):
        from src.health.cli import health_cli
        monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "ws"))
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
        monkeypatch.delenv("NEAR_API_KEY", raising=False)

        # The CLI's load_config() re-reads .env, which on this dev box
        # repopulates NEAR_API_KEY. Stub the dotenv loader so the test
        # actually exercises the "key missing" branch.
        monkeypatch.setattr("src.config._load_dotenv", lambda: None)
        # Also delete again *after* config load — load_config doesn't
        # actually mutate os.environ for NEAR_API_KEY since we stubbed
        # _load_dotenv, but a previous load may have left it set.
        monkeypatch.delenv("NEAR_API_KEY", raising=False)

        rc = health_cli(["prefs", "set", "--privacy-mode", "private_tee"])
        captured = capsys.readouterr()
        assert rc == 0
        # Set succeeded but warning printed
        assert "NEAR_API_KEY" in captured.out
        assert "fall back" in captured.out
