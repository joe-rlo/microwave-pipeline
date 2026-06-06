"""Tests for build_main_llm() — the main-LLM factory.

What we verify:
- Empty env → returns LLMClient (legacy). Behavior unchanged.
- LLM_STAGE_MAIN="near:<model>" with NEAR_API_KEY → returns LLMSession.
- LLM_STAGE_MAIN without NEAR_API_KEY → factory raises at construction
  (we want fail-fast at startup, not silent fallback that masks
  misconfiguration).
- Unknown provider in LLM_STAGE_MAIN → warns and falls back to legacy.
- Tools from build_provider_tools() are passed through to LLMSession.

We do NOT actually invoke the returned object — that hits the network /
SDK. Constructor-shape checks are enough; the session and client both
have separate end-to-end tests.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.llm.client import LLMClient
from src.llm.factory import build_main_llm
from src.llm.session import LLMSession


def _make_config(**overrides) -> SimpleNamespace:
    """Build a Config-shaped stub with the fields build_main_llm reads."""
    defaults = {
        "model_main": "sonnet",
        "auth_mode": "max",
        "anthropic_api_key": "",
        "cli_path": "",
        "output_dir": Path("/tmp/x"),
        "workspace_dir": Path("/tmp/x"),
        "instacart_api_key": "",
        "instacart_partner_linkback_url": "",
        "github_token": "",
        "bot_builtin_tools": (),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# --- Legacy path ---


class TestLegacyDefault:
    def test_no_env_returns_legacy(self, monkeypatch):
        monkeypatch.delenv("LLM_STAGE_MAIN", raising=False)
        config = _make_config()
        llm = build_main_llm(config)
        assert isinstance(llm, LLMClient)

    def test_unknown_provider_warns_and_falls_back(self, monkeypatch, caplog):
        monkeypatch.setenv("LLM_STAGE_MAIN", "unknown_provider:model_x")
        config = _make_config()
        with caplog.at_level("WARNING"):
            llm = build_main_llm(config)
        assert isinstance(llm, LLMClient)
        assert any("unknown" in r.message.lower() for r in caplog.records)


# --- NEAR path ---


class TestNearPath:
    def test_near_override_returns_session(self, monkeypatch):
        monkeypatch.setenv("LLM_STAGE_MAIN", "near:claude-sonnet-4-6")
        monkeypatch.setenv("NEAR_API_KEY", "k")
        config = _make_config()
        llm = build_main_llm(config)
        assert isinstance(llm, LLMSession)
        assert llm.model == "claude-sonnet-4-6"

    def test_near_falls_back_to_config_model_when_override_missing_model(
        self, monkeypatch
    ):
        monkeypatch.setenv("LLM_STAGE_MAIN", "near:")
        monkeypatch.setenv("NEAR_API_KEY", "k")
        config = _make_config(model_main="claude-opus-4-7")
        llm = build_main_llm(config)
        assert isinstance(llm, LLMSession)
        assert llm.model == "claude-opus-4-7"

    def test_near_without_api_key_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_STAGE_MAIN", "near:m")
        monkeypatch.delenv("NEAR_API_KEY", raising=False)
        config = _make_config()
        with pytest.raises(RuntimeError, match="NEAR_API_KEY"):
            build_main_llm(config)


# --- Tool wiring on NEAR path ---


class TestToolsPassthrough:
    def test_no_tools_when_no_keys(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LLM_STAGE_MAIN", "near:m")
        monkeypatch.setenv("NEAR_API_KEY", "k")
        # Disable always-on web + file tools so this assertion isolates
        # to instacart/github wiring. Scheduler tools are always-on (they
        # only need the local SQLite path that's already in config) so
        # they appear here unconditionally. Point Blink creds at a
        # nonexistent path so the dev machine's real creds don't leak in.
        monkeypatch.setenv("WEB_TOOLS_DISABLED", "1")
        monkeypatch.setenv("FILE_TOOLS_DISABLED", "1")
        monkeypatch.setenv("WEBSEARCH_DISABLED", "1")
        monkeypatch.setenv("MEDICAL_TOOLS_DISABLED", "1")
        monkeypatch.setenv("BLINK_CREDENTIALS_PATH", str(tmp_path / "no-blink.json"))
        config = _make_config()  # neither instacart nor github
        llm = build_main_llm(config)
        names = sorted(td.name for td in llm._tools)
        assert names == sorted([
            "scheduler_list",
            "scheduler_get",
            "scheduler_add",
            "scheduler_remove",
            "scheduler_set_enabled",
        ])
        assert set(llm._tool_handlers) == set(names)

    def test_github_tools_registered_on_session(self, monkeypatch):
        monkeypatch.setenv("LLM_STAGE_MAIN", "near:m")
        monkeypatch.setenv("NEAR_API_KEY", "k")
        monkeypatch.setenv("WEB_TOOLS_DISABLED", "1")
        monkeypatch.setenv("FILE_TOOLS_DISABLED", "1")
        monkeypatch.setenv("WEBSEARCH_DISABLED", "1")
        monkeypatch.setenv("MEDICAL_TOOLS_DISABLED", "1")
        config = _make_config(github_token="ghp_fake")
        llm = build_main_llm(config)
        # Three github tools should land on the session, alongside the
        # always-on scheduler tools.
        names = {td.name for td in llm._tools}
        assert {
            "github_list_repos",
            "github_repo_summary",
            "github_recent_activity",
        } <= names
        assert set(llm._tool_handlers.keys()) == names

    def test_webfetch_registered_by_default(self, monkeypatch):
        # Without WEB_TOOLS_DISABLED, webfetch appears on the NEAR path
        # automatically — no env key required.
        monkeypatch.setenv("LLM_STAGE_MAIN", "near:m")
        monkeypatch.setenv("NEAR_API_KEY", "k")
        monkeypatch.delenv("WEB_TOOLS_DISABLED", raising=False)
        config = _make_config()
        llm = build_main_llm(config)
        names = [td.name for td in llm._tools]
        assert "webfetch" in names


# --- build_baa_llm (Phase D.2) ---


from src.health.config import HealthConfig
from src.llm.factory import build_baa_llm
from src.llm.session import LLMSession


def _make_health_baa_config(**overrides) -> HealthConfig:
    """HealthConfig with the BAA path fully populated."""
    defaults = {
        "enabled": True,
        "baa_provider": "bedrock",
        "baa_model_main": "anthropic.claude-sonnet-4-x",
        "baa_model_escalation": "anthropic.claude-opus-4-x",
    }
    defaults.update(overrides)
    return HealthConfig(**defaults)


class TestBuildBaaLlm:
    def test_returns_none_when_health_module_disabled(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        config = _make_config()
        config.health = HealthConfig(enabled=False)
        assert build_baa_llm(config) is None

    def test_returns_none_when_baa_provider_is_none(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        config = _make_config()
        config.health = HealthConfig(
            enabled=True, baa_provider="none", baa_model_main="x"
        )
        assert build_baa_llm(config) is None

    def test_returns_none_when_no_model_main(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        config = _make_config()
        config.health = HealthConfig(
            enabled=True, baa_provider="bedrock", baa_model_main=""
        )
        # phi_path_available is False when baa_model_main is empty
        assert build_baa_llm(config) is None

    def test_returns_none_when_aws_region_missing(self, monkeypatch, caplog):
        monkeypatch.delenv("AWS_REGION", raising=False)
        config = _make_config()
        config.health = _make_health_baa_config()
        with caplog.at_level("ERROR"):
            result = build_baa_llm(config)
        assert result is None
        assert any("AWS_REGION" in r.message for r in caplog.records)

    def test_returns_none_for_unsupported_baa_provider(
        self, monkeypatch, caplog
    ):
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        config = _make_config()
        # vertex isn't wired in D.1 — should warn and return None
        config.health = HealthConfig(
            enabled=True, baa_provider="vertex", baa_model_main="x",
        )
        with caplog.at_level("WARNING"):
            result = build_baa_llm(config)
        assert result is None
        assert any("vertex" in r.message for r in caplog.records)

    def _stub_boto3(self, monkeypatch):
        """Shared boto3 stub so BedrockProvider's lazy import succeeds in
        tests without the real dependency installed."""
        class _StubBoto3:
            @staticmethod
            def client(name, **kwargs):
                return object()
        import sys
        monkeypatch.setitem(sys.modules, "boto3", _StubBoto3)

    def test_returns_llmsession_when_fully_configured(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        # Isolate from always-on tools we don't care about for this assertion
        monkeypatch.setenv("WEB_TOOLS_DISABLED", "1")
        monkeypatch.setenv("FILE_TOOLS_DISABLED", "1")
        monkeypatch.setenv("WEBSEARCH_DISABLED", "1")
        monkeypatch.setenv("MEDICAL_TOOLS_DISABLED", "1")
        monkeypatch.setenv("BLINK_CREDENTIALS_PATH", str(tmp_path / "no-blink.json"))
        self._stub_boto3(monkeypatch)
        config = _make_config()
        config.health = _make_health_baa_config()

        llm = build_baa_llm(config)
        assert isinstance(llm, LLMSession)
        assert llm.model == "anthropic.claude-sonnet-4-x"
        # Health-profile tools are now wired on BAA so the LLM can answer
        # "what's in my profile?" instead of confabulating "empty".
        names = {td.name for td in llm._tools}
        assert names == {
            "health_profile_summary",
            "health_profile_show",
            "health_profile_audit",
        }
        # Handlers agree
        assert set(llm._tool_handlers) == names

    def test_external_tools_never_register_on_baa_path(self, monkeypatch, tmp_path):
        """PHI privacy regression: webfetch / github / instacart / blink
        / scheduler MUST NOT be exposed to the BAA model even when their
        own gate conditions are met. They could leak PHI to external
        services. Adding any of them to the BAA path requires explicit
        opt-in via _BAA_ALLOWED_TOOLS, which this test guards."""
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        # Try to enable EVERY external tool — they should still be filtered out
        monkeypatch.delenv("WEB_TOOLS_DISABLED", raising=False)
        monkeypatch.delenv("FILE_TOOLS_DISABLED", raising=False)
        monkeypatch.delenv("WEBSEARCH_DISABLED", raising=False)
        # Real blink creds present on the dev machine — point elsewhere
        # so the test isn't sensitive to machine state.
        monkeypatch.setenv("BLINK_CREDENTIALS_PATH", str(tmp_path / "no-blink.json"))
        self._stub_boto3(monkeypatch)

        config = _make_config(
            github_token="ghp_fake",
            instacart_api_key="ic_fake",
        )
        config.workspace_dir = tmp_path / "ws"
        config.health = _make_health_baa_config()

        llm = build_baa_llm(config)
        names = {td.name for td in llm._tools}
        # Forbidden families on BAA
        forbidden_prefixes = ("github_", "instacart_", "blink_", "webfetch", "read_file", "websearch", "scheduler_")
        leaked = [n for n in names if n.startswith(forbidden_prefixes)]
        assert leaked == [], f"BAA path leaked external tools: {leaked}"
        # And we still have the health-profile tools
        assert "health_profile_summary" in names

    def test_baa_omits_health_tools_when_module_off(self, monkeypatch, tmp_path):
        """Sanity: the gating that makes health-profile tools register
        (config.health.enabled) is the SAME gate that lets build_baa_llm
        return non-None. So a disabled health module → no BAA LLM at all,
        not a BAA LLM with empty tools. This test just pins that path."""
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        config = _make_config()
        # phi_path_available will be False here
        config.health = HealthConfig(
            enabled=False, baa_provider="bedrock", baa_model_main="x",
        )
        assert build_baa_llm(config) is None

    def test_builtin_tools_warn_on_near_path(self, monkeypatch, caplog):
        # BOT_BUILTIN_TOOLS is an SDK-only feature. When set alongside
        # the NEAR path, the factory should warn loudly so the user
        # knows their setting isn't taking effect.
        monkeypatch.setenv("LLM_STAGE_MAIN", "near:m")
        monkeypatch.setenv("NEAR_API_KEY", "k")
        config = _make_config(bot_builtin_tools=("WebFetch", "WebSearch"))
        with caplog.at_level("WARNING"):
            build_main_llm(config)
        assert any(
            "BOT_BUILTIN_TOOLS" in r.message for r in caplog.records
        )
