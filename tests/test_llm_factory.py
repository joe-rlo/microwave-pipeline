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
    def test_no_tools_when_no_keys(self, monkeypatch):
        monkeypatch.setenv("LLM_STAGE_MAIN", "near:m")
        monkeypatch.setenv("NEAR_API_KEY", "k")
        config = _make_config()  # neither instacart nor github
        llm = build_main_llm(config)
        # No tool defs registered on the session
        assert llm._tools == []
        assert llm._tool_handlers == {}

    def test_github_tools_registered_on_session(self, monkeypatch):
        monkeypatch.setenv("LLM_STAGE_MAIN", "near:m")
        monkeypatch.setenv("NEAR_API_KEY", "k")
        config = _make_config(github_token="ghp_fake")
        llm = build_main_llm(config)
        # Three github tools should land on the session
        names = sorted(td.name for td in llm._tools)
        assert names == sorted([
            "github_list_repos",
            "github_repo_summary",
            "github_recent_activity",
        ])
        assert set(llm._tool_handlers.keys()) == set(names)

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
