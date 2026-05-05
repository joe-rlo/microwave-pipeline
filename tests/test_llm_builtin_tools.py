"""Tests for the BOT_BUILTIN_TOOLS plumbing.

We don't connect to the real Agent SDK — we patch ClaudeAgentOptions
to capture the kwargs the orchestrator constructs, then assert the
shape (allowed_tools list, setting_sources list). That's the
load-bearing surface: if these get passed correctly, the SDK does
the right thing with them.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.client import LLMClient


def _captured_options(**llm_kwargs):
    """Construct an LLMClient, run _connect_max with everything patched,
    and return the kwargs that ClaudeAgentOptions was called with."""
    captured: dict = {}

    class _FakeOptions:
        def __init__(self, **kw):
            captured.update(kw)

    fake_client = MagicMock()
    fake_client.connect = AsyncMock()

    class _FakeSDKClient:
        def __init__(self, options):
            captured["__options"] = options

        connect = AsyncMock()

    with patch.dict(
        "sys.modules",
        {
            "claude_agent_sdk": MagicMock(
                ClaudeAgentOptions=_FakeOptions,
                ClaudeSDKClient=_FakeSDKClient,
            ),
        },
    ):
        client = LLMClient(**llm_kwargs)
        client._stable_prompt = "system prompt"
        import asyncio
        asyncio.run(client._connect_max())
    return captured


class TestBuiltinTools:
    def test_default_no_builtins_no_setting_sources(self):
        """Stock install: empty builtin_tools, no setting_sources passed."""
        opts = _captured_options(model="sonnet", auth_mode="max")
        assert opts.get("allowed_tools") == []
        # setting_sources is intentionally absent — defaults to None,
        # which means the SDK doesn't load any settings files. Don't
        # silently pull the user's settings.local.json without their
        # explicit opt-in via builtin_tools.
        assert "setting_sources" not in opts

    def test_builtins_listed_in_allowed_tools(self):
        opts = _captured_options(
            model="sonnet", auth_mode="max",
            builtin_tools=["WebFetch", "WebSearch"],
        )
        assert opts["allowed_tools"] == ["WebFetch", "WebSearch"]

    def test_builtins_enable_settings_sourcing(self):
        """The whole point: when builtins are enabled, the SDK is told
        to read settings.local.json so the user's permission patterns
        gate actual calls. Without this the bot would have unrestricted
        Bash access regardless of what the user has in their settings."""
        opts = _captured_options(
            model="sonnet", auth_mode="max",
            builtin_tools=["Bash"],
        )
        assert opts["setting_sources"] == ["user", "project", "local"]

    def test_mcp_and_builtins_compose(self):
        """When both an MCP tool bundle (Instacart) and builtins are
        configured, allowed_tools holds both — the model sees the full
        union."""
        from src.tools import ToolBundle
        bundle = ToolBundle(
            mcp_servers={"microwave": object()},
            allowed_tools=["mcp__microwave__instacart_create_cart"],
            catalog_text="…",
        )
        opts = _captured_options(
            model="sonnet", auth_mode="max",
            tool_bundle=bundle,
            builtin_tools=["WebFetch"],
        )
        assert "mcp__microwave__instacart_create_cart" in opts["allowed_tools"]
        assert "WebFetch" in opts["allowed_tools"]
        # MCP bundle still drives mcp_servers
        assert "microwave" in opts["mcp_servers"]
        # Settings are sourced because builtins are present
        assert opts["setting_sources"] == ["user", "project", "local"]

    def test_mcp_only_no_settings_source(self):
        """Pure MCP-only path (Instacart but no built-ins): we don't
        enable settings-sourcing, since MCP tools don't need it."""
        from src.tools import ToolBundle
        bundle = ToolBundle(
            mcp_servers={"microwave": object()},
            allowed_tools=["mcp__microwave__instacart_create_cart"],
            catalog_text="…",
        )
        opts = _captured_options(
            model="sonnet", auth_mode="max",
            tool_bundle=bundle,
        )
        assert "setting_sources" not in opts
