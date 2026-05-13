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
        """Stock install: empty builtin_tools, no setting_sources passed,
        no permission_mode override (SDK default)."""
        opts = _captured_options(model="sonnet", auth_mode="max")
        assert opts.get("allowed_tools") == []
        # setting_sources is intentionally absent — defaults to None,
        # which means the SDK doesn't load any settings files. Don't
        # silently pull the user's settings.local.json without their
        # explicit opt-in via builtin_tools.
        assert "setting_sources" not in opts
        # permission_mode also unset — SDK uses its default (interactive
        # approval for tools not on the allowlist). Fine when there are
        # no tools to approve anyway.
        assert "permission_mode" not in opts

    def test_builtins_listed_in_allowed_tools(self):
        opts = _captured_options(
            model="sonnet", auth_mode="max",
            builtin_tools=["WebFetch", "WebSearch"],
        )
        assert opts["allowed_tools"] == ["WebFetch", "WebSearch"]

    def test_builtins_disable_settings_inheritance(self):
        """`setting_sources=[]` shuts off the SDK's invisible inheritance
        from `~/.claude/` and project-local `.claude/`. Previously we
        loaded those for the permission patterns, but bypassPermissions
        skips the patterns anyway — leaving them loaded was just a
        silent context leak (model prefs, hooks, sub-agents could
        quietly affect the bot). Pin the lockdown explicitly so a
        future "tighten settings" refactor can't accidentally
        re-enable inheritance."""
        opts = _captured_options(
            model="sonnet", auth_mode="max",
            builtin_tools=["Bash"],
        )
        assert opts["setting_sources"] == []

    def test_builtins_bypass_interactive_permission_prompts(self):
        """Critical for messaging-channel UX: the bot has no terminal
        to display a permission prompt. Without bypass, the SDK hangs
        on any tool call not pre-approved by the allowlist patterns,
        and the LLM rationalizes confused workarounds. Lock the bypass
        so a future "tighten this" refactor doesn't quietly break the
        bot for everyone using BOT_BUILTIN_TOOLS."""
        opts = _captured_options(
            model="sonnet", auth_mode="max",
            builtin_tools=["Write", "Edit", "Bash"],
        )
        assert opts["permission_mode"] == "bypassPermissions"

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
        # No silent .claude/ inheritance even though builtins are on
        assert opts["setting_sources"] == []

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
