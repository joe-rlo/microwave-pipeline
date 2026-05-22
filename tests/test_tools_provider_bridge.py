"""Tests for build_provider_tools() — the bridge from existing MCP-shape
tool handlers to the provider abstraction's (ToolDefinition, handler)
pairs.

We exercise:
- Registry is empty when no env keys are set
- Instacart key present → one ProviderTool registered
- GitHub token present → three ProviderTools registered
- Bridge handlers correctly unwrap MCP-shaped success responses
- Bridge handlers raise on MCP-shaped error responses (so the session's
  tool loop converts to is_error=True for the model)
- Both shapes use the same JSON schema (no drift between SDK and
  provider surfaces)
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.llm.provider import ToolDefinition
from src.tools import ProviderTool, build_provider_tools


# --- Registry ---


class TestProviderRegistry:
    def test_empty_when_no_keys(self):
        config = SimpleNamespace(instacart_api_key="", github_token="")
        tools = build_provider_tools(config)
        assert tools == []

    def test_instacart_only(self):
        config = SimpleNamespace(
            instacart_api_key="fake",
            instacart_partner_linkback_url="",
            github_token="",
        )
        tools = build_provider_tools(config)
        names = [t.definition.name for t in tools]
        assert names == ["instacart_create_cart"]

    def test_github_only_registers_three(self):
        config = SimpleNamespace(
            instacart_api_key="",
            github_token="ghp_fake",
        )
        tools = build_provider_tools(config)
        names = [t.definition.name for t in tools]
        assert names == [
            "github_list_repos",
            "github_repo_summary",
            "github_recent_activity",
        ]

    def test_both_registered_when_both_keys_set(self):
        config = SimpleNamespace(
            instacart_api_key="fake",
            instacart_partner_linkback_url="",
            github_token="ghp_fake",
        )
        tools = build_provider_tools(config)
        names = sorted(t.definition.name for t in tools)
        assert names == sorted([
            "instacart_create_cart",
            "github_list_repos",
            "github_repo_summary",
            "github_recent_activity",
        ])

    def test_returned_definitions_are_real_ToolDefinitions(self):
        config = SimpleNamespace(
            instacart_api_key="fake",
            instacart_partner_linkback_url="",
            github_token="",
        )
        tools = build_provider_tools(config)
        for pt in tools:
            assert isinstance(pt, ProviderTool)
            assert isinstance(pt.definition, ToolDefinition)
            assert pt.definition.input_schema  # non-empty schema


# --- Schema drift guards ---


class TestSchemaConsistency:
    """The Agent SDK shape and the provider shape must use the SAME
    underlying JSON schema. If someone updates one and forgets the
    other, these tests fail loudly."""

    def test_instacart_schema_matches_module_source(self):
        from src.tools import instacart as mod

        config = SimpleNamespace(
            instacart_api_key="k",
            instacart_partner_linkback_url="",
        )
        provider_tools = build_provider_tools(config)
        provider_schema = provider_tools[0].definition.input_schema
        assert provider_schema is mod.INSTACART_CREATE_CART_SCHEMA

    def test_github_schemas_match_module_sources(self):
        from src.tools import github as mod

        config = SimpleNamespace(github_token="ghp_fake")
        provider_tools = build_provider_tools(config)
        by_name = {pt.definition.name: pt.definition.input_schema for pt in provider_tools}
        assert by_name["github_list_repos"] is mod.LIST_REPOS_SCHEMA
        assert by_name["github_repo_summary"] is mod.REPO_SUMMARY_SCHEMA
        assert by_name["github_recent_activity"] is mod.RECENT_ACTIVITY_SCHEMA


# --- Bridge behavior: unwrapping MCP responses ---


@pytest.mark.asyncio
class TestHandlerUnwrap:
    async def test_success_returns_text(self):
        # Patch the underlying instacart handler to return MCP success
        config = SimpleNamespace(
            instacart_api_key="k",
            instacart_partner_linkback_url=None,
        )
        with patch("src.tools.instacart._handle_create_cart") as h:
            h.return_value = {
                "content": [{"type": "text", "text": '{"url":"x","item_count":1}'}],
            }
            tools = build_provider_tools(config)
            result = await tools[0].handler({"title": "x", "items": [{"name": "y"}]})
        # The handler should return the inner text verbatim.
        # The model will parse the JSON on its side.
        assert json.loads(result) == {"url": "x", "item_count": 1}

    async def test_error_raises(self):
        config = SimpleNamespace(
            instacart_api_key="k",
            instacart_partner_linkback_url=None,
        )
        with patch("src.tools.instacart._handle_create_cart") as h:
            h.return_value = {
                "content": [{"type": "text", "text": "Instacart API error: 401"}],
                "is_error": True,
            }
            tools = build_provider_tools(config)
            with pytest.raises(RuntimeError, match="Instacart API error"):
                await tools[0].handler({"title": "x", "items": [{"name": "y"}]})

    async def test_github_success_unwrap(self):
        config = SimpleNamespace(github_token="t")
        payload = '{"count":2,"repos":[]}'
        with patch("src.tools.github._handle_list_repos") as h:
            h.return_value = {
                "content": [{"type": "text", "text": payload}],
            }
            tools = build_provider_tools(config)
            list_repos_tool = next(
                t for t in tools if t.definition.name == "github_list_repos"
            )
            result = await list_repos_tool.handler({"limit": 5})
        assert result == payload

    async def test_github_error_raises_with_message(self):
        config = SimpleNamespace(github_token="t")
        with patch("src.tools.github._handle_repo_summary") as h:
            h.return_value = {
                "content": [{"type": "text", "text": "GitHub API error: rate limited"}],
                "is_error": True,
            }
            tools = build_provider_tools(config)
            summary_tool = next(
                t for t in tools if t.definition.name == "github_repo_summary"
            )
            with pytest.raises(RuntimeError, match="rate limited"):
                await summary_tool.handler({"repo": "joe/repo"})

    async def test_handler_passes_args_through(self):
        # The bridge mustn't drop or mangle the args dict.
        config = SimpleNamespace(github_token="t")
        captured = {}
        with patch("src.tools.github._handle_list_repos") as h:
            async def capture(args, token):
                captured["args"] = args
                captured["token"] = token
                return {"content": [{"type": "text", "text": "ok"}]}

            h.side_effect = capture
            tools = build_provider_tools(config)
            list_repos_tool = next(
                t for t in tools if t.definition.name == "github_list_repos"
            )
            await list_repos_tool.handler({"visibility": "public", "limit": 10})
        assert captured["args"] == {"visibility": "public", "limit": 10}
        assert captured["token"] == "t"
