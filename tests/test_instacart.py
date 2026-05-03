"""Tests for the Instacart integration + tool wrapper.

Covers:
- LineItem payload shaping (only-what's-set, no None drift)
- IDP client happy path with a stubbed aiohttp response
- IDP client error paths (4xx, malformed JSON, missing URL)
- Tool handler translating LLM args → client call → MCP result shape
- Tool registry: empty when key missing, populated when set

We deliberately don't exercise `claude_agent_sdk` in tests — the SDK
wraps our handler in an MCP server, but the handler itself is just an
async function we can call directly. Verifying the *contract* (input
schema, output shape) at this layer protects us from SDK upgrades.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.integrations.instacart import (
    InstacartClient,
    InstacartError,
    LineItem,
    ProductsLinkResult,
)


# --- LineItem payload shaping ---


class TestLineItem:
    def test_minimal_payload(self):
        item = LineItem(name="milk")
        assert item.to_payload() == {"name": "milk"}

    def test_quantity_and_unit(self):
        item = LineItem(name="chicken thighs", quantity=2, unit="lb")
        payload = item.to_payload()
        assert payload["name"] == "chicken thighs"
        assert payload["quantity"] == 2
        assert payload["unit"] == "lb"
        assert "filters" not in payload  # no filters configured

    def test_filters_aggregated(self):
        item = LineItem(
            name="milk",
            brand_filters=["Horizon"],
            health_filters=["ORGANIC"],
        )
        payload = item.to_payload()
        assert payload["filters"] == {
            "brand_filters": ["Horizon"],
            "health_filters": ["ORGANIC"],
        }

    def test_empty_filter_lists_omitted(self):
        # If both filter lists are empty, the `filters` key should not
        # appear at all — Instacart treats `{filters: {}}` as a no-op
        # but it's noise we'd rather not send.
        item = LineItem(name="milk", brand_filters=[], health_filters=[])
        payload = item.to_payload()
        assert "filters" not in payload

    def test_zero_quantity_kept(self):
        # 0 is a legitimate (if weird) quantity; must not be dropped
        # the way a None would be.
        item = LineItem(name="ice", quantity=0)
        assert item.to_payload()["quantity"] == 0


# --- IDP client ---


def _make_response(status: int, body: str | dict):
    """Build a stand-in for an aiohttp ClientResponse.

    The real response is an async context manager returned by
    `session.post(...)`. We mock the bare minimum the client uses:
    `.status` and `.text()`.
    """
    text_body = body if isinstance(body, str) else json.dumps(body)

    resp = MagicMock()
    resp.status = status
    resp.text = AsyncMock(return_value=text_body)
    # The client uses `async with session.post(...)` — make __aenter__
    # return the response object, __aexit__ a no-op.
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _stub_session(post_cm) -> MagicMock:
    """Build a stand-in aiohttp.ClientSession.post returning `post_cm`."""
    session = MagicMock()
    session.closed = False
    session.post = MagicMock(return_value=post_cm)
    session.close = AsyncMock()
    return session


@pytest.mark.asyncio
class TestInstacartClient:
    async def test_create_products_link_happy_path(self):
        post_cm = _make_response(
            200,
            {"products_link_url": "https://instacart.com/store/recipes/abc"},
        )
        session = _stub_session(post_cm)
        client = InstacartClient(api_key="fake", session=session)

        result = await client.create_products_link(
            title="Pasta night",
            line_items=[LineItem(name="pasta"), LineItem(name="sauce")],
        )

        assert isinstance(result, ProductsLinkResult)
        assert result.url == "https://instacart.com/store/recipes/abc"
        # Verify request body shape
        call = session.post.call_args
        body = call.kwargs["json"]
        assert body["title"] == "Pasta night"
        assert body["link_type"] == "shopping_list"
        assert body["line_items"] == [{"name": "pasta"}, {"name": "sauce"}]
        # Auth header present
        headers = call.kwargs["headers"]
        assert headers["Authorization"] == "Bearer fake"

    async def test_4xx_raises_with_status_and_body(self):
        post_cm = _make_response(401, "unauthorized")
        session = _stub_session(post_cm)
        client = InstacartClient(api_key="bad", session=session)

        with pytest.raises(InstacartError) as exc_info:
            await client.create_products_link(
                title="x", line_items=[LineItem(name="x")]
            )
        assert exc_info.value.status == 401
        assert "unauthorized" in (exc_info.value.body or "")

    async def test_malformed_json_raises(self):
        post_cm = _make_response(200, "not json at all")
        session = _stub_session(post_cm)
        client = InstacartClient(api_key="fake", session=session)

        with pytest.raises(InstacartError) as exc_info:
            await client.create_products_link(
                title="x", line_items=[LineItem(name="x")]
            )
        # Status was 200 (parsed body, not the HTTP layer) — but it's
        # still an InstacartError because the response was unusable.
        assert "Malformed" in str(exc_info.value)

    async def test_missing_url_field_raises(self):
        post_cm = _make_response(200, {"unrelated": "field"})
        session = _stub_session(post_cm)
        client = InstacartClient(api_key="fake", session=session)

        with pytest.raises(InstacartError) as exc_info:
            await client.create_products_link(
                title="x", line_items=[LineItem(name="x")]
            )
        assert "products_link_url" in str(exc_info.value)

    async def test_empty_line_items_rejected(self):
        client = InstacartClient(api_key="fake")
        with pytest.raises(ValueError):
            await client.create_products_link(title="x", line_items=[])

    async def test_optional_fields_passed_through(self):
        post_cm = _make_response(
            200, {"products_link_url": "https://x"}
        )
        session = _stub_session(post_cm)
        client = InstacartClient(api_key="fake", session=session)

        await client.create_products_link(
            title="weekly",
            line_items=[LineItem(name="milk")],
            expires_in_days=7,
            instructions=["Leave at door"],
            partner_linkback_url="https://my.site/done",
        )
        body = session.post.call_args.kwargs["json"]
        assert body["expires_in"] == 7
        assert body["instructions"] == ["Leave at door"]
        assert body["landing_page_configuration"]["partner_linkback_url"] == "https://my.site/done"
        assert body["landing_page_configuration"]["enable_pantry_items"] is True


# --- Tool handler ---


@pytest.mark.asyncio
class TestToolHandler:
    async def test_handler_returns_mcp_shape_on_success(self):
        from src.tools.instacart import _handle_create_cart

        # Patch InstacartClient inside the tool module so the handler
        # uses our fake instead of a real one.
        fake_result = ProductsLinkResult(
            url="https://instacart.com/store/abc", raw={}
        )
        with patch("src.tools.instacart.InstacartClient") as MockClient:
            instance = MockClient.return_value
            instance.create_products_link = AsyncMock(return_value=fake_result)

            args = {
                "title": "Pasta night",
                "items": [
                    {"name": "pasta", "quantity": 1, "unit": "lb"},
                    {"name": "sauce"},
                ],
            }
            result = await _handle_create_cart(args, "fake-key", None)

        assert "content" in result
        assert "is_error" not in result
        # The handler serializes the structured payload as JSON inside
        # a single text block — verify the LLM can parse it back.
        text = result["content"][0]["text"]
        payload = json.loads(text)
        assert payload["url"] == "https://instacart.com/store/abc"
        assert payload["item_count"] == 2
        assert payload["title"] == "Pasta night"

    async def test_handler_rejects_no_valid_items(self):
        from src.tools.instacart import _handle_create_cart

        # Items missing `name` are silently dropped; if nothing
        # survives we error out before hitting the API.
        args = {
            "title": "junk",
            "items": [{"quantity": 1}, {"unit": "lb"}],
        }
        result = await _handle_create_cart(args, "fake-key", None)
        assert result.get("is_error") is True
        assert "name" in result["content"][0]["text"]

    async def test_handler_surfaces_instacart_error(self):
        from src.tools.instacart import _handle_create_cart

        with patch("src.tools.instacart.InstacartClient") as MockClient:
            instance = MockClient.return_value
            instance.create_products_link = AsyncMock(
                side_effect=InstacartError("boom", status=500)
            )

            args = {"title": "x", "items": [{"name": "milk"}]}
            result = await _handle_create_cart(args, "fake-key", None)

        assert result.get("is_error") is True
        assert "Instacart API error" in result["content"][0]["text"]


# --- Registry ---


class TestRegistry:
    def test_no_key_means_empty_bundle(self):
        from src.tools import build_tools

        config = SimpleNamespace(instacart_api_key="")
        bundle = build_tools(config)
        assert bundle.is_empty
        assert bundle.allowed_tools == []
        assert bundle.mcp_servers == {}

    def test_key_present_registers_tool(self):
        from src.tools import build_tools

        config = SimpleNamespace(
            instacart_api_key="fake-key",
            instacart_partner_linkback_url="",
        )
        bundle = build_tools(config)
        assert not bundle.is_empty
        assert any(
            "instacart_create_cart" in t for t in bundle.allowed_tools
        ), bundle.allowed_tools
        # Catalog text should mention the tool by name so the model
        # can find it in its system context.
        assert "instacart_create_cart" in bundle.catalog_text
