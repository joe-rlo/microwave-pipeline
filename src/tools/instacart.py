"""Agent SDK tool wrapper for Instacart IDP.

Exposes one tool to the LLM: `instacart_create_cart`. Given a list of
items, returns a Shop with Instacart URL the user taps to checkout.

Why this is a real SDK tool (not a `<instacart>` extract-and-replace
block in the response): the LLM needs to *see* the URL come back so it
can incorporate it into its reply naturally — confirm the items, hand
over the link, suggest follow-ups. Extract-and-replace blocks break the
LLM's mental model ("did the cart actually get built? what's in it?")
because the model never sees the result of its own action.

Tool ergonomics:
- One call per cart. Multi-step "add then add then checkout" flows
  would need cart-state persistence we don't have. Today: build a full
  list, get a link, done.
- Item shape is intentionally narrow. `name` + optional `quantity`,
  `unit`, `brand`, `health_filters`. The LLM extracts these from natural
  language ("two pounds of organic chicken thighs" → name=chicken
  thighs, quantity=2, unit=lb, health_filters=["ORGANIC"]).
- The tool returns a structured dict, not just a URL — so the LLM
  knows how many items resolved and can mention it ("Cart's ready,
  12 items, here's the link.").
"""

from __future__ import annotations

import logging
from typing import Any

from src.integrations.instacart import (
    InstacartClient,
    InstacartError,
    LineItem,
)

log = logging.getLogger(__name__)


# What the LLM reads in its system context. Concise on purpose — token
# weight matters and the JSON schema (passed separately to the tool
# decorator) carries the formal contract.
INSTACART_TOOL_DOCS = """\
**instacart_create_cart** — Build a Shop with Instacart cart and return a checkout URL.

When to use:
- User says "order groceries", "build a shopping list", "ingredients for [recipe]", "add X to my cart".
- You have a clear list of items. If items are vague ("snacks for the week"), ask before calling.

How to use:
- Pass `items` as a list. Each item needs `name`; `quantity`, `unit`, `brand_filters`, `health_filters` are optional.
- Use `health_filters` for things like ORGANIC, GLUTEN_FREE, VEGAN (uppercase, Instacart's enum).
- Set `title` to something the user will recognize on the Instacart page ("Pasta night", "Weekly groceries").
- The call returns `{ url, item_count }`. Hand the URL to the user — they tap it, finish checkout in Instacart.

Don't fabricate a URL. If the tool errors, say so plainly and ask if they want to retry.
"""


# JSON schema for the tool input. The Agent SDK validates against this
# before invoking the handler — saves us writing defensive parsing in
# the handler body. Keep this in lockstep with `_handle_create_cart`.
INSTACART_CREATE_CART_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": (
                "Short, human label for the cart shown on the Instacart page. "
                "Examples: 'Weekly groceries', 'Pasta night', 'Sunday meal prep'."
            ),
        },
        "items": {
            "type": "array",
            "minItems": 1,
            "description": "Items to add to the cart.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Product name. Be specific when it matters ('chicken thighs' beats 'chicken').",
                    },
                    "quantity": {
                        "type": "number",
                        "description": "Quantity in `unit`. Omit if not specified by the user.",
                    },
                    "unit": {
                        "type": "string",
                        "description": "Unit for quantity: 'each', 'lb', 'oz', 'gallon', 'jar', etc.",
                    },
                    "display_text": {
                        "type": "string",
                        "description": "How this line shows in the Instacart UI. Defaults to `name`.",
                    },
                    "brand_filters": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Preferred brands. Instacart narrows results to these.",
                    },
                    "health_filters": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Instacart health filters (UPPERCASE): "
                            "ORGANIC, GLUTEN_FREE, VEGAN, KOSHER, FAT_FREE, SUGAR_FREE, LOW_FAT."
                        ),
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        },
        "expires_in_days": {
            "type": "integer",
            "minimum": 1,
            "maximum": 365,
            "description": "How long the link stays valid. Defaults to 30.",
        },
        "instructions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional notes for the shopper, one per line.",
        },
    },
    "required": ["title", "items"],
    "additionalProperties": False,
}


def build_instacart_tools(config) -> list:
    """Build the SdkMcpTool list for Instacart. Returns [] if SDK is
    unavailable (caller already checks the API key)."""
    try:
        from claude_agent_sdk import tool
    except ImportError:
        return []

    api_key = getattr(config, "instacart_api_key", "")
    partner_linkback_url = getattr(config, "instacart_partner_linkback_url", "") or None

    @tool(
        name="instacart_create_cart",
        description=(
            "Build a Shop with Instacart cart from a list of items and return a "
            "checkout URL. The user taps the URL to land in Instacart with the "
            "cart pre-loaded and finishes checkout there."
        ),
        input_schema=INSTACART_CREATE_CART_SCHEMA,
    )
    async def instacart_create_cart(args: dict[str, Any]) -> dict[str, Any]:
        return await _handle_create_cart(args, api_key, partner_linkback_url)

    return [instacart_create_cart]


async def _handle_create_cart(
    args: dict[str, Any],
    api_key: str,
    partner_linkback_url: str | None,
) -> dict[str, Any]:
    """Translate args → IDP call → SDK-shaped tool response.

    The SDK expects the handler to return a dict in MCP tool-result
    shape: `{"content": [{"type": "text", "text": "..."}]}` for success
    or with `"is_error": True` on failure. We wrap the structured result
    as JSON inside the text block so the LLM can read fields like
    `url` and `item_count` without us inventing a richer block type.
    """
    import json

    raw_items = args.get("items") or []
    line_items: list[LineItem] = []
    for raw in raw_items:
        if not isinstance(raw, dict) or not raw.get("name"):
            continue
        line_items.append(
            LineItem(
                name=raw["name"],
                quantity=raw.get("quantity"),
                unit=raw.get("unit"),
                display_text=raw.get("display_text"),
                brand_filters=list(raw.get("brand_filters") or []),
                health_filters=list(raw.get("health_filters") or []),
            )
        )

    if not line_items:
        return _error("No valid items provided. Each item needs a `name` field.")

    title = args.get("title") or "Shopping list"
    expires_in_days = args.get("expires_in_days")
    instructions = args.get("instructions") or None

    client = InstacartClient(api_key=api_key)
    try:
        result = await client.create_products_link(
            title=title,
            line_items=line_items,
            expires_in_days=expires_in_days,
            instructions=instructions,
            partner_linkback_url=partner_linkback_url,
        )
    except InstacartError as e:
        log.warning("Instacart tool call failed: %s", e)
        return _error(f"Instacart API error: {e}")
    except Exception as e:
        # Belt and braces — the SDK will surface the exception text to
        # the LLM, but we want a structured error so the model can react.
        log.exception("Unexpected Instacart tool failure")
        return _error(f"Unexpected error building cart: {e}")

    payload = {
        "url": result.url,
        "item_count": len(line_items),
        "title": title,
    }
    return {
        "content": [
            {"type": "text", "text": json.dumps(payload)},
        ],
    }


def _error(message: str) -> dict[str, Any]:
    """SDK tool-result shape for an error response."""
    return {
        "content": [{"type": "text", "text": message}],
        "is_error": True,
    }
