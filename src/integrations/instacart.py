"""Instacart Developer Platform (IDP) API client.

Wraps `POST /idp/v1/products/products_link`. Joe says "build a cart of
groceries" and we hand back a `products_link_url` — Instacart's hosted
"Shop with Instacart" page where the user lands with the cart pre-loaded
and finishes checkout (address, time, payment) themselves.

We only need the IDP. Real fulfillment (Connect API) is B2B and not
self-serve. Docs: https://docs.instacart.com/developer_platform_api/

What we deliberately do NOT do here:
- No Agent SDK imports. This is a pure HTTP client; the tool wrapper
  in `src/tools/instacart.py` adapts it to the SDK's `@tool` shape.
- No retry-with-jitter / circuit-breaking. IDP calls are user-initiated
  one-shots — if it fails the user just asks again. Premature complexity.
- No persistent session state. Every call is fresh; the API key lives
  in env, not in the client object's lifetime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import aiohttp

log = logging.getLogger(__name__)


# IDP base URL. Instacart uses `connect.instacart.com` as the hostname
# for both Connect (B2B) and IDP (public dev platform); the path prefix
# `/idp/` is what segregates the public surface.
IDP_BASE_URL = "https://connect.instacart.com"
IDP_PRODUCTS_LINK_PATH = "/idp/v1/products/products_link"

# IDP also exposes `/idp/v1/products/recipe` for recipe-format pages
# (ingredient list + cooking instructions). We don't expose that yet —
# add it when there's a real use case. Shopping list covers the
# "order me groceries" surface cleanly.

DEFAULT_TIMEOUT_SEC = 20.0


class InstacartError(RuntimeError):
    """Raised on any IDP API failure — auth, validation, network, etc.

    Carries the HTTP status (when available) and the response body so
    the caller can surface a useful error to the user. The tool wrapper
    catches this and converts it to a clean tool-error response.
    """

    def __init__(self, message: str, *, status: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass
class LineItem:
    """One line on the shopping list.

    `name` is the only required field. Everything else narrows the
    match (quantity/unit) or filters which products qualify (brand,
    health attributes). Instacart matches `name` against its catalog
    fuzzily — "milk" returns reasonable defaults, "Horizon Organic
    whole milk" gets you closer to a specific SKU.
    """

    name: str
    quantity: float | None = None
    unit: str | None = None  # e.g. "each", "lb", "oz", "gallon"
    display_text: str | None = None  # how the line shows in the UI; defaults to `name`
    brand_filters: list[str] = field(default_factory=list)
    # Instacart's documented health filters include ORGANIC, GLUTEN_FREE,
    # FAT_FREE, VEGAN, KOSHER, SUGAR_FREE, LOW_FAT — case matters (uppercase).
    # We don't validate values here; the API rejects bad ones with a 400 and
    # we surface that to the user. Whitelisting in code would silently drop
    # filters Instacart adds later.
    health_filters: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name}
        if self.quantity is not None:
            payload["quantity"] = self.quantity
        if self.unit:
            payload["unit"] = self.unit
        if self.display_text:
            payload["display_text"] = self.display_text
        filters: dict[str, list[str]] = {}
        if self.brand_filters:
            filters["brand_filters"] = list(self.brand_filters)
        if self.health_filters:
            filters["health_filters"] = list(self.health_filters)
        if filters:
            payload["filters"] = filters
        return payload


@dataclass
class ProductsLinkResult:
    """What we return to the tool layer (and ultimately the user)."""

    url: str
    raw: dict[str, Any]  # full IDP response, in case the tool wants more fields


class InstacartClient:
    """Thin async HTTP client around the IDP `products_link` endpoint."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = IDP_BASE_URL,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        session: aiohttp.ClientSession | None = None,
    ):
        if not api_key:
            raise ValueError("Instacart API key required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        # Optional caller-supplied session — used by tests to inject a
        # mock. In production we open a session per request, which is
        # cheap given how rarely this is called.
        self._session = session

    async def create_products_link(
        self,
        *,
        title: str,
        line_items: list[LineItem],
        image_url: str | None = None,
        expires_in_days: int | None = None,
        instructions: list[str] | None = None,
        partner_linkback_url: str | None = None,
        enable_pantry_items: bool = True,
    ) -> ProductsLinkResult:
        """Create a Shop with Instacart link from a list of items.

        Returns the URL the user taps to land in Instacart with the cart
        pre-loaded. Raises InstacartError on any failure.
        """
        if not line_items:
            raise ValueError("line_items must be non-empty")

        body: dict[str, Any] = {
            "title": title,
            "link_type": "shopping_list",
            "line_items": [item.to_payload() for item in line_items],
        }
        if image_url:
            body["image_url"] = image_url
        if expires_in_days is not None:
            body["expires_in"] = int(expires_in_days)
        if instructions:
            body["instructions"] = list(instructions)
        landing: dict[str, Any] = {"enable_pantry_items": enable_pantry_items}
        if partner_linkback_url:
            landing["partner_linkback_url"] = partner_linkback_url
        body["landing_page_configuration"] = landing

        url = f"{self.base_url}{IDP_PRODUCTS_LINK_PATH}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Use the injected session if present (tests), else a fresh one.
        # `closed` check guards against tests that pass an already-closed
        # session by accident.
        own_session = self._session is None or self._session.closed
        session = (
            aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_sec))
            if own_session
            else self._session
        )

        try:
            async with session.post(url, json=body, headers=headers) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    log.warning(
                        "Instacart IDP returned %s: %s", resp.status, text[:500]
                    )
                    raise InstacartError(
                        f"Instacart IDP {resp.status}",
                        status=resp.status,
                        body=text,
                    )
                try:
                    data = await _parse_json(resp, text)
                except ValueError as e:
                    raise InstacartError(f"Malformed IDP response: {e}", body=text) from e
        except aiohttp.ClientError as e:
            raise InstacartError(f"Network error calling Instacart IDP: {e}") from e
        finally:
            if own_session:
                await session.close()

        link_url = data.get("products_link_url")
        if not link_url or not isinstance(link_url, str):
            raise InstacartError(
                "IDP response missing 'products_link_url'", body=str(data)
            )
        return ProductsLinkResult(url=link_url, raw=data)


async def _parse_json(resp: aiohttp.ClientResponse, text: str) -> dict[str, Any]:
    """Parse JSON without trusting Content-Type.

    Some gateways strip Content-Type on edge responses. We've already
    read the body for logging; just json.loads it and let the caller
    handle malformed payloads.
    """
    import json
    if not text:
        raise ValueError("empty response body")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"not valid JSON: {e}") from e
