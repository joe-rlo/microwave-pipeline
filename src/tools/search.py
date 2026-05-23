"""Native websearch tool — pluggable backend, DuckDuckGo by default.

Phase C.5 part 2. Third SDK built-in replacement (after webfetch and
read_file). Replaces Agent SDK's WebSearch with a MicrowaveOS-owned
implementation that:

- Takes a query string, returns up to N results as title + URL + snippet
- Backend chosen at startup by `WEBSEARCH_BACKEND` env (ddg | brave |
  serper | tavily); default is `ddg`
- DDG is the only backend wired in this phase. The others are stubs
  that raise a clear error so misconfig fails loudly rather than
  silently routing to an unrelated backend

Why pluggable: the user picked DDG with "willing to switch later" as
their stated preference. Swap is a one-line env change once another
backend is implemented.

Honest caveat — DDG limitation:

DuckDuckGo doesn't have an official search API. The
`duckduckgo-search` package scrapes the DDG HTML endpoint. It can
break when DDG changes their HTML (has happened a few times). For
a personal use case the trade-off is fine; for higher reliability
swap to Brave Search API once you're willing to manage a key.

The tool is registered automatically; no API key required for DDG.
Set `WEBSEARCH_DISABLED=true` to opt out entirely.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


DEFAULT_MAX_RESULTS = 8
HARD_MAX_RESULTS = 25  # cap per call; bigger N rarely useful
DEFAULT_BACKEND = "ddg"


# --- Result type ---


@dataclass(frozen=True)
class SearchResult:
    """One search result. All three fields are best-effort strings —
    different backends populate them with different fidelity."""

    title: str
    url: str
    snippet: str

    def to_payload(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


# --- Backend protocol ---


class SearchBackend(ABC):
    """A search backend. `search()` returns at most `max_results`."""

    name: str

    @abstractmethod
    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        ...


# --- DDG backend ---


class DDGBackend(SearchBackend):
    name = "ddg"

    async def search(
        self, query: str, *, max_results: int,
    ) -> list[SearchResult]:
        # Lazy import — keeps the module loadable when duckduckgo-search
        # isn't installed; the error surfaces at first use with a
        # clear pip install hint.
        try:
            from duckduckgo_search import DDGS
        except ImportError as e:
            raise SearchToolError(
                "DDG backend requires the duckduckgo-search package. "
                "Install with `pip install duckduckgo-search`."
            ) from e

        # DDGS is synchronous; wrap in to_thread so the async event
        # loop isn't blocked. Same pattern as the Bedrock provider's
        # boto3 wrap.
        def _do_search() -> list[dict]:
            try:
                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=max_results))
            except Exception as e:
                # The package surfaces a mix of exceptions (rate limit,
                # parse errors when DDG changes HTML, network). Surface
                # them all uniformly so the LLM gets a clean error.
                raise SearchToolError(f"DDG search failed: {e}") from e

        raw = await asyncio.to_thread(_do_search)

        out: list[SearchResult] = []
        for r in raw:
            if not isinstance(r, dict):
                continue
            # duckduckgo-search returns {title, href, body}
            title = (r.get("title") or "").strip()
            url = (r.get("href") or "").strip()
            snippet = (r.get("body") or "").strip()
            if not title and not url:
                continue
            out.append(SearchResult(title=title, url=url, snippet=snippet))
        return out


# --- Stub backends (Phase C.5 ships DDG only; others land later) ---


class _UnimplementedBackend(SearchBackend):
    """Common shape for backends that need an API key + impl. Raises
    on `search()` with a message pointing the user at the env switch."""

    KEY_ENV: str = ""
    SIGNUP_URL: str = ""

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        raise SearchToolError(
            f"{self.name!r} backend is not yet implemented. To use it, "
            f"either (a) implement the backend and submit a PR, or "
            f"(b) set WEBSEARCH_BACKEND=ddg in your env. Signup info "
            f"for this provider when implementation lands: {self.SIGNUP_URL}"
        )


class BraveBackend(_UnimplementedBackend):
    name = "brave"
    KEY_ENV = "BRAVE_SEARCH_API_KEY"
    SIGNUP_URL = "https://brave.com/search/api/"


class SerperBackend(_UnimplementedBackend):
    name = "serper"
    KEY_ENV = "SERPER_API_KEY"
    SIGNUP_URL = "https://serper.dev"


class TavilyBackend(_UnimplementedBackend):
    name = "tavily"
    KEY_ENV = "TAVILY_API_KEY"
    SIGNUP_URL = "https://tavily.com"


# --- Backend selection ---


def get_backend(name: str | None = None) -> SearchBackend:
    """Resolve the backend by name or env. Unknown names raise so
    typos in WEBSEARCH_BACKEND surface immediately."""
    selected = (name or os.environ.get("WEBSEARCH_BACKEND") or DEFAULT_BACKEND).strip().lower()
    if selected == "ddg":
        return DDGBackend()
    if selected == "brave":
        return BraveBackend()
    if selected == "serper":
        return SerperBackend()
    if selected == "tavily":
        return TavilyBackend()
    raise SearchToolError(
        f"Unknown WEBSEARCH_BACKEND={selected!r}. "
        "Valid options: ddg | brave | serper | tavily"
    )


# --- Public surface ---


class SearchToolError(RuntimeError):
    """Raised on any search tool failure that should surface to the model."""


async def search(
    query: str, *, max_results: int = DEFAULT_MAX_RESULTS,
    backend: SearchBackend | None = None,
) -> list[SearchResult]:
    """Run a web search. `backend` is injectable for tests; production
    uses `get_backend()` from env."""
    if not isinstance(query, str) or not query.strip():
        raise SearchToolError("query must be a non-empty string")

    max_results = min(max(1, int(max_results)), HARD_MAX_RESULTS)
    be = backend if backend is not None else get_backend()
    return await be.search(query.strip(), max_results=max_results)


# --- Tool docs + schema ---


WEBSEARCH_TOOL_DOCS = """\
**websearch** — Search the public web and return a list of results.

When to use:
- The user asks about a current event, recent news, or anything you
  can't reason about from training alone.
- You need to find pages to fetch (then follow up with `webfetch`).
- The user explicitly says "search for X".

When NOT to use:
- For information that lives in the user's own files (use `read_file`).
- For URLs the user already gave you (use `webfetch` directly).

How to use:
- `query`: search query string. Be specific.
- `max_results`: defaults to 8, max 25. Bigger N is rarely useful.
- Returns a list of {title, url, snippet}. You can then pick the most
  relevant URL and call `webfetch` to read the full page.
"""


WEBSEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query string. Be specific.",
        },
        "max_results": {
            "type": "integer",
            "minimum": 1,
            "maximum": HARD_MAX_RESULTS,
            "description": (
                f"Max results to return. Defaults to {DEFAULT_MAX_RESULTS}. "
                "Bigger N rarely useful; pick 3–8 for most queries."
            ),
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}


# --- SDK-shape registration ---


def build_websearch_sdk_tools() -> list:
    """SdkMcpTool wrapper. Returns [] if the SDK isn't installed."""
    try:
        from claude_agent_sdk import tool
    except ImportError:
        return []

    @tool(
        name="websearch",
        description=(
            "Search the public web. Returns up to N {title, url, snippet} "
            "results. Use webfetch afterwards to read interesting pages "
            "in full."
        ),
        input_schema=WEBSEARCH_SCHEMA,
    )
    async def websearch_tool(args: dict[str, Any]) -> dict[str, Any]:
        return await _handle_websearch(args)

    return [websearch_tool]


async def _handle_websearch(args: dict[str, Any]) -> dict[str, Any]:
    """Shared handler — returns MCP shape, used by both SDK and provider paths."""
    query = args.get("query")
    max_results = args.get("max_results") or DEFAULT_MAX_RESULTS

    if not isinstance(query, str) or not query.strip():
        return _error("query must be a non-empty string")

    try:
        results = await search(query, max_results=max_results)
    except SearchToolError as e:
        log.info("websearch failed: %s", e)
        return _error(str(e))
    except Exception as e:
        log.exception("Unexpected websearch failure")
        return _error(f"Unexpected error: {e}")

    payload = {
        "query": query,
        "backend": get_backend().name,
        "count": len(results),
        "results": [r.to_payload() for r in results],
    }
    return {
        "content": [{"type": "text", "text": json.dumps(payload)}],
    }


def _error(message: str) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": message}],
        "is_error": True,
    }


def websearch_disabled() -> bool:
    """True when WEBSEARCH_DISABLED env flag is set."""
    return os.environ.get("WEBSEARCH_DISABLED", "").strip().lower() in (
        "1", "true", "yes", "on",
    )
